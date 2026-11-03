"""Exercise the Python-free downloader without network or large weight files."""

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "download_models.sh"
PAYLOAD = b"test weights\n"


@pytest.fixture(params=["wget", "curl"])
def downloader(tmp_path, request):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("mkdir", "sha256sum", "mv", "ln"):
        (bindir / name).symlink_to(shutil.which(name))
    transport = bindir / request.param
    transport.write_text('''#!/bin/bash
while [[ $# -gt 0 ]]; do
    case "$1" in
        -O|-o) dest="$2"; shift 2 ;;
        -*) shift ;;
        *) url="$1"; shift ;;
    esac
done
printf '%s\n' "$url" >> "$CALL_LOG"
if [[ "$url" == *deit_base* && "${FAIL_MODE:-}" != "" ]]; then
    printf 'incomplete' > "$dest"
    [[ "$FAIL_MODE" == checksum ]] && exit 0
    exit 1
fi
printf 'test weights\n' > "$dest"
''')
    transport.chmod(0o755)
    # Fixture data replaces only content hashes; URLs, revisions and cache logic
    # are exercised by running the complete production shell script.
    script = tmp_path / "download_models.sh"
    script.write_text(re.sub(r'"[0-9a-f]{64}"',
                             f'"{hashlib.sha256(PAYLOAD).hexdigest()}"',
                             SCRIPT.read_text()))
    env = {key: value for key, value in os.environ.items()
           if key not in ("TORCH_HOME", "HF_HOME", "HF_HUB_CACHE", "FAIL_MODE")}
    env.update(PATH=str(bindir), MODEL_CACHE=str(tmp_path / "cache"),
               CALL_LOG=str(tmp_path / "calls"))

    def run(**overrides):
        current = env | overrides
        hub = Path(current.get("TORCH_HOME", tmp_path / "cache/torch")) / "hub"
        (hub / "facebookresearch_dinov2_main").mkdir(parents=True, exist_ok=True)
        return subprocess.run(["/bin/bash", str(script)], env=current,
                              capture_output=True, text=True)

    return run, tmp_path


def test_cache_layout_reuse_and_corrupt_file_repair(downloader):
    run, root = downloader
    result = run()
    assert result.returncode == 0, result.stdout + result.stderr
    hub = root / "cache/huggingface/hub"
    repos = list(hub.glob("models--*"))
    assert len(repos) == 4
    for repo in repos:
        revision = (repo / "refs/main").read_text()
        weights = list((repo / "snapshots" / revision).glob("*.safetensors"))
        assert len(weights) == 1
        assert weights[0].is_symlink()
        assert weights[0].read_bytes() == PAYLOAD
    calls = (root / "calls").read_text()
    assert len(calls.splitlines()) == 6  # 4 HF models + FID + DINOv2
    assert run().returncode == 0
    assert (root / "calls").read_text() == calls
    blob = next(repos[0].joinpath("blobs").iterdir())
    blob.write_bytes(b"truncated")
    assert run().returncode == 0
    assert blob.read_bytes() == PAYLOAD
    assert len((root / "calls").read_text().splitlines()) == 7


@pytest.mark.parametrize("mode", ["transfer", "checksum"])
def test_failed_download_is_not_published_and_can_retry(downloader, mode):
    run, root = downloader
    result = run(FAIL_MODE=mode)
    assert result.returncode != 0
    repo = root / "cache/huggingface/hub/models--timm--deit_base_distilled_patch16_224.fb_in1k"
    assert not (repo / "refs/main").exists()
    assert not list((repo / "snapshots").rglob("*.safetensors"))
    assert run().returncode == 0
    assert (repo / "refs/main").exists()


def test_explicit_library_cache_overrides(downloader):
    run, root = downloader
    overrides = dict(TORCH_HOME=str(root / "torch override"),
                     HF_HOME=str(root / "hf home"))
    assert run(**overrides).returncode == 0
    assert (root / "torch override/hub/checkpoints").is_dir()
    assert len(list((root / "hf home/hub").glob("models--*"))) == 4
    overrides["HF_HUB_CACHE"] = str(root / "hub override")
    assert run(**overrides).returncode == 0
    assert len(list((root / "hub override").glob("models--*"))) == 4
