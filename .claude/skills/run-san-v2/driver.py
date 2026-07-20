#!/usr/bin/env python
"""Smoke driver for san-v2 (StyleSAN-XL GAN training/generation repo).

This repo's "app" is four click CLIs (train / gen_images / dataset_tool /
calc_metrics) plus the StyleGAN3 model stack with custom CUDA ops. Real
training/generation needs multi-GB dataset zips + trained checkpoints that live
on the H200 cluster, not in a clean checkout. This driver instead exercises
every surface that IS reachable in a bare container + single GPU:

  cli      import + --help contract for the 3 importable CLIs
  test     the CPU pytest suite (tests/)
  dataset  REAL end-to-end: synth tiny labeled images -> `dataset_tool convert`
           -> verify the produced .zip + dataset.json labels/class_names
  gen      GPU: build a tiny conditional StyleGAN3 Generator (no checkpoint
           needed), run a forward pass through the custom CUDA ops, save a PNG
  all      everything above (default)

Run with the `san` conda env's interpreter:
  /home/david/anaconda3/envs/san/bin/python .claude/skills/run-san-v2/driver.py all

Artifacts (dataset zip, generated PNG) land in --workdir (default: a temp dir;
path is printed). Exit code is nonzero if any selected check fails.
"""
import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def find_repo_root() -> Path:
    """Walk up from this file until we find the repo (has train.py + gen_images.py)."""
    for p in [Path(__file__).resolve()] + list(Path(__file__).resolve().parents):
        d = p if p.is_dir() else p.parent
        if (d / "train.py").exists() and (d / "gen_images.py").exists():
            return d
    # Fallback: env override or CWD.
    env = os.environ.get("SAN_V2_ROOT")
    if env and (Path(env) / "train.py").exists():
        return Path(env)
    raise SystemExit("Could not locate san-v2 repo root (no train.py found). Set SAN_V2_ROOT.")


REPO = find_repo_root()
# The custom CUDA ops JIT-link via `ninja`, which torch invokes through PATH.
# When you call the env's python by absolute path, its bin/ is NOT on PATH, so
# add it (that's where the env's ninja lives).
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

OK = "\033[32m[ OK ]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


def run_py(args, timeout=600):
    """Run `python <args>` from the repo root; return (rc, stdout+stderr)."""
    p = subprocess.run(
        [sys.executable, *args], cwd=REPO, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout,
    )
    return p.returncode, p.stdout


def check_cli() -> bool:
    """--help must exit 0 and expose the flags the smoke tests contract on."""
    expect = {
        "train.py": ["--precision", "--tf32", "--num-fid-samples", "--path-stem", "--syn-layers"],
        "gen_images.py": ["--network", "--classes", "--samples-per-class", "--save-mode"],
        "dataset_tool.py": ["convert"],  # click group; `convert` is the subcommand
    }
    ok = True
    for script, flags in expect.items():
        rc, out = run_py([script, "--help"], timeout=180)
        missing = [f for f in flags if f not in out]
        if rc == 0 and not missing:
            print(f"{OK} {script} --help  (flags present)")
        else:
            ok = False
            print(f"{FAIL} {script} --help  rc={rc} missing={missing}")
            print("   " + "\n   ".join(out.strip().splitlines()[-6:]))
    # calc_metrics.py is known to fail at import (missing CLIP vocab .gz) -- see SKILL Gotchas.
    return ok


def check_test() -> bool:
    rc, out = run_py(["-m", "pytest", "tests/", "-q"], timeout=600)
    tail = out.strip().splitlines()[-1] if out.strip() else "(no output)"
    print(f"{OK if rc == 0 else FAIL} pytest tests/  -> {tail}")
    if rc != 0:
        print("   " + "\n   ".join(out.strip().splitlines()[-15:]))
    return rc == 0


def check_dataset(workdir: Path) -> bool:
    """Synthesize a tiny labeled folder and pack it into a resolution zip."""
    import numpy as np
    from PIL import Image

    raw = workdir / "raw_ds"
    classes = ["Ultra_Co11", "Ultra_Co25", "Ultra_Co6_2"]  # the 3 grain classes
    rng = np.random.default_rng(0)
    for cls in classes:
        (raw / cls).mkdir(parents=True, exist_ok=True)
        for i in range(4):
            arr = (rng.random((96, 96, 3)) * 255).astype("uint8")
            Image.fromarray(arr).save(raw / cls / f"img{i}.png")

    dest = workdir / "ds_64x64.zip"
    if dest.exists():
        dest.unlink()
    rc, out = run_py(
        ["dataset_tool.py", "convert",
         f"--source={raw}", f"--dest={dest}", "--resolution=64x64"],
        timeout=300,
    )
    if rc != 0 or not dest.exists():
        print(f"{FAIL} dataset_tool convert  rc={rc}")
        print("   " + "\n   ".join(out.strip().splitlines()[-8:]))
        return False

    import json
    with zipfile.ZipFile(dest) as z:
        names = z.namelist()
        has_json = "dataset.json" in names
        meta = json.loads(z.read("dataset.json")) if has_json else {}
    n_imgs = sum(n.endswith(".png") for n in names)
    good = (
        has_json
        and n_imgs == 12
        and meta.get("class_names") == sorted(classes)  # labels follow sorted()
        and len(meta.get("labels", [])) == 12
    )
    print(f"{OK if good else FAIL} dataset_tool convert -> {dest.name} "
          f"({n_imgs} imgs, class_names={meta.get('class_names')})")
    return good


def check_gen(workdir: Path) -> bool:
    """Build a tiny conditional StyleGAN3 G (no checkpoint) and generate a PNG on GPU."""
    import pickle

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        print(f"{FAIL} gen  -> no CUDA device (generation requires a GPU)")
        return False

    sys.path.insert(0, str(REPO))
    import dnnlib  # noqa: E402
    from training.networks_stylegan3_resetting import Generator  # noqa: E402

    # The MappingNetwork loads pretrained ImageNet class embeddings from
    # in_embeddings/tf_efficientnet_lite0.pkl (absent in a clean checkout).
    # Point SAN_EMBED at a tiny synthetic table so the tiny G is self-contained.
    embed_path = workdir / "embed_tiny.pkl"
    with open(embed_path, "wb") as f:
        pickle.dump({"embed": torch.nn.Embedding(3, 128)}, f)
    os.environ["SAN_EMBED"] = str(embed_path)

    device = torch.device("cuda")
    # 16x16 stem config, conditional (3 classes), stylegan3-r (conv_kernel=1, radial).
    G = Generator(
        z_dim=64, c_dim=3, w_dim=512, img_resolution=16, img_channels=3,
        channel_base=32768 * 2, channel_max=512 * 2, num_layers=6,
        conv_kernel=1, use_radial_filters=True, magnitude_ema_beta=0.999,
        num_fp16_res=0, conv_clamp=None,
        mapping_kwargs=dnnlib.EasyDict(rand_embedding=False, num_layers=2),
    ).eval().requires_grad_(False).to(device)

    z = torch.randn(3, 64, device=device)
    c = torch.eye(3, device=device)  # one sample per class
    with torch.inference_mode():
        img = G(z, c, truncation_psi=0.7)  # triggers custom-op JIT/cache load

    from PIL import Image
    arr = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    sheet = np.concatenate(list(arr), axis=1)
    out_png = workdir / "gen_forward.png"
    Image.fromarray(sheet).resize((16 * 3 * 8, 16 * 8), Image.NEAREST).save(out_png)

    good = tuple(img.shape) == (3, 3, 16, 16) and out_png.exists()
    params = sum(p.numel() for p in G.parameters()) / 1e6
    print(f"{OK if good else FAIL} gen forward on {torch.cuda.get_device_name(0)} "
          f"({params:.1f}M params) -> {out_png}")
    return good


def main():
    ap = argparse.ArgumentParser(description="san-v2 smoke driver")
    ap.add_argument("cmd", nargs="?", default="all",
                    choices=["all", "cli", "test", "dataset", "gen"])
    ap.add_argument("--workdir", default=None,
                    help="where artifacts land (default: a temp dir)")
    args = ap.parse_args()

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="san_v2_smoke_"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"repo: {REPO}")
    print(f"python: {sys.executable}")
    print(f"workdir: {workdir}\n")

    results = {}
    if args.cmd in ("all", "cli"):
        results["cli"] = check_cli()
    if args.cmd in ("all", "test"):
        results["test"] = check_test()
    if args.cmd in ("all", "dataset"):
        results["dataset"] = check_dataset(workdir)
    if args.cmd in ("all", "gen"):
        results["gen"] = check_gen(workdir)

    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  {k:8} {'PASS' if v else 'FAIL'}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
