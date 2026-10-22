"""The backbone-weight check is the first thing a training start does.

combra.metrics.require_backbones() checks the local caches for the weights
download_models.sh fetches. train.py calls it before the dataset is read and before
any process is spawned, whenever the combra metrics will run; a missing file exits
non-zero. CPU-only, no dataset, no network.
"""

import importlib.util

import pytest
from click.testing import CliRunner

pytestmark = pytest.mark.skipif(importlib.util.find_spec("combra") is None,
                                reason="combra is not installed")

ARGV = ["--outdir", "out", "--cfg", "stylegan3-r", "--gpus", "1", "--batch-gpu", "2",
        "--data", "missing.zip"]


def _patch(monkeypatch, require_backbones):
    import combra.metrics

    import train
    calls = []
    monkeypatch.setattr(combra.metrics, "require_backbones", require_backbones)

    def build_config(opts):
        calls.append("build_config")
        return {}, "desc"

    monkeypatch.setattr(train, "build_config", build_config)
    monkeypatch.setattr(train, "launch_training", lambda **kw: calls.append("launch_training"))
    return train, calls


def test_missing_backbones_abort_before_anything(monkeypatch):
    def require_backbones():
        raise RuntimeError("metric backbone weights are not downloaded -- run the model "
                           "repo's download_models.sh first. Missing:\n  fid: x")

    train, calls = _patch(monkeypatch, require_backbones)
    result = CliRunner().invoke(train.main, ARGV)
    assert result.exit_code != 0
    assert "download_models.sh" in result.output
    assert calls == []  # no dataset read, no config, no launch


@pytest.mark.parametrize("args, checked", [
    ([], True),
    (["--combra-metrics", "False"], False),
    (["--num-fid-samples", "0"], False),
])
def test_check_runs_only_when_metrics_run(monkeypatch, args, checked):
    checks = []

    def require_backbones():
        checks.append(1)
        return {"fid": "a", "cmmd": "b", "fd_dinov2": "c"}

    train, calls = _patch(monkeypatch, require_backbones)
    result = CliRunner().invoke(train.main, ARGV + args)
    assert result.exit_code == 0, result.output
    assert len(checks) == int(checked)
    assert calls == ["build_config", "launch_training"]
    found = "Backbone weights found (download_models.sh): fid, cmmd, fd_dinov2"
    assert (found in result.output) is checked
