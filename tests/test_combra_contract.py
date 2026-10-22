"""Assert every combra symbol this repo imports actually exists.

The eval path is deliberately fault-tolerant: a missing or broken combra must
never abort a training run. That tolerance is also how a real breakage hid for a
whole release. combra 0.5.0 removed ``angle_density_metrics_from_pooled``,
``fid_from_features`` and ``fd_dinov2_from_features`` and renamed
``combra_smoke_test``; this repo kept importing all four, the ``except`` around
the eval swallowed the ImportError every tick, and ``--combra-metrics true``
quietly produced nothing for months.

Nothing caught it because nothing asserted the symbols. This does — CPU-only, no
GPU, no dataset, no network, so it runs in every CI job. It is skipped only when
combra is genuinely absent, which is a different (and visible) condition.
"""

import importlib.util
import os

import pytest

# Every (module, name) this repo imports from combra, in one place. Adding a
# combra call anywhere in the repo means adding its name here. The training loop
# reaches the feature / angle / Gaussian-fit functions only through
# combra.metrics.distributed, so those four symbols are the load-bearing ones.
REQUIRED = [
    ("combra.metrics.distributed", "all_ranks_ok"),
    ("combra.metrics.distributed", "distributed_metrics"),
    ("combra.metrics.distributed", "gather_generated"),
    ("combra.metrics.distributed", "precompute_reference"),
    ("combra.metrics", "require_backbones"),  # the first check in train.py
    ("combra.io", "write_hparams"),
    ("combra", "__version__"),  # the [startup] line in train.py
]

combra_installed = importlib.util.find_spec("combra") is not None
requires_combra = pytest.mark.skipif(not combra_installed, reason="combra is not installed")


def test_combra_is_installed_when_required():
    """CI sets COMBRA_REQUIRED=1 once it has installed combra; from then on an
    absent combra is a FAILURE, not a skip. Every test below is skipif-guarded, so
    without this one the whole file can go green by doing nothing -- which is the
    exact failure mode it exists to prevent."""
    if os.environ.get("COMBRA_REQUIRED") == "1":
        assert combra_installed, "COMBRA_REQUIRED=1 but combra is not importable"


@requires_combra
@pytest.mark.parametrize("module, name", REQUIRED)
def test_combra_exports_symbol(module, name):
    mod = importlib.import_module(module)

    assert hasattr(mod, name), (
        f"{module}.{name} is missing. This repo imports it; without it the combra "
        "metrics silently disappear. Check combra's CHANGELOG for a rename."
    )


@requires_combra
def test_combra_import_block_resolves():
    # The exact imports the training loop performs. Guarded there, unguarded here.
    from combra.io import write_hparams  # noqa: F401
    from combra.metrics.distributed import (  # noqa: F401
        all_ranks_ok,
        distributed_metrics,
        gather_generated,
        precompute_reference,
    )


# The keys combra >= 0.25 returns from distributed_metrics, which the loop logs as
# combra_<key>: v1 (P0 angles) has two fitted amplitudes, v2 (P6 angles) one mass
# share (pi); then the image metrics.
WASS_KEYS = ("w1", "w2", "circular_w1", "circular_w2")
V1_KEYS = (*WASS_KEYS, "mu1", "mu2", "sigma1", "sigma2", "amp1", "amp2")
V2_KEYS = (*WASS_KEYS, "mu1", "mu2", "sigma1", "sigma2", "pi")
EXPECTED_KEYS = ({f"v1_{k}" for k in V1_KEYS} | {f"v2_{k}" for k in V2_KEYS}
                 | {"fid", "cmmd", "fd_dinov2"})


@requires_combra
def test_distributed_metrics_returns_the_logged_keys():
    # Not just importable -- callable on the reference / gathered shapes the loop
    # hands it, and returning the keys the loop logs.
    import numpy as np
    from combra.metrics.distributed import distributed_metrics

    # The sample must be genuinely BIMODAL. These are WC-Co vertex angles -- a
    # convex mode and a reflex one -- and the gauss half of this metric fits two
    # Gaussians to them. A single normal leaves the second Gaussian with nothing to
    # sit on, so combra reports the relative errors as nan rather than dividing by a
    # phantom mode.
    rng = np.random.default_rng(0)

    def angles(mu1, sigma1, mu2, sigma2, n=4000, share=0.7):
        k = int(n * share)
        both = [rng.normal(mu1, sigma1, k), rng.normal(mu2, sigma2, n - k)]
        return (np.concatenate(both) % 360).astype(np.float32)

    def feats():
        return {name: rng.normal(size=(16, 8)).astype(np.float32) for name in ("fid", "cmmd", "fd_dinov2")}

    ref_angles = angles(100, 20, 240, 25)
    gen_angles = angles(104, 21, 236, 26)
    reference = {"angles": {"v1": ref_angles, "v2": ref_angles}, "feat": feats()}
    out = distributed_metrics(reference, {"v1": gen_angles, "v2": gen_angles}, feats(), device="cpu")

    assert set(out) == EXPECTED_KEYS
    for key in EXPECTED_KEYS:
        assert np.isfinite(out[key]), f"{key} is not finite"


@requires_combra
def test_logged_keys_map_to_three_sections():
    # combra_<key> -> Metrics/ (image metrics), Metrics_v1/ (ten), Metrics_v2/ (nine).
    pytest.importorskip("torch")
    from training.training_loop import metric_tag

    tags = [metric_tag(f"combra_{key}") for key in EXPECTED_KEYS]
    sections = {}
    for tag in tags:
        section, name = tag.split("/")
        sections.setdefault(section, set()).add(name)
    assert sections == {
        "Metrics": {"combra_fid", "combra_cmmd", "combra_fd_dinov2"},
        "Metrics_v1": {f"combra_{k}" for k in V1_KEYS},
        "Metrics_v2": {f"combra_{k}" for k in V2_KEYS},
    }


@requires_combra
def test_write_hparams_accepts_step():
    # The loop passes step=cur_nimg so the HPARAMS land in the run's own event file
    # (combra >= 0.15.3); an older combra raises TypeError at the end of every run.
    import inspect

    from combra.io import write_hparams

    assert "step" in inspect.signature(write_hparams).parameters


@requires_combra
def test_precompute_reference_accepts_dihedral():
    # --augment passes dihedral= so the reference covers all 8 dihedral transforms
    # (combra >= 0.19.0); an older combra raises TypeError and the metrics disappear.
    import inspect

    from combra.metrics.distributed import precompute_reference

    assert "dihedral" in inspect.signature(precompute_reference).parameters


def test_no_startup_smoke_test():
    # v0.8.1 removed the rank-0 startup self_test: it loaded the backbones on rank 0
    # alone while the other ranks waited in a collective, and a stalled load on an
    # offline node tripped the NCCL watchdog. The reference precompute loads them on
    # every rank together; nothing else may call self_test at startup.
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "training/training_loop.py"
    calls = [n for n in ast.walk(ast.parse(src.read_text()))
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "self_test"]
    assert calls == []


@requires_combra
def test_startup_line_records_combra_version():
    # The [startup] line names the combra version right after cuda (all four model
    # repos), so a run killed before the end-of-run HPARAMS still records it.
    pytest.importorskip("torch")
    import combra

    from train import _startup_header

    parts = _startup_header(1).split(" | ")
    assert parts[0].startswith("torch ") and parts[1].startswith("cuda ")
    assert parts[2] == f"combra {combra.__version__}" and parts[3] == "gpus 1"
