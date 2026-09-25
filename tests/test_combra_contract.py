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
    ("combra.metrics", "self_test"),
    ("combra.io", "write_hparams"),
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
    from combra.metrics import self_test  # noqa: F401
    from combra.metrics.distributed import (  # noqa: F401
        all_ranks_ok,
        distributed_metrics,
        gather_generated,
        precompute_reference,
    )


@requires_combra
def test_angle_metrics_run_on_pooled_angles():
    # Not just importable -- callable, and returning the keys the loop logs.
    import numpy as np
    from combra.metrics import angle_density_metrics_from_pooled

    # The sample must be genuinely BIMODAL. These are WC-Co vertex angles -- a
    # convex mode and a reflex one -- and the gauss half of this metric fits two
    # Gaussians to them. A single normal (which is what this fixture used to pass)
    # leaves the second Gaussian with nothing to sit on, so combra reports the
    # relative errors as nan rather than dividing by a phantom mode.
    rng = np.random.default_rng(0)

    def angles(mu1, sigma1, mu2, sigma2, n=4000, share=0.7):
        k = int(n * share)
        both = [rng.normal(mu1, sigma1, k), rng.normal(mu2, sigma2, n - k)]
        return np.concatenate(both) % 360

    out = angle_density_metrics_from_pooled(
        angles(100, 20, 240, 25), angles(104, 21, 236, 26)
    )
    for key in ("w1", "w2", "circular_w1", "circular_w2", "mu1", "sigma1", "pi"):
        assert np.isfinite(out[key]), f"{key} is not finite"


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


def test_startup_smoke_test_uses_synthetic_grains():
    # The startup self_test must probe the backends on combra's synthetic grains:
    # with images=<4 training images> it found no angles at 16^2 and a degenerate
    # bimodal fit at 64^2, so strict=True aborted every low-res run before tick 0.
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "training/training_loop.py"
    calls = [n for n in ast.walk(ast.parse(src.read_text()))
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "self_test"]
    assert len(calls) == 1
    kwargs = {k.arg: k.value for k in calls[0].keywords}
    assert "images" not in kwargs and not calls[0].args
    assert kwargs["strict"].value is True and kwargs["image_metrics"].value is True
