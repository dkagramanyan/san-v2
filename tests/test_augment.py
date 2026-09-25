"""--augment: per-item dihedral transform of the raw uint8 training reals.

CPU-only. Covers the transform itself (only the 8 dihedral elements, roughly
uniform, deterministic, dtype/shape/order preserved), the CLI wiring, and that the
combra reference is asked for the dihedral expansion exactly when --augment is on.
"""

import importlib.util
import sys
import types

import numpy as np
import pytest
import torch

from training.training_loop import _combra_precompute_reference, dihedral_augment


def _dihedral(x):
    """The 8 dihedral transforms of one CHW image, in dihedral_augment's code order."""
    rots = [torch.rot90(x, k, dims=(-2, -1)) for k in range(4)]
    return rots + [r.flip(-1) for r in rots]


def _batch(n, res=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, [n, 3, res, res], generator=g, dtype=torch.uint8)


def _which(src, out):
    """Index of the dihedral transform that maps src to out (asserting exactly one)."""
    hits = [i for i, t in enumerate(_dihedral(src)) if torch.equal(t, out)]
    assert len(hits) == 1, "output is not a (unique) dihedral transform of its input"
    return hits[0]


def test_only_dihedral_transforms_and_roughly_uniform():
    n = 8000
    x = _batch(n)
    y = dihedral_augment(x, torch.Generator().manual_seed(0))
    assert y.dtype == torch.uint8 and y.shape == x.shape
    # Item order is preserved (item i is a transform of input i), so the batch labels
    # drawn alongside it stay aligned.
    counts = np.bincount([_which(x[i], y[i]) for i in range(n)], minlength=8)
    assert counts.sum() == n
    # Expected 1000 each; binomial sd ~ 30, so +-150 is a 5-sigma band.
    assert np.all(np.abs(counts - n / 8) < 150), counts


def test_deterministic_per_seed_and_input_untouched():
    x = _batch(64)
    x0 = x.clone()
    a = dihedral_augment(x, torch.Generator().manual_seed(42))
    b = dihedral_augment(x, torch.Generator().manual_seed(42))
    c = dihedral_augment(x, torch.Generator().manual_seed(43))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert torch.equal(x, x0)


def test_refuses_non_square():
    with pytest.raises(AssertionError, match="square"):
        dihedral_augment(torch.zeros([2, 3, 8, 16], dtype=torch.uint8), torch.Generator())


def test_training_loop_gates_augmentation_on_the_flag():
    # The fetch path calls dihedral_augment only under `if augment:`; the grid,
    # reference and eval loaders read the dataset directly.
    import inspect

    from training import training_loop as tl
    src = inspect.getsource(tl.training_loop)
    assert "if augment:\n                phase_real_img = dihedral_augment(" in src
    assert src.count("dihedral_augment(") == 1
    assert inspect.signature(tl.training_loop).parameters["augment"].default is True


@pytest.mark.skipif(importlib.util.find_spec("combra") is None, reason="combra is not installed")
@pytest.mark.parametrize("augment", [True, False])
def test_reference_call_passes_dihedral(monkeypatch, augment):
    calls = []
    fake = types.ModuleType("combra.metrics.distributed")
    fake.all_ranks_ok = lambda ok, device, n: ok

    def precompute_reference(images_u8, device, rank, world_size, *, amp=False, dihedral=False):
        calls.append(dict(n=len(images_u8), rank=rank, dihedral=dihedral, dtype=images_u8.dtype))
        return {"ok": True}, True

    fake.precompute_reference = precompute_reference
    monkeypatch.setitem(sys.modules, "combra.metrics.distributed", fake)
    x = _batch(10).numpy()
    dataset = [(x[i], np.zeros([3], np.float32)) for i in range(len(x))]
    ref, ok = _combra_precompute_reference(dataset, "cpu", 1, 2, dihedral=augment)
    assert ok and ref == {"ok": True}
    # The reference slice is the raw originals (rank 1 of 2 -> 5 images); combra does
    # the x8 expansion itself.
    assert calls == [dict(n=5, rank=1, dihedral=augment, dtype=np.uint8)]
