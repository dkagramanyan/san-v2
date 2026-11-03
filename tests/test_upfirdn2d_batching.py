"""Large resampling batches must obey the CUDA plugin's 32-bit index limit.

Use a small limit and the reference operator as a CPU plugin stand-in so CI can
exercise oversized intermediate tensors, gradients, and strided inputs cheaply.
"""

from types import SimpleNamespace

import pytest
import torch

from torch_utils.ops import upfirdn2d


@pytest.fixture
def plugin(monkeypatch):
    calls = []

    def run(x, f, upx, upy, downx, downy, px0, px1, py0, py1, flip, gain):
        limit = upfirdn2d._CUDA_INDEX_LIMIT
        assert x.numel() <= limit
        assert sum((n - 1) * s for n, s in zip(x.shape, x.stride())) <= limit
        y = upfirdn2d._upfirdn2d_ref(
            x, f, up=(upx, upy), down=(downx, downy),
            padding=(px0, px1, py0, py1), flip_filter=flip, gain=gain,
        )
        assert y.numel() <= limit
        calls.append(x.shape[0])
        return y

    monkeypatch.setattr(upfirdn2d, "_plugin", SimpleNamespace(upfirdn2d=run))
    monkeypatch.setattr(upfirdn2d, "_CUDA_INDEX_LIMIT", 400)
    return calls


@pytest.mark.parametrize("separable", [False, True])
@pytest.mark.parametrize("flip", [False, True])
def test_oversized_resampling_and_gradients_match_reference(plugin, separable, flip):
    torch.manual_seed(42)
    x = torch.randn(5, 2, 4, 4, dtype=torch.float64, requires_grad=True)
    f = torch.tensor([1, 2, 3], dtype=torch.float32) / 6
    if not separable:
        f = f.outer(f)
    kwargs = dict(up=3, down=2, padding=(2, 1, 1, 2), gain=1.3, flip_filter=flip)
    actual = upfirdn2d._upfirdn2d_cuda(**kwargs).apply(x, f)
    expected = upfirdn2d._upfirdn2d_ref(x, f, **kwargs)
    torch.testing.assert_close(actual, expected)
    assert len(plugin) > (2 if separable else 1)

    grad = torch.autograd.grad(actual.square().sum(), x, create_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), x, create_graph=True)[0]
    torch.testing.assert_close(grad, expected_grad)
    second = torch.autograd.grad(grad.sum(), x)[0]
    expected_second = torch.autograd.grad(expected_grad.sum(), x)[0]
    torch.testing.assert_close(second, expected_second)


def test_separable_intermediate_is_checked_even_when_final_output_fits(plugin):
    x = torch.randn(5, 2, 4, 4)
    f = torch.ones(3) / 3
    kwargs = dict(up=(4, 1), down=(1, 4), padding=1)
    actual = upfirdn2d._upfirdn2d_cuda(**kwargs).apply(x, f)
    expected = upfirdn2d._upfirdn2d_ref(x, f, **kwargs)
    assert x.numel() <= 400 and expected.numel() <= 400
    assert len(plugin) > 2
    torch.testing.assert_close(actual, expected)


def test_oversized_strided_footprint_is_compacted(plugin):
    x = torch.randn(5, 2, 4, 40)[..., ::10]
    assert x.numel() < 400
    actual = upfirdn2d._upfirdn2d_cuda().apply(x, None)
    torch.testing.assert_close(actual, x)


def test_small_batch_uses_one_call(plugin):
    x = torch.randn(2, 2, 4, 4)
    actual = upfirdn2d._upfirdn2d_cuda().apply(x, None)
    assert plugin == [2]
    torch.testing.assert_close(actual, x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_cuda_plugin_chunking_matches_reference(monkeypatch):
    upfirdn2d._init()
    monkeypatch.setattr(upfirdn2d, "_CUDA_INDEX_LIMIT", 400)
    x = torch.randn(5, 2, 4, 4, device="cuda", requires_grad=True)
    f = torch.ones(3, device="cuda") / 3
    kwargs = dict(up=3, down=2, padding=1)
    actual = upfirdn2d.upfirdn2d(x, f, **kwargs)
    expected = upfirdn2d.upfirdn2d(x, f, impl="ref", **kwargs)
    torch.testing.assert_close(actual, expected)
    grad = torch.autograd.grad(actual.square().sum(), x)[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), x)[0]
    torch.testing.assert_close(grad, expected_grad)
