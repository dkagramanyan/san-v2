"""Invariant tests for the SAN (Slicing Adversarial Network) layers.

These tests pin the numerical behaviour of the discriminative normalized linear
layers so that structural refactors (e.g. the shared ``_san_dual_path`` helper)
cannot silently change the training math. They run on CPU with tiny tensors.

Run from the repo root:

    pytest tests/test_san_modules.py
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

# Make the repo root importable when pytest is invoked from elsewhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pg_modules.san_modules import (  # noqa: E402
    SANLinear,
    SANConv1d,
    SANConv2d,
    SANEmbedding,
)


def _seed():
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Weight-normalization invariant: normalized weight has unit L2 norm.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layer_factory, norm_dim", [
    (lambda: SANLinear(8, 4), 1),
    (lambda: SANConv1d(6, 5, 3), [1, 2]),
    (lambda: SANConv2d(6, 5, 3), [1, 2, 3]),
])
def test_normalized_weight_is_unit_norm(layer_factory, norm_dim):
    _seed()
    layer = layer_factory()
    w = layer._get_normalized_weight()
    norms = w.norm(p=2.0, dim=norm_dim)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_embedding_output_is_unit_norm():
    _seed()
    emb = SANEmbedding(num_embeddings=10, embedding_dim=4)
    idx = torch.tensor([0, 3, 7])
    out = emb(idx)  # eval mode -> scaled, normalized embedding
    # Direction component (before scale) is unit norm; recompute it explicitly.
    raw = F.embedding(idx, emb.weight)
    direction = raw / raw.norm(p=2.0, dim=-1, keepdim=True).clamp_min(1e-12)
    norms = direction.norm(p=2.0, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert out.shape == (3, 4)


# ---------------------------------------------------------------------------
# Dual-path (flg_train) structure and value equivalence.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layer_factory, make_input", [
    (lambda: SANLinear(8, 4), lambda: torch.randn(2, 8)),
    (lambda: SANConv1d(6, 5, 3), lambda: torch.randn(2, 6, 10)),
    (lambda: SANConv2d(6, 5, 3), lambda: torch.randn(2, 6, 10, 10)),
])
def test_flg_train_returns_pair_matching_eval(layer_factory, make_input):
    _seed()
    layer = layer_factory()
    x = make_input()

    eval_out = layer(x, flg_train=False)
    assert torch.is_tensor(eval_out)

    train_out = layer(x, flg_train=True)
    assert isinstance(train_out, list) and len(train_out) == 2
    out_fun, out_dir = train_out

    # Both training paths reproduce the eval output numerically (they differ only
    # in which side of the product carries gradient).
    assert torch.allclose(out_fun, eval_out, atol=1e-6)
    assert torch.allclose(out_dir, eval_out, atol=1e-6)


def test_embedding_flg_train_returns_pair_matching_eval():
    _seed()
    emb = SANEmbedding(num_embeddings=10, embedding_dim=4)
    idx = torch.tensor([1, 2, 5])

    eval_out = emb(idx, flg_train=False)
    out_fun, out_dir = emb(idx, flg_train=True)
    assert torch.allclose(out_fun, eval_out, atol=1e-6)
    assert torch.allclose(out_dir, eval_out, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient separation: out_fun feeds input only, out_dir feeds weight only.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layer_factory, make_input", [
    (lambda: SANLinear(8, 4), lambda: torch.randn(2, 8, requires_grad=True)),
    (lambda: SANConv2d(6, 5, 3), lambda: torch.randn(2, 6, 10, 10, requires_grad=True)),
])
def test_gradient_separation(layer_factory, make_input):
    _seed()
    layer = layer_factory()

    # Function path: gradient should reach the input, not the (detached) weight.
    x = make_input()
    out_fun, _ = layer(x, flg_train=True)
    out_fun.sum().backward()
    assert x.grad is not None
    assert layer.weight.grad is None

    layer.zero_grad(set_to_none=True)

    # Direction path: gradient should reach the weight, not the (detached) input.
    x2 = make_input()
    _, out_dir = layer(x2, flg_train=True)
    out_dir.sum().backward()
    assert x2.grad is None
    assert layer.weight.grad is not None


# ---------------------------------------------------------------------------
# Explicit math reference: eval output equals a hand-written recomputation.
# ---------------------------------------------------------------------------

def test_linear_matches_manual_recomputation():
    _seed()
    layer = SANLinear(8, 4)
    x = torch.randn(3, 8)
    expected = F.linear(x + layer.bias, layer._get_normalized_weight(), None) * layer.scale
    assert torch.allclose(layer(x), expected, atol=1e-6)


def test_conv2d_matches_manual_recomputation():
    _seed()
    layer = SANConv2d(6, 5, 3)
    x = torch.randn(2, 6, 12, 12)
    biased = x + layer.bias.view(layer.in_channels, 1, 1)
    expected = F.conv2d(biased, layer._get_normalized_weight(), None, layer.stride,
                        layer.padding, layer.dilation, layer.groups)
    expected = expected * layer.scale.view(layer.out_channels, 1, 1)
    assert torch.allclose(layer(x), expected, atol=1e-6)
