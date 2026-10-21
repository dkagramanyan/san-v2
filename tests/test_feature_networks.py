"""The projected discriminator must build on the installed timm.

``_make_efficientnet`` referenced ``model.act1``, which timm >= 0.6 folded into
``bn1`` (``BatchNormAct2d``). This repo requires timm >= 1.0, so every fresh install
died constructing D before the first tick -- and nothing caught it, because no test
built D. This builds the real discriminator with the backbones ``train.py``
configures (CPU, random weights, so no downloads) and runs a forward pass.
"""

import ast
import os
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT))


def _train_backbones():
    # The D backbones train.py actually configures, so a change there is tested here.
    for node in ast.walk(ast.parse((ROOT / "train.py").read_text())):
        if isinstance(node, ast.keyword) and node.arg == "backbones":
            return ast.literal_eval(node.value)
    raise AssertionError("train.py sets no backbones= for the discriminator")


@pytest.fixture
def random_weights(monkeypatch):
    # The builders hardcode pretrained=True; random weights exercise the same modules.
    create = timm.create_model
    monkeypatch.setattr(timm, "create_model", lambda name, pretrained=False, **kw: create(name, **kw))


def test_efficientnet_split_is_timm_forward(random_weights):
    # Mathematically identical to timm's own stem + blocks: conv_stem -> bn1 (BN and
    # the stem activation) -> every block, split into the four layers.
    from feature_networks.pretrained_builder import _make_efficientnet

    model = timm.create_model("tf_efficientnet_lite0").eval()
    assert not isinstance(model.bn1.act, torch.nn.Identity)  # the activation lives in bn1
    pretrained = _make_efficientnet(model).eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = x
        for layer in (pretrained.layer0, pretrained.layer1, pretrained.layer2, pretrained.layer3):
            out = layer(out)
        assert torch.equal(out, model.blocks(model.bn1(model.conv_stem(x))))


@pytest.mark.parametrize("cond", [True, False])
def test_discriminator_builds_and_runs(random_weights, cond):
    import dnnlib

    D = dnnlib.util.construct_class_by_name(
        class_name="pg_modules.discriminator.ProjectedDiscriminator",
        backbones=_train_backbones(), diffaug=True, interp224=True,
        backbone_kwargs=dnnlib.EasyDict(cout=64, expand=True, proj_type=2, num_discs=4, cond=cond),
        c_dim=3, img_resolution=64, img_channels=3,
    ).eval()
    assert list(D.feature_networks) == _train_backbones()
    c = torch.nn.functional.one_hot(torch.tensor([0, 2]), 3).float()
    with torch.no_grad():
        logits = D(torch.rand(2, 3, 64, 64) * 2 - 1, c)
    # A list of per-image logit rows, one batch per backbone; the loss torch.cat()s it.
    assert len(logits) == 2 * len(_train_backbones())
    assert torch.isfinite(torch.cat(logits)).all()
