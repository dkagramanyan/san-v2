"""Inference-snapshot round trip (§3): save_inference_snapshot -> load_generator.

Builds tiny stylegan3-r generators on CPU (the custom ops fall back to their
reference paths), writes each as a snapshot and rebuilds it from current code. The
stage-2 case is exactly what --path-stem does: a SuperresGenerator constructed on a
stage-1 snapshot, itself saved and reloaded.
"""

import os
import pickle
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_utils import checkpoint  # noqa: E402


@pytest.fixture
def stem_path(tmp_path, monkeypatch):
    # MappingNetwork loads its class embedding from SAN_EMBED; a tiny stand-in will do.
    embed = tmp_path / "embed.pkl"
    with open(embed, "wb") as f:
        pickle.dump({"embed": torch.nn.Embedding(1000, 8)}, f)
    monkeypatch.setenv("SAN_EMBED", str(embed))

    from training.networks_stylegan3_resetting import Generator
    torch.manual_seed(0)
    G = Generator(z_dim=8, c_dim=3, w_dim=16, img_resolution=16, img_channels=3,
                  mapping_kwargs=dict(num_layers=2), channel_base=256, channel_max=16,
                  num_layers=4, conv_kernel=1, use_radial_filters=True).eval()
    path = str(tmp_path / "stem.pt")
    checkpoint.save_inference_snapshot(path, G, dict(n_classes=3))
    return path, G


def _same_output(a, b):
    z = torch.randn([2, 8], generator=torch.Generator().manual_seed(1))
    c = torch.eye(3)[:2]
    with torch.no_grad():
        return torch.equal(a(z, c, noise_mode="const"), b(z, c, noise_mode="const"))


def test_generator_round_trip(stem_path):
    path, G = stem_path
    blob = torch.load(path, weights_only=False)["G_ema"]
    assert blob["class_name"] == "training.networks_stylegan3_resetting.Generator"
    assert _same_output(G, checkpoint.load_generator(path))


def test_superres_generator_round_trip(stem_path, tmp_path):
    from training.networks_stylegan3_resetting import SuperresGenerator
    stem, _ = stem_path
    S = SuperresGenerator(img_resolution=32, path_stem=stem, head_layers=3, up_factor=2,
                          conv_kernel=1, use_radial_filters=True).eval()
    path = str(tmp_path / "sr.pt")
    checkpoint.save_inference_snapshot(path, S, {})
    assert _same_output(S, checkpoint.load_generator(path))


def test_snapshot_with_decorator_class_name_still_loads(stem_path, tmp_path):
    # Snapshots up to v0.5.0 stored the persistence wrapper's unimportable name.
    path, G = stem_path
    data = torch.load(path, weights_only=False)
    data["G_ema"]["class_name"] = "torch_utils.persistence.persistent_class.<locals>.Decorator"
    legacy = str(tmp_path / "legacy.pt")
    torch.save(data, legacy)
    assert _same_output(G, checkpoint.load_generator(legacy))
