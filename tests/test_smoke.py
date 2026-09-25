"""CPU smoke tests: CLI-contract parsing and pure-Python label/class logic.

These assert the v2 API surface (models_api_proposal §2/§4/§5) without running the
model: click --help introspection for each console script, plus unit tests for the
dataset label contract and the generation class resolver.
"""

import pytest
from click.testing import CliRunner


def _help(cmd):
    result = CliRunner().invoke(cmd, ["--help"])
    assert result.exit_code == 0, result.output
    return result.output


def test_train_cli_contract():
    from train import main
    out = _help(main)
    for flag in ["--precision", "--tf32", "--bench", "--snapshot-keep-last",
                 "--num-fid-samples", "--combra-ref-count", "--augment",
                 "--path-stem", "--up-factor", "--syn-layers"]:
        assert flag in out, f"missing {flag} in san-train --help"
    # Removed flags must be gone.
    for flag in ["--resume", "--metrics", "--fp32", "--nobench",
                 "--save-inference-only", "--restart_every", "--mirror"]:
        assert flag not in out, f"{flag} should have been removed from san-train"


def test_gen_images_cli_contract():
    from gen_images import generate_images
    out = _help(generate_images)
    for flag in ["--network", "--classes", "--samples-per-class", "--batch-gpu",
                 "--gpus", "--save-mode", "--merge"]:
        assert flag in out, f"missing {flag} in san-gen-images --help"


def test_prepare_data_is_group_with_convert():
    from dataset_tool import prepare_data
    out = _help(prepare_data)
    assert "convert" in out


def test_resolve_classes_by_index_range_and_name():
    from gen_images import resolve_classes
    names = ["Ultra_Co11", "Ultra_Co25", "Ultra_Co6_2"]
    assert resolve_classes("0,2", 3, names) == [0, 2]
    assert resolve_classes("0-2", 3, names) == [0, 1, 2]
    assert resolve_classes("Ultra_Co11,Ultra_Co6_2", 3, names) == [0, 2]


def test_resolve_classes_validates_range_and_names():
    from click import ClickException

    from gen_images import resolve_classes
    with pytest.raises(ClickException):
        resolve_classes("5", 3, None)          # out of range
    with pytest.raises(ClickException):
        resolve_classes("Nope", 3, ["a", "b", "c"])  # unknown name


def test_label_metadata_alphabetical_with_class_names():
    from dataset_tool import _build_label_metadata
    arch = ["00000/a.png", "00000/b.png", "00000/c.png"]
    # Folder classes are given out of alphabetical order; labels must follow sorted().
    classes = ["Ultra_Co25", "Ultra_Co11", "Ultra_Co6_2"]
    meta = _build_label_metadata(arch, classes, [None, None, None])
    assert meta["class_names"] == ["Ultra_Co11", "Ultra_Co25", "Ultra_Co6_2"]
    idx = {name: i for i, name in enumerate(meta["class_names"])}
    assert meta["labels"] == [[arch[0], idx["Ultra_Co25"]],
                              [arch[1], idx["Ultra_Co11"]],
                              [arch[2], idx["Ultra_Co6_2"]]]


def test_label_metadata_errors_on_missing_label():
    from dataset_tool import _build_label_metadata
    arch = ["00000/a.png", "b.png"]  # second image has no class folder
    with pytest.raises(SystemExit):
        _build_label_metadata(arch, ["ClassA", None], [None, None])


def test_label_metadata_refuses_integer_labels_without_names():
    # §5 Rule 2: bare integers must not be stamped with invented names ('0', '1', ...).
    from dataset_tool import _build_label_metadata
    with pytest.raises(SystemExit):
        _build_label_metadata(["00000/a.png", "00000/b.png"], [None, None], [0, 1])


def _zip(tmp_path, class_names):
    import io
    import json
    import zipfile

    import numpy as np
    import PIL.Image
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as zf:
        labels = []
        for i in range(2):
            buf = io.BytesIO()
            PIL.Image.fromarray(np.zeros([16, 16, 3], np.uint8)).save(buf, format="png")
            zf.writestr(f"00000/img{i:08d}.png", buf.getvalue())
            labels.append([f"00000/img{i:08d}.png", i])
        meta = {"labels": labels}
        if class_names is not None:
            meta["class_names"] = class_names
        zf.writestr("dataset.json", json.dumps(meta))
    return str(path)


def _config(tmp_path, *args, class_names=("A", "B")):
    import dnnlib
    from train import build_config, main
    argv = ["--outdir", str(tmp_path), "--cfg", "stylegan3-r", "--gpus", "1", "--batch-gpu", "2",
            "--data", _zip(tmp_path, None if class_names is None else list(class_names)), *args]
    return build_config(dnnlib.EasyDict(main.make_context("san-train", argv).params))


def test_cond_requires_class_names(tmp_path):
    from click import ClickException
    with pytest.raises(ClickException, match="class_names"):
        _config(tmp_path, "--cond", "True", class_names=None)
    _config(tmp_path, "--cond", "True")  # named zip is accepted


def test_precision_bf16_is_refused_and_fp32_survives_superres(tmp_path):
    from click import ClickException
    with pytest.raises(ClickException, match="bf16"):
        _config(tmp_path, "--precision", "bf16")
    c, _ = _config(tmp_path, "--precision", "fp32", "--superres", "True", "--path-stem", "stem.pt")
    assert c.G_kwargs.class_name.endswith("SuperresGenerator")
    assert c.G_kwargs.num_fp16_res == 0 and c.G_kwargs.conv_clamp is None


def test_one_denorm_rounds_to_nearest():
    # §5: every uint8 conversion is rint((x + 1) * 127.5), clamped -- also gen_images'.
    import numpy as np
    import torch

    from torch_utils import gen_utils, misc
    x = torch.tensor([-1.5, -1.0, -0.9168, 0.0, 0.3, 1.0, 1.5]).reshape(1, 1, 1, 7)
    want = np.rint((x.numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    assert np.array_equal(misc.denorm_to_uint8(x).numpy(), want)

    class _G:
        def synthesis(self, ws, noise_mode):
            return x.expand(1, 3, 1, 7)
    img = gen_utils.w_to_img(_G(), torch.zeros([1, 2, 4]))
    assert np.array_equal(img[0, :, :, 0], want[0, 0])


def test_augment_defaults_on_and_can_be_disabled(tmp_path):
    c, _ = _config(tmp_path)
    assert c.augment is True
    c, _ = _config(tmp_path, "--augment", "False")
    assert c.augment is False
