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
                 "--num-fid-samples", "--combra-ref-count", "--mirror",
                 "--path-stem", "--up-factor", "--syn-layers"]:
        assert flag in out, f"missing {flag} in san-train --help"
    # Removed flags must be gone.
    for flag in ["--resume", "--metrics", "--fp32", "--nobench",
                 "--save-inference-only", "--restart_every"]:
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
