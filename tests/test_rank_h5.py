"""h5 schema contract for generated images (models_api_proposal §4/§5).

CPU-only, tiny arrays: asserts the unified shard signature
(``format="generated_images_shard"``, ``schema_version=1``), the merged-file
roundtrip, the hard-fail on missing slots, the zero-sample-rank teardown
(``samples_per_class < world_size``), and that ``class_names`` is mandatory.
No GPU, no checkpoint, no network.
"""

import h5py
import numpy as np
import pytest

from gen_images import (
    H5_FORMAT_SHARD,
    H5_SCHEMA_VERSION,
    RankH5Writer,
    _merge_shards_to_one_h5,
)

CLASS_NAMES = ["Ultra_Co11", "Ultra_Co25", "Ultra_Co6_2"]
CLASSES = [0, 2]
SPC = 4  # samples per class
RES = 8


def _writer(shards_dir, rank, class_names=CLASS_NAMES):
    w = RankH5Writer(
        shard_path=shards_dir / f"rank_{rank:03d}.h5",
        classes=CLASSES,
        samples_per_class=SPC,
        compression=None,
        chunk_images=2,
        class_names=class_names,
    )
    w.open()
    return w


def _write(w, class_idx, idxs):
    idxs = np.asarray(idxs, dtype=np.int64)
    images = np.full((idxs.size, RES, RES, 3), class_idx + 1, dtype=np.uint8)
    seeds = 1000 + class_idx * SPC + idxs
    w.write_batch(class_idx, idxs, seeds, images)


def _merge(tmp_path, world_size):
    merged = tmp_path / "merged.h5"
    _merge_shards_to_one_h5(
        merged_path=merged,
        shards_dir=tmp_path / "shards",
        classes=CLASSES,
        samples_per_class=SPC,
        compression=None,
        chunk_images=2,
        world_size=world_size,
        class_names=CLASS_NAMES,
    )
    return merged


def test_shard_and_merged_roundtrip(tmp_path):
    sh = tmp_path / "shards"
    for rank, idxs in [(0, [0, 1]), (1, [2, 3])]:
        w = _writer(sh, rank)
        for c in CLASSES:
            _write(w, c, idxs)
        w.close()

    # Shard signature (§4): sniffable without knowing which model produced it.
    with h5py.File(sh / "rank_000.h5", "r") as f:
        assert f.attrs["format"] == H5_FORMAT_SHARD
        assert int(f.attrs["schema_version"]) == H5_SCHEMA_VERSION
        assert list(f.attrs["class_names"]) == CLASS_NAMES
        assert tuple(f.attrs["image_shape_hwc"]) == (RES, RES, 3)
        assert int(f.attrs["samples_per_class"]) == SPC

    merged = _merge(tmp_path, world_size=2)
    with h5py.File(merged, "r") as f:
        assert f.attrs["format"] == H5_FORMAT_SHARD
        assert int(f.attrs["schema_version"]) == H5_SCHEMA_VERSION
        assert list(f.attrs["class_names"]) == CLASS_NAMES
        assert tuple(f.attrs["image_shape_hwc"]) == (RES, RES, 3)
        assert int(f.attrs["samples_per_class"]) == SPC
        assert int(f.attrs["missing_count"]) == 0
        for c in CLASSES:
            g = f[f"class_{c}"]
            assert int(g.attrs["class_idx"]) == c
            assert g.attrs["class_name"] == CLASS_NAMES[c]
            assert g["images"].dtype == np.uint8
            assert g["images"].shape == (SPC, RES, RES, 3)
            assert (g["images"][:] == c + 1).all()
            assert g["seeds"].dtype == np.int64
            assert list(g["seeds"][:]) == [1000 + c * SPC + i for i in range(SPC)]
            assert g["written"].dtype == np.bool_
            assert g["written"][:].all()


def test_merge_hard_fails_on_missing_slots(tmp_path):
    sh = tmp_path / "shards"
    w = _writer(sh, 0)
    _write(w, 0, [0, 1])       # 2 of 4
    _write(w, 2, [0, 1, 2])    # 3 of 4
    assert w.close() == 3
    with pytest.raises(RuntimeError, match="missing"):
        _merge(tmp_path, world_size=1)
    # The partial merged file must not survive as a fake-complete artifact (§4).
    assert not (tmp_path / "merged.h5").exists()


def test_zero_sample_rank_deletes_empty_shard_and_merge_succeeds(tmp_path):
    # samples_per_class < world_size: a tail rank opens a shard but never writes.
    sh = tmp_path / "shards"
    w0 = _writer(sh, 0)
    for c in CLASSES:
        _write(w0, c, range(SPC))
    w0.close()

    w1 = _writer(sh, 1)  # no work assigned
    assert (sh / "rank_001.h5").exists()
    assert w1.close() == 0
    assert not (sh / "rank_001.h5").exists()

    merged = _merge(tmp_path, world_size=2)
    with h5py.File(merged, "r") as f:
        assert int(f.attrs["missing_count"]) == 0

    # An absent rank file must NOT mask real gaps: the coverage gate still fires.
    (sh / "rank_000.h5").unlink()
    w0 = _writer(sh, 0)
    _write(w0, 0, [0])
    _write(w0, 2, [0])
    w0.close()
    with pytest.raises(RuntimeError, match="missing"):
        _merge(tmp_path, world_size=2)


def test_class_names_is_mandatory(tmp_path):
    with pytest.raises(ValueError, match="class_names"):
        RankH5Writer(
            shard_path=tmp_path / "shards" / "rank_000.h5",
            classes=CLASSES,
            samples_per_class=SPC,
            compression=None,
            chunk_images=2,
            class_names=None,
        )
    with pytest.raises(ValueError, match="class_names"):
        _merge_shards_to_one_h5(
            merged_path=tmp_path / "merged.h5",
            shards_dir=tmp_path / "shards",
            classes=CLASSES,
            samples_per_class=SPC,
            compression=None,
            chunk_images=2,
            world_size=1,
            class_names=None,
        )
