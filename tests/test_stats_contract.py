"""The row this repo writes to ``stats.jsonl`` must be readable by combra.

``combra.metrics.load_fid_by_kimg`` reads ``Metrics/combra_fid`` and
``Progress/kimg`` from the same JSON line, and shape-filters away any record whose
values are not plain scalars -- silently, returning ``{}``. That is exactly how
san-v2 and StyleSwin runs produced an unreadable metric history while every
combra-side test passed: the reader was tested against a synthetic *flat* row, and
nothing tested the producer.

This is the producer half. It builds the real row through the training loop's own
function, so a change to the row shape fails here instead of silently emptying the
analysis layer.
"""

import importlib.util
import json

import pytest

pytest.importorskip("torch")  # the training-loop module imports torch at module level

requires_combra = pytest.mark.skipif(
    importlib.util.find_spec("combra") is None, reason="combra is not installed"
)


def _row():
    from types import SimpleNamespace

    from training.training_loop import build_stats_row

    # The collector nests every scalar as {num, mean, std}; build_stats_row must
    # flatten it to the mean, which is what TensorBoard receives too.
    stats_dict = {
        "Progress/kimg": SimpleNamespace(num=1, mean=403.2),
        "Progress/tick": SimpleNamespace(num=1, mean=7.0),
        "Loss/G/loss": SimpleNamespace(num=1, mean=1.25),
        "Timing/Greg": SimpleNamespace(num=1, mean=float("nan")),
        "Timing/eval_sec": SimpleNamespace(num=0, mean=float("nan")),  # not reported this tick
    }
    return build_stats_row(stats_dict, {"combra_fid": 12.5, "combra_cmmd": float("inf")}, 1000.0, 900.0)


def test_row_is_strict_json_with_null_for_non_finite():
    row = _row()
    line = json.dumps(row, allow_nan=False)  # raises on a bare NaN / Infinity
    back = json.loads(line)
    assert back["Timing/Greg"] is None
    assert back["Metrics/combra_cmmd"] is None


def test_row_types_and_eval_keys():
    row = _row()
    assert row["Progress/tick"] == 7 and isinstance(row["Progress/tick"], int)
    assert row["Metrics/combra_fid"] == 12.5  # metrics live in the tick row
    # A scalar nobody reported this tick is absent, never NaN or a stale repeat.
    assert "Timing/eval_sec" not in row



def test_row_and_log_lines_have_three_metric_sections():
    # combra >= 0.25: v1 / v2 angle keys go to Metrics_v1/ and Metrics_v2/, the image
    # metrics and this repo's own scalars stay under Metrics/ (load_fid_by_kimg).
    from types import SimpleNamespace

    from training.training_loop import build_stats_row, metrics_log_lines

    metrics = {"combra_v1_w1": 1.5, "combra_v1_amp2": float("nan"), "combra_v2_pi": 2.5,
               "combra_fid": 12.5, "combra_cmmd": 0.25, "combra_fd_dinov2": 300.0,
               "combra_num_fid_samples": 256.0, "combra_fid_best": 12.5}
    row = build_stats_row({"Progress/kimg": SimpleNamespace(num=1, mean=4.0)}, metrics, 1000.0, 900.0)
    assert {k for k in row if k.startswith("Metrics")} == {
        "Metrics_v1/combra_w1", "Metrics_v1/combra_amp2", "Metrics_v2/combra_pi",
        "Metrics/combra_fid", "Metrics/combra_cmmd", "Metrics/combra_fd_dinov2",
        "Metrics/combra_num_fid_samples", "Metrics/combra_fid_best"}
    assert row["Metrics_v1/combra_amp2"] is None
    assert metrics_log_lines(metrics) == [
        "Metrics: combra_fid 12.5000  combra_cmmd 0.2500  combra_fd_dinov2 300.0000  "
        "combra_num_fid_samples 256.0000  combra_fid_best 12.5000",
        "Metrics v1: combra_w1 1.5000  combra_amp2 nan",
        "Metrics v2: combra_pi 2.5000",
    ]


def test_row_contains_only_json_scalars():
    for key, value in _row().items():
        assert value is None or isinstance(value, (int, float, str)), (
            f"{key} is {type(value).__name__}, not a JSON scalar -- "
            "load_fid_by_kimg will shape-filter this record away"
        )


@requires_combra
def test_row_round_trips_through_load_fid_by_kimg(tmp_path):
    from combra.metrics import load_fid_by_kimg

    path = tmp_path / "stats.jsonl"
    path.write_text(json.dumps(_row(), allow_nan=False) + "\n")
    assert load_fid_by_kimg(str(path)) == {"000403": 12.5}
