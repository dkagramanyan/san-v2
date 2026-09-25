"""Snapshot retention: the newest snapshot plus the best by each combra metric.

Replays the loop's save -> eval -> update_best_snapshots -> prune_snapshots order on
empty files, so no model is built.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_utils import checkpoint  # noqa: E402


def _run(tmp_path, evals, keep_last=1):
    best = {}
    for kimg, metrics in evals:
        path = str(tmp_path / f"san-snapshot-{kimg:06d}-inference.pt")
        open(path, "wb").close()
        checkpoint.update_best_snapshots(best, path, metrics)
        checkpoint.prune_snapshots(str(tmp_path), keep_last, keep=[p for _, p in best.values()])
    kept = sorted(int(n.split("-")[2]) for n in os.listdir(tmp_path))
    return kept, {k: int(os.path.basename(p).split("-")[2]) for k, (_, p) in best.items()}


def _m(fid, dino, cmmd):
    return dict(combra_fid=fid, combra_fd_dinov2=dino, combra_cmmd=cmmd)


def test_keeps_last_plus_best_per_metric(tmp_path):
    kept, best = _run(tmp_path, [
        (0, {}),                   # tick 0: saved, never evaluated
        (100, _m(50, 500, 5.0)),
        (200, _m(30, 600, 6.0)),   # best fid
        (300, _m(40, 400, 7.0)),   # best fd_dinov2
        (400, _m(45, 450, 1.0)),   # best cmmd
        (500, _m(60, 700, 9.0)),   # last
    ])
    assert best == dict(combra_fid=200, combra_fd_dinov2=300, combra_cmmd=400)
    assert kept == [200, 300, 400, 500]


def test_one_file_serves_several_roles_and_nan_is_ignored(tmp_path):
    kept, best = _run(tmp_path, [
        (100, _m(20, 300, 2.0)),
        (200, _m(math.nan, math.nan, math.nan)),
        (300, _m(25, math.nan, 3.0)),
    ])
    assert best == dict(combra_fid=100, combra_fd_dinov2=100, combra_cmmd=100)
    assert kept == [100, 300]


def test_best_is_never_pruned_and_newest_can_be_best(tmp_path):
    kept, best = _run(tmp_path, [
        (100, _m(20, 300, 2.0)),
        (200, _m(10, 200, 1.0)),
    ])
    assert best == dict(combra_fid=200, combra_fd_dinov2=200, combra_cmmd=200)
    assert kept == [200]


def test_without_metrics_only_last_is_kept(tmp_path):
    kept, best = _run(tmp_path, [(0, {}), (100, {}), (200, {})])
    assert best == {}
    assert kept == [200]


def test_keep_last_zero_keeps_all(tmp_path):
    kept, _ = _run(tmp_path, [(100, _m(1, 1, 1)), (200, _m(2, 2, 2)), (300, {})], keep_last=0)
    assert kept == [100, 200, 300]
