"""Run-folder layout and checkpoint retention (run_layout, identical in all four repos).

The retention tests replay the loop's eval -> save_snapshot order with a writer that
stores the kimg as text, so no model is built.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_utils import run_layout  # noqa: E402


def _m(fid, dino, cmmd):
    return dict(combra_fid=fid, combra_fd_dinov2=dino, combra_cmmd=cmmd)


def _run(tmp_path, evals, mode='best', suffixes=('',)):
    best = {}
    for kimg, metrics in evals:
        def write(path, suffix, kimg=kimg):
            with open(path, 'w') as f:
                f.write(f'{kimg}{suffix}')
        run_layout.save_snapshot(str(tmp_path), 'm', kimg, write, metrics, best, mode, suffixes)
    files = {n: open(tmp_path / n).read() for n in os.listdir(tmp_path)}
    return files, {k: kimg for k, (_, kimg) in best.items()}


EVALS = [
    (0, {}),                   # tick 0: never evaluated
    (100, _m(50, 500, 5.0)),
    (200, _m(30, 600, 6.0)),   # best fid
    (300, _m(40, 400, 7.0)),   # best fd_dinov2
    (400, _m(45, 450, 1.0)),   # best cmmd
    (500, _m(60, 700, 9.0)),   # last
]


def test_best_keeps_only_the_bests(tmp_path):
    files, best = _run(tmp_path, EVALS)
    assert best == dict(combra_fid=200, combra_fd_dinov2=300, combra_cmmd=400)
    assert files == {'m-best-fid.pt': '200', 'm-best-fd_dinov2.pt': '300', 'm-best-cmmd.pt': '400'}


def test_one_snapshot_best_by_all_is_one_file(tmp_path):
    files, _ = _run(tmp_path, [(100, _m(20, 300, 2.0)), (200, _m(math.nan, math.nan, math.nan)),
                               (300, _m(25, math.nan, 3.0))])
    assert files == {'m-best-fid.pt': '100', 'm-best-fd_dinov2.pt': '100', 'm-best-cmmd.pt': '100'}
    inodes = {os.stat(tmp_path / n).st_ino for n in files}
    assert len(inodes) == 1
    assert not [n for n in os.listdir(tmp_path) if n.endswith('.tmp')]


def test_without_metrics_last_is_kept(tmp_path):
    files, best = _run(tmp_path, [(0, {}), (100, {}), (200, {})])
    assert best == {}
    assert files == {'m-last.pt': '200'}


def test_last_goes_once_a_best_exists(tmp_path):
    files, _ = _run(tmp_path, [(0, {}), (100, _m(1, 1, 1)), (200, _m(2, 2, 2))])
    assert files == {'m-best-fid.pt': '100', 'm-best-fd_dinov2.pt': '100', 'm-best-cmmd.pt': '100'}


def test_best_plus_last(tmp_path):
    files, _ = _run(tmp_path, EVALS, mode='best+last')
    assert files == {'m-best-fid.pt': '200', 'm-best-fd_dinov2.pt': '300', 'm-best-cmmd.pt': '400',
                     'm-last.pt': '500'}


def test_all_keeps_every_snapshot(tmp_path):
    files, _ = _run(tmp_path, EVALS, mode='all')
    assert {n for n in files if 'kimg' in n} == {f'm-kimg{k:06d}.pt' for k, _ in EVALS}
    assert files['m-best-fid.pt'] == '200'
    assert 'm-last.pt' not in files


def test_suffixes_one_file_per_ema(tmp_path):
    files, _ = _run(tmp_path, [(100, _m(1, 1, 1)), (200, _m(0.5, 2, 2))],
                    suffixes=('-ema0.050', '-ema0.100'))
    assert files['m-best-fid-ema0.050.pt'] == '200-ema0.050'
    assert files['m-best-fid-ema0.100.pt'] == '200-ema0.100'
    assert files['m-best-cmmd-ema0.100.pt'] == '100-ema0.100'
    assert len(files) == 6


def test_best_line():
    best = {'combra_cmmd': (1.0, 400), 'combra_fid': (30.0, 200)}
    assert run_layout.best_line(best) == 'Best snapshots: combra_fid 30.0000 kimg000200  combra_cmmd 1.0000 kimg000400'


def test_run_desc():
    data = './datasets/imagenet_9to4_1024x1024_256x256.zip'
    assert run_layout.run_desc('san', 'stylegan3-r', data, 2, 84) == \
        'san-stylegan3-r-imagenet_9to4_1024x1024_256x256-gpus2-batch84'
    assert run_layout.run_desc('edm2', 'edm2-img256-s', data, 2, 128, 'try1') == \
        'edm2-img256-s-imagenet_9to4_1024x1024_256x256-gpus2-batch128-try1'
    assert run_layout.run_desc('styleswin', None, data, 1, 8) == \
        'styleswin-imagenet_9to4_1024x1024_256x256-gpus1-batch8'
    assert run_layout.dataset_tag('/x/my-data set.zip') == 'my_data_set'
    assert run_layout.dataset_tag('/x/folder/') == 'folder'


def test_paths(tmp_path):
    run = str(tmp_path / '00003-x-d-gpus1-batch8')
    run_layout.make_run_dir(run)
    assert sorted(os.listdir(run)) == ['checkpoints', 'samples']
    assert run_layout.train_log_path(run).endswith('/00003-x-d-gpus1-batch8/00003-x-d-gpus1-batch8-train.log')
    assert run_layout.fakes_path(run).endswith('/samples/fakes-init.png')
    assert run_layout.fakes_path(run, 1234).endswith('/samples/fakes-kimg001234.png')
    assert run_layout.reals_path(run).endswith('/samples/reals.png')


def test_checkpoint_desc():
    run = '/x/runs/00003-san-stylegan3-r-d-gpus2-batch84'
    assert run_layout.checkpoint_desc(f'{run}/checkpoints/san-best-fid.pt') == \
        '00003-san-stylegan3-r-d-gpus2-batch84-best-fid'
    assert run_layout.checkpoint_desc(f'{run}/checkpoints/edm2-best-cmmd-ema0.050.pt') == \
        '00003-san-stylegan3-r-d-gpus2-batch84-best-cmmd-ema0.050'
    assert run_layout.checkpoint_desc('/x/old/san-snapshot-000400-inference.pt') == 'san-snapshot-000400-inference'


def test_adopt_console_log(tmp_path, monkeypatch):
    launch = tmp_path / '.launch'
    launch.mkdir()
    log = launch / 'm-train_256-20261104-120000.log'
    log.write_text('Run settings:\n')
    (launch / 'm-train_256-20261104-120000.pid').write_text('123\n')
    run = str(tmp_path / '00000-r')
    run_layout.make_run_dir(run)
    monkeypatch.setenv('RUN_LOG', str(log))
    with open(log, 'a') as writer:           # an open writer (tee) keeps writing after the move
        dst = run_layout.adopt_console_log(run)
        writer.write('after\n')
    assert dst == os.path.join(run, '00000-r-console.log')
    assert open(dst).read() == 'Run settings:\nafter\n'
    assert open(os.path.join(run, '00000-r-console.pid')).read() == '123\n'
    assert os.listdir(launch) == []
    assert os.environ['RUN_LOG'] == dst


def test_adopt_console_log_without_launcher(tmp_path, monkeypatch):
    monkeypatch.delenv('RUN_LOG', raising=False)
    assert run_layout.adopt_console_log(str(tmp_path)) is None
