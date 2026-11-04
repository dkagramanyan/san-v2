"""Run-folder layout and checkpoint retention (combra_docs spec §2, §3, §7).

This file is identical in san, DiffiT, edm2 and StyleSwin. One run folder::

    <outdir>/<id:05d>-<model>[-<cfg>]-<dataset>-gpus<G>-batch<B>[-desc]/
        training_options.json            resolved launch config
        <run>-train.log                  rank-0 transcript, one [timestamp] per line
        <run>-console.log                raw output of the sh/ launcher (all ranks, tracebacks)
        <run>-console.pid                pid of a detached launch
        stats.jsonl                      one scalar row per tick
        events.out.tfevents.*.<run>      TensorBoard
        checkpoints/<model>-best-fid.pt  best by combra_fid (also -best-fd_dinov2, -best-cmmd)
        checkpoints/<model>-last.pt      newest snapshot (--snapshots best+last, or no best yet)
        checkpoints/<model>-kimg<k>.pt   every snapshot (--snapshots all)
        samples/reals.png  samples/fakes-init.png  samples/fakes-kimg<k>.png
"""

import math
import os
import re
import shutil

CHECKPOINTS = 'checkpoints'
SAMPLES = 'samples'

# --snapshots: which checkpoints a run keeps.
#   best       the best snapshot by each of BEST_METRICS; <model>-last.pt only while no
#              best exists (combra off, before the first eval, or every eval nan)
#   best+last  the bests plus <model>-last.pt, always
#   all        the bests plus <model>-kimg<k>.pt for every snapshot tick
SNAPSHOT_MODES = ('best', 'best+last', 'all')

# Lower is better for all three.
BEST_METRICS = ('combra_fid', 'combra_fd_dinov2', 'combra_cmmd')

#----------------------------------------------------------------------------
# Names.

def dataset_tag(data):
    """The dataset part of a run name: the zip/folder name without extension.

    Characters other than letters, digits, '.' and '_' become '_', so the dataset
    never adds a '-' and the dashes in a run name only separate fields.
    """
    stem = os.path.splitext(os.path.basename(os.path.normpath(data)))[0]
    return re.sub(r'[^A-Za-z0-9._]', '_', stem)

def run_desc(model, cfg, data, gpus, batch, desc=None):
    """``<model>[-<cfg>]-<dataset>-gpus<G>-batch<B>[-desc]``; the model is prefixed only
    when the cfg does not already start with it (san's ``stylegan3-r`` does not)."""
    name = cfg if (cfg and cfg.startswith(model)) else '-'.join(x for x in (model, cfg) if x)
    name += f'-{dataset_tag(data)}-gpus{gpus}-batch{batch}'
    return f'{name}-{desc}' if desc else name

def run_name(run_dir):
    return os.path.basename(os.path.normpath(run_dir))

def train_log_path(run_dir):
    return os.path.join(run_dir, f'{run_name(run_dir)}-train.log')

def console_log_path(run_dir):
    return os.path.join(run_dir, f'{run_name(run_dir)}-console.log')

def reals_path(run_dir):
    return os.path.join(run_dir, SAMPLES, 'reals.png')

def fakes_path(run_dir, kimg=None):
    """``samples/fakes-init.png`` for ``kimg=None``, else ``samples/fakes-kimg<k>.png``."""
    name = 'fakes-init.png' if kimg is None else f'fakes-kimg{int(kimg):06d}.png'
    return os.path.join(run_dir, SAMPLES, name)

def checkpoint_dir(run_dir):
    return os.path.join(run_dir, CHECKPOINTS)

def checkpoint_desc(path):
    """Default name of what is generated from a checkpoint (e.g. the merged ``.h5``).

    A checkpoint in ``<run>/checkpoints/`` gives ``<run>-<role>``, e.g.
    ``00003-san-stylegan3-r-<dataset>-gpus2-batch84-best-fid`` for
    ``san-best-fid.pt``, so files generated from different runs never share a name.
    Any other file gives its stem.
    """
    path = os.path.abspath(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.dirname(path)
    if os.path.basename(parent) != CHECKPOINTS or '-' not in stem:
        return stem
    return f'{run_name(os.path.dirname(parent))}-{stem.split("-", 1)[1]}'

#----------------------------------------------------------------------------
# Run folder.

def make_run_dir(run_dir, exist_ok=False):
    """Create the run folder with its checkpoints/ and samples/ subfolders.

    The launcher creates a fresh folder; the training loop calls it again with
    ``exist_ok=True`` so a loop started on a plain folder has its subfolders too.
    """
    os.makedirs(run_dir, exist_ok=exist_ok)
    os.makedirs(checkpoint_dir(run_dir), exist_ok=True)
    os.makedirs(os.path.join(run_dir, SAMPLES), exist_ok=True)

def adopt_console_log(run_dir):
    """Move the sh/ launcher's console log into the run folder.

    The launch scripts write their whole output to ``$RUN_LOG`` (under
    ``<outdir>/.launch/`` until the run folder exists) and a detached launch writes
    its pid beside it. Both move to ``<run>-console.log`` / ``<run>-console.pid``.
    The writers keep their open file, so the output keeps arriving after the move.
    Returns the new path, or ``None`` when there is nothing to move or the move fails
    (e.g. a different filesystem), in which case the log stays where it is.
    """
    src = os.environ.get('RUN_LOG')
    if not src or not os.path.isfile(src):
        return None
    dst = console_log_path(run_dir)
    try:
        os.replace(src, dst)
    except OSError as e:
        print(f'Console log stays at {src}: {e}', flush=True)
        return None
    pid = os.path.splitext(src)[0] + '.pid'
    if os.path.isfile(pid):
        try:
            os.replace(pid, os.path.splitext(dst)[0] + '.pid')
        except OSError:
            pass
    os.environ['RUN_LOG'] = dst
    print(f'Console log moved to {dst}', flush=True)
    return dst

#----------------------------------------------------------------------------
# Checkpoints.

def _link(src, dst):
    # Atomic: hard link (copy where links are unsupported) to a temp name, then replace.
    tmp = f'{dst}.tmp'
    if os.path.lexists(tmp):
        os.remove(tmp)
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copyfile(src, tmp)
    os.replace(tmp, dst)

def update_best(best, metrics, kimg):
    """Record ``kimg`` in ``best`` (``{metric: (value, kimg)}``) for every metric it
    improves; return the improved metric names. Missing or non-finite values never
    count and a tie keeps the earlier snapshot, so a best only gets strictly better."""
    improved = []
    for key in BEST_METRICS:
        value = (metrics or {}).get(key)
        if value is None or not math.isfinite(value):
            continue
        if key not in best or value < best[key][0]:
            best[key] = (float(value), int(kimg))
            improved.append(key)
    return improved

def best_metrics(metrics):
    """The BEST_METRICS values of one eval, for a checkpoint's metadata."""
    return {k: float(metrics[k]) for k in BEST_METRICS if k in (metrics or {})}

def save_snapshot(ckpt_dir, model, kimg, write, metrics, best, mode='best', suffixes=('',)):
    """Save one snapshot tick under the names ``mode`` keeps; return the file names.

    ``write(path, suffix)`` writes the checkpoint for one suffix to ``path`` (edm2
    writes one file per EMA std, suffix ``-ema<std>``; the others pass ``('',)``).
    ``metrics`` is this tick's eval of the same weights (empty without an eval);
    ``best`` is updated in place. Each file is written once to a temp name and hard
    linked to every name it holds (``<model>-best-fid.pt``, ``-best-fd_dinov2``,
    ``-best-cmmd``, ``-last``, ``-kimg<k>``), so one snapshot that is best by several
    metrics takes the disk space of one. A tick that is no best writes nothing in
    ``best`` mode once a best exists.
    """
    if mode not in SNAPSHOT_MODES:
        raise ValueError(f'snapshot mode {mode!r} not in {SNAPSHOT_MODES}')
    improved = update_best(best, metrics, kimg)
    roles = [f'best-{k[len("combra_"):]}' for k in improved]
    if mode == 'all':
        roles.append(f'kimg{int(kimg):06d}')
    keep_last = (mode == 'best+last') or not best
    if keep_last:
        roles.append('last')
    written = []
    for suffix in suffixes:
        if roles:
            tmp = os.path.join(ckpt_dir, f'.{model}-kimg{int(kimg):06d}{suffix}.pt.tmp')
            write(tmp, suffix)
            for role in roles:
                name = f'{model}-{role}{suffix}.pt'
                _link(tmp, os.path.join(ckpt_dir, name))
                written.append(name)
            os.remove(tmp)
        if not keep_last:
            last = os.path.join(ckpt_dir, f'{model}-last{suffix}.pt')
            if os.path.exists(last):
                os.remove(last)
    return written

def best_line(best):
    """``Best snapshots: combra_fid <v> kimg<k>  combra_fd_dinov2 <v> kimg<k>  ...``"""
    return 'Best snapshots: ' + '  '.join(
        f'{k} {best[k][0]:.4f} kimg{best[k][1]:06d}' for k in BEST_METRICS if k in best)

#----------------------------------------------------------------------------
