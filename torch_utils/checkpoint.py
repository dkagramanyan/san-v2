# State-dict checkpoint I/O for san-v2.
#
# The v2 model API (see the wc_cv models_api_proposal, §3) mandates ONE artifact
# kind: an EMA-only inference snapshot stored as a `.pt` **state dict** -- weight
# tensors keyed by parameter name plus the persistence init kwargs needed to
# rebuild the module from current code. No pickled `nn.Module` objects ever touch
# disk, so loading never depends on the exact `timm` version that trained the
# discriminator (the discriminator is not saved at all).
#
# Writes are atomic: every snapshot is streamed to a temp file in the same
# directory and moved into place with ``os.replace``, so a snapshot that exists
# under its final name is always complete.

import glob
import os
import re

import torch

import dnnlib

#----------------------------------------------------------------------------

CHECKPOINT_FORMAT = 'san_inference_state_dict'
CHECKPOINT_VERSION = 1

#----------------------------------------------------------------------------

def _module_blob(module):
    """Serialize a persistence-decorated ``nn.Module`` to a state-dict blob.

    Relies on the StyleGAN persistence mechanism (``torch_utils.persistence``),
    which records the constructor ``init_args`` / ``init_kwargs`` on every
    instance, so the module can be rebuilt from current code and re-filled with
    the stored weights.
    """
    return dict(
        class_name=f'{type(module).__module__}.{type(module).__qualname__}',
        init_args=list(module.init_args),
        init_kwargs=dict(module.init_kwargs),
        state_dict={k: v.detach().cpu() for k, v in module.state_dict().items()},
    )

def _rebuild_module(blob, device):
    module = dnnlib.util.construct_class_by_name(
        *blob['init_args'], class_name=blob['class_name'], **blob['init_kwargs'])
    module.load_state_dict(blob['state_dict'], strict=True)
    return module.to(device).eval().requires_grad_(False)

#----------------------------------------------------------------------------

def save_inference_snapshot(path, G_ema, metadata):
    """Atomically write an EMA-only inference snapshot as a `.pt` state dict.

    ``metadata`` carries the self-describing keys
    ``{n_classes, resolution, class_names, cur_nimg}`` (§3) so downstream code
    reads grain-class *names* from the checkpoint instead of guessing integer
    conventions.
    """
    data = dict(
        format=CHECKPOINT_FORMAT,
        version=CHECKPOINT_VERSION,
        model='san',
        metadata=dict(metadata),
        G_ema=_module_blob(G_ema),
    )
    tmp = f'{path}.tmp'
    torch.save(data, tmp)
    os.replace(tmp, path)  # atomic: a file under its final name is always complete

#----------------------------------------------------------------------------

def load_snapshot(path, device='cpu'):
    """Load an inference snapshot; return ``(G_ema, metadata)``.

    ``G_ema`` is rebuilt from current code via the stored persistence kwargs and
    filled with the saved weights.
    """
    with dnnlib.util.open_url(path) as f:
        data = torch.load(f, map_location='cpu', weights_only=False)
    if data.get('format') != CHECKPOINT_FORMAT:
        raise IOError(f'{path}: not a san inference snapshot (format={data.get("format")!r})')
    G_ema = _rebuild_module(data['G_ema'], device)
    return G_ema, dict(data.get('metadata', {}))

def load_generator(path, device='cpu'):
    """Load just the EMA generator from an inference snapshot."""
    G_ema, _ = load_snapshot(path, device)
    return G_ema

def read_metadata(path):
    """Read just the self-describing metadata dict from a snapshot (no rebuild)."""
    with dnnlib.util.open_url(path) as f:
        data = torch.load(f, map_location='cpu', weights_only=False)
    return dict(data.get('metadata', {}))

#----------------------------------------------------------------------------

def prune_snapshots(run_dir, keep_last):
    """Keep only the ``keep_last`` newest ``san-snapshot-<kimg>-inference.pt``.

    ``keep_last == 0`` keeps everything. Pruning runs after the newest snapshot
    is safely on disk, so the retained history is always complete.
    """
    if keep_last is None or keep_last <= 0:
        return
    pat = re.compile(r'san-snapshot-(\d+)-inference\.pt$')
    snaps = []
    for p in glob.glob(os.path.join(run_dir, 'san-snapshot-*-inference.pt')):
        m = pat.search(os.path.basename(p))
        if m is not None:
            snaps.append((int(m.group(1)), p))
    snaps.sort()  # ascending kimg; newest last
    for _, p in snaps[:-keep_last]:
        try:
            os.remove(p)
        except OSError:
            pass

#----------------------------------------------------------------------------
