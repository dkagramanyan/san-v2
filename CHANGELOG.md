# Changelog

All notable changes to this fork (`san-v2`) are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased] — 2026-07-16

### Changed
- **Production `sbatch/train_*.sbatch` now pass `--save-inference-only 0`** (was `True`).
  A prod run keeps only the rolling `network-snapshot-latest.pt` resume checkpoint and the
  best-FID `best_model.pkl` — no per-tick `G_ema` history is accumulated. Set
  `--save-inference-only 1` to restore the per-tick inference snapshots.
- **Full resume checkpoint renamed** `network-snapshot.pkl` → **`network-snapshot-latest.pt`**
  (`torch_utils/misc.get_ckpt_path`). Behaviour is unchanged — it is still a single file
  overwritten in place every snapshot tick (never accumulates) and still carries the
  `G`/`D`/`G_ema` networks + resume `progress` (no optimizer state, as before). The
  per-tick `network-snapshot-<kimg>-inference.pkl` history snapshots serve as the
  accumulating record; the `-latest.pt` file is purely for `--resume`.
  **Migration:** existing runs must rename their on-disk `network-snapshot.pkl` to
  `network-snapshot-latest.pt` to keep auto-resuming.
- **`timm` unpinned to 1.x** (`timm>=1.0.0`, was `timm==0.4.12`) in `requirements.txt`
  and `pyproject.toml`. Updated feature-network model ids for the new timm registry:
  `tf_efficientnet_b0_ns` → `tf_efficientnet_b0.ns_jft_in1k` (`feature_networks/constants.py`);
  `vit_deit_base_patch16_384` → `deit_base_patch16_384`,
  `vit_deit_base_distilled_patch16_384` → `deit_base_distilled_patch16_384`,
  `vit_base_resnet50_384` → `vit_base_r50_s16_384` (`feature_networks/vit.py`, legacy
  DPT-ViT helpers not on the default projected-discriminator path). All other deps
  already used `>=` (latest-compatible); `glob` is stdlib.
  **⚠ Breaking:** checkpoints that embed the projected discriminator's timm feature
  networks (`best_model.pkl` stems used as `--path_stem`, and `network-snapshot-latest.pt`)
  were saved under timm 0.4.12 and will **not** unpickle under timm 1.x — the progressive
  16²→1024² stem chain must be regenerated from scratch. Inference-only `G_ema` snapshots
  are unaffected. This migration was **not** runtime-tested (timm 1.x unavailable in the
  authoring env); verify with `tests/test_san_modules.py` on the cluster before a long run.

## [Unreleased] — 2026-06-25

### Added
- **`--save-inference-only` training flag** — writes a tiny `network-snapshot-<kimg>-inference.pkl`
  containing only `G_ema` (no discriminator, no resume state) each snapshot tick; the
  smallest artifact, intended for `gen_images.py` / `calc_metrics.py`.
  (`train.py`, `training/training_loop.py`)
- **`--combra-metrics` training flag** (default `true`) — computes the combra
  generative-quality metrics each snapshot tick, **independent of `--metrics`** (you can
  now use `--metrics none` and still get combra, or vice-versa).
- **combra image-feature metrics** (`fid`, `cmmd`, `fd_dinov2`) are now computed during
  training (the loop passes `image_metrics=True`), in addition to the angle-density
  metrics. All are logged to TensorBoard under `Metrics/combra_*`.
- **Startup warning** when `--combra-metrics=true` but the `combra` package is not
  installed (instead of a silent skip).
- **Pretrained-model downloader** — running `python tests/test_san_modules.py` directly
  pre-fetches every weight training + the combra metrics need (timm backbones,
  InceptionV3, CLIP, DINOv2). `download_models.sh` does the same via pure
  `wget`/`curl`/`git` into the right caches, for offline compute nodes.
- **Generate sbatch scripts** for 512×512 and 1024×1024.

### Changed
- **sbatch scripts** are now self-contained and submittable from the `sbatch/` folder
  (each resolves the repo root), target **2× H200**, and queue on the **`rocky`
  partition** (no reservation). They load **no system CUDA module** — the custom ops
  build against the conda toolkit via `CUDA_HOME=$CONDA_PREFIX`. Train scripts pass
  `--save-inference-only True`.
- **Hydra entry point** — `train_hydra.py` now derives all option defaults from the
  `train.py` click CLI (single source of truth), so `configs/config.yaml` only declares
  the required fields and new flags propagate automatically.
- **`legacy.load_network_pkl`** mirrors `G_ema` onto `G` when a pickle has only `G_ema`,
  so inference-only snapshots load through the existing pipeline.
- **Install** — PyTorch wheels bumped to the CUDA 13.2 index (`cu132`); install the CUDA
  compiler (`cuda-nvcc`) from conda since no system CUDA module is loaded. `timm==0.4.12`
  is required (newer timm cannot unpickle the trained `best_model.pkl` stems); re-pin it
  after installing `requirements.txt` (combra's `open-clip-torch` otherwise pulls a newer
  timm).
- **Tests** — `test.py` moved to `tests/test_cuda_ops.py` (skips cleanly under CPU-only
  CI; still runnable as a script).
- **Docs** — README rewritten as an install → test → train → generate guide; combra
  `san_v2.md` doc kept in sync.

### Removed
- FFHQ leftovers from upstream: `scripts/train_ffhq16.sh`,
  `configs/experiment/ffhq16_stem.yaml`, `configs/experiment/ffhq32_superres.yaml`.
- Dead dependencies from `requirements.txt`: `imgui`, `glfw`, `pyopengl`,
  `imageio-ffmpeg`, and the pip `ninja` (ninja is installed from conda).
- `scripts/run_tests.sh` (use `python -m pytest tests/ -v`).
- `generate_64x64` sbatch script.

### Related (combra submodule, separate repo)
- CMMD now uses the `ViT-L-14-336-quickgelu` CLIP variant to match the `openai` weights
  (silences the QuickGELU activation-mismatch warning).
