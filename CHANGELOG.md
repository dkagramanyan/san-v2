# Changelog

All notable changes to this fork (`san-v2`) are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

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
