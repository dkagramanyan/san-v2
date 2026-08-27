# Slicing Adversarial Network (SAN) [ICLR 2024]

This repository contains a fork of the official PyTorch implementation of **"SAN: Inducing Metrizability of GAN with Discriminative Normalized Linear Layer"** (*[arXiv 2301.12811](https://arxiv.org/abs/2301.12811)*).
Please cite [[1](#citation)] in your work when using this code in your experiments.

### [[Project Page]](https://ytakida.github.io/san/)

This fork (`san-v2`) is specialised for generating **WC-Co microstructure SEM images**
(the `imagenet_9to4` dataset, three grain classes). It is trained **progressively**
(low → high resolution), every stage warm-starting its frozen stem from the previous
stage's inference snapshot via `--path-stem`. See [Differences from upstream](#differences-from-the-original-sony-stylesan-xl)
for the engineering changes, and the combra docs page (`san_v2`) for how training
evaluation is wired into [combra](https://github.com/dkagramanyan/combra).

The guide below walks through **install → test → train → generate**. On the cluster
the training and generation steps run through the launch scripts in [`sh/`](sh/)
(`sbatch --account=<proj> --partition=<part> --gpus=2 sh/train_256.sh`); the same
scripts run unmodified on a workstation.


## 1. Installation

Create and activate a Python 3.12 conda env:

```bash
conda create -n san-v2 python=3.12 -y
conda activate san-v2
```

Install, all into the conda env:

- the latest **PyTorch** (CUDA 13.2 wheels for H200; the wheel bundles the CUDA runtime),
- the **CUDA compiler** `nvcc`, used to JIT-build the custom ops — get it from conda's
  `nvidia` channel so it matches torch's CUDA (the pip wheel ships no `nvcc`),
- **ninja**, **from conda only** — a pip-installed ninja conflicts with it and the
  custom ops then fail to build.

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip3 install ninja
conda install -c nvidia cuda-nvcc -y      # match torch's CUDA major (13.x)
```

The training/inference scripts use this conda toolkit directly — they set
`CUDA_HOME=$CONDA_PREFIX` and load **no** system CUDA module.

Install the package (torch and ninja are intentionally *not* declared as
dependencies, so this won't disturb the versions above). Add the `combra` extra to
enable the in-training generative-quality metrics:

```bash
cd san-v2
pip install -e .                 # or: pip install -e '.[combra]'
```

The `combra` extra pulls `combra[metrics]`, not bare `combra` — since combra 0.5.0
the torch / `pytorch-fid` / `open-clip-torch` stack lives behind that extra, and
without it `combra_fid`, `combra_cmmd` and `combra_fd_dinov2` come back `nan`.
combra also floors Python at **3.12**, which is why this package does too.

`timm` (`>=1.0`) is a train-time-only dependency of the projected discriminator.
Checkpoints are EMA-only `.pt` state dicts, so loading one never needs `timm`; the
pickled `best_model.pkl` stems that once forced a `timm==0.4.12` pin are gone.

Verify the toolchain:

```bash
conda list | grep -E "torch|cuda|cudnn|ninja|timm"
```


## 2. Test the build

The custom CUDA ops (`bias_act`, `filtered_lrelu`, `upfirdn2d`, …) are JIT-compiled on
first use. Compile and check them against PyTorch references — including the H200
(Hopper `sm_90`) path — before training:

```bash
python tests/test_cuda_ops.py
```

The SAN-layer unit tests run on CPU and are wired into CI (the GPU CUDA-op tests above
are auto-skipped without a GPU):

```bash
python -m pytest tests/ -v
```

### Pre-download pretrained weights (offline nodes)

Training pulls pretrained backbones from the network on first use — the discriminator
backbones (`deit_base_distilled_patch16_224`, `tf_efficientnet_lite0`), the
classifier-guidance model (`deit_small_distilled_patch16_224`), and, for
`--combra-metrics`, the InceptionV3 / CLIP / DINOv2 weights. If your compute nodes have
no internet, fetch them once on a login node — they cache under `$HOME` (shared with the
compute nodes), so the training job then needs no network:

```bash
bash download_models.sh        # pure wget/curl/git, no Python
```

`download_models.sh` downloads each weight file straight into the caches the libraries
read (`$HOME/.cache/torch/hub/checkpoints`, `$HOME/.cache/clip`, …) — override the root
with `MODEL_CACHE=/shared/path` to land them on a shared filesystem. (A Python-driven
equivalent that always resolves the correct URLs is `python tests/test_san_modules.py`,
for when you have a working interpreter.)


## 3. Data preparation

`dataset_tool.py` packs a preprocessed image folder into a resolution-specific `.zip`.
The flags live under the `convert` subcommand (the CLI is a click group).
Build one zip per stage of the progressive recipe (16² → 1024²):

```bash
for res in 16x16 32x32 64x64 128x128 256x256 512x512 1024x1024; do
  python dataset_tool.py convert \
    --source=/home/david/mnt/ssd_2_sata/python/phd/datasets/preprocessed/imagenet_9to4_1024x1024 \
    --dest=./datasets/imagenet_9to4_1024x1024_${res}.zip \
    --resolution=${res}
done
```


## 4. Training

Training is **progressive**: the 16² stem trains from scratch, and every higher
resolution is a super-resolution stage that warm-starts its frozen stem from the
previous stage's newest inference snapshot via `--path-stem` (weights only — there
is no resume; see [Checkpoints](#checkpoints)).

```bash
# Stage 0 — 16x16 stem (no superres)
python train.py --outdir=./runs/wc-cv_h200 --cfg=stylegan3-r --cond True \
        --data=./datasets/imagenet_9to4_1024x1024_16x16.zip \
        --gpus=2 --mirror False --snap 500 --batch-gpu 320 --kimg 20000 --syn-layers 6

# Stage N — superres, warm-starting from the previous stage's snapshot
python train.py --outdir=./runs/wc-cv_h200 --cfg=stylegan3-r --cond True \
        --data=./datasets/imagenet_9to4_1024x1024_32x32.zip \
        --gpus=2 --mirror False --snap 100 --batch-gpu 96 --kimg 20000 --syn-layers 6 \
        --superres True --up-factor 2 --head-layers 7 \
        --path-stem ./runs/wc-cv_h200/00000-stylegan3-r-gpus2-batch640/san-snapshot-020000-inference.pt
```

Per-stage tuned settings (resolution → per-GPU batch on 2× H200; `--batch-gpu` is
per GPU, so total batch = `batch-gpu × gpus`):

| stage | resolution | `--batch-gpu` | stem from |
|---|---|---|---|
| 0 | 16×16   | 320 | — (stem) |
| 1 | 32×32   | 96  | stage 0 |
| 2 | 64×64   | 120 | stage 1 |
| 3 | 128×128 | 64  | stage 2 |
| 4 | 256×256 | 42  | stage 3 |
| 5 | 512×512 | 25  | stage 4 |
| 6 | 1024×1024 | 14 | stage 5 |

### Launching (workstation or SLURM)

[`sh/`](sh/) holds one script per resolution and task — `train_{256,512,1024}.sh` and
`generate_{256,512,1024}.sh`. Each contains only the compute-node environment (conda
env `san-v2`, `CUDA_HOME=$CONDA_PREFIX` for the JIT ops, `TORCH_CUDA_ARCH_LIST` from the
GPUs present, the offline-hub flags) and one `san-train` / `san-gen-images` call whose
knobs are env vars with defaults; anything after the script name is appended.

```bash
bash sh/train_256.sh                                      # workstation, defaults
sbatch --account=<proj> --partition=<part> --gpus=2 sh/train_256.sh   # cluster
# superres stage on top of the previous resolution's newest snapshot:
PATH_STEM=./runs/00000-stylegan3-r-gpus2-batch64/san-snapshot-020000-inference.pt \
    bash sh/train_512.sh
DATA=./datasets/my.zip KIMG=200 SNAP=2 bash sh/train_256.sh   # smoke run
```

No account, partition or node names live in the scripts — SLURM specifics are supplied
on the `sbatch` line.

### Hydra entry point

The same runs can be launched through [Hydra](https://hydra.cc) — `train_hydra.py`
shares `train.py`'s `build_config()`, so checkpoints and resume are interchangeable:

```bash
python train_hydra.py outdir=./runs cfg=stylegan3-r cond=true \
        data=./datasets/imagenet_9to4_1024x1024_16x16.zip gpus=2 batch_gpu=320
```

The click CLI is the single source of truth for defaults, so
[`configs/config.yaml`](configs/config.yaml) only declares the required fields
(`outdir`/`cfg`/`data`/`gpus`/`batch_gpu`); override any other `train.py` flag on the
command line using its Python name (e.g. `syn_layers=6`, `superres=true`).


## 5. Generating samples

```bash
python gen_images.py \
  --outdir=./generated/ \
  --trunc=0.7 \
  --samples-per-class 1000 \
  --classes 0,1,2 \
  --gpus 2 \
  --batch-gpu 60 \
  --network=./runs/wc-cv_h200/00004-stylegan3-r-gpus2-batch84/san-snapshot-020000-inference.pt
```

By default (`--save-mode hdf5`) each GPU writes a shard and rank 0 merges them into
`<outdir>/<desc>.h5` — the angle-pipeline input — refusing to merge an incomplete run;
`--save-mode dir` writes `class_<c>/idx_<i:06d>_seed_<s>.png` plus a `classes.json`
manifest. `--gpus N` self-spawns one worker per GPU (no `torchrun`). On the cluster use
`NETWORK=<snapshot> sbatch --account=<proj> --partition=<part> --gpus=2 sh/generate_256.sh`
(or `_512` / `_1024`).

> **Class index → grain morphology** is documented in the combra `san_v2` docs page;
> note the SAN index order differs from DiffiT (the `Co11`↔`Co25` swap).


## Quality Metrics

Score a trained snapshot with the StyleGAN-XL metric runners (build the matching
dataset zip first, per [Data preparation](#3-data-preparation)):

```bash
python calc_metrics.py --metrics=fid50k_full --network=<path_to_checkpoint>
python calc_metrics.py --metrics=is50k       --network=<path_to_checkpoint>
```

Metric runners gather features across GPUs via NCCL all-gathers (no per-rank
broadcast loop), and in distributed mode the workload is evenly partitioned so every
GPU stays busy. During training the metric evaluators inherit a dynamic per-GPU batch
size from the current run (capped between 32 and 512), keeping the detector queues
full and cutting evaluation latency.


## Checkpoints

There is exactly one artifact kind: `san-snapshot-<kimg:06d>-inference.pt` — the EMA
generator's weights plus self-describing metadata (`n_classes`, `resolution`,
`class_names`, `cur_nimg`), as a plain `.pt` state dict. No discriminator, no
optimizer state, no pickled modules, so loading never depends on `timm`.

- Written every `--snap` ticks **and always at the last tick**, so the newest snapshot
  is the final model; the write is atomic (temp file + `os.replace`).
- History is pruned to `--snapshot-keep-last` (default 3; `0` keeps all). Pick the best
  checkpoint post-hoc from `stats.jsonl` (`Metrics/combra_fid` per kimg) — there is no
  `best_model.*`.
- **There is no resume.** A run goes launch → `--kimg` → stop; size `--kimg` (or split
  stages) to fit the job's walltime. `--path-stem <snapshot>` is a weights-only warm
  start of the frozen lower-resolution stem, not a resume.
- `gen_images.py` and `calc_metrics.py` load these snapshots directly.


## Differences from the original Sony StyleSAN-XL

This repository is a fork of Sony's
[StyleSAN-XL](https://github.com/sony/san/tree/main/stylesan-xl) (which builds on
StyleGAN-XL → StyleGAN3 + Projected GAN). The **SAN training objective, optimizer
settings, and weight initialization are unchanged from upstream** — the changes here
are engineering / infrastructure improvements:

- **H200 kernel optimization** — a batched-matmul (BMM) path for 1×1 modulated
  convolutions in the generator (`training/networks_stylegan3_resetting.py`).
- **Fused Adam** for both G and D optimizers (`train.py`, `fused=True`).
- **CUDA-kernel warmup** that JIT-compiles all kernel configurations before the loop
  starts to avoid mid-training stalls (`training/training_loop.py:warmup_cuda_kernels`).
- **Distributed image generation with HDF5 output** in `gen_images.py`.
- **Dynamic per-GPU metric batch sizing** and NCCL all-gather based metric collection.
- **Unified, opt-in debug/timing instrumentation** across the training stack
  (`--debug`), writing to `<run_dir>/debug.txt` (no hardcoded paths).
- **EMA-only inference snapshots** (`san-snapshot-<kimg:06d>-inference.pt`, atomic,
  pruned to `--snapshot-keep-last`) as the single checkpoint kind — no resume state, no
  discriminator, so loading never needs `timm`.
- **CWD-independent ImageNet embedding loading** — the `in_embeddings/*.pkl` path is
  resolved relative to the repo root and overridable via the `SAN_EMBED` env var.
- **ImageNet 1024×1024 progressive-superres recipe** (the training commands above).
- **combra training-evaluation integration** — optional per-snapshot scoring of
  generated samples with `combra.metrics.compute_all_metrics`, logged to TensorBoard as
  `Metrics/combra_*`. Toggled by `--combra-metrics` (default `true`), **independent of
  `--metrics`**; warns at startup if enabled but combra is not installed (see the combra
  `san_v2` docs).
