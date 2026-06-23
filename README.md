# Slicing Adversarial Network (SAN) [ICLR 2024]

This repository contains a fork of the official PyTorch implementation of **"SAN: Inducing Metrizability of GAN with Discriminative Normalized Linear Layer"** (*[arXiv 2301.12811](https://arxiv.org/abs/2301.12811)*).
Please cite [[1](#citation)] in your work when using this code in your experiments.

### [[Project Page]](https://ytakida.github.io/san/)

This fork (`san-v2`) is specialised for generating **WC-Co microstructure SEM images**
(the `imagenet_9to4` dataset, three grain classes). It is trained **progressively**
(low → high resolution), every stage resuming from the previous stage's
`best_model.pkl`. See [Differences from upstream](#differences-from-the-original-sony-stylesan-xl)
for the engineering changes, and the combra docs page (`san_v2`) for how training
evaluation is wired into [combra](https://github.com/dkagramanyan/combra).

The guide below walks through **install → test → train → generate**. On the cluster
all four steps run on **H200 GPUs** via the ready-made Slurm scripts in
[`sbatch/`](sbatch/).


## 1. Installation

Create and activate a Python 3.12 conda env:

```bash
conda create -n san python=3.12 -y
conda activate san
```

Install, all into the conda env:

- the latest **PyTorch** (CUDA 13.2 wheels for H200; the wheel bundles the CUDA runtime),
- the **CUDA compiler** `nvcc`, used to JIT-build the custom ops — get it from conda's
  `nvidia` channel so it matches torch's CUDA (the pip wheel ships no `nvcc`),
- **ninja**, **from conda only** — a pip-installed ninja conflicts with it and the
  custom ops then fail to build.

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip3 install ninja    # match torch's CUDA major (13.x)
```

The training/inference scripts use this conda toolkit directly — they set
`CUDA_HOME=$CONDA_PREFIX` and load **no** system CUDA module.

Install the remaining dependencies (torch and ninja are intentionally not in
`requirements.txt`, so this won't disturb the versions above):

```bash
cd san-v2
pip install -r requirements.txt
```

Verify the toolchain:

```bash
conda list | grep -E "torch|cuda|cudnn|ninja"
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


## 3. Data preparation

`dataset_tool.py` packs a preprocessed image folder into a resolution-specific `.zip`.
Build one zip per stage of the progressive recipe (16² → 1024²):

```bash
for res in 16x16 32x32 64x64 128x128 256x256 512x512 1024x1024; do
  python dataset_tool.py \
    --source=/home/david/mnt/ssd_2_sata/python/phd/datasets/preprocessed/imagenet_9to4_1024x1024 \
    --dest=./datasets/imagenet_9to4_1024x1024_${res}.zip \
    --resolution=${res}
done
```


## 4. Training

Training is **progressive**: the 16² stem trains from scratch, and every higher
resolution is a super-resolution stage that resumes from the previous stage's
`best_model.pkl` via `--path_stem`.

```bash
# Stage 0 — 16x16 stem (no superres)
python train.py --outdir=./runs/wc-cv_h200 --cfg=stylegan3-r --cond True \
        --data=./datasets/imagenet_9to4_1024x1024_16x16.zip \
        --gpus=2 --mirror=0 --snap 500 --batch-gpu 320 --kimg 20000 --syn_layers 6

# Stage N — superres, resuming from the previous stage's best_model.pkl
python train.py --outdir=./runs/wc-cv_h200 --cfg=stylegan3-r --cond True \
        --data=./datasets/imagenet_9to4_1024x1024_32x32.zip \
        --gpus=2 --mirror=0 --snap 100 --batch-gpu 96 --kimg 20000 --syn_layers 6 \
        --superres --up_factor 2 --head_layers 7 \
        --path_stem ./runs/wc-cv_h200/00000-stylegan3-r-imagenet_9to4_1024x1024_16x16-gpus4-batch560/best_model.pkl
```

Per-stage tuned settings (resolution → per-GPU batch on 2× H200; `--batch-gpu` is
per GPU, so total batch = `batch-gpu × gpus`):

| stage | resolution | `--batch-gpu` | resumes from |
|---|---|---|---|
| 0 | 16×16   | 320 | — (stem) |
| 1 | 32×32   | 96  | stage 0 |
| 2 | 64×64   | 120 | stage 1 |
| 3 | 128×128 | 64  | stage 2 |
| 4 | 256×256 | 42  | stage 3 |
| 5 | 512×512 | 25  | stage 4 |
| 6 | 1024×1024 | 14 | stage 5 |

### Launching on H200 (Slurm)

Each stage has a ready-made script in [`sbatch/`](sbatch/) (2× H200 each). They are
written to be **submitted from the sbatch folder** and jobs land on the `rocky`
partition (no reservation needed):

```bash
cd sbatch
sbatch train_16x16.sbatch
sbatch train_32x32.sbatch
# … through train_1024x1024.sbatch
```

Each script resolves the repo root itself, loads the H200 toolchain
(`CUDA/12.9`, `TORCH_CUDA_ARCH_LIST=9.0`), activates the per-node conda env and
uses a persistent kernel cache, so a resubmit skips JIT recompilation.

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
  --network=./runs/wc-cv_h200/00004-stylegan3-r-imagenet_9to4_1024x1024_256x256-gpus4-batch168/best_model.pkl
```

Images are written per class into `class_<id>/<class>_<index>.png`; pass `--gpus`
(or launch with `torchrun`) to distribute generation across GPUs. On the cluster use the
per-resolution scripts [`sbatch/generate_256x256.sbatch`](sbatch/generate_256x256.sbatch),
`generate_512x512.sbatch` or `generate_1024x1024.sbatch` (submit from the sbatch folder,
same as training).

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


## Saving checkpoints vs. weights-only vs. inference-only snapshots

There are three kinds of artifacts the training loop can write — in **decreasing size**:

- **Full resume checkpoint** — `network-snapshot.pkl` (written when `--restart_every`
  is set). It contains the `G`/`D`/`G_ema` networks **and** the training progress
  (`cur_nimg`, tick, augmentation `p`, `pl_mean`, best FID) needed to resume training.
- **Weights-only snapshot** — pass `--save-weights-only=true` to also write
  `network-snapshot-<kimg>.pkl` every snapshot tick. These contain the `G`/`D`/`G_ema`
  weights (no resume state) — smaller than the full checkpoint, but still carry the
  large projected discriminator.
- **Inference-only snapshot** — pass `--save-inference-only=true` to also write
  `network-snapshot-<kimg>-inference.pkl` every snapshot tick. This holds **only
  `G_ema`** (no discriminator, no non-EMA generator, no resume state), so it is by far
  the smallest. It is exactly what `gen_images.py` / `calc_metrics.py` load
  (`legacy.load_network_pkl` mirrors `G_ema` onto `G` on load). **Not** for resuming.

The weights-only and inference-only snapshots are for inference/evaluation; use the
full resume checkpoint to continue training. The `sbatch/train_*.sbatch` scripts pass
`--save-inference-only True` so every run leaves small, ready-to-ship generators.

```bash
python train.py --outdir=./runs/wc-cv_h200 --cfg=stylegan3-r --cond True \
        --data=./datasets/imagenet_9to4_1024x1024_16x16.zip \
        --gpus=2 --batch-gpu 320 --snap 10 --save-inference-only=true
```


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
- **Weights-only** (`--save-weights-only`) and **inference-only** (`--save-inference-only`,
  `G_ema` only — the smallest artifact) snapshots, in addition to the full resume checkpoint.
- **CWD-independent ImageNet embedding loading** — the `in_embeddings/*.pkl` path is
  resolved relative to the repo root and overridable via the `SAN_EMBED` env var.
- **ImageNet 1024×1024 progressive-superres recipe** (the training commands above).
- **combra training-evaluation integration** — optional per-tick scoring of generated
  samples with `combra.metrics.compute_all_metrics` (see the combra `san_v2` docs).
