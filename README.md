# Slicing Adversarial Network (SAN) [ICLR 2024]

This repository contains the official PyTorch implementation of **"SAN: Inducing Metrizability of GAN with Discriminative Normalized Linear Layer"** (*[arXiv 2301.12811](https://arxiv.org/abs/2301.12811)*).
Please cite [[1](#citation)] in your work when using this code in your experiments.

### [[Project Page]](https://ytakida.github.io/san/)


## Installation

Create conda env with python=3.12

```
cd san
pip install -r requirements.txt
```


Uninstall old anaconda and cuda

```
pip uninstall torch torchvision -y
pip uninstall nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12  -y
```

Install new versions

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

You should install only 1 version of ninja, using anaconda. Without it you will get errors
```
conda install anaconda::ninja -y
```

Check 
```
conda list | grep -E "torch|cuda|cudnn"
```

After it is neccesary to check the plugins

```bash
python test.py
```


## FFHQ

### Data preparation  

```
python dataset_tool.py --source=/home/david/mnt/ssd_2_sata/python/phd/datasets/preprocessed/imagenet_9to4_1024x1024 \
                         --dest=/home/david/mnt/ssd_2_sata/python/phd/datasets/preprocessed/imagenet_9to4_1024x1024_16x16.zip \
                         --resolution=16x16

python dataset_tool.py --source=/home/david/mnt/ssd_2_sata/python/phd/datasets/preprocessed/imagenet_9to4_1024x1024 \
                         --dest=/home/david/mnt/ssd_2_sata/python/phd/datasets/preprocessed/imagenet_9to4_1024x1024_32x32.zip \
                         --resolution=32x32

python dataset_tool.py --source=/home/david/mnt/ssd_2_sata/python/phd/datasets/preprocessed/imagenet_9to4_1024x1024 \
                         --dest=/home/david/mnt/ssd_2_sata/python/phd/datasets/preprocessed/imagenet_9to4_1024x1024_64x64.zip \
                         --resolution=64x64                      

python dataset_tool.py --source=/home/david/mnt/ssd_2_sata/python/phd/datasets/preprocessed/imagenet_9to4_1024x1024 \
                         --dest=/home/david/mnt/ssd_2_sata/python/phd/datasets/preprocessed/imagenet_9to4_1024x1024_128x128.zip \
                         --resolution=128x128

python dataset_tool.py --source=/home/david/mnt/ssd_2_sata/python/phd/datasets/preprocessed/imagenet_9to4_1024x1024 \
                         --dest=/home/david/mnt/ssd_2_sata/python/phd/datasets/preprocessed/imagenet_9to4_1024x1024_256x256.zip \
                         --resolution=256x256

python dataset_tool.py --source=/home/david/mnt/ssd_2_sata/python/phd/datasets/preprocessed/imagenet_9to4_1024x1024 \
                         --dest=/home/david/mnt/ssd_2_sata/python/phd/datasets/preprocessed/imagenet_9to4_1024x1024_512x512.zip \
                         --resolution=512x512

python dataset_tool.py --source=/home/david/mnt/ssd_2_sata/python/phd/datasets/preprocessed/imagenet_9to4_1024x1024 \
                         --dest=/home/david/mnt/ssd_2_sata/python/phd/datasets/preprocessed/imagenet_9to4_1024x1024_1024x1024.zip \
                         --resolution=1024x1024

```

### Training
```
python train.py --outdir=./training-runs/ffhq --cfg=stylegan3-r --data=./data/ffhq16.zip \
        --gpus=8 --batch=2048 --mirror=1 --snap 10 --batch-gpu 8 squeue --syn_layers 6

python train.py --outdir=./training-runs/ffhq --cfg=stylegan3-r --data=./data/ffhq32.zip \
        --gpus=8 --batch=2048 --mirror=1 --snap 10 --batch-gpu 8 --kimg 175000 --syn_layers 6 \
        --superres --up_factor 2 --head_layers 7 \
        --path_stem training-runs/ffhq/00000-stylegan3-r-ffhq16-gpus8-batch2048/best_model.pkl

python train.py --outdir=./training-runs/ffhq --cfg=stylegan3-t --data=./data/ffhq64.zip \
        --gpus=8 --batch=256 --mirror=1 --snap 10 --batch-gpu 8 --kimg 95000 --syn_layers 6 \
        --superres --up_factor 2 --head_layers 4 \
        --path_stem training-runs/ffhq/00001-stylegan3-r-ffhq32-gpus8-batch2048/best_model.pkl

python train.py --outdir=./training-runs/ffhq --cfg=stylegan3-t --data=./data/ffhq128.zip \
        --gpus=8 --batch=256 --mirror=1 --snap 10 --batch-gpu 8 --kimg 57000 --syn_layers 6 \
        --superres --up_factor 2 --head_layers 4 \
        --path_stem training-runs/ffhq/00002-stylegan3-t-ffhq64-gpus8-batch256/best_model.pkl

python train.py --outdir=./training-runs/ffhq --cfg=stylegan3-t --data=./data/ffhq256.zip \
        --gpus=8 --batch=256 --mirror=1 --snap 10 --batch-gpu 8 --kimg 11000 --syn_layers 6 \
        --superres --up_factor 2 --head_layers 4 \
        --path_stem training-runs/ffhq/00003-stylegan3-t-ffhq128-gpus8-batch256/best_model.pkl

python train.py --outdir=./training-runs/ffhq --cfg=stylegan3-t --data=./data/ffhq512.zip \
        --gpus=8 --batch=128 --mirror=1 --snap 10 --batch-gpu 8 --kimg 4000 --syn_layers 6 \
        --superres --up_factor 2 --head_layers 4 \
        --path_stem training-runs/ffhq/00004-stylegan3-t-ffhq256-gpus8-batch256/best_model.pkl

python train.py --outdir=./training-runs/ffhq --cfg=stylegan3-t --data=./data/ffhq1024.zip \
        --gpus=8 --batch=128 --mirror=1 --snap 10 --batch-gpu 8 --kimg 4000 --syn_layers 6 \
        --superres --up_factor 2 --head_layers 4 \
        --path_stem training-runs/ffhq/00005-stylegan3-t-ffhq512-gpus8-batch128/best_model.pkl
```


## Generating Samples
```python
python gen_images.py \
  --outdir=./generated/ \
  --trunc=0.7 \
  --samples-per-class 1000 \
  --classes 0,1,2 \
  --gpus 4 \
  --batch-gpu 60 \
  --network=./runs/wc-cv_h200/00002-stylegan3-r-imagenet_9to4_1024x1024_64x64-gpus4-batch480/best_model.pkl
```
Images are written per class into `class_<id>/<class>_<index>.png`; run with `torchrun` or the `--gpus` flag to distribute work across available GPUs.

## Quality Metrics
You need to preprocess a dataset in advance, following Data Preparation.
To calculate metrics for a specific network snapshot, run
```
python calc_metrics.py --metrics=fid50k_full --network=<path_to_checkpoint>
python calc_metrics.py --metrics=is50k --network=<path_to_checkpoint>
```

The metric runners now gather features across GPUs via NCCL all-gathers,
eliminating the previous per-rank broadcast loop. When you launch metrics in
distributed mode the workload is evenly partitioned and every GPU remains busy.
During training the metric evaluators also inherit a dynamic per-GPU batch size
from the current run (capped between 32 and 512), which keeps the detector
queues full and cuts evaluation latency substantially.

## Saving checkpoints vs. weights-only snapshots

There are two kinds of artifacts the training loop can write:

- **Full resume checkpoint** — `network-snapshot.pkl` (written when `--restart_every`
  is set). It contains the `G`/`D`/`G_ema` networks **and** the training progress
  (`cur_nimg`, tick, augmentation `p`, `pl_mean`, best FID) needed to resume training.
  This feature is unchanged.
- **Weights-only snapshot** — pass `--save-weights-only=true` to also write
  `network-snapshot-<kimg>.pkl` every snapshot tick. These contain only the
  `G`/`D`/`G_ema` weights (no resume state), so they are smaller and are intended for
  inference/evaluation (`gen_images.py`, `calc_metrics.py`) — **not** for resuming
  training. Use the full checkpoint above to resume.

```
python train.py --outdir=./training-runs/ffhq --cfg=stylegan3-r --data=./data/ffhq16.zip \
        --gpus=8 --batch-gpu 8 --snap 10 --save-weights-only=true
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
- **Weights-only snapshots** (`--save-weights-only`, see above) in addition to the
  full resume checkpoint.
- **CWD-independent ImageNet embedding loading** — the `in_embeddings/*.pkl` path is
  resolved relative to the repo root and overridable via the `SAN_EMBED` env var.
- **ImageNet 1024×1024 progressive-superres recipe** (see the training commands above).
