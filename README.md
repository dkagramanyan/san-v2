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

```python

# Pre-compile CUDA extensions (run this once before multi-GPU training)
import os
os.chdir('/home/dgkagramanyan/san-v2')
print(f"Working directory: {os.getcwd()}")

import torch
from torch_utils import custom_ops

# Set verbosity to see compilation progress
custom_ops.verbosity = 'full'

# Force compilation by importing the ops
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import filtered_lrelu

# Trigger compilation
print("Compiling upfirdn2d...")
upfirdn2d._init()

print("Compiling bias_act...")
bias_act._init()

print("Compiling filtered_lrelu...")
filtered_lrelu._init()

print("Pre-compilation complete!")
```

If there are no error, the training will be successful 

if there are errors
1) clearn torch_extentions cache
2) install nvcc

```
conda install -c conda-forge cuda-toolkit=13.0
conda install -c nvidia cuda-nvcc=13.0
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
```
python gen_class_samplesheet.py --outdir=generated --trunc=0.7 \
  --samples-per-class 100000 --classes 0,1,2 --grid-width 8 --batch-gpu=8 --batch-latent=4 \
  --network=./runs/ffhq/00017-stylegan3-r-o_bc_left_4x_1536_1024x1024_rgb_512x512-gpus2-batch20/best_model.pkl
```

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
