#!/bin/bash
#SBATCH --job-name=SAN-128
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --time=3-0:0
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --account=proj_1661
#SBATCH --reservation=rocky
#SBATCH --nodelist=cn-050
#SBATCH --gres-flags=enforce-binding

module purge
module load Python
module load CUDA/12.9

conda activate san_rocky_50

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

nvidia-smi

which nvcc
nvcc --version

which gcc
gcc --version

which g++
g++ --version

conda list | grep -E "torch|cuda|cudnn|ninja"

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export TORCH_CUDA_ARCH_LIST="9.0a"


python gen_images.py \
  --outdir=./generated/ \
  --trunc=0.7 \
  --samples-per-class 10000 \
  --classes 0,1,2 \
  --gpus 4 \
  --batch-gpu 60 \
  --network=./runs/wc-cv_h200/00002-stylegan3-r-imagenet_9to4_1024x1024_64x64-gpus4-batch480/best_model.pkl

