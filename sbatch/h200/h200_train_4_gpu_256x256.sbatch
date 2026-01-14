#!/bin/bash
#SBATCH --job-name=SAN-128
#SBATCH --gpus=8
#SBATCH --cpus-per-task=16
#SBATCH --time=3-0:0
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --account=proj_1661
#SBATCH --reservation=rocky
#SBATCH --nodelist=cn-051
#SBATCH --gres-flags=enforce-binding

module purge
module load Python
module load CUDA/12.9

conda activate san_rocky_51

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
export TORCH_CUDA_ARCH_LIST="9.0"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# export PYTHONWARNINGS=all

export TORCH_EXTENSIONS_DIR="${HOME}/.cache/torch_extensions"
export CUDA_CACHE_PATH="${HOME}/.cache/cuda_cache"



python train.py --outdir=./runs/wc-cv_h200_51 \
        --cfg=stylegan3-r \
        --data=./datasets/imagenet_9to4_1024x1024_256x256.zip \
        --gpus=4 \
        --mirror=0 \
        --tick 1 \
        --workers 2 \
        --snap 100 \
        --batch-gpu 42 \
        --kimg 20000 \
        --cond True \
        --syn_layers 6\
        --superres \
        --up_factor 2 \
        --head_layers 7 \
        --path_stem ./runs/wc-cv_h200/00003-stylegan3-r-imagenet_9to4_1024x1024_128x128-gpus4-batch256/best_model.pkl
        