#!/bin/bash
#SBATCH --job-name=san-train-32
#SBATCH --partition=rocky
#SBATCH --account=proj_1631
#SBATCH --nodelist=cn-049
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=3-0:0
#SBATCH --gres-flags=enforce-binding

# Hop to the repo root so the script can be submitted from the sbatch folder,
# e.g. `cd sbatch && sbatch train_32x32.sbatch`.
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
while [[ ! -f train.py && "$PWD" != / ]]; do cd ..; done

conda activate san_rocky

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# H200 (Hopper sm_90) + persistent kernel cache for fast startup.
# Use the CUDA toolkit from the active conda env (no system CUDA module loaded).
export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_EXTENSIONS_DIR="${HOME}/.cache/torch_extensions"
export CUDA_CACHE_PATH="${HOME}/.cache/cuda_cache"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1

nvidia-smi
nvcc --version
conda list | grep -E "torch|cuda|cudnn|ninja"

python train.py --outdir=./runs/wc-cv_h200 \
        --cfg=stylegan3-r \
        --data=./datasets/imagenet_9to4_1024x1024_32x32.zip \
        --gpus=2 \
        --mirror=0 \
        --snap 100 \
        --batch-gpu 96 \
        --kimg 20000 \
        --cond True \
        --save-inference-only True \
        --syn_layers 6\
        --superres \
        --up_factor 2 \
        --head_layers 7 \
        --path_stem runs/wc-cv_h200/00000-stylegan3-r-imagenet_9to4_1024x1024_16x16-gpus4-batch560/best_model.pkl
