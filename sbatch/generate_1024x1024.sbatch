#!/bin/bash
#SBATCH --job-name=san-gen-1024
#SBATCH --partition=rocky
#SBATCH --account=proj_1661
#SBATCH --nodelist=cn-051
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=3-0:0
#SBATCH --gres-flags=enforce-binding

# Hop to the repo root so the script can be submitted from the sbatch folder,
# e.g. `cd sbatch && sbatch generate_1024x1024.sbatch`.
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
while [[ ! -f gen_images.py && "$PWD" != / ]]; do cd ..; done

module purge
module load Python

conda activate san_rocky_51

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
conda list | grep -E "torch|cuda|cudnn|ninja|pigz"

python gen_images.py \
  --outdir=./generated/ \
  --trunc=0.7 \
  --samples-per-class 100000 \
  --classes 0,1,2 \
  --gpus 2 \
  --batch-gpu 16 \
  --network=./runs/wc-cv_h200/00006-stylegan3-r-imagenet_9to4_1024x1024_1024x1024-gpus4-batch56/best_model.pkl
