#!/bin/bash
#SBATCH --job-name=SAN-128
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
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

# Use persistent cache directories for compiled CUDA kernels (critical for fast startup!)
export TORCH_EXTENSIONS_DIR="${HOME}/.cache/torch_extensions"
export CUDA_CACHE_PATH="${HOME}/.cache/cuda_cache"
# export CUDA_CACHE_MAXSIZE=4294967296  # 4GB cache for JIT compiled kernels
# export CUDA_CACHE_DISABLE=0           # Ensure caching is enabled
# export CUDA_MODULE_LOADING=EAGER      # Load all CUDA modules at startup (reduces first-use latency)


python test.py