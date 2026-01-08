#!/bin/bash
#SBATCH --job-name=SAN-GAN-16
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --time=3-0:0
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --account=proj_1631
#SBATCH --reservation=rocky
#SBATCH --nodelist=cn-049

module purge
module load Python

conda activate san_rocky

nvidia-smi
gcc --version
conda list | grep -E "torch|cuda|cudnn"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

rm -rf ~/.cache/torch_extensions

python train.py --outdir=./runs/wc-cv_h200 \
        --cfg=stylegan3-r \
        --data=./datasets/imagenet_9to4_1024x1024_16x16.zip \
        --gpus=4 \
        --batch=1280 \
        --mirror=0 \
        --snap 500 \
        --batch-gpu 320 \
        --kimg 20000 \
        --cond True \
        --syn_layers 6
        