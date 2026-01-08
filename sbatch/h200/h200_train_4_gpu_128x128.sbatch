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

conda activate san_rocky_50

nvidia-smi
gcc --version
conda list | grep -E "torch|cuda|cudnn"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

rm -rf ~/.cache/torch_extensions

python train.py --outdir=./runs/wc-cv_h200 \
        --cfg=stylegan3-r \
        --data=./datasets/imagenet_9to4_1024x1024_128x128.zip \
        --gpus=4 \
        --batch=240 \
        --mirror=0 \
        --snap 100 \
        --batch-gpu 60 \
        --kimg 20000 \
        --cond True \
        --syn_layers 6\
        --superres \
        --up_factor 2 \
        --head_layers 7 \
        --path_stem runs/wc-cv_h200/00002-stylegan3-r-imagenet_9to4_1024x1024_64x64-gpus4-batch480/best_model.pkl
        