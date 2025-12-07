#!/bin/bash
#SBATCH --job-name=SAN-GAN-16
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --time=3-0:0
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --account=proj_1631
#SBATCH --constraint="type_e"

module purge
module load Python/Anaconda
module load CUDA/12.4
module load gnu13/13.3

source activate san

nvidia-smi
gcc --version
conda list | grep cuda

python train.py --outdir=./training-runs/wc-cv \
        --cfg=stylegan3-r \
        --data=./datasets/imagenet_9to4_1024x1024_16x16.zip \
        --gpus=4 \
        --batch=384 \
        --mirror=0 \
        --snap 500 \
        --batch-gpu 96 \
        --kimg 20000 \
        --cond True \
        --syn_layers 6

