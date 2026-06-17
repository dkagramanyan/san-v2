#!/bin/bash
#SBATCH --job-name=GAN
#SBATCH --gpus=2
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

cd ..

python train.py --outdir=./training-runs/ffhq \
        --cfg=stylegan3-r \
        --data=./datasets/o_bc_left_4x_1536_1024x1024_rgb_16x16.zip \
        --gpus=2 \
        --batch=192 \
        --mirror=1 \
        --snap 500 \
        --batch-gpu 96 \
        --kimg 10000 \
        --cond True

