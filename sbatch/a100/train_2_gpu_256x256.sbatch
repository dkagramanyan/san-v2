#!/bin/bash
#SBATCH --job-name=SAN-GAN-256
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --time=4-0:0
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
        --data=./datasets/o_bc_left_4x_1536_1024x1024_rgb_256x256.zip \
        --gpus=2 \
        --batch=52 \
        --mirror=1 \
        --snap 100 \
        --batch-gpu 26 \
        --kimg 10000 \
        --cond True \
        --syn_layers 6 \
        --superres \
        --up_factor 2 \
        --head_layers 7 \
        --path_stem training-runs/ffhq/00014-stylegan3-r-o_bc_left_4x_1536_1024x1024_rgb_128x128-gpus2-batch80/best_model.pkl

