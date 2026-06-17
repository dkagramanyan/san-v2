#!/bin/bash
#SBATCH --job-name=generate-GAN-SAN
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-4:0
#SBATCH --partition=normal
#SBATCH --account=proj_1631
#SBATCH --constraint="type_e"

module purge
module load Python/Anaconda
module load CUDA/12.4
module load gnu13/13.3

source activate san

cd ..

python gen_class_samplesheet.py \
  --outdir=sample_sheets \
  --trunc=0.7 \
  --samples-per-class 10000 \
  --classes 0,1,2 \
  --grid-width 1 \
  --network=./training-runs/ffhq/00016-stylegan3-r-o_bc_left_4x_1536_1024x1024_rgb_256x256-gpus2-batch52/best_model.pkl

