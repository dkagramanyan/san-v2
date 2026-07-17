#!/bin/bash
# Train san-v2 at 1024x1024. Override DATA/OUTDIR/GPUS via env, e.g.
#   DATA=./datasets/wc_co_1024.zip GPUS=2 bash sh/train_1024.sh
source "$(dirname "$0")/_env.sh"

san-train \
    --outdir "${OUTDIR}" \
    --cfg stylegan3-r \
    --data "${DATA}" \
    --gpus "${GPUS}" \
    --batch-gpu 8 \
    --cond True \
    --mirror False \
    --precision fp16 \
    --kimg 20000 \
    --tick 4 \
    --snap 100 \
    --snapshot-keep-last 3 \
    --combra-metrics True \
    --num-fid-samples 10000
