#!/bin/bash
# Train san-v2 at 512x512. Override DATA/OUTDIR/GPUS via env, e.g.
#   DATA=./datasets/wc_co_512.zip GPUS=2 bash sh/train_512.sh
source "$(dirname "$0")/_env.sh"

san-train \
    --outdir "${OUTDIR}" \
    --cfg stylegan3-r \
    --data "${DATA}" \
    --gpus "${GPUS}" \
    --batch-gpu 16 \
    --cond True \
    --mirror False \
    --precision fp16 \
    --kimg 20000 \
    --tick 4 \
    --snap 100 \
    --snapshot-keep-last 3 \
    --combra-metrics True \
    --num-fid-samples 10000
