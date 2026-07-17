#!/bin/bash
# Generate 1024x1024 samples. Override NETWORK/OUTDIR/CLASSES/GPUS via env, e.g.
#   NETWORK=./runs/00000-stylegan3-r-gpus2-batch64/san-snapshot-020000-inference.pt \
#   CLASSES=Ultra_Co11,Ultra_Co25 bash sh/generate_1024.sh
source "$(dirname "$0")/_env.sh"

san-gen-images \
    --network "${NETWORK:?set NETWORK to a san-snapshot-*-inference.pt}" \
    --outdir "${OUTDIR:-./generated}" \
    --classes "${CLASSES:-0,1,2}" \
    --samples-per-class "${SAMPLES_PER_CLASS:-1000}" \
    --gpus "${GPUS}" \
    --batch-gpu 8 \
    --trunc 0.7 \
    --save-mode hdf5
