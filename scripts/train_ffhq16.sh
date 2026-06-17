#!/bin/bash
# Example: launch the FFHQ 16x16 stem stage via the Hydra entry point.
# Edit outdir/data to match your environment, then: bash scripts/train_ffhq16.sh
set -euo pipefail
cd "$(dirname "$0")/.."

python train_hydra.py +experiment=ffhq16_stem \
    outdir=./training-runs/ffhq \
    data=./data/ffhq16.zip
