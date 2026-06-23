#!/bin/bash
# Download every pretrained weight that training + the combra metrics need, into one
# cache root the (offline) compute nodes share. Run this on a node WITH internet
# (e.g. a login node); the training jobs then need no network.
#
# Models fetched (URLs/filenames are resolved by timm / open_clip / torch.hub):
#   - timm:        deit_base_distilled_patch16_224, tf_efficientnet_lite0,
#                  deit_small_distilled_patch16_224           -> $TORCH_HOME/hub/checkpoints
#   - combra fid:  InceptionV3 (pytorch-fid weights)          -> $TORCH_HOME/hub/checkpoints
#   - combra cmmd: CLIP ViT-L-14-336 (open_clip, openai)      -> $XDG_CACHE_HOME/clip
#   - combra fd:   DINOv2 dinov2_vitb14 (torch.hub)           -> $TORCH_HOME/hub
#
# Usage:
#   bash download_models.sh                      # caches under $HOME/.cache (the defaults)
#   MODEL_CACHE=/shared/team/caches bash download_models.sh   # custom shared cache root
set -euo pipefail

# Run from the repo root (this script lives there) so `python tests/...` resolves.
cd "$(dirname "$(readlink -f "$0")")"

# Single cache root for every framework so training reads exactly what we fetch here.
# The default ($HOME/.cache) is the libraries' normal location, so with a shared $HOME
# the training jobs find the weights with no extra configuration.
export MODEL_CACHE="${MODEL_CACHE:-$HOME/.cache}"
export TORCH_HOME="${MODEL_CACHE}/torch"               # timm / torch.hub / pytorch-fid
export HF_HOME="${MODEL_CACHE}/huggingface"            # HuggingFace hub
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export XDG_CACHE_HOME="${MODEL_CACHE}"                 # open_clip / CLIP -> $XDG_CACHE_HOME/clip
mkdir -p "${TORCH_HOME}/hub/checkpoints" "${HUGGINGFACE_HUB_CACHE}"

echo "Caching pretrained models under: ${MODEL_CACHE}"
echo "  TORCH_HOME=${TORCH_HOME}"
echo "  HF_HOME=${HF_HOME}"
echo "  XDG_CACHE_HOME=${XDG_CACHE_HOME}"
echo

# Fail clearly if the interpreter itself is broken, instead of dumping a raw traceback.
if ! python -c "import sys" >/dev/null 2>&1; then
    echo "ERROR: '$(command -v python)' won't start. Fix the conda env first" >&2
    echo "       (see the Installation section of the README), then re-run." >&2
    exit 1
fi

# Python-driven so each library resolves the correct URL and filename into the caches
# exported above. Reuses the model list in tests/test_san_modules.py (single source).
python tests/test_san_modules.py

echo
echo "Done. If you used a custom MODEL_CACHE, export the same vars before training so the"
echo "jobs read this cache (add them to your sbatch files):"
echo "  export TORCH_HOME=${TORCH_HOME} HF_HOME=${HF_HOME} XDG_CACHE_HOME=${XDG_CACHE_HOME}"
