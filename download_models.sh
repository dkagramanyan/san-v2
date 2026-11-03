#!/bin/bash
# Pure-bash prefetch of every pretrained weight training + the combra metrics need,
# straight into the caches the libraries look in -- no Python, no GPU required. Run on a
# node WITH internet (e.g. a login node); the weights cache under $HOME, shared with the
# offline compute nodes, so the training jobs then need no network.
#
# timm 1.x and open_clip weights use the HuggingFace hub cache; pytorch-fid and
# torch.hub dinov2 use the Torch cache. HuggingFace revisions and SHA256s are pinned.
#
# Usage:
#   bash download_models.sh                      # caches under $HOME/.cache (the defaults)
#   MODEL_CACHE=/shared/team/caches bash download_models.sh
# With MODEL_CACHE set, point the jobs at it:
#   export TORCH_HOME=$MODEL_CACHE/torch HF_HOME=$MODEL_CACHE/huggingface
set -uo pipefail

MODEL_CACHE="${MODEL_CACHE:-$HOME/.cache}"
# Explicit library cache overrides take precedence, just as in the training scripts.
HUB_DIR="${TORCH_HOME:-${MODEL_CACHE}/torch}/hub"
HUB_CKPT="$HUB_DIR/checkpoints"
HF_HUB="${HF_HUB_CACHE:-${HF_HOME:-${MODEL_CACHE}/huggingface}/hub}"
mkdir -p "$HUB_CKPT" "$HUB_DIR" "$HF_HUB" || exit 1

if ! command -v wget >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: need wget or curl on PATH." >&2
    exit 1
fi
if ! command -v sha256sum >/dev/null 2>&1; then
    echo "ERROR: need sha256sum on PATH." >&2
    exit 1
fi

status=0
fetch() {  # fetch <url> <dest> [sha256]
    local url="$1" dest="$2" sha="${3:-}" part="${2}.part"
    if [[ -s "$dest" ]]; then
        if [[ -z "$sha" ]] || printf '%s  %s\n' "$sha" "$dest" | sha256sum --check --status; then
            echo "  exists: ${dest##*/}"
            return 0
        fi
        echo "  checksum mismatch; downloading again: ${dest##*/}"
    fi
    echo "  downloading: ${url##*/}"
    # Publish only complete files. A killed transfer leaves .part, never a cache hit.
    if command -v wget >/dev/null 2>&1; then
        wget -O "$part" "$url" || { echo "  FAILED: $url"; status=1; return 1; }
    else
        curl -fL -o "$part" "$url" || { echo "  FAILED: $url"; status=1; return 1; }
    fi
    if [[ ! -s "$part" ]] || { [[ -n "$sha" ]] && ! printf '%s  %s\n' "$sha" "$part" | sha256sum --check --status; }; then
        echo "  FAILED: empty file or checksum mismatch: $url" >&2
        status=1
        return 1
    fi
    mv -f "$part" "$dest" || { status=1; return 1; }
}

fetch_hf() {  # fetch_hf <repo> <revision> <filename> <sha256>
    local repo="$1" rev="$2" filename="$3" sha="$4"
    local cache="$HF_HUB/models--${1//\//--}"
    echo "  $repo/$filename"
    mkdir -p "$cache/blobs" "$cache/refs" "$cache/snapshots/$rev" || { status=1; return 1; }
    fetch "https://huggingface.co/$repo/resolve/$rev/$filename" "$cache/blobs/$sha" "$sha" || return 1
    ln -sfn "../../blobs/$sha" "$cache/snapshots/$rev/$filename" || { status=1; return 1; }
    # refs/main lets huggingface_hub resolve these pinned snapshots while offline.
    printf '%s' "$rev" > "$cache/refs/main.part" &&
        mv -f "$cache/refs/main.part" "$cache/refs/main" || { status=1; return 1; }
}

echo "Caching pretrained models under: $MODEL_CACHE"

echo; echo "[1/4] timm discriminator / classifier backbones -> $HF_HUB"
fetch_hf "timm/deit_base_distilled_patch16_224.fb_in1k" \
    "ae7e500ba066fc77dcce2bf518f54a093b9f9c01" "model.safetensors" \
    "ccc9d1bbeede1fc8609a7a7482773a35057e9f38035b9b804bced9126c5a70dc"
fetch_hf "timm/deit_small_distilled_patch16_224.fb_in1k" \
    "b53f4f1eb32f73f8a02164626258a9d3c5ad8a62" "model.safetensors" \
    "81d5995a0066bf8c861effc54841f894fc73a5074834985e2d406b55eeb1e3f6"
fetch_hf "timm/tf_efficientnet_lite0.in1k" \
    "e074755b39c3cd8ca77d917a7cfb62e70da6bf88" "model.safetensors" \
    "1b2a452450b5cf3f8f42da5b6214e71d5fda5666bef4dfae0699407e5bef13e1"

echo; echo "[2/4] InceptionV3 FID weights (combra fid) -> $HUB_CKPT"
fetch "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth" "$HUB_CKPT/pt_inception-2015-12-05-6726825d.pth"

echo; echo "[3/4] CLIP ViT-L-14-336 'openai' (combra cmmd) -> $HF_HUB"
fetch_hf "timm/vit_large_patch14_clip_336.openai" \
    "81e38efc4637de5023b10e75a7f9bd1c6fa6b010" "open_clip_model.safetensors" \
    "fbc415c3d0d7b79faed8f5ccfb740c32b7c4f5ffe7283b851f89c6231c01a8e0"

echo; echo "[4/4] DINOv2 dinov2_vitl14 (combra fd_dinov2) -> $HUB_DIR"
fetch "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth" "$HUB_CKPT/dinov2_vitl14_pretrain.pth"
# torch.hub also needs the dinov2 model code (it normally fetches the repo itself).
if [[ -d "$HUB_DIR/facebookresearch_dinov2_main" ]]; then
    echo "  exists: facebookresearch_dinov2_main/"
elif command -v git >/dev/null 2>&1; then
    echo "  cloning facebookresearch/dinov2"
    git clone --depth 1 https://github.com/facebookresearch/dinov2 "$HUB_DIR/facebookresearch_dinov2_main" \
        || { echo "  FAILED: git clone dinov2"; status=1; }
else
    echo "  SKIPPED dinov2 repo: git not on PATH"; status=1
fi

echo
if [[ $status -eq 0 ]]; then
    echo "Done. Weights cached in $HUB_CKPT and $HF_HUB."
else
    echo "Some downloads failed (see above)."
fi
exit $status
