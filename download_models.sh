#!/bin/bash
# Pure-bash prefetch of every pretrained weight training + the combra metrics need,
# straight into the caches the libraries look in -- no Python, no GPU required. Run on a
# node WITH internet (e.g. a login node); the weights cache under $HOME, shared with the
# offline compute nodes, so the training jobs then need no network.
#
# URLs and on-disk filenames are pinned to what the installed libraries expect
# (timm==0.4.12, pytorch-fid, open_clip 'openai', torch.hub dinov2). If a library version
# differs it may still re-download at runtime -- verify the files below exist afterwards.
#
# Usage:
#   bash download_models.sh                      # caches under $HOME/.cache (the defaults)
#   MODEL_CACHE=/shared/team/caches bash download_models.sh
set -u

MODEL_CACHE="${MODEL_CACHE:-$HOME/.cache}"
HUB_CKPT="${MODEL_CACHE}/torch/hub/checkpoints"   # timm + pytorch-fid + dinov2 weights
HUB_DIR="${MODEL_CACHE}/torch/hub"                # torch.hub repo code (dinov2)
CLIP_DIR="${MODEL_CACHE}/clip"                    # open_clip 'openai' weights
mkdir -p "$HUB_CKPT" "$HUB_DIR" "$CLIP_DIR"

if ! command -v wget >/dev/null 2>&1 && ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: need wget or curl on PATH." >&2
    exit 1
fi

status=0
fetch() {  # fetch <url> <dest>
    local url="$1" dest="$2"
    if [[ -s "$dest" ]]; then
        echo "  exists: ${dest##*/}"
        return 0
    fi
    echo "  downloading: ${url##*/}"
    if command -v wget >/dev/null 2>&1; then
        wget -c -O "$dest" "$url" || { echo "  FAILED: $url"; rm -f "$dest"; status=1; return 1; }
    else
        curl -fL -o "$dest" "$url" || { echo "  FAILED: $url"; rm -f "$dest"; status=1; return 1; }
    fi
}

echo "Caching pretrained models under: $MODEL_CACHE"

echo; echo "[1/4] timm discriminator / classifier backbones -> $HUB_CKPT"
fetch "https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth"  "$HUB_CKPT/deit_base_distilled_patch16_224-df68dfff.pth"
fetch "https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth" "$HUB_CKPT/deit_small_distilled_patch16_224-649709d9.pth"
fetch "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_lite0-0aa007d2.pth" "$HUB_CKPT/tf_efficientnet_lite0-0aa007d2.pth"

echo; echo "[2/4] InceptionV3 FID weights (combra fid) -> $HUB_CKPT"
fetch "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth" "$HUB_CKPT/pt_inception-2015-12-05-6726825d.pth"

echo; echo "[3/4] CLIP ViT-L-14-336 'openai' (combra cmmd) -> $CLIP_DIR"
fetch "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt" "$CLIP_DIR/ViT-L-14-336px.pt"

echo; echo "[4/4] DINOv2 dinov2_vitb14 (combra fd_dinov2) -> $HUB_DIR"
fetch "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth" "$HUB_CKPT/dinov2_vitb14_pretrain.pth"
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
    echo "Done. All weights cached under $MODEL_CACHE."
else
    echo "Some downloads failed (see above)."
fi
exit $status
