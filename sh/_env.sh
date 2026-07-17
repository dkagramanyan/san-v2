#!/bin/bash
# Shared environment for the san-v2 launch scripts (§9). Sourced by every
# sh/train_*.sh and sh/generate_*.sh. Contains ONLY the compute-node environment;
# the actual `san-train` / `san-gen-images` call lives in each script.
#
# No hardcoded user homes, --nodelist or account IDs live here: SLURM specifics
# are supplied at submission time, e.g.
#   sbatch --account=<proj> --partition=rocky --gpus=2 sh/train_256.sh
# The same script also runs unmodified on a workstation: `bash sh/train_256.sh`.

set -euo pipefail

# Self-locate the repo root (walk up to pyproject.toml) so the script works from
# anywhere it is submitted.
_dir="${SLURM_SUBMIT_DIR:-$PWD}"
while [[ ! -f "$_dir/pyproject.toml" && "$_dir" != / ]]; do _dir="$(dirname "$_dir")"; done
cd "$_dir"

# conda env name == repo name.
conda activate san-v2 || true

export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

# CUDA toolkit from the active conda env; Hopper (sm_90) + persistent kernel cache.
export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX:-}}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${HOME}/.cache/torch_extensions}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${HOME}/.cache/cuda_cache}"

# Offline-cluster contract: backbones are prefetched once on a login node via
# `bash download_models.sh`; compute nodes never reach the network.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Common overridable knobs.
export OUTDIR="${OUTDIR:-./runs}"
export DATA="${DATA:-./datasets/dataset.zip}"
export GPUS="${GPUS:-2}"
