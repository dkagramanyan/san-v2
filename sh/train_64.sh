#!/usr/bin/env bash
# san -- train at 64x64.
#
# Workstation:  bash sh/train_64.sh                (detaches, prints the log path)
#               FOREGROUND=1 bash sh/train_64.sh   (stays attached; output still goes to the log)
# SLURM:        sbatch --account=<proj> --partition=<part> --nodes=1 --gpus=2 --cpus-per-task=8 --time=3-0:0 sh/train_64.sh
#
# Defaults target the production allocation: 2x H200 (sm_90), 8 CPUs, fixed seed 42.
#
# Every knob is in the run settings block below: edit it there, or override one for a
# single launch with an env var (DATA=<zip> GPUS=<n> bash sh/train_64.sh). Anything
# after the script name is appended to the command (e.g. `... --kimg 200 --snap 2` for a
# smoke run). No user homes, --nodelist or account IDs live here -- SLURM specifics come
# from the sbatch line (spec §9).
set -euo pipefail

# --- Run settings --------------------------------------------------------------
# NAME="${NAME:-default}": the default is used unless NAME is set in the environment.
CONDA_ENV="${CONDA_ENV:-san-v2}"   # env name = repo name
OUTDIR="${OUTDIR:-./runs}"
CFG="${CFG:-stylegan3-r}"
DATA="${DATA:-./datasets/imagenet_9to4_1024x1024_64x64.zip}"
GPUS="${GPUS:-2}"
BATCH_GPU="${BATCH_GPU:-120}"         # per GPU
SYN_LAYERS="${SYN_LAYERS:-6}"
PRECISION="${PRECISION:-fp16}"
KIMG="${KIMG:-20000}"
TICK="${TICK:-4}"                     # kimg per tick
SNAP="${SNAP:-100}"                   # ticks per snapshot + combra eval
KEEP_LAST="${KEEP_LAST:-1}"           # newest snapshots kept (bests are never pruned)
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-10000}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-3}"               # data-loader workers per rank
PATH_STEM="${PATH_STEM:-}"            # previous stage's snapshot; empty = train from scratch
UP_FACTOR="${UP_FACTOR:-2}"           # superres stage only (PATH_STEM set)
HEAD_LAYERS="${HEAD_LAYERS:-7}"       # superres stage only (PATH_STEM set)

# --- Environment -------------------------------------------------------------
# Repo root: under SLURM the script runs from a spool copy, so walk up from the submit
# dir there and from this file's own location on a workstation.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
while [[ ! -f "$REPO_DIR/pyproject.toml" && "$REPO_DIR" != / ]]; do REPO_DIR="$(dirname "$REPO_DIR")"; done
[[ -f "$REPO_DIR/pyproject.toml" ]] || { echo "cannot find the repo root -- submit from inside the repo" >&2; exit 1; }
# The stem path is stored in every snapshot of this stage and re-read whenever one is
# loaded, so make it absolute (relative to where the job was launched) before the cd.
# Exported so the detached re-launch below (which starts in the repo root) keeps it.
if [[ -n "$PATH_STEM" ]]; then PATH_STEM="$(realpath -e "$PATH_STEM")"; export PATH_STEM; fi
SELF="$(realpath "${BASH_SOURCE[0]}")"   # this file, for the detached re-launch below
cd "$REPO_DIR"

# --- Launch: detach and log ----------------------------------------------------
# On a workstation the script re-launches itself in its own session (setsid nohup) and
# returns at once: the run survives closing the terminal, and everything it prints
# goes to logs/<name>-<date>.log (with a .pid file beside it). FOREGROUND=1 keeps it
# attached; the output is still copied to the log. Under SLURM it never detaches (the
# job already runs unattended); the output goes both to the slurm .out and to the log.
RUN_NAME=san-train_64
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs}"
if [[ -z "${RUN_LOG:-}" ]]; then
    mkdir -p "$LOG_DIR"
    RUN_LOG="$LOG_DIR/$RUN_NAME-$(date +%Y%m%d-%H%M%S).log"
    export RUN_LOG
    if [[ -z "${SLURM_JOB_ID:-}" && "${FOREGROUND:-0}" != 1 ]]; then
        setsid nohup bash "$SELF" "$@" > "$RUN_LOG" 2>&1 < /dev/null &
        pid=$!
        echo "$pid" > "${RUN_LOG%.log}.pid"
        echo "Started $RUN_NAME in the background (pid $pid, its own process group)."
        echo "  log:    $RUN_LOG"
        echo "  follow: tail -f $RUN_LOG"
        echo "  stop:   kill -- -$pid"
        exit 0
    fi
    exec > >(tee -a "$RUN_LOG") 2>&1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
# Custom CUDA ops JIT-compile on first import: nvcc from the conda env, arch list from
# H200 (sm_90) by default; explicit values win. Persistent kernel caches skip the JIT rebuild.
export CC="${CC:-gcc}" CXX="${CXX:-g++}"
export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX}}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"   # H200 = sm_90
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${HOME}/.cache/torch_extensions}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${HOME}/.cache/cuda_cache}"
# Offline-cluster contract: backbones are prefetched once on a login node
# (bash download_models.sh); compute nodes never reach the network.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"      # CLIP (CMMD) weights
export TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"      # torch.hub DINOv2 + Inception weights

# GPUs / CPUs: 2x H200 and 8 CPUs. SLURM sets CUDA_VISIBLE_DEVICES itself; the default
# only applies on a workstation. 8 CPUs / 2 ranks -> 4 threads per rank, 3 loader
# workers per rank (WORKERS) so the two main processes keep a core each.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# Determinism / logging: PYTHONHASHSEED pins Python hashing alongside --seed; NCCL
# surfaces a dead rank as an error instead of a hang; Python output is unbuffered so
# the SLURM log follows the run.
export PYTHONHASHSEED=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1

# --- One console-command call ------------------------------------------------
# Progressive recipe (README §4): train_16.sh trains the 16x16 stem from scratch; every
# higher resolution is a superres stage -- set PATH_STEM to one of the previous stage's
# kept snapshots, normally its best-FID one (named in that run's last "Best snapshots:"
# log line; weights-only warm start of the frozen stem); leave it unset to train this
# resolution from scratch instead. Defaults follow the README's per-stage table
# (--syn-layers 6, --head-layers 7, per-GPU batch).
# Retention: KEEP_LAST newest snapshots plus the best by combra_fid / combra_fd_dinov2 /
# combra_cmmd, which are never pruned (at most 4 files with the default KEEP_LAST=1).
STEM_ARGS=()
if [[ -n "$PATH_STEM" ]]; then
    STEM_ARGS=(--superres True --up-factor "$UP_FACTOR" --head-layers "$HEAD_LAYERS" --path-stem "$PATH_STEM")
fi

CMD=(
    san-train
    --outdir "$OUTDIR"
    --cfg "$CFG"
    --data "$DATA"
    --gpus "$GPUS"
    --batch-gpu "$BATCH_GPU"
    --cond True --syn-layers "$SYN_LAYERS"
    --precision "$PRECISION"
    --kimg "$KIMG" --tick "$TICK" --snap "$SNAP" --snapshot-keep-last "$KEEP_LAST"
    --combra-metrics True --num-fid-samples "$NUM_FID_SAMPLES"
    --seed "$SEED" --workers "$WORKERS"
    ${STEM_ARGS[@]+"${STEM_ARGS[@]}"}
    "$@"
)

# The log alone reproduces the run: settings, code version, machine, command.
commit="$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
[[ -z "$(git -C "$REPO_DIR" status --porcelain 2>/dev/null)" ]] || commit+=-dirty
echo "Run settings:"
for name in CONDA_ENV OUTDIR CFG DATA GPUS BATCH_GPU SYN_LAYERS PRECISION KIMG TICK SNAP KEEP_LAST \
            NUM_FID_SAMPLES SEED WORKERS PATH_STEM UP_FACTOR HEAD_LAYERS; do
    echo "  $name=${!name}"
done
echo "  commit=$commit"
echo "  host=$HOSTNAME"
echo "  date=$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  command=$(printf '%q ' "${CMD[@]}")"

"${CMD[@]}"
