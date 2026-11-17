#!/bin/bash
# Run a Python script from collect-prompts/ inside the LAIFS multitorch
# container (vLLM 0.26, ROCm, gfx90a). Use on LUMI-G nodes (inside Slurm jobs).
#
# Usage:  ./run_gpu.sh compile/llm_runner.py --requests ... --out ...
set -euo pipefail

SIF_GPU=${SIF_GPU:-/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706/lumi-multitorch-full-u24r72f21m50t211-20261028_090706.sif}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

export HF_HOME=${HF_HOME:-/pfs/lustrep3/scratch/project_465003466/dasamuel/hf_cache}
export HF_HUB_OFFLINE=1        # models are read from local paths only
# keep all compile/kernel caches off the (small, nearly full) home directory
CACHE=/pfs/lustrep3/scratch/project_465003466/dasamuel/gpu_cache
export VLLM_CACHE_ROOT=$CACHE/vllm VLLM_CONFIG_ROOT=$CACHE/vllm_config
export TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor
export XDG_CACHE_HOME=$CACHE/xdg VLLM_NO_USAGE_STATS=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

exec singularity exec -B /pfs "$SIF_GPU" bash -c \
    'cd "$0" && exec python -I -u "$@"' "$HERE" "$@"
