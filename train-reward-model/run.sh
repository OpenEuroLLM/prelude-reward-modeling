#!/bin/bash
# Run a Python script from train-reward-model/ inside the LAIFS multitorch container (torch 2.11, ROCm 7.2,
# transformers 5.10, flash-attention). On the login node for data preparation, inside Slurm jobs for training.
#
# Usage:  ./run.sh prepare_data.py split
set -euo pipefail

SIF=${SIF:-/appl/local/laifs/containers/lumi-multitorch-u24r72f21m50t211-20261028_090706/lumi-multitorch-full-u24r72f21m50t211-20261028_090706.sif}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

export HF_HOME=${HF_HOME:-/pfs/lustrep3/scratch/project_465003466/dasamuel/hf_cache}
CACHE=/pfs/lustrep3/scratch/project_465003466/dasamuel/gpu_cache
export TRITON_CACHE_DIR=$CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor XDG_CACHE_HOME=$CACHE/xdg
export MIOPEN_USER_DB_PATH=$CACHE/miopen MIOPEN_CUSTOM_CACHE_DIR=$CACHE/miopen
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}

# the bindings of the lumi-aif-singularity-bindings module: /var/spool/slurmd lets the CXI provider of the RCCL
# network plugin create its fabric domain (without it, inter-node traffic silently falls back to sockets)
BIND=/pfs,/scratch,/projappl,/project,/flash,/appl
[ -d /var/spool/slurmd ] && BIND=/var/spool/slurmd,$BIND,/boot
export SLURM_MPI_TYPE=pmi2

exec singularity exec -B "$BIND" "$SIF" bash -c 'cd "$0" && exec python -I -u "$@"' "$HERE" "$@"
