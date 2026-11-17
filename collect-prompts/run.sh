#!/bin/bash
# Run a Python script from collect-prompts/ inside the LUMI PyTorch container.
#
# Usage:  ./run.sh download.py --dry-run
#         SIF=/path/to/other.sif ./run.sh some_script.py ...
#
# Expects HF_TOKEN in the environment (e.g. exported from ~/.bashrc);
# singularity passes environment variables into the container.
set -euo pipefail

SIF=${SIF:-/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

# HF cache on project scratch (home is nearly full). Only cache/xet chunks
# live here; the token is read from HF_TOKEN, never written to disk.
export HF_HOME=${HF_HOME:-/pfs/lustrep3/scratch/project_465003466/dasamuel/hf_cache}
export PYTHONNOUSERSITE=1   # ignore ~/.local site-packages
export PYTHONUNBUFFERED=1

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is not set" >&2
    exit 1
fi

exec singularity exec -B /pfs "$SIF" bash -c \
    'source /opt/miniconda3/bin/activate pytorch && cd "$0" && exec python -I -u "$@"' \
    "$HERE" "$@"
