#!/bin/bash
# Upload the figures the dataset card (README.md) embeds to
# https://huggingface.co/datasets/openeurollm/prelude-reward-modeling, and nothing else.
#
# Usage (login node):   collect-preferences/upload_figures.sh <HF_TOKEN>
#
# The set of figures is read from README.md's image links (figures/*.png); a referenced file that is missing
# stops the run before anything is uploaded, and files in figures/ that the card does not use are left alone.
# Figures already in the repo with the same content are skipped. The token is handed to the container as the
# HF_TOKEN variable (run.sh requires it) and is never printed. upload_hf.sh uploads the whole release instead.
set -euo pipefail
[ $# -eq 1 ] && [ -n "$1" ] || { echo "usage: $0 <HF_TOKEN>" >&2; exit 2; }
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
HF_TOKEN="$1" exec "$HERE/run.sh" upload_hf.py --only figures
