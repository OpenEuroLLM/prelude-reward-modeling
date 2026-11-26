#!/bin/bash
#SBATCH --job-name=rm-bwdprec
#SBATCH --account=project_465003445
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=00:50:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/reward_model/logs/%x-%j.out
# Per-layer precision of the backward pass (bwd_precision.py): transformers' model on one node, one process.
#   MODEL=gemma-4-31B-it sbatch train-reward-model/slurm/bwd_precision.sbatch
set -uo pipefail
ROOT=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
cd $ROOT/train-reward-model
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=32
srun --ntasks=1 --gpus-per-node=8 --cpus-per-task=56 ./run.sh bwd_precision.py --model "${MODEL:-gemma-4-31B-it}" \
    --out $ROOT/data/reward_model/logs/bwd_precision-$SLURM_JOB_ID.json ${ARGS:-}
