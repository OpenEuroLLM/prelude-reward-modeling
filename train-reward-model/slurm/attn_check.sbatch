#!/bin/bash
#SBATCH --job-name=rm-attncheck
#SBATCH --account=project_465003445
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:20:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/reward_model/logs/%x-%j.out
# Attention kernel checks on one GCD (attn_check.py; ARGS=--real for the model's layout and scales).
set -uo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/train-reward-model
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=7
srun --ntasks=1 --gpus-per-node=1 ./run.sh attn_check.py ${ARGS:-}
