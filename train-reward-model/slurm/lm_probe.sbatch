#!/bin/bash
#SBATCH --job-name=rm-lmprobe
#SBATCH --account=project_465003445
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/reward_model/logs/%x-%j.out
# Forward-pass reference check (lm_ref.py): transformers' from_pretrained model on one node, one process.
#   MODEL=gemma-4-31B-it sbatch train-reward-model/slurm/lm_ref.sbatch
set -uo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/train-reward-model
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
srun --ntasks=1 --gpus-per-node=8 --cpus-per-task=56 ./run.sh lm_probe.py "${MODEL:-gemma-4-31B-it}"
