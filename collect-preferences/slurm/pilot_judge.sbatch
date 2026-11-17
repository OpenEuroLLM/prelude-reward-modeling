#!/bin/bash
# Pilot: one judge over the pilot pairs on one standard-g node (pilot_node.sh: dense judges with tp 2 run as
# four replicas that share the pairs, MoE/large judges with tp 8 as one replica).
#   JUDGE=gemma-4-31B-it [PAIRS=...] [OUT=...] [LIMIT=n] sbatch collect-preferences/slurm/pilot_judge.sbatch
# (LIMIT: pairs per replica, for a short test, e.g. with `sbatch -p dev-g -t 00:30:00`)
#SBATCH --job-name=pilot-judge
#SBATCH --account=project_465003445
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=02:30:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/logs/slurm-%x-%j.out

cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
: "${JUDGE:?set JUDGE=<name from judges.yaml>}"
exec bash collect-preferences/slurm/pilot_node.sh
