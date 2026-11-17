#!/bin/bash
# Full run: one judge over all pairs (data/preferences/pairs.parquet) as a job array, one standard-g node per
# array task. Each node runs 8/tp replicas (judge_node.sh); replica i of node k takes shard k*(8/tp)+i of
# NODES*(8/tp), with the own-family pairs left out. NODES fixes the shards, so a task that fails or runs out of
# time is resubmitted alone with the same NODES and resumes where it stopped:
#   JUDGE=gemma-4-31B-it NODES=6 sbatch --array=0-5 -J judge-gemma collect-preferences/slurm/judge_full.sbatch
#   JUDGE=gemma-4-31B-it NODES=6 sbatch --array=3 -J judge-gemma collect-preferences/slurm/judge_full.sbatch
# Environment: JUDGE, NODES, PAIRS (default data/preferences/pairs.parquet), OUT (default data/preferences/votes),
# THINKING (default: judge.py's, on), LIMIT (pairs per replica, for a short test, with OUT elsewhere).
#SBATCH --job-name=judge
#SBATCH --account=project_465003445
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=16:00:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/logs/slurm-%x-%A_%a.out

cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
: "${JUDGE:?set JUDGE=<name from judges.yaml>}" "${NODES:?set NODES=<array size>, fixed for the whole run}"
export NODE=${SLURM_ARRAY_TASK_ID:-0}
exec bash collect-preferences/slurm/judge_node.sh
