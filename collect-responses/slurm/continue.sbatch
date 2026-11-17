#!/bin/bash
# Continuation of the truncated responses (continue_truncated.py): on every node one worker per GCD
# (continue_node.sh); the workers claim the units of $BASE/plan.json. Runs on standard-g (dev-g only for a
# short test). Workers claim no new unit 15 minutes before the time limit and flush their outputs on SIGTERM;
# resubmit until `continue_truncated.py status` shows everything continued:
#   sbatch -N 4 -t 03:00:00 collect-responses/slurm/continue.sbatch
#   BASE=data/responses/continue_test sbatch -p dev-g -N 1 -t 00:30:00 collect-responses/slurm/continue.sbatch
#SBATCH --job-name=continue
#SBATCH --account=project_465003445
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/logs/slurm-%x-%j.out

set -euo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
export BASE=${BASE:-data/responses/continue}
echo "start $(date -Is): $SLURM_NNODES nodes ($SLURM_JOB_NODELIST), base $BASE"
limit=$(squeue -h -j "$SLURM_JOB_ID" -o %l)
secs=$(echo "$limit" | awk -F'[-:]' '{d=h=m=s=0; if ($0 ~ /-/) {d=$1; h=$2; m=$3; s=$4} else if (NF==3) {h=$1; m=$2; s=$3}
    else if (NF==2) {m=$1; s=$2} else {m=$1}; print d*86400+h*3600+m*60+s}')
export MAX_HOURS=${MAX_HOURS:-$(awk -v s="$secs" 'BEGIN{printf "%.2f", (s-900)/3600}')}
echo "time limit $limit: workers claim new units for $MAX_HOURS h"
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash collect-responses/slurm/continue_node.sh
echo "end $(date -Is)"
