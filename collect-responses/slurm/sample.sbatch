#!/bin/bash
# Response sampling: on every node, one `sample.py work` per GCD (sample_node.sh); the workers claim the
# work units of data/responses/samples/plan.json. Runs on standard-g (dev-g only for short tests).
# Workers claim no new unit after MAX_HOURS (default: 15 minutes before the time limit; on SIGTERM at the
# limit they flush their finished outputs). Resubmit until
# `sample.py status` shows 100%:
#   sbatch -N 4 -t 08:00:00 collect-responses/slurm/sample.sbatch
#SBATCH --job-name=sample
#SBATCH --account=project_465003445
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/logs/slurm-%x-%j.out

set -euo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
echo "start $(date -Is): $SLURM_NNODES nodes ($SLURM_JOB_NODELIST)"
limit=$(squeue -h -j "$SLURM_JOB_ID" -o %l)          # e.g. 8:00:00 or 1-00:00:00
secs=$(echo "$limit" | awk -F'[-:]' '{d=h=m=s=0; if ($0 ~ /-/) {d=$1; h=$2; m=$3; s=$4} else if (NF==3) {h=$1; m=$2; s=$3}
    else if (NF==2) {m=$1; s=$2} else {m=$1}; print d*86400+h*3600+m*60+s}')
export MAX_HOURS=${MAX_HOURS:-$(awk -v s="$secs" 'BEGIN{printf "%.2f", (s-900)/3600}')}
echo "time limit $limit: workers claim new units for $MAX_HOURS h"
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash collect-responses/slurm/sample_node.sh
echo "end $(date -Is)"
