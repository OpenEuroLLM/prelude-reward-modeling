#!/bin/bash
# Generic CPU job: run one collect-prompts/ script in the CPU container.
# Submit from the repo root, e.g.
#   sbatch -J select collect-prompts/compile/slurm/cpu_job.sbatch compile/select_candidates.py
# Override resources on the command line (-c, --mem, -t, -p) as needed.
#SBATCH --job-name=cpu_job
#SBATCH --account=project_465003445
#SBATCH --partition=largemem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/logs/slurm-%x-%j.out

set -euo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
echo "start $(date -Is) on $(hostname): $*"
collect-prompts/run.sh "$@"
echo "end $(date -Is)"
sacct -j "$SLURM_JOB_ID" -o JobID,Elapsed,MaxRSS,State -n || true
