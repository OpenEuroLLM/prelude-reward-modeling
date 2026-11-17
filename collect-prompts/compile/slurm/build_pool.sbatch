#!/bin/bash
# Stage 1: index all extracted records into data/pool/index/ (CPU only).
# Submit from the repo root, with HF_TOKEN exported (run.sh checks it):
#   sbatch collect-prompts/compile/slurm/build_pool.sbatch [build_pool.py args]
# Memory: ~1.3 GB shared LID model + <1 GB per worker; 40G leaves headroom.
#SBATCH --job-name=build_pool
#SBATCH --account=project_465003445
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/logs/slurm-%x-%j.out

set -euo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
echo "start $(date -Is) on $(hostname)"
collect-prompts/run.sh compile/build_pool.py --workers $(( SLURM_CPUS_PER_TASK - 1 )) "$@"
echo "end $(date -Is)"
sacct -j "$SLURM_JOB_ID" -o JobID,Elapsed,MaxRSS,State -n || true
