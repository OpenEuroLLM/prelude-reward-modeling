#!/bin/bash
#SBATCH --job-name=pilot-judges
#SBATCH --account=project_465003445
#SBATCH --partition=standard-g
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/logs/slurm-%x-%j.out

set -euo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
IFS=, read -ra list <<< "${JUDGES:?set JUDGES=a,b,c (one node per judge)}"
[ "${#list[@]}" -eq "$SLURM_NNODES" ] || { echo "${#list[@]} judges but $SLURM_NNODES nodes"; exit 2; }
echo "start $(date -Is): $SLURM_NNODES nodes ($SLURM_JOB_NODELIST): $JUDGES"
rc=0
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=0 \
    bash collect-preferences/slurm/pilot_node.sh || rc=$?
echo "end $(date -Is) rc=$rc"
exit $rc
