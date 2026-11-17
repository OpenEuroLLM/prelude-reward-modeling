#!/bin/bash
# Smoke test of the whole model pool on one node: 8 workers (one GCD each), each loading its share
# of models.yaml in turn with smoke_test.py run. Resumable: models with a .meta.json are skipped.
#   sbatch collect-responses/slurm/smoke.sbatch            (from the repo root)
#   ONLY=name1,name2 sbatch ...                            (a subset)
#SBATCH --job-name=smoke-pool
#SBATCH --account=project_465003445
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=01:30:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/logs/slurm-%x-%j.out

set -euo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
if [ -n "${ONLY:-}" ]; then
    names=(${ONLY//,/ })
else
    names=($(grep -E '^- name: ' collect-responses/models.yaml | awk '{print $3}'))
fi
W=8
echo "start $(date -Is) on $(hostname): ${#names[@]} models over $W workers"
pids=()
for i in $(seq 0 $(( W - 1 ))); do
    mine=()
    for k in $(seq "$i" "$W" $(( ${#names[@]} - 1 ))); do mine+=("${names[$k]}"); done
    [ ${#mine[@]} -eq 0 ] && continue
    list=$(IFS=,; echo "${mine[*]}")
    HIP_VISIBLE_DEVICES=$i CUDA_VISIBLE_DEVICES=$i \
        collect-responses/run_gpu.sh smoke_test.py run --models "$list" \
        > "data/logs/smoke-${SLURM_JOB_ID}.w$i.log" 2>&1 &
    pids+=($!)
    echo "worker $i: $list"
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
echo "end $(date -Is) rc=$rc"
exit 0
