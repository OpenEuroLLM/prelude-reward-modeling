#!/bin/bash
# Sampling-parameter comparison on the smoke test's general prompts: one run per "model|override|tag" entry
# in RUNS (space-separated; override is JSON without spaces), 8 workers (one GCD each), entries round-robin.
#   RUNS='prelude-w2-rl|{"top_p":0.95}|@p95 salamandra-7b-v1.1|{"top_p":0.95,"repetition_penalty":null}|@p95norep' \
#       sbatch collect-responses/slurm/sampling_test.sbatch
#SBATCH --job-name=sampling-test
#SBATCH --account=project_465003445
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=01:00:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/logs/slurm-%x-%j.out

set -euo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
read -r -a runs <<< "$RUNS"
W=8
echo "start $(date -Is) on $(hostname): ${#runs[@]} runs over $W workers"
pids=()
for i in $(seq 0 $(( W - 1 ))); do
    (
        for k in $(seq "$i" "$W" $(( ${#runs[@]} - 1 ))); do
            IFS='|' read -r name override tag <<< "${runs[$k]}"
            HIP_VISIBLE_DEVICES=$i CUDA_VISIBLE_DEVICES=$i collect-responses/run_gpu.sh smoke_test.py run \
                --models "$name" --override "$override" --tag "$tag"
        done
    ) > "data/logs/sampling-test-${SLURM_JOB_ID}.w$i.log" 2>&1 &
    pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || true; done
echo "end $(date -Is)"
