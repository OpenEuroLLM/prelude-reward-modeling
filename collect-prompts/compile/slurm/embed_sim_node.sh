#!/bin/bash
# One GPU node: embed candidates + hold-out prompts with bge-m3 (one process per
# GPU), then similarity search + k-means (similarity.py). Started by
# stage3.sbatch via srun, or by embed_sim.sbatch.
# Outputs: data/llm/embed*/out/, data/pool/neighbors/shard*.parquet, data/pool/clusters.parquet
set -euo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
D=$PWD/data/llm
GPUS=8
LOG_PREFIX=${LOG_PREFIX:-data/logs/embed_sim-${SLURM_JOB_ID}}
run_per_gpu() {   # $1 = log tag, rest = script + args (shard args appended)
    local tag=$1; shift
    local pids=()
    for i in $(seq 0 $(( GPUS - 1 ))); do
        HIP_VISIBLE_DEVICES=$i CUDA_VISIBLE_DEVICES=$i \
            collect-prompts/run_gpu.sh "$@" --shard "$i" --num-shards "$GPUS" \
            > "${LOG_PREFIX}.$tag.gpu$i.log" 2>&1 &
        pids+=($!)
    done
    local rc=0
    for p in "${pids[@]}"; do wait "$p" || rc=$?; done
    return $rc
}
echo "$(hostname): embed_sim start $(date -Is)"
run_per_gpu embc compile/embed_runner.py --input "$D/embed/input.parquet" --out "$D/embed/out"
echo "candidates embedded $(date -Is)"
run_per_gpu embh compile/embed_runner.py --input "$D/embed_holdout/input.parquet" --out "$D/embed_holdout/out"
echo "holdout embedded $(date -Is)"
run_per_gpu sim compile/similarity.py
echo "$(hostname): embed_sim end $(date -Is)"
