#!/bin/bash
# Per-node launcher (started by srun from stage3.sbatch): REPLICAS vLLM engines
# on this node, each with TP = 8 / REPLICAS, on global shards
#   NODE_IDX*REPLICAS .. NODE_IDX*REPLICAS+REPLICAS-1  of  NUM_NODES*REPLICAS.
# Env: REPLICAS (4), NUM_NODES (required), NODE_IDX (default SLURM_NODEID),
#      MODEL (default Gemma-4-31B-it), LOG_PREFIX. Args go to llm_runner.py.
set -euo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
MODEL=${MODEL:-$PWD/data/models/gemma-4-31B-it}
GPUS=8
REPLICAS=${REPLICAS:-4}
TP=$(( GPUS / REPLICAS ))
NODE_IDX=${NODE_IDX:-$SLURM_NODEID}
TOTAL=$(( NUM_NODES * REPLICAS ))
LOG_PREFIX=${LOG_PREFIX:-data/logs/llm-${SLURM_JOB_ID}}
echo "$(hostname): node $NODE_IDX/$NUM_NODES, $REPLICAS replicas x TP=$TP, $(date -Is)"
pids=()
for i in $(seq 0 $(( REPLICAS - 1 ))); do
    shard=$(( NODE_IDX * REPLICAS + i ))
    devs=$(seq -s, $(( i * TP )) $(( i * TP + TP - 1 )))
    HIP_VISIBLE_DEVICES=$devs CUDA_VISIBLE_DEVICES=$devs \
        collect-prompts/run_gpu.sh compile/llm_runner.py --model "$MODEL" --tp "$TP" \
        --shard "$shard" --num-shards "$TOTAL" "$@" > "${LOG_PREFIX}.shard${shard}.log" 2>&1 &
    pids+=($!)
done
# A failing replica (e.g. a GPU hang) must not take down the other replicas or
# nodes: srun cancels the whole step when any task exits non-zero. Failures are
# logged here; the runner is resumable, so a resubmission finishes the shard.
for i in "${!pids[@]}"; do
    rc=0; wait "${pids[$i]}" || rc=$?
    echo "$(hostname): replica $i (shard $(( NODE_IDX * REPLICAS + i ))) exited rc=$rc"
done
echo "$(hostname): done $(date -Is)"
exit 0
