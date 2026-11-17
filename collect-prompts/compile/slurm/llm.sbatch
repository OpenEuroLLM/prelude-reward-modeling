#!/bin/bash
# Batch LLM inference: REPLICAS vLLM engines per node, each with TP = GPUS/REPLICAS.
# Submit from the repo root, e.g.
#   sbatch -p standard-g --gpus-per-node=2 --cpus-per-task=14 --mem=120G -t 01:00:00 \
#       --export=ALL,REPLICAS=1 collect-prompts/compile/slurm/llm.sbatch \
#       --requests $PWD/data/llm/smoke/requests.parquet --out $PWD/data/llm/smoke/out
# (paths must be absolute: run_gpu.sh changes into collect-prompts/)
# Extra args are passed to compile/llm_runner.py (model defaults to Gemma-4-31B-it).
# Several jobs can split one request set: NUM_NODES=N NODE_IDX=k gives job k the
# global shards k*REPLICAS .. k*REPLICAS+REPLICAS-1 of N*REPLICAS (resumable, so
# a later job with the same NODE_IDX continues where the previous one stopped).
# Never run two jobs with the same NODE_IDX at the same time.
#SBATCH --job-name=llm
#SBATCH --account=project_465003445
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/logs/slurm-%x-%j.out

set -euo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
MODEL=${MODEL:-$PWD/data/models/gemma-4-31B-it}
GPUS=${SLURM_GPUS_ON_NODE:-8}
REPLICAS=${REPLICAS:-$(( GPUS / 2 ))}
TP=$(( GPUS / REPLICAS ))
NUM_NODES=${NUM_NODES:-1}
NODE_IDX=${NODE_IDX:-0}
TOTAL_SHARDS=$(( NUM_NODES * REPLICAS ))
echo "start $(date -Is) on $(hostname): $GPUS GPUs, $REPLICAS replicas x TP=$TP, node $NODE_IDX/$NUM_NODES"
rocm-smi --showmeminfo vram 2>/dev/null | grep -E 'GPU\[|Total' | head -4 || true

pids=()
for i in $(seq 0 $(( REPLICAS - 1 ))); do
    devs=$(seq -s, $(( i * TP )) $(( i * TP + TP - 1 )))
    HIP_VISIBLE_DEVICES=$devs CUDA_VISIBLE_DEVICES=$devs \
        collect-prompts/run_gpu.sh compile/llm_runner.py --model "$MODEL" --tp "$TP" \
        --shard "$(( NODE_IDX * REPLICAS + i ))" --num-shards "$TOTAL_SHARDS" "$@" \
        > "data/logs/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.shard$(( NODE_IDX * REPLICAS + i )).log" 2>&1 &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
echo "end $(date -Is) rc=$rc"
sacct -j "$SLURM_JOB_ID" -o JobID,Elapsed,MaxRSS,State -n || true
exit $rc
