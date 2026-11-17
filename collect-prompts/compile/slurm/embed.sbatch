#!/bin/bash
# Embeddings: one embed_runner.py process per GPU. Submit from the repo root:
#   sbatch collect-prompts/compile/slurm/embed.sbatch --input $PWD/X.parquet --out $PWD/DIR
# (paths must be absolute: run_gpu.sh changes into collect-prompts/)
#SBATCH --job-name=embed
#SBATCH --account=project_465003445
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/logs/slurm-%x-%j.out

set -euo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
GPUS=${SLURM_GPUS_ON_NODE:-8}
echo "start $(date -Is) on $(hostname): $GPUS GPUs"
pids=()
for i in $(seq 0 $(( GPUS - 1 ))); do
    HIP_VISIBLE_DEVICES=$i CUDA_VISIBLE_DEVICES=$i \
        collect-prompts/run_gpu.sh compile/embed_runner.py --shard "$i" --num-shards "$GPUS" "$@" \
        > "data/logs/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.gpu$i.log" 2>&1 &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
echo "end $(date -Is) rc=$rc"
exit $rc
