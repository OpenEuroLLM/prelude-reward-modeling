#!/bin/bash
# Generic multi-node batch LLM inference.
# Each node runs llm_node.sh (4 Gemma replicas, TP=2); shards = nodes * 4.
# Keep the node count fixed across resubmissions so shards resume consistently.
#SBATCH --job-name=llm_multi
#SBATCH --account=project_465003445
#SBATCH --partition=standard-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/logs/slurm-%x-%j.out

set -euo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
: "${REQ:?set REQ=<abs path to requests.parquet>}" "${OUT:?set OUT=<abs output dir>}"
MAXLEN=${MAXLEN:-8192}
echo "start $(date -Is): $SLURM_NNODES nodes ($SLURM_JOB_NODELIST) REQ=$REQ"
srun --nodes="$SLURM_NNODES" --ntasks="$SLURM_NNODES" --ntasks-per-node=1 \
    env NUM_NODES="$SLURM_NNODES" LOG_PREFIX="data/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}" \
    bash collect-prompts/compile/slurm/llm_node.sh \
    --requests "$REQ" --out "$OUT" --max-model-len "$MAXLEN" --chunk 2000 --max-num-seqs 256 \
    ${EXTRA:-}
echo "end $(date -Is)"
