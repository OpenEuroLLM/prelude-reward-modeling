#!/bin/bash
# Stage 3 on several GPU nodes in ONE job:
#   nodes 0 .. ANNOT_NODES-1  annotation with Gemma-4-31B-it (llm_node.sh)
#   last node (if EMBED=1)    bge-m3 embeddings + similarity + k-means (embed_sim_node.sh)
# Annotation is split into ANNOT_NODES*4 global shards; keep ANNOT_NODES fixed
# across rounds so a later round resumes exactly the same shards.
# Submit from the repo root, e.g.
#   sbatch -p standard-g -N 5 -t 03:00:00 --export=ALL,ANNOT_NODES=4,EMBED=1 \
#       collect-prompts/compile/slurm/stage3.sbatch
#SBATCH --job-name=stage3
#SBATCH --account=project_465003445
#SBATCH --partition=standard-g
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=400G
#SBATCH --time=03:00:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/logs/slurm-%x-%j.out

set -euo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
# srun refuses to start when several memory variables are set (LUMI defaults)
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
ANNOT_NODES=${ANNOT_NODES:-4}
EMBED=${EMBED:-1}
D=$PWD/data/llm/annotate
echo "start $(date -Is): $SLURM_NNODES nodes ($SLURM_JOB_NODELIST), annot=$ANNOT_NODES embed=$EMBED"
pids=()
srun --nodes="$ANNOT_NODES" --ntasks="$ANNOT_NODES" --ntasks-per-node=1 --relative=0 \
    env NUM_NODES="$ANNOT_NODES" LOG_PREFIX="data/logs/annot-${SLURM_JOB_ID}" \
    bash collect-prompts/compile/slurm/llm_node.sh \
    --requests "$D/requests.parquet" --out "$D/out" \
    --max-model-len 8192 --chunk 4000 --max-num-seqs 256 &
pids+=($!)
if [ "$EMBED" = "1" ]; then
    srun --nodes=1 --ntasks=1 --relative="$ANNOT_NODES" \
        env LOG_PREFIX="data/logs/embed_sim-${SLURM_JOB_ID}" \
        bash collect-prompts/compile/slurm/embed_sim_node.sh &
    pids+=($!)
fi
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
echo "end $(date -Is) rc=$rc"
exit $rc
