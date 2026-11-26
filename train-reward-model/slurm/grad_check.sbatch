#!/bin/bash
#SBATCH --job-name=rm-gradcheck
#SBATCH --account=project_465003445
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=01:15:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/reward_model/logs/%x-%j.out
# Gradient and attention checks of the training step (grad_check.py) on 16 GCDs, without training:
#   MODEL=gemma-4-31B-it sbatch train-reward-model/slurm/grad_check.sbatch
# A: embeddings and norm weights frozen (as in the Gemma full run), all checks.
# B: only the embeddings frozen (as in the earlier 16-GCD debug run), the two plan gradients.
# ARGS="--tests lm,train32,train16,update" runs one step with these arguments instead (nothing frozen unless given).
set -uo pipefail
ROOT=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
cd "$ROOT"
MODEL=${MODEL:-gemma-4-31B-it}
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
OUT=$ROOT/data/reward_model/logs/grad_check-$SLURM_JOB_ID
step() {
    MASTER_PORT=$1; shift
    MASTER_PORT=$MASTER_PORT srun --nodes="$SLURM_NNODES" --ntasks=$((8 * SLURM_NNODES)) --ntasks-per-node=8 \
        --gpus-per-node=8 --cpus-per-task=7 --mem=480G \
        --cpu-bind=mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000 \
        bash -c 'export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID MASTER_ADDR MASTER_PORT
                 export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB OMP_NUM_THREADS=7
                 export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
                 cd train-reward-model && exec ./run.sh grad_check.py "$@"' _ "$@"
}
if [ -n "${ARGS:-}" ]; then        # one step with these grad_check.py arguments instead of A and B
    echo "$(date -Is) grad_check.py $ARGS"
    # shellcheck disable=SC2086
    step 29612 --model "$MODEL" $ARGS --out "$OUT.json"
    if [ -n "${ARGS2:-}" ]; then   # optional second step: MODEL2 (default MODEL) with ARGS2
        echo "$(date -Is) grad_check.py --model ${MODEL2:-$MODEL} $ARGS2"
        # shellcheck disable=SC2086
        step 29613 --model "${MODEL2:-$MODEL}" $ARGS2 --out "$OUT-2.json"
    fi
    echo "$(date -Is) done"
    exit 0
fi
echo "$(date -Is) A: --freeze-embed --freeze-norms, all checks"
step 29610 --model "$MODEL" --freeze-embed --freeze-norms --out "$OUT-A.json"
echo "$(date -Is) B: --freeze-embed, plan gradients"
step 29611 --model "$MODEL" --freeze-embed --tests train32,train16 --out "$OUT-B.json"
echo "$(date -Is) done"
