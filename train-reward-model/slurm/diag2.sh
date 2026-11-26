#!/bin/bash
# Gemma 4 31B per-parameter gradient norms without updates (bf16) on 2 nodes, same batches as the 1-node check
# (slurm/diag.sh): does the gradient spike of the multi-node runs need optimizer steps?
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
MASK=mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000
echo "== $(date -Is) Gemma 4 31B gradient norms, 2 nodes"
MASTER_ADDR=${NODELIST%%,*} MASTER_PORT=29740 srun --nodes=2 --ntasks=16 --ntasks-per-node=8 -w "$NODELIST" \
    --gpus-per-node=8 --cpus-per-task=7 --mem=480G --cpu-bind=$MASK \
    bash train-reward-model/slurm/rank.sh --model gemma-4-31B-it --run diag2-gemma31 --batch 64 --diagnose-grads 4 \
    2>&1 | grep -E "\] (model|step|  by)|Error|Traceback"
echo "== $(date -Is) end"
