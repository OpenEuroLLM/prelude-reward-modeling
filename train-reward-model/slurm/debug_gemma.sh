#!/bin/bash
# Gemma 4 31B full fine-tuning on 2 nodes with frozen embeddings and norm weights, the full run's warmup: does the
# gradient-norm spike after the first updates disappear?
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
MASK=mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000
echo "== $(date -Is) Gemma 4 31B, frozen norms"
MASTER_ADDR=${NODELIST%%,*} MASTER_PORT=29750 srun --nodes=2 --ntasks=16 --ntasks-per-node=8 -w "$NODELIST" \
    --gpus-per-node=8 --cpus-per-task=7 --mem=480G --cpu-bind=$MASK \
    bash train-reward-model/slurm/rank.sh --model gemma-4-31B-it --run debug-gemma31-normfrozen --batch 64 --max-steps 10 \
    --lr 2e-6 --warmup 0.005 --freeze-embed --freeze-norms --center 0.01 --micro-tokens 12288 \
    2>&1 | grep -E "\] (model|step|EVAL|done)|Error|Traceback|error"
echo "== $(date -Is) end"
