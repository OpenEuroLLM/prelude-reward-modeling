#!/bin/bash
# One-node checks on $NODELIST: speed of the Qwen3.8 linear-attention layer with the triangular-solve gated delta
# rule, then Gemma 4 31B per-parameter gradient norms (3 steps, bf16, no updates).
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
MASK=mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000
echo "== $(date -Is) Qwen3.8 layers, fast gated delta rule"
GDN_FAST=1 srun --nodes=1 --ntasks=1 -w "$NODELIST" --gpus-per-node=8 --cpus-per-task=56 --mem=480G \
    train-reward-model/run.sh bench_kernels.py --layers-only 2>&1 | grep -E "layer|whole|FAILED"
echo "== $(date -Is) Gemma 4 31B gradient norms"
MASTER_ADDR=${NODELIST%%,*} MASTER_PORT=29720 srun --nodes=1 --ntasks=8 --ntasks-per-node=8 -w "$NODELIST" \
    --gpus-per-node=8 --cpus-per-task=7 --mem=480G --cpu-bind=$MASK \
    bash train-reward-model/slurm/rank.sh --model gemma-4-31B-it --run diag-gemma31 --batch 64 --diagnose-grads 3 \
    2>&1 | grep -E "\] (model|step|  by)|Error|Traceback"
echo "== $(date -Is) end"
