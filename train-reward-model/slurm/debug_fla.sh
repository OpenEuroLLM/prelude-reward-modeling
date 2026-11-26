#!/bin/bash
# Debugging checks on two nodes ($NODELIST), one after another: flash-linear-attention against the PyTorch gated
# delta rule and the Qwen3.8 layer speed with it (one GPU), a few Qwen3.8-27B full fine-tuning steps with fla (2
# nodes), then Gemma 4 31B full fine-tuning steps with the full run's warmup, frozen embeddings and centring.
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
IFS=, read -r -a H <<< "$NODELIST"
MASK=mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000
step() {  # nodes hostlist port args...
    local n=$1 w=$2 port=$3; shift 3
    MASTER_ADDR=${w%%,*} MASTER_PORT=$port srun --nodes=$n --ntasks=$((8 * n)) --ntasks-per-node=8 -w "$w" \
        --gpus-per-node=8 --cpus-per-task=7 --mem=480G --cpu-bind=$MASK "$@"
}
F="\] (model|step|EVAL|done)|Error|Traceback|error"
echo "== $(date -Is) fla vs PyTorch gated delta rule"
srun --nodes=1 --ntasks=1 -w "${H[0]}" --gpus-per-node=8 --cpus-per-task=56 --mem=480G \
    train-reward-model/run.sh fla_check.py 2>&1 | grep -E "^fla|T=|Error|Traceback|error"
echo "== $(date -Is) Qwen3.8 layers with fla"
srun --nodes=1 --ntasks=1 -w "${H[0]}" --gpus-per-node=8 --cpus-per-task=56 --mem=480G \
    train-reward-model/run.sh bench_kernels.py --layers-only 2>&1 | grep -E "layer|whole|FAILED|Error"
echo "== $(date -Is) Qwen3.8-27B full fine-tuning with fla, 2 nodes"
step 2 "$NODELIST" 29731 bash train-reward-model/slurm/rank.sh --model qwen3.8-27b --run debug-qwen27-fla --batch 128 \
    --max-steps 5 --lr 2e-6 --center 0.01 2>&1 | grep -E "$F"
echo "== $(date -Is) Gemma 4 31B full fine-tuning, full-run warmup, 2 nodes"
step 2 "$NODELIST" 29732 bash train-reward-model/slurm/rank.sh --model gemma-4-31B-it --run debug-gemma31-warm --batch 128 \
    --max-steps 12 --lr 2e-6 --freeze-embed --center 0.01 2>&1 | grep -E "$F"
echo "== $(date -Is) end"
