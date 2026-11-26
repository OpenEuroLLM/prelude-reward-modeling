#!/bin/bash
# Two-node checks, one after another on $NODELIST: RCCL bandwidth across nodes, then a few full-fine-tuning steps
# of Gemma 4 31B and Qwen3.8-27B (throughput with the Slingshot plugin), then 1-node LoRA steps of both.
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
IFS=, read -r -a H <<< "$NODELIST"
MASK=mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000
step() {  # nodes hostlist port args...
    local n=$1 w=$2 port=$3; shift 3
    MASTER_ADDR=${w%%,*} MASTER_PORT=$port srun --nodes=$n --ntasks=$((8 * n)) --ntasks-per-node=8 -w "$w" \
        --gpus-per-node=8 --cpus-per-task=7 --mem=480G --cpu-bind=$MASK "$@"
}
echo "== $(date -Is) RCCL across 2 nodes"
step 2 "$NODELIST" 29710 bash train-reward-model/slurm/nccl_rank.sh 2>&1 | grep -E "GB/s|NET/OFI|rror" | grep -v "Selected provider" | sort | uniq -c
echo "== $(date -Is) Gemma 4 31B full fine-tuning, 2 nodes"
step 2 "$NODELIST" 29711 bash train-reward-model/slurm/rank.sh --model gemma-4-31B-it --run smoke2-gemma31 --batch 64 --max-steps 6 --lr 1e-6 2>&1 | grep -E "\] (model|step|EVAL|done)|Error|Traceback"
echo "== $(date -Is) Qwen3.8-27B full fine-tuning, 2 nodes"
step 2 "$NODELIST" 29712 bash train-reward-model/slurm/rank.sh --model qwen3.8-27b --run smoke2-qwen27 --batch 64 --max-steps 4 --lr 1e-6 2>&1 | grep -E "\] (model|step|EVAL|done)|Error|Traceback"
echo "== $(date -Is) LoRA r64, 1 node each"
step 1 "${H[0]}" 29713 bash train-reward-model/slurm/rank.sh --model gemma-4-31B-it --run smoke2-gemma31-lora --lora 64 --lora-alpha 64 --batch 64 --max-steps 6 --lr 2e-5 2>&1 | grep -E "\] (model|step|EVAL|done)|Error|Traceback" | sed 's/^/gemma-lora /' &
step 1 "${H[1]}" 29714 bash train-reward-model/slurm/rank.sh --model qwen3.8-27b --run smoke2-qwen27-lora --lora 64 --lora-alpha 64 --batch 64 --max-steps 4 --lr 2e-5 2>&1 | grep -E "\] (model|step|EVAL|done)|Error|Traceback" | sed 's/^/qwen-lora /' &
wait
echo "== $(date -Is) end"
