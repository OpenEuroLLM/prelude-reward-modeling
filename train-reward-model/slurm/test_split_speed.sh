#!/bin/bash
# Cross-rank splitting of long pairs, on 4 nodes ($NODELIST from runs.sbatch):
#  1. correctness: Prelude step-1 gradients (no updates) without splits (--micro-tokens 32768) and with many splits
#     spread over the ranks (--micro-tokens 6000), one node each, in parallel; the gradient norms should agree
#  2. speed: Qwen3.8-27B full fine-tuning on all 4 nodes, the real run's first 6 batches (its step times were 78-87 s)
set -uo pipefail
ROOT=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
cd "$ROOT"
IFS=, read -r -a H <<< "$NODELIST"
CPUB=mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000
step() {  # nodes(comma) port args...
    local w=$1 port=$2; shift 2
    local n; n=$(tr ',' '\n' <<< "$w" | grep -c .)
    MASTER_ADDR=${w%%,*} MASTER_PORT=$port srun --nodes="$n" --ntasks=$((8 * n)) --ntasks-per-node=8 -w "$w" \
        --gpus-per-node=8 --cpus-per-task=7 --mem=480G --cpu-bind=$CPUB bash train-reward-model/slurm/rank.sh "$@"
}
echo "== $(date -Is) correctness"
rm -rf data/reward_model/runs/test-xsplit-*
step "${H[0]}" 29601 --model prelude-threeway-dpo --run test-xsplit-none --micro-tokens 32768 --diagnose-grads 2 --lr 5e-6 \
    > data/reward_model/logs/test-xsplit-none.log 2>&1 &
step "${H[1]}" 29602 --model prelude-threeway-dpo --run test-xsplit-many --micro-tokens 6000 --diagnose-grads 2 --lr 5e-6 \
    > data/reward_model/logs/test-xsplit-many.log 2>&1 &
wait
for f in none many; do echo "-- $f"; grep -h "grad norm" data/reward_model/logs/test-xsplit-$f.log | cut -c1-200; done
echo "== $(date -Is) speed"
rm -rf data/reward_model/runs/test-xsplit-q27
step "$NODELIST" 29603 --model qwen3.8-27b --run test-xsplit-q27 --loss soft --center 0.01 --lr 2e-6 --train-limit 1280 \
    --max-steps 6 --profile-step 3
python3 - <<'EOF'
import json
for l in open("/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/reward_model/runs/test-xsplit-q27/log.jsonl"):
    d = json.loads(l)
    if "loss" in d:
        print("step", d["step"], "seconds %.0f" % d["seconds"], "loss %.4f" % d["loss"], "gnorm %.2f" % d["gnorm"], "mem", d.get("mem_gb"), "split", d.get("n_split"))
EOF
echo "== $(date -Is) end"
