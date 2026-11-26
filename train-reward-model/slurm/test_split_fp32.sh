#!/bin/bash
# Prelude gradients with fp32 accumulation (normal training mode, LR 0 so nothing changes; --grad-report 1 logs the
# per-type norms): without splits (--micro-tokens 32768) vs many splits over the ranks (--micro-tokens 6000)
set -uo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
IFS=, read -r -a H <<< "$NODELIST"
CPUB=mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000
for k in 0 1; do
    mt=$([ $k = 0 ] && echo 32768 || echo 6000)
    rm -rf data/reward_model/runs/test-fp32-$mt; export RM_CHECK_PARTNER=1
    MASTER_ADDR=${H[$k]} MASTER_PORT=$((29611 + k)) srun --nodes=1 --ntasks=8 --ntasks-per-node=8 -w "${H[$k]}" --gpus-per-node=8 \
        --cpus-per-task=7 --mem=480G --cpu-bind=$CPUB bash train-reward-model/slurm/rank.sh --model prelude-threeway-dpo \
        --run test-fp32-$mt --micro-tokens $mt --lr 0 --train-limit 1280 --max-steps 2 --grad-report 1 \
        > data/reward_model/logs/test-fp32-$mt.log 2>&1 &
done
wait
python3 - <<'PY'
import json
R = "/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/reward_model/runs/test-fp32-{}/log.jsonl"
rep = {mt: [json.loads(l) for l in open(R.format(mt)) if "grad_report" in l] for mt in (32768, 6000)}
for a, b in zip(rep[32768], rep[6000]):
    print("step", a["step"], "gnorm %.3f vs %.3f" % (a["gnorm"], b["gnorm"]))
    ta, tb = dict(map(tuple, a["grad_report"]["types"])), dict(map(tuple, b["grad_report"]["types"]))
    for k in list(ta)[:8]:
        print("   %-55s %8.3f %8.3f" % (k, ta[k], tb.get(k, float("nan"))))
PY
