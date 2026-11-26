#!/bin/bash
# Which packing gives the right gradient? Prelude, LR 0, fp32 accumulation, per-type gradient norms, 2 steps, one node
# each: --micro-tokens 1 (every sequence alone in its own pass: the reference), 6000, 16384 (training), 32768 (no splits)
set -uo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
IFS=, read -r -a H <<< "$NODELIST"
CPUB=mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000
MT=(1 6000 16384 32768)
for k in 0 1 2 3; do
    mt=${MT[$k]}
    rm -rf data/reward_model/runs/test-ref-$mt
    MASTER_ADDR=${H[$k]} MASTER_PORT=$((29621 + k)) srun --nodes=1 --ntasks=8 --ntasks-per-node=8 -w "${H[$k]}" --gpus-per-node=8 \
        --cpus-per-task=7 --mem=480G --cpu-bind=$CPUB bash train-reward-model/slurm/rank.sh --model prelude-threeway-dpo \
        --run test-ref-$mt --micro-tokens $mt --lr 0 --train-limit 1280 --max-steps 2 --grad-report 1 \
        > data/reward_model/logs/test-ref-$mt.log 2>&1 &
done
wait
python3 - <<'PY'
import json
R = "/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/reward_model/runs/test-ref-{}/log.jsonl"
MT = (1, 6000, 16384, 32768)
rep = {mt: [json.loads(l) for l in open(R.format(mt)) if "grad_report" in l] for mt in MT}
keys = ["head.weight", "backbone.embed_tokens.weight", "backbone.layers.*.input_layernorm.weight",
        "backbone.layers.*.self_attn.v_proj.weight", "backbone.layers.*.mlp.down_proj.weight"]
for s in range(2):
    print("step", s + 1, "  micro-tokens:", "  ".join("%8d" % mt for mt in MT))
    print("   %-45s" % "total", "  ".join("%8.3f" % rep[mt][s]["gnorm"] for mt in MT))
    for k in keys:
        print("   %-45s" % k, "  ".join("%8.3f" % dict(map(tuple, rep[mt][s]["grad_report"]["types"])).get(k, float("nan")) for mt in MT))
PY
