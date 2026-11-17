#!/bin/bash
# One node of a pilot job: one judge over the pilot pairs as 8/tp replicas that share the pairs (judge.py
# --shard). Started by pilot_judge.sbatch (one judge, batch step) or by srun from pilot_judges.sbatch (one
# node per entry of JUDGES=a,b,c; the node's rank picks its entry). An entry is one item or several joined by
# "+", run one after the other on the node. An item is <judge>[:<out dir>], e.g. gemma-4-31B-it or
# gemma-4-31B-it:data/preferences/pilot/v3/votes; the default out dir is OUT.
# Environment: JUDGE or JUDGES, PAIRS, OUT, LIMIT (pairs per replica, for short tests).
set -euo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
if [ -n "${JUDGES:-}" ]; then
    IFS=, read -ra list <<< "$JUDGES"
    ENTRY=${list[${SLURM_PROCID:-0}]}
else
    ENTRY=${JUDGE:?set JUDGE=<name>[:<out dir>] or JUDGES=a,b,c}
fi
PAIRS=${PAIRS:-data/preferences/pilot/pairs.parquet}
rc=0
IFS=+ read -ra items <<< "$ENTRY"
for ITEM in "${items[@]}"; do
    IFS=: read -r JUDGE_NAME OUTI <<< "$ITEM"
    OUT_DIR=${OUTI:-${OUT:-data/preferences/pilot/v3/votes}}
    TP=$(python3 -c "import yaml,sys; print([j for j in yaml.safe_load(open('collect-preferences/judges.yaml'))['judges'] if j['name']==sys.argv[1]][0]['tp'])" "$JUDGE_NAME")
    N=$(( 8 / TP ))
    echo "start $(date -Is) on $(hostname): $JUDGE_NAME tp=$TP replicas=$N out=$OUT_DIR"
    pids=()
    for i in $(seq 0 $(( N - 1 ))); do
        devs=$(seq -s, $(( i * TP )) $(( i * TP + TP - 1 )))
        HIP_VISIBLE_DEVICES=$devs CUDA_VISIBLE_DEVICES=$devs collect-preferences/run_gpu.sh judge.py \
            --judge "$JUDGE_NAME" --pairs "$PAIRS" --out "$OUT_DIR" --shard "$i/$N" --tp "$TP" \
            ${LIMIT:+--limit "$LIMIT"} \
            > "data/logs/pilot-${JUDGE_NAME}-${SLURM_JOB_ID}.w$i.log" 2>&1 &
        pids+=($!)
    done
    for i in "${!pids[@]}"; do
        r=0; wait "${pids[$i]}" || r=$?
        [ "$r" -eq 0 ] || { echo "$(hostname): $JUDGE_NAME replica $i exited rc=$r"; rc=$r; }
    done
    echo "end $(date -Is) on $(hostname): $JUDGE_NAME"
done
exit $rc
