#!/bin/bash
# One node of the full run (judge_full.sbatch): 8/tp replicas of $JUDGE, replica i on GPUs i*tp .. i*tp+tp-1,
# taking shard NODE*(8/tp)+i of NODES*(8/tp) with --skip-own-family. Environment: JUDGE, NODE, NODES, PAIRS, OUT,
# THINKING, LIMIT.
set -euo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
PAIRS=${PAIRS:-data/preferences/pairs.parquet}
OUT=${OUT:-data/preferences/votes}
TP=$(python3 -c "import yaml,sys; print([j for j in yaml.safe_load(open('collect-preferences/judges.yaml'))['judges'] if j['name']==sys.argv[1]][0]['tp'])" "$JUDGE")
R=$(( 8 / TP )); N=$(( NODES * R ))
[ "$NODE" -lt "$NODES" ] || { echo "NODE $NODE outside 0..$(( NODES - 1 ))"; exit 2; }
echo "start $(date -Is) on $(hostname): $JUDGE tp=$TP, shards $(( NODE * R ))..$(( NODE * R + R - 1 )) of $N, out=$OUT"
pids=()
for i in $(seq 0 $(( R - 1 ))); do
    s=$(( NODE * R + i ))
    devs=$(seq -s, $(( i * TP )) $(( i * TP + TP - 1 )))
    HIP_VISIBLE_DEVICES=$devs CUDA_VISIBLE_DEVICES=$devs collect-preferences/run_gpu.sh judge.py \
        --judge "$JUDGE" --pairs "$PAIRS" --out "$OUT" --shard "$s/$N" --tp "$TP" --skip-own-family \
        ${THINKING:+--thinking "$THINKING"} ${LIMIT:+--limit "$LIMIT"} \
        > "data/logs/judge-${JUDGE}-s${s}of${N}-${SLURM_JOB_ID:-local}.log" 2>&1 &
    pids+=($!)
done
rc=0
for i in "${!pids[@]}"; do
    r=0; wait "${pids[$i]}" || r=$?
    [ "$r" -eq 0 ] || { echo "$(hostname): shard $(( NODE * R + i ))/$N exited rc=$r"; rc=$r; }
done
echo "end $(date -Is) on $(hostname): $JUDGE rc=$rc"
exit $rc
