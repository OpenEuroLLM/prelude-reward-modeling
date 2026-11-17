#!/bin/bash
# One refill round of the compiled prompt set (stages 6 -> 6b -> 5 -> 6c -> 6 -> 7).
# Run from the repo root, on the login / interactive node (it only submits and
# waits;
#   collect-prompts/compile/slurm/refill_round.sh N [ALTERNATES=4] [MAX_CHARS=4000] [--assemble-first]
# Env: LOCAL=1 runs the CPU steps here (niced, 8 threads; logs data/logs/local-<step>.out)
#      instead of Slurm; CPU_PART (default largemem) for Slurm CPU steps; EXTRA_ALLOC extra
#      allocate.py args; GPU_TIME (default 00:40:00) for the translation job.
# N is the new round; data/compiled/failed_slots.parquet must hold the failures
# of round N-1 (i.e. assembly ran after round N-1), unless --assemble-first.
set -euo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
N=$1; K=${2:-4}; MAXC=${3:-4000}; FIRST=${4:-}
P=$((N - 1)); C=data/compiled; CJ="-p ${CPU_PART:-largemem} $([[ ${CPU_PART:-} == debug ]] && echo -t 00:30:00 || true) collect-prompts/compile/slurm/cpu_job.sbatch"
waitj() { until ! timeout 60 squeue -h -j "$1" 2>/dev/null | grep -q .; do sleep 20; done
          st=$(timeout 60 sacct -j "$1" -X -n -o State | head -1 | tr -d ' ')
          echo "job $1: $st"; [[ $st == COMPLETED ]]; }
log() { echo "[$(date -Is)] $*"; }
# cpu NAME "SLURM RESOURCES" SCRIPT ARGS... -> runs the step, sets CPU_LOG to its log file
cpu() {
    local name=$1 res=$2; shift 2
    if [[ ${LOCAL:-} == 1 ]]; then
        CPU_LOG=data/logs/local-$name.out
        log "$name (local)"
        nice -n 10 env OMP_NUM_THREADS=8 collect-prompts/run.sh "$@" > $CPU_LOG 2>&1 \
            || { log "$name FAILED, see $CPU_LOG"; tail -20 $CPU_LOG; return 1; }
    else
        local J; J=$(sbatch --parsable -J $name $res $CJ "$@")
        CPU_LOG=data/logs/slurm-$name-$J.out
        log "$name $J"; waitj $J
    fi
}

if [[ $FIRST == --assemble-first ]]; then
    cpu assemble_pre$N "-c 8 --mem 120G -t 01:00:00" compile/assemble.py
fi
n_failed=$(python3 -c "import json;print(json.load(open('$C/assemble_report.json'))['failed'])")
log "round $N: $n_failed failed slots of round $P"
[[ $n_failed == 0 ]] && { log "nothing to refill"; exit 0; }
cp $C/failed_slots.parquet $C/failed_slots.round$P.parquet
cp $C/assemble_report.json $C/assemble_report.round$P.json
cp $C/allocation.parquet $C/allocation.round$P.parquet

cpu alloc_refill$N "" compile/allocate.py --order native-first --origin-free \
    --refill $PWD/$C/failed_slots.round$P.parquet --refill-max-chars $MAXC --alternates $K --out $PWD/$C ${EXTRA_ALLOC:-}
grep -E "refill|pass |alternate" $CPU_LOG | tail -40
cpu trreq$N "-c 2 --mem 32G -t 00:30:00" compile/build_translation_requests.py --round $N
grep -E "translation requests|alternates" $CPU_LOG

if grep -qE "^0 translation requests" $CPU_LOG; then
    log "translate: 0 requests, skipped"
else
    G=""; t0=$(date +%s)
    while [[ -z $G ]] && (( $(date +%s) - t0 < 10800 )); do
        G=$(sbatch --parsable -J translate$N -N 1 -t ${GPU_TIME:-00:40:00} --exclude=nid005019 \
            --export=ALL,REQ=$PWD/data/llm/translate/round$N/requests.parquet,OUT=$PWD/data/llm/translate/round$N/out,MAXLEN=32768 \
            collect-prompts/compile/slurm/llm_multi.sbatch 2>/dev/null) || G=""
        [[ -z $G ]] && sleep 60
    done
    [[ -z $G ]] && { log "standard-g never free"; exit 1; }
    log "translate $G"; waitj $G || true
    for f in data/logs/translate$N-$G.shard*.log; do grep -hE "^shard [0-9]+: " $f | tail -1 || true; done
fi

if (( K > 1 )); then
    cpu pick$N "-c 4 --mem 48G -t 00:30:00" compile/pick_alternates.py --round $N
    grep -E "round|chosen|allocation updated|Traceback|Error" $CPU_LOG
fi
cpu assemble$N "-c 8 --mem 120G -t 01:00:00" compile/assemble.py
python3 -c "import json;r=json.load(open('$C/assemble_report.json'));print('rows',r['rows'],'failed',r['failed'],r['failed_reasons'])"
cpu stats$N "-c 4 --mem 48G -t 00:30:00" compile/dataset_stats.py
tail -8 $CPU_LOG
