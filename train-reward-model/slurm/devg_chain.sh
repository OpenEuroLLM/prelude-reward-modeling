#!/bin/bash
# Keep the full-data reward-model runs moving on dev-g, in chunks: whenever there is no dev-g job of mine, submit
# one with the unfinished runs that no running job holds (priority order below, up to MAXN nodes, at most 960
# node-minutes and 3 h). The chunks run with TAKEOVER=0: they skip runs that a standard-g job holds, and a
# standard-g job that starts takes its runs over (STOP file, checkpoint, resume; see managed.sh).
# No chunk runs past END; dev-g needs IDLE_SPARE idle nodes beyond the request, so that others still find room.
# Run on the login node as a background task of the session (not detached: it should end with the session):
#   bash train-reward-model/slurm/devg_chain.sh >> data/reward_model/logs/devg_chain.log 2>&1
set -uo pipefail
ROOT=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
RUNS=$ROOT/data/reward_model/runs
SPECDIR=$ROOT/data/reward_model/logs/devg-specs
# PRIO entries are run[:nodes] (default: the "# nodes:" line of slurm/runs/<run>.sh); Qwen LoRA, the slowest run, gets 2
PRIO=${PRIO:-"g31-full-soft-lr2e-6 q27-full-lora64-lr2e-5:2 g31-full-lora64-lr2e-5 q27-full-soft-lr2e-6 p9-full-soft-lr5e-6 p9-full-soft-lr3e-6"}
MAXN=${MAXN:-6}
END=${END:?set END, e.g. 2026-11-21T07:30}
IDLE_SPARE=${IDLE_SPARE:-4}
cd "$ROOT"
mkdir -p "$SPECDIR"
say() { echo "$(date -Is) $*"; }
held() { local o; o=$(cat "$RUNS/$1/OWNER" 2>/dev/null); [ -n "$o" ] && [ "$(squeue -h -j "$o" -o %T 2>/dev/null)" = RUNNING ]; }
say "chain started: until $END, at most $MAXN nodes; priority $PRIO"
while true; do
    mins=$(( ($(date -d "$END" +%s) - $(date +%s)) / 60 ))
    if [ "$mins" -lt 60 ]; then say "less than an hour before $END: no more chunks"; exit 0; fi
    if [ -n "$(squeue -h -u "$USER" -p dev-g -o %i)" ]; then sleep 60; continue; fi
    spec=$SPECDIR/chunk-$(date +%Y%m%d-%H%M%S).txt
    : > "$spec"
    nodes=0 left=0
    for e in $PRIO; do
        r=${e%%:*}
        [ -e "$RUNS/$r/DONE" ] && continue
        left=$((left + 1))
        held "$r" && continue
        n=$(awk '/^# nodes:/ {print $3; exit}' "train-reward-model/slurm/runs/$r.sh")
        [[ $e == *:* ]] && n=${e##*:}
        [ $((nodes + n)) -gt "$MAXN" ] && continue
        echo "$n @train-reward-model/slurm/runs/$r.sh" >> "$spec"
        nodes=$((nodes + n))
    done
    if [ "$left" -eq 0 ]; then say "all runs are done"; exit 0; fi
    if [ "$nodes" -eq 0 ]; then sleep 300; continue; fi
    t=$((960 / nodes)); [ "$t" -gt 180 ] && t=180; [ "$t" -gt "$mins" ] && t=$mins
    idle=$(sinfo -p dev-g -h -t idle -o %D | awk '{s += $1} END {print s + 0}')
    if [ "$idle" -lt $((nodes + IDLE_SPARE)) ]; then sleep 120; continue; fi
    out=$(SPEC=$spec TAKEOVER=0 sbatch -J rm-chunk -p dev-g -N "$nodes" -t "$t" train-reward-model/slurm/runs.sbatch 2>&1)
    say "$out: $nodes nodes, $t min: $(awk '{print $2}' "$spec" | xargs -n1 basename | tr '\n' ' ')"
    sleep 120
done
