#!/bin/bash
# One train_rm.py run on the nodes in $NODELIST (comma-separated; runs.sbatch sets it for "@script" spec lines),
# written so that a run can move between jobs (dev-g chunks, standard-g jobs) without losing work:
#   - a run with runs/<run>/DONE is skipped;
#   - runs/<run>/OWNER holds the job that runs it; if that job is still running, TAKEOVER=1 (the default) asks it
#     to stop (runs/<run>/STOP: the trainer checkpoints after the current step and exits) and waits, TAKEOVER=0
#     (opportunistic jobs) skips the run;
#   - --time-limit is set from the job's remaining wall time (TIME_LIMIT overrides, in minutes), so the trainer
#     checkpoints in time; train arguments should include --resume and --ckpt-every.
# Usage (from a spec line): 4 @train-reward-model/slurm/runs/<run>.sh   (which calls this script with the arguments)
set -uo pipefail
ROOT=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
RUNS=$ROOT/data/reward_model/runs
TAKEOVER=${TAKEOVER:-1}
cd "$ROOT"
: "${NODELIST:?}" "${MASTER_ADDR:?}" "${MASTER_PORT:?}"

run=""
prev=""
for a in "$@"; do [ "$prev" = "--run" ] && run=$a; prev=$a; done
: "${run:?no --run in the arguments}"
dir=$RUNS/$run
mkdir -p "$dir"
nn=$(tr ',' '\n' <<< "$NODELIST" | grep -c .)

say() { echo "$(date -Is) [$run, job $SLURM_JOB_ID] $*"; }
owner() { cat "$dir/OWNER" 2>/dev/null; }
running() { [ -n "$1" ] && [ "$1" != "$SLURM_JOB_ID" ] && [ "$(squeue -h -j "$1" -o %T 2>/dev/null)" = RUNNING ]; }
minutes_left() {
    local L d=0 h=0 m=0 s=0 a b c
    L=$(squeue -h -j "$SLURM_JOB_ID" -o %L)
    if [[ $L == *-* ]]; then d=${L%%-*}; L=${L#*-}; fi
    IFS=: read -r a b c <<< "$L"
    if [ -n "${c:-}" ]; then h=$a; m=$b; elif [ -n "${b:-}" ]; then m=$a; fi
    echo $(( (10#$d * 24 + 10#$h) * 60 + 10#$m ))
}

if [ -e "$dir/DONE" ]; then say "done already, skipped"; exit 0; fi
o=$(owner)
if running "$o"; then
    if [ "$TAKEOVER" != 1 ]; then say "running in job $o, skipped"; exit 0; fi
    say "running in job $o: asking it to stop and waiting"
    touch "$dir/STOP"
    while running "$(owner)"; do sleep 30; done
    say "job $o has let go"
    if [ -e "$dir/DONE" ]; then rm -f "$dir/STOP"; say "finished meanwhile, skipped"; exit 0; fi
fi
echo "$SLURM_JOB_ID" > "$dir/OWNER"
rm -f "$dir/STOP"
left=$(minutes_left)
tl=${TIME_LIMIT:-$((left - 4))}
say "starting on $NODELIST ($nn nodes, $left min left in the job, --time-limit $tl)"
srun --nodes="$nn" --ntasks=$((8 * nn)) --ntasks-per-node=8 -w "$NODELIST" --gpus-per-node=8 --cpus-per-task=7 \
    --mem=480G \
    --cpu-bind=mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000 \
    bash train-reward-model/slurm/rank.sh "$@" --time-limit "$tl"
rc=$?
[ "$(owner)" = "$SLURM_JOB_ID" ] && rm -f "$dir/OWNER"
say "exited rc=$rc$([ -e "$dir/DONE" ] && echo ', run complete')"
exit $rc
