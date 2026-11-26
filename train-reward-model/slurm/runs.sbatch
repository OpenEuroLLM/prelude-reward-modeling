#!/bin/bash
#SBATCH --job-name=rm-runs
#SBATCH --account=project_465003445
#SBATCH --partition=dev-g
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=480G
#SBATCH --time=00:40:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/reward_model/logs/%x-%j.out
# Several reward-model runs in one allocation, each on its own nodes (dev-g allows one job per user).
#   SPEC=train-reward-model/slurm/specs/smoke.txt sbatch -N 5 -t 00:40:00 train-reward-model/slurm/runs.sbatch
# SPEC lines: "<nodes> <train_rm.py arguments>" or "<nodes> @<script> [arguments]" (a script that starts its own
# srun steps on $NODELIST, e.g. slurm/runs/<run>.sh, which go through managed.sh and can move between jobs);
# '#' starts a comment. Runs get the job's nodes in order.
# A run's stdout goes to data/reward_model/logs/<job>-<line>.log.
set -uo pipefail
ROOT=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
cd "$ROOT"
: "${SPEC:?set SPEC to a spec file}"
mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
mapfile -t lines < <(grep -v '^\s*#' "$SPEC" | grep -v '^\s*$')
used=0
pids=()
names=()
for k in "${!lines[@]}"; do
    read -r nn args <<< "${lines[$k]}"
    if [ $((used + nn)) -gt "${#hosts[@]}" ]; then
        echo "not enough nodes for line $k: ${lines[$k]}"; continue
    fi
    nodes=$(IFS=,; echo "${hosts[*]:$used:$nn}")
    master=${hosts[$used]}
    used=$((used + nn))
    log=$ROOT/data/reward_model/logs/${SLURM_JOB_ID}-${k}.log
    echo "$(date -Is) line $k on $nodes: $args -> $log"
    if [[ "$args" == @* ]]; then          # "@script": a script that runs its own steps on these nodes
        NODELIST=$nodes MASTER_ADDR=$master MASTER_PORT=$((29500 + k)) bash ${args#@} > "$log" 2>&1 &
    else
    MASTER_ADDR=$master MASTER_PORT=$((29500 + k)) srun --nodes="$nn" --ntasks=$((8 * nn)) --ntasks-per-node=8 \
        -w "$nodes" --gpus-per-node=8 --cpus-per-task=7 --mem=480G \
        --cpu-bind=mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000 \
        bash train-reward-model/slurm/rank.sh $args > "$log" 2>&1 &
    fi
    pids+=($!)
    names+=("$k")
done
rc=0
for n in "${!pids[@]}"; do
    r=0; wait "${pids[$n]}" || r=$?
    echo "$(date -Is) line ${names[$n]} exited rc=$r"
    [ "$r" -eq 0 ] || rc=$r
done
exit $rc
