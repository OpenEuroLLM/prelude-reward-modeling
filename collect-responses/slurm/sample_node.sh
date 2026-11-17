#!/bin/bash
# Per-node launcher (started by srun from sample.sbatch): one sample.py worker per GCD. A failing worker
# must not take down the others; the work is resumable, so a resubmission finishes it.
set -uo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
pids=()
for i in 0 1 2 3 4 5 6 7; do
    HIP_VISIBLE_DEVICES=$i CUDA_VISIBLE_DEVICES=$i collect-responses/run_gpu.sh sample.py work --max-hours "${MAX_HOURS:-2.6}" \
        > "data/logs/sample-${SLURM_JOB_ID}.$(hostname).g$i.log" 2>&1 &
    pids+=($!)
done
for i in "${!pids[@]}"; do
    rc=0; wait "${pids[$i]}" || rc=$?
    echo "$(hostname): worker $i exited rc=$rc"
done
exit 0
