#!/bin/bash
# Checkpoint / resume tests through managed.sh on the nodes in $NODELIST (one model per spec line):
#   test_ckpt.sh stop  MODEL ARGS...  run A until step 3 is logged, ask it to stop (STOP file), then resume (B)
#                                     to --max-steps; the steps after the resume should look like A's
#   test_ckpt.sh time  MODEL ARGS...  A with a 12-minute time limit (stops by itself after a step or two), then B
#   test_ckpt.sh ref   MODEL ARGS...  an uninterrupted reference run (same steps) to compare a resumed run with
set -uo pipefail
mode=$1 model=$2; shift 2
ROOT=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
cd "$ROOT"
run=test-$mode-$model
rm -rf "$ROOT/data/reward_model/runs/$run"
M="bash train-reward-model/slurm/managed.sh --model $model --run $run --train-limit 2560 --ckpt-every 3 --resume"
case $mode in
  ref)  $M "$@" ;;
  stop) $M "$@" &
        pid=$!
        until grep -q '"step": 4,' "$ROOT/data/reward_model/runs/$run/log.jsonl" 2>/dev/null; do
            sleep 10; kill -0 $pid 2>/dev/null || break; done
        echo "$(date -Is) touching STOP"; touch "$ROOT/data/reward_model/runs/$run/STOP"
        wait $pid
        echo "$(date -Is) resuming"; $M "$@" ;;
  time) TIME_LIMIT=12 $M "$@"
        echo "$(date -Is) resuming"; $M "$@" ;;
esac
grep -h '"step"\|stopped' "$ROOT/data/reward_model/runs/$run/log.jsonl" | cut -c1-220
