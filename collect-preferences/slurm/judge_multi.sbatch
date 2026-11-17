#!/bin/bash
#SBATCH --job-name=judge-multi
#SBATCH --account=project_465003445
#SBATCH --partition=standard-g
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=01:30:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/logs/slurm-%x-%j.out
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
: "${SPECS:?set SPECS=JUDGE:NODES:NODE:PAIRS ...}" "${OUT:?set OUT}"
mapfile -t hosts < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
read -r -a specs <<< "$SPECS"
[ "${#specs[@]}" -le "${#hosts[@]}" ] || { echo "${#specs[@]} specs but ${#hosts[@]} nodes"; exit 2; }
pids=()
for n in "${!specs[@]}"; do
    IFS=: read -r j nn k p <<< "${specs[$n]}"
    echo "$(date -Is) ${hosts[$n]}: $j node $k of $nn, pairs $p"
    JUDGE=$j NODES=$nn NODE=$k PAIRS=$p OUT=$OUT srun --nodes=1 --ntasks=1 -w "${hosts[$n]}" --gpus-per-node=8 \
        --cpus-per-task=56 --mem=480G bash collect-preferences/slurm/judge_node.sh &
    pids+=($!)
done
rc=0
for n in "${!pids[@]}"; do
    r=0; wait "${pids[$n]}" || r=$?
    [ "$r" -eq 0 ] || { echo "step ${specs[$n]} exited rc=$r"; rc=$r; }
done
echo "$(date -Is) end rc=$rc"
exit $rc
