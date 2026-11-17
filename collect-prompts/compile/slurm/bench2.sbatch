#!/bin/bash
#SBATCH --job-name=bench2
#SBATCH --account=project_465003445
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=400G
#SBATCH --time=01:30:00
#SBATCH --exclude=nid005019
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/logs/slurm-%x-%j.out

set -uo pipefail
R=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
cd $R
unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU
LP=data/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}
B=${B:-$R/data/llm/bench2}       # request dir: <B>/<phase>/requests.parquet

phase() {   # name model replicas [runner args...]
    local name=$1 model=$2 reps=$3; shift 3
    echo "[$(date -Is)] phase $name: $model ($reps replicas)"
    srun --nodes=1 --ntasks=1 env NUM_NODES=1 REPLICAS=$reps MODEL="$model" LOG_PREFIX="$LP.$name" \
        bash collect-prompts/compile/slurm/llm_node.sh --requests $B/$name/requests.parquet --out $B/$name/out \
        --max-model-len ${MAXLEN:-4096} --window 2000 --max-num-seqs 256 "$@" \
        || echo "[$(date -Is)] phase $name FAILED"
    for f in $LP.$name.shard*.log; do grep -hE "^shard [0-9]+: " "$f" | tail -1; done
}

PHASES=${PHASES:-"salta eurollm tilde tower"}
on() { [[ " $PHASES " == *" $1 "* ]]; }
on salta   && phase salta   $R/data/models/salamandraTA-7b-instruct   8
on eurollm && phase eurollm $R/data/models/EuroLLM-22B-Instruct-2512  4
on tilde   && phase tilde   $R/data/models/TildeOpen-30b              4 --save-token-ids
on tower   && phase tower   $R/data/models/Tower-Plus-72B             2
echo "[$(date -Is)] end"
