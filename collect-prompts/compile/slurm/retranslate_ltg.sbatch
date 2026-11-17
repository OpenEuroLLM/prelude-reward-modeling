#!/bin/bash
# One standard-g node, four phases (each phase: 4 vLLM replicas, TP=2, resumable):
#   1. TranslateGemma-27B: targeted re-translation + Latgalian pivot leg 1 + FLORES en->lv
#   2. CPU: build the Latgalian candidate requests (pivots use phase-1 Latvian)
#   3. Gemma-4-31B-it: Latgalian candidates G0..G3, G2p
#   4. TildeOpen-30b:  Latgalian candidates T1..T4, T2p, T4p
# Prepare first: collect-prompts/run.sh compile/tg_requests.py
# Submit from the repo root:  sbatch collect-prompts/compile/slurm/retranslate_ltg.sbatch
# A failing phase is logged and the next one still runs; resubmitting resumes.
# PHASES (default "tg ltgreq gemma tilde") selects phases, e.g.
#   sbatch --export=ALL,PHASES=tilde collect-prompts/compile/slurm/retranslate_ltg.sbatch
# TildeOpen: transformers 5 in this container converts its SentencePiece model into
# a tokenizer that encodes every space twice, so its prompts are pre-tokenized with
# SentencePiece (ltg_requests.py, column prompt_ids) and the generated token ids are
# saved and decoded with SentencePiece (ltg_score.py).
#SBATCH --job-name=retr_ltg
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

phase() {   # name model requests out [runner args...]
    local name=$1 model=$2 req=$3 out=$4; shift 4
    echo "[$(date -Is)] phase $name: $model"
    srun --nodes=1 --ntasks=1 env NUM_NODES=1 MODEL="$model" LOG_PREFIX="$LP.$name" \
        bash collect-prompts/compile/slurm/llm_node.sh --requests "$req" --out "$out" \
        --max-model-len 16384 --window 2000 --max-num-seqs 256 "$@" \
        || echo "[$(date -Is)] phase $name FAILED"
    for f in $LP.$name.shard*.log; do grep -hE "^shard [0-9]+: " "$f" | tail -1; done
}

PHASES=${PHASES:-"tg ltgreq gemma tilde"}
on() { [[ " $PHASES " == *" $1 "* ]]; }
on tg && phase tg $R/data/models/translategemma-27b-it ${TG_REQ:-$R/data/llm/translate_tg/requests.parquet} \
    ${TG_OUT:-$R/data/llm/translate_tg/out} --chat-tokens
if on ltgreq; then
    echo "[$(date -Is)] build Latgalian requests"
    srun --nodes=1 --ntasks=1 collect-prompts/run.sh compile/ltg_requests.py || echo "ltg_requests FAILED"
fi
on gemma && phase gemma $R/data/models/gemma-4-31B-it ${GEMMA_REQ:-$R/data/ltg/eval/gemma/requests.parquet} \
    ${GEMMA_OUT:-$R/data/ltg/eval/gemma/out}
on tilde && phase tilde $R/data/models/TildeOpen-30b ${TILDE_REQ:-$R/data/ltg/eval/tilde/requests.parquet} \
    ${TILDE_OUT:-$R/data/ltg/eval/tilde/out} --save-token-ids
echo "[$(date -Is)] end"
