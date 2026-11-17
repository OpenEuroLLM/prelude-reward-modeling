#!/bin/bash
# Stage 3b on one GPU node: embed candidates + hold-out prompts with bge-m3,
# then similarity search (candidate kNN + hold-out max-sim).
# Outputs: data/llm/embed*/out/, data/pool/neighbors/shard*.parquet, data/pool/clusters.parquet
#SBATCH --job-name=embed_sim
#SBATCH --account=project_465003445
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/logs/slurm-%x-%j.out

set -euo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling
exec bash collect-prompts/compile/slurm/embed_sim_node.sh
