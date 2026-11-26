#!/bin/bash
#SBATCH --job-name=rm-bench
#SBATCH --account=project_465003445
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --output=/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/reward_model/logs/%x-%j.out
# Kernel and layer benchmarks on one GCD (bench_kernels.py).
set -euo pipefail
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/train-reward-model
export HF_HUB_OFFLINE=1
srun --ntasks=1 ./run.sh bench_kernels.py
