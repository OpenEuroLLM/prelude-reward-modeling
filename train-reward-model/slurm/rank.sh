#!/bin/bash
# One rank of a distributed reward-model run (started by srun, one task per GCD).
export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
export MASTER_ADDR MASTER_PORT
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB
export OMP_NUM_THREADS=7 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# expandable_segments is not supported on this ROCm build (PyTorch warns and ignores it). Without it, cached blocks
# of >= 256 MiB are never split for smaller tensors, so a free large block can always be returned. Gemma 4 31B
# otherwise fragmented 10 GiB at its 60 GiB cap (49 GiB allocated) and a 672 MiB MLP tensor failed (2026-11-22).
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:256
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/train-reward-model
exec ./run.sh train_rm.py "$@"
