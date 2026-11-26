#!/bin/bash
# RCCL bandwidth test with debug output (nccl_test.py), one task per GCD.
export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 NCCL_NET_GDR_LEVEL=PHB NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
cd /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/train-reward-model
exec ./run.sh nccl_test.py
