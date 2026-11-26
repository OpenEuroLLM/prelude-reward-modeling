#!/bin/bash
# nodes: 4
# Gemma 4 31B, full fine-tuning, full training set. Nothing frozen. --fp32-residual: the residual stream, norms and
# layer scalars in fp32 (matrix multiplications and attention in bf16): in plain bf16 the backbone gradient is
# mostly rounding noise (grad_check.py: two packings of the same batch agree with cosine 0.13; 0.95 with fp32
# residual). Tokens with the generation prompt (empty thinking block), 2026-11-22.
exec bash train-reward-model/slurm/managed.sh --model gemma-4-31B-it --run g31-full-soft-lr2e-6 --loss soft --center 0.01 --lr 2e-6 --fp32-residual --eval-every 200 --ckpt-every 50 --profile-step 3 --resume --save --score /pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/reward_model/bench/gemma-4-31B-it/rb2.parquet,/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/reward_model/bench/gemma-4-31B-it/rmbench.parquet,/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/reward_model/bench/gemma-4-31B-it/mrb.parquet,/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/reward_model/bench/gemma-4-31B-it/mrb_small.parquet,/pfs/lustrep3/scratch/project_465003466/dasamuel/prelude-reward-modeling/data/reward_model/bench/gemma-4-31B-it/probes.parquet
