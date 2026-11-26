# train-reward-model

Scalar reward models trained on the preference pairs (`data/preferences/preferences.jsonl`, see
`collect-preferences/`): a decoder backbone, text part only, with a linear head on the last hidden state of the
answer's end-of-turn token. The literature review behind the design is
[`reports/Reward model training and evaluation.md`](../reports/).

## Data

`prepare_data.py split` produces the training and validation sets:
- **Validation:** the 1,008 pilot pairs (`data/preferences/pilot/pairs.parquet`), the ones with Claude Opus
  reference labels. All 38 languages are present.
- **Training:** the other 126,918 pairs.

Each pair produces two sequences: the conversation rendered as the backbone's own generation prompt, followed
by one answer and the end-of-turn token. The model sees what the judges saw: no reasoning, no model name, no added
system prompt. Templates are rendered with thinking disabled, because otherwise Qwen3.8's template injects a
"Reasoning effort" system prompt. Answers are tokenized as plain text, so special-token strings inside them stay
text.

**Generation prompt, not a finished turn** (fixed 2026-11-22). The prefix is exactly what the model sees when it
starts an answer. Gemma 4 always opens its answer with an empty thinking block (`<|turn>model\n<|channel>thought\n<channel|>`),
but its template strips that block from finished turns. The Gemma tokens made before this fix used the
finished-turn rendering, so every answer started where the model is certain the block comes: it predicts
`<|channel>` with a logit of about 27 there, against about 10 for the next-best token.
- Effect: the pretrained model's next-token loss on validation answers was 9.2 nats/token in that format and is
  4.5 with the block, the same with transformers' own model and our pipeline (`lm_ref.py`, `lm_probe.py`).
- Qwen3.8's two renderings are the same: 4,008 prompts, 806 of them multi-turn, give byte-identical tokens.
- The old Gemma tokens are in `data/reward_model/_old-history-format/`.

`prepare_data.py tokenize --model M` writes one tokenized copy per backbone. Qwen3.8 tokens per training epoch:

| Max sequence length | Sequences cut | Tokens kept |
|---|---|---|
| 8,192 | 2.0% | 88% |
| 16,384 (default) | 0.8% | 97% |

**Truncation.** A cut keeps the prompt's first 16 tokens and its last tokens, up to half the length, and keeps
the start of the answer. The end-of-turn token always closes the sequence. No pair is dropped.

`prepare_bench.py` renders the public benchmarks the same way:
- RewardBench 2;
- RM-Bench;
- M-RewardBench in the 11 European languages of our set (cs, de, el, es, fr, it, nl, pl, pt, ro, uk), plus
  `mrb_small` with 300 pairs per language.

The prompt set was decontaminated against all three.

`prepare_probes.py` builds length probes from 507 validation answers in all 38 languages. Each answer won its
pair with at least 5 of 6 votes, ended normally and has three or more paragraphs. It is paired with two copies:
- **padded:** its last paragraph repeated;
- **truncated:** cut after the paragraph where it passes half its length.

The original should win both. A length-biased RM prefers the padded copy; one that rewards brevity for its own
sake prefers the truncated one.

## Training (`train_rm.py`)

FSDP2 over every GCD of the job, bf16 compute with fp32 master weights. Gradient checkpointing is on, AdamW with
betas (0.9, 0.95), cosine decay to 10% after 3% warmup, clipping at 1.0, and one epoch.

**Packing.** Several pairs are packed into one row without padding, and a pair is never split. Each layer type
gets the sequence boundaries:

| Layer type | Implementation |
|---|---|
| Head dim ≤ 256 (incl. Gemma's 1,024-token sliding window) | flash-attention varlen |
| Gemma 4's global layers (head dim 512) | PyTorch SDPA per sequence; its flash backend supports 512 on MI250X |
| Qwen3.8's Gated-DeltaNet layers | run once per sequence |

**Qwen3.8 linear attention.** These layers use the Triton kernels of flash-linear-attention v0.6.0. It is
unpacked from its source release into `/pfs/lustrep3/scratch/project_465003466/dasamuel/vendor/fla-v0.6.0`
(outside the container and the repository; `rm_model.py` puts it on the path, and `FLA_PATH=` with an empty
value turns it off). Its kernel matches the PyTorch reference to about 5e-3 in outputs and gradients
(`attn_check.py`) and is 16× faster. A Qwen3.8 linear-attention layer now costs the same as a full-attention
layer. Without fla, `gdn_fast.py` replaces transformers' slow fallback loop with one triangular solve (18% faster,
same result to about 1e-7).

**Gemma 4 numerics.** Gemma 4 31B has extreme scales in its checkpoint:
- post-attention norm weights of about 1e-4 in layers 2–11;
- pre-attention norm weights up to 54 and pre-MLP norm weights up to 268 in the early layers;
- final norm weights up to 510;
- per-layer output scalars down to 0.036.

Its backbone gradient amplifies rounding errors by orders of magnitude. `grad_check.py` computes the gradient of
one batch from the same weights in several ways. On the batch of step 1, the training step's own procedure gives:

| Precision | 32- vs 64-rank packing: backbone gradient cosine |
|---|---|
| bf16 (as trained before) | 0.13 |
| `--fp32-residual` | 0.95 |
| Qwen3.8 in bf16, for comparison | 0.94 |

On 128 short pairs, the comparison of sequences packed against sequences alone gives:

| Precision | Packed vs alone: backbone gradient cosine |
|---|---|
| bf16 | 0.18 |
| fp32 | 0.99 |

Against the fp32 gradient, computing only the attention core or only the attention projections in bf16 already
drops the backbone cosine to 0.15 and 0.01. The rewards agree throughout (correlation 0.996–0.999): the forward
pass is accurate, the backbone gradient is not. On 128 short pairs, each sequence alone (`grad_check.py --tests
prec --prec-variants coef`):

| Gradient, training precision (`--fp32-residual`, kernels) vs fp32 | Backbone cosine | Norm vs fp32 |
|---|---|---|
| the loss's own coefficients | 0.06 | 0.62 |
| the first answer of each pair only | 0.07 | 0.56 |
| all coefficients positive (sum of the rewards) | 0.63 | 0.82 |

The bf16 gradient is smaller than the fp32 one: it misses part of the true gradient rather than adding noise.
It is not the kernels (exact attention in bf16 gives 0.13), not the packing, and not the difference within a pair.
transformers' own model, one sequence at a time with eager attention (`bwd_precision.py`), shows where the gradient
of a reward on the last token gets lost: it is exact down to layer 40 (cosine 0.998), then falls to 0.6–0.8 below
layer 25 with about 0.7 of the fp32 norm. An fp32 attention core restores 0.97, and bf16 in layers 40–59 alone
costs nothing. Feeding the kernels keys and values minus their mean (the same function) does not help.
Only fp32 throughout gives the exact gradient; fp32 matrix multiplications on MI250X have a quarter of the bf16
throughput.
- `--fp32-residual` keeps the residual stream, the norms and the layer scalars in fp32, and runs the matrix
  multiplications and attention in bf16 (autocast). It needs more memory, so the run uses 4 nodes.
- `--mem-headroom` (default 4 GiB) caps PyTorch's caching allocator below the GPU's memory. Without the cap the
  Gemma run reserved 62.8 of 64 GiB, and the flash-attention backward aborted with
  `HSA_STATUS_ERROR_OUT_OF_RESOURCES`: the ROCm runtime had no memory left for kernel scratch.
- `slurm/rank.sh` sets `PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:256`. `expandable_segments` is not supported on
  this ROCm build: PyTorch warns and ignores it. Without the setting, 10 GiB of the 60 GiB cap was cached in
  partly used blocks while 49 GiB was allocated, and a 672 MiB MLP tensor failed with a CUDA out-of-memory error
  at step 336.
- The packed attention kernels are correct with the model's own layout and scales: `attn_check.py --real`
  matches an fp32 reference to about 3e-3 in outputs and gradients, for up to 4 sequences per row and up to
  8,000 tokens.
- The earlier runs froze the norm weights (`--freeze-norms`) against the growing gradient norm. That treated a
  symptom. The current run freezes nothing.

**Losses (`--loss`):**
- `soft`: Bradley–Terry cross-entropy against the vote share;
- `hard`: the majority, with even splits left out;
- `unanimous`: unanimous pairs only;
- `weighted`: the majority, weighted by |2p−1|.

`--center` adds reward centring. `--lora R` trains LoRA on all linear layers instead of full fine-tuning.
`--freeze-embed` / `--freeze-norms` keep the input embeddings and the RMSNorm weights fixed.

**Long pairs.** A pass holds whole pairs, so a pair of two long answers made a single pass of up to 32k tokens. 2.35%
of pairs exceed 16k tokens, nearly one per batch. Gemma 31B ran out of memory on one (batch 5, 29,856 tokens), and on
Qwen's 4 nodes the GCD holding such a pair kept the other 31 waiting: rank 0 had 32,176 tokens, the median rank 10,400.

A pair longer than `--micro-tokens` is therefore split into its two sequences, which the planner may place on
different ranks:
1. a no-grad forward gives every half's reward on the rank that holds it, and one all-reduce shares them;
2. each half then runs with its partner's reward as a constant.

The gradients are exact: the no-grad and grad-pass rewards of the same half agree to 4e-4. Against a reference with
one sequence per pass (`--micro-tokens 1`, LR 0, two batches), the head and the attention and MLP weights agree in
every packing. Prelude's embedding gradient norm varies with the packing even in the reference (0.31–0.58, once 1.36).
That variation seems to be a property of the model, like Gemma's norm weights, and the planner leaves it as it was.

Qwen3.8-27B full fine-tuning on 4 nodes went from 78–83 s to 47 s per step with this planner.

**Linear attention.** Qwen3.8's linear-attention layers now run once on the packed row. The projections cover all
tokens; the short convolution and the gated delta rule use flash-linear-attention's varlen kernels (`cu_seqlens`).
Against the earlier once-per-sequence loop, outputs agree to 5e-4 and gradients to 5e-3 (`gdn_varlen_check.py`),
and the layer is about 17% faster. `GDN_LOOP=1` restores the loop.

**Diagnostics.**
- `--profile-step K` profiles one step: rank 0's kernel table goes to `profile_<step>.txt`, and every rank's token
  count and its forward, no-grad and pass times go to the log.
- Every step logs the peak memory, the allocator retries and the number of split pairs.
- `--grad-report X` logs the largest gradients by parameter, type and layer when the norm exceeds X.

**Checkpoints.** `--ckpt-every` / `--resume` write and read sharded model and optimizer checkpoints (DCP;
any number of nodes can resume them). With `--ckpt-every`, the trainer writes a checkpoint and exits, without the
final phase, in two cases:
- when it is asked to stop (a `STOP` file in the run directory);
- when the next step and a checkpoint would no longer fit in `--time-limit`, unless the rest of the run and the final
  phase still fit. The checkpoint margin is 10 minutes or twice the last checkpoint's time, whichever is longer.

Rank 0's file work during training goes through a background thread: the log, the `STOP` poll (every 20 s) and the
validation parquet. Lustre can stall for half an hour. A rank blocked in a stat or write while the others wait in
a collective once ended a run with an NCCL timeout. The NCCL timeout is now 2 h, which also covers stalled checkpoint
writes.

The next job continues from that checkpoint. The final phase (evaluation, save, scoring) is resumable too:
`final.json` records what is done, and `DONE` marks a finished run. A resumed run matches an uninterrupted one
within run-to-run noise (Prelude, 8 steps: validation loss 0.6199 vs 0.6198). A Gemma 31B checkpoint (370 GB
with the optimizer) usually takes 40–50 s to write and 130–200 s to load. On a slow Lustre a write took 380–515 s,
and once over 9 minutes: Slurm killed that job during the closing checkpoint and its last 49 steps were lost.

**Outputs** go to `data/reward_model/runs/<run>/`:
- `log.jsonl`;
- `val_<step>.parquet`, the validation rewards;
- `scores_<bench>.parquet`, the benchmark rewards;
- `model/` with `--save`: bf16, in the original checkpoint's key names plus `score.weight` / `score.bias`.

`eval_report.py RUN ...` computes the metrics (definitions in its docstring):
- accuracy against the vote majority;
- expected agreement with one vote;
- agreement with Opus on the pairs where Opus is consistent;
- log loss;
- how often the RM prefers the longer answer, next to the labels;
- per tier, category, length band and language;
- the official RewardBench 2, RM-Bench and M-RewardBench scores.

## Running on LUMI

`slurm/runs.sbatch` runs several runs in one allocation, each on its own nodes. It reads a spec file:
`SPEC=train-reward-model/slurm/specs/<file>.txt sbatch -N <nodes> -t <time> [-p standard-g] train-reward-model/slurm/runs.sbatch`.

The full-data runs have one launcher each, `slurm/runs/<run>.sh` (spec line `<nodes> @train-reward-model/slurm/runs/<run>.sh`).
It goes through `slurm/managed.sh`, which lets a run move between jobs:
- it skips finished runs;
- it records the job that holds a run in `OWNER`;
- it sets `--time-limit` from the job's remaining time;
- if another running job holds the run, it either takes the run over (`STOP`, wait for the checkpoint, continue)
  or, with `TAKEOVER=0`, skips it.

`slurm/devg_chain.sh` runs on the login node. It keeps the unfinished runs moving on dev-g in chunks of at most
960 node-minutes, until `END`. The chunks use `TAKEOVER=0`, so a standard-g job that starts takes its runs over.
`slurm/test_ckpt.sh` tests the stop, resume and time-limit paths.

`run.sh` uses the LAIFS multitorch container: torch 2.11, ROCm 7.2, transformers 5.10. It must bind
`/var/spool/slurmd`; without it RCCL's Slingshot plugin cannot create its fabric domain and inter-node traffic
silently falls back to sockets. With the bind, 2 nodes reach about 89 GB/s all-gather (`nccl_test.py`).

Measured training throughput, tokens per second per GCD (full data is about 330M tokens per epoch):

| Backbone | Setup | Tokens/s per GCD |
|---|---|---|
| Prelude 9B (threeway-dpo), full fine-tuning | 1 node | about 1,000 |
| Gemma 4 31B, full fine-tuning | 2 nodes | 130–210 |
| Gemma 4 31B, full fine-tuning, `--fp32-residual` | 4 nodes | 118–163 (median 140) |
| Gemma 4 31B, LoRA r64 | 1 node | 310–390 |
| Qwen3.8-27B, full fine-tuning | 2 nodes | 70–125 with the PyTorch fallback; 180–370 with flash-linear-attention |
| Qwen3.8-27B, LoRA r64 | 1 node | 100–190 with the PyTorch fallback; 370–400 with flash-linear-attention |

## Results so far

**Sweep 1** (2026-11-20). Prelude 9B (`prelude-threeway-dpo`, the Prelude checkpoint with the best Elo in the
data) is a same-lineage control for the policy. Every run uses the same 18,000 training pairs, 140 steps of 128
pairs, and one node. Validation is on the 1,008 pilot pairs. Opus agreement is on its 935 consistent pairs. M-RB
is `mrb_small`.

| Run | Acc | Opus | Log loss | Longer wins, RM / labels | RewardBench 2 | RM-Bench (hard) | M-RB |
|---|---|---|---|---|---|---|---|
| soft, LR 1e-6 | 80.9 | 79.4 | 0.496 | 50.8 / 46.1 | 63.4 | 64.5 (42.3) | 77.1 |
| soft, LR 2e-6 | 84.9 | 84.0 | 0.428 | 48.6 / 46.1 | 64.9 | 65.9 (47.0) | 77.2 |
| soft, LR 5e-6 | **87.7** | **86.6** | **0.371** | 47.6 / 46.1 | 64.7 | **67.9 (50.2)** | **80.3** |
| hard, LR 2e-6 | 84.3 | 83.2 | 0.434 | 49.2 / 46.1 | **65.6** | 66.1 (46.0) | 77.8 |
| unanimous, LR 2e-6 | 84.1 | 82.8 | 0.435 | 48.9 / 46.1 | 64.7 | 65.4 (45.1) | 77.4 |
| LoRA r64, soft, LR 2e-5 | 84.4 | 84.5 | 0.401 | 46.7 / 46.1 | 64.0 | 65.7 (49.3) | 77.8 |

The labels (the pool majority) agree with Opus on 95.0% of these pairs; a tied vote counts half.

- **Learning rate:** at this short length, the highest LR is clearly best, and every curve is still rising at
  step 140.
- **Labels:** soft, hard and unanimous-only are within noise of each other on accuracy. Soft has the best log
  loss.
- **LoRA:** matches full fine-tuning at the same LR scale.
- **Length:** the RM is length-neutral in every length-ratio band. It prefers the longer answer about as often
  as the labels do, including below 50% when one answer is over 4× longer.
- **Tiers:** English is the weakest tier (78% accuracy, against 88–91% elsewhere), as it was for the judges.

**Qwen3.8-27B, full fine-tuning** (`q27-full-soft-lr2e-6`, finished 2026-11-22). The run uses all training pairs
for one epoch: 991 steps of 128 pairs. Labels are soft, the centring weight is 0.01, and the LR is 2e-6 with a
cosine decay to 10%. It ran on 4 nodes, with the last 22 steps on 2.

| Step | Acc | Opus | Log loss | Longer wins, RM / labels |
|---|---|---|---|---|
| 200 | 90.5 | 90.2 | 0.330 | 47.8 / 44.8 |
| 400 | 90.3 | 90.1 | 0.320 | 44.9 / 44.8 |
| 600 | 91.1 | 91.3 | 0.317 | 43.9 / 44.8 |
| 800 | 91.2 | 91.6 | 0.315 | 45.9 / 44.8 |
| 991 | **91.3** | **91.7** | **0.313** | 45.7 / 44.8 |

The longer-wins column is the in-training measure over 519 pairs. `eval_report.py` counts its length pairs by
token ratio and gives 47.2 / 46.3.

Benchmarks at step 991:

| Benchmark | Score |
|---|---|
| RewardBench 2 | 79.2 |
| RM-Bench | 85.1 (hard 78.7) |
| M-RewardBench | 90.0 (small 90.7) |

The best Prelude 9B run of sweep 1 scored 64.7, 67.9 (50.2) and 80.3.

- **Training curve:** most of the gain came early. From step 400 to 991, 25 Opus pairs were fixed and 10
  broken (sign test p = 0.017). From step 800 to 991 the change was 7 fixed, 6 broken. The training-batch
  accuracy, scored before each update, rose from 92.5% (steps 400–499) to 93.5% (steps 900–991).
- **Judges:** on the same 935 pairs, with each of a judge's two calls counting half, the RM's Opus agreement
  (91.7) is behind Gemma 4 31B (92.7) and Qwen3.8-27B (92.5), and ahead of Qwen3.5-122B (90.9) and gpt-oss high
  (88.4). Per tier:

  | Tier | RM | Gemma 4 31B | Qwen3.8-27B | Qwen3.5-122B | gpt-oss high |
  |---|---|---|---|---|---|
  | en (183) | 85.2 | 89.1 | 90.2 | 86.9 | 81.7 |
  | large (235) | 94.9 | 94.9 | 93.4 | 92.1 | 89.1 |
  | mid (283) | 95.1 | 93.8 | 94.2 | 93.8 | 94.0 |
  | small (234) | 89.3 | 91.9 | 91.2 | 89.3 | 86.1 |

  The RM matches or beats the best judges on the large and mid tiers. All of its gap to them is in English and
  the small languages.
- **Length:** the RM prefers the longer answer as often as the labels do in every length-ratio band:
  50.6 / 50.0 at 1.5–2×, 48.5 / 48.3 at 2–4×, and 44.1 / 42.3 above 4×.
- **Length probes:** the original answer beats its padded copy in 95.1% of probes. It beats its copy cut in
  half in 64.9%: 85% in English (87 probes) and 61% in the other languages (420).
  - The truncated probe has no reference. The cut falls at a paragraph boundary, and 88% of the cut copies end
    in sentence punctuation. The originals only won their pilot pair, so they are not flawless.
  - In the cases the RM gets "wrong", the removed part often holds the flaw: a notes section that breaks a
    one-paragraph instruction, a switch into Chinese, or invented details. Preferring the cut copy can be right.
  - The rate needs a judge baseline on the same probes before it can be read as a weakness.

**Gemma 4 31B, full fine-tuning** (`g31-full-soft-lr2e-6`, finished 2026-11-23). The data, labels, schedule and
LR are the same as for the Qwen run, with `--fp32-residual`. It ran on 4 nodes, in dev-g chunks and then one
standard-g job, at a median of 140 tokens/s per GCD.

| Step | Acc | Opus | Log loss | Longer wins, RM / labels |
|---|---|---|---|---|
| 200 | 91.6 | 90.8 | 0.332 | 45.0 / 44.7 |
| 400 | 91.4 | 91.0 | 0.315 | 45.4 / 44.7 |
| 600 | 91.9 | 90.7 | 0.323 | 45.0 / 44.7 |
| 800 | 92.4 | **91.8** | 0.314 | 45.2 / 44.7 |
| 991 | **92.6** | 91.7 | **0.312** | 44.8 / 44.7 |

The in-training longer-wins measure covers 522 pairs here (Gemma's tokenizer); `eval_report.py` gives 46.9 / 46.4.

Benchmarks at step 991, next to Qwen:

| Benchmark | Gemma 4 31B | Qwen3.8-27B |
|---|---|---|
| RewardBench 2 | **81.2** | 79.2 |
| RM-Bench | **86.3** (hard **80.5**) | 85.1 (hard 78.7) |
| M-RewardBench | **90.5** (small 90.8) | 90.0 (small 90.7) |

- **Against Qwen, on the same pairs:** Gemma is ahead on the vote majority at every evaluation, by 0.8–1.3 points,
  but never significantly. At step 991 the difference is +1.3 [−0.3, +2.9] (McNemar p = 0.13, 38 pairs only Gemma
  gets right, 25 only Qwen). On Opus the two are tied: 91.7 each, with 26 pairs each way. The two RMs' reward
  differences correlate at 0.98.
- **Benchmarks:** RewardBench 2 is +2.0, from Factuality, Focus, Math and Ties. Gemma is behind on Precise IF
  (52.5 / 55.0) and Safety (87.1 / 89.6). In RM-Bench the gain is in math (86.8 / 82.2) and the hard split.
- **Tiers** (majority accuracy, Gemma / Qwen): en 87.2 / 83.1, large 91.9 / 92.7, mid 97.0 / 94.3, small 92.2 /
  92.7. On Opus, the English tier is 86.9 (Qwen RM 85.2), still behind the best judges (89.1–90.2). The per-tier
  differences are not tested.
- **Training curve:** flat after step 400. From step 400 to 991, 15 Opus pairs were fixed and 9 broken
  (p = 0.31); from 800 to 991, 5 and 6. The training-batch accuracy rose from 92.1% (steps 400–499) to 92.7%
  (steps 900–991).
- **Length:** the RM is length-neutral in every ratio band: 50.6 / 51.4 at 1.5–2×, 48.5 / 47.5 at 2–4×, and
  43.2 / 42.4 above 4×. The original answer beats its padded copy in 92.7% of probes (Qwen 95.1) and its
  half-length copy in 64.5% (Qwen 64.9).
- **Gradient spikes:** the clipped norm is usually 15–60. Single steps reached 49,711 (step 162), 74,328 (221),
  75,120 (412) and 108,310 (626), always mostly in `post_attention_layernorm.weight` of layers 2–3 and 13–14. The
  clip contained them; the loss and the next steps were normal.
