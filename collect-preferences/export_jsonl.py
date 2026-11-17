#!/usr/bin/env python3
"""Export the preference dataset as one JSONL file: prompt, two responses, label and the votes behind it.

  collect-preferences/run.sh export_jsonl.py [--out data/preferences/preferences.jsonl]

One row per labelled pair, in the order of data/compiled/prompts.jsonl (already shuffled, seed 20261113).
Field order: messages, response_a, response_b, preference, then the other fields of the prompts.jsonl row
(id, language, ...) in its order. The fields, their values and how they were made are described in
data/preferences/preferences.README.md, written by this script from CARD below. Inputs:
  data/compiled/prompts.jsonl            the prompt and its fields, copied as they are (prompts.README.md)
  data/responses/responses.parquet       the two answers (slot 0 -> response_a, slot 1 -> response_b), their
                                         hidden reasoning, model (Hugging Face id and revision), mode, finish
  data/preferences/pairs.parquet         OpenLID language of each answer
  data/preferences/labels.parquet        label, rule, judges (tie-breaker last), every valid vote
  data/preferences/votes/*/*.parquet     the judges' answers: the comparison sentence (short_rationale) and
                                         the whole JSON answer (full_rationale), re-parsed from the raw output
                                         with judge.py's final_text and judge_prompt.extract; votes whose JSON
                                         did not parse in the run (status ok_regex) are parsed leniently
                                         (control characters allowed, invalid escapes doubled, then a regex on
                                         the five known keys, strict and then tolerant of missing quotes and
                                         commas; on the answer part, then on the whole output, since a leaked
                                         think tag quoted by the judge makes final_text split inside the JSON)
                                         and kept only if the verdict is the stored one
  collect-preferences/judges.yaml        judge pool name -> Hugging Face id
Left out: the 70 pairs without a label (no judge could read them, see build_labels.py; they stay in
pairs.parquet), the 4 prompts whose two answers are both empty (not in pairs.parquet), the Opus silver votes
of the pilot (evaluation only), seeds, token counts, the judges' thinking, and the pipeline's internal columns
(all kept in the parquet files).
The file is written to <out>.tmp and renamed when complete, after checks on the written file: field order,
row count and ids against labels.parquet, messages against prompts.jsonl, answers against pairs.parquet,
labels in [0, 1] and equal to labels.parquet, vote counts against labels.parquet, the think-tag rule's loser
containing a think tag, the too-long rule's rows having no vote.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from judge import final_text  # noqa: E402  (vLLM is imported lazily inside judge.main, so this is CPU-only)
from judge_prompt import extract  # noqa: E402

ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
PREF = os.path.join(DATA, "preferences")
THINK_TAGS = ("<think>", "</think>")
SIDE = {"x": "a", "y": "b"}
MODE = {"direct": "direct", "think": "thinking"}
RULES = (None, "think_tag", "too_long")
PROMPT_FIELDS = ("id", "language", "messages", "translation_status", "translation_model", "translated_from_language",
                 "category", "task_type", "subject_domain", "author_type", "safety_label", "locale_dependence",
                 "complexity", "explicit_constraint_types", "request_summary", "verifiable_instructions", "source")
OUT_FIELDS = ("messages", "response_a", "response_b", "preference") + tuple(f for f in PROMPT_FIELDS if f != "messages")
JUDGE_KEYS = (("request", "Request"), ("response_a", "Response A"), ("response_b", "Response B"),
              ("comparison", "Comparison"), ("verdict", "Verdict"))
FIELDS_RE = re.compile(r'"request"\s*:\s*"(?P<request>.*?)"\s*,\s*"response_a"\s*:\s*"(?P<response_a>.*?)"\s*,\s*'
                       r'"response_b"\s*:\s*"(?P<response_b>.*?)"\s*,\s*"comparison"\s*:\s*"(?P<comparison>.*?)"\s*,\s*'
                       r'"verdict"\s*:\s*"(?P<verdict>[AB])"', re.S)
SEP = r'["\']?\s*,?\s*'                                  # a value's closing quote and the comma may be missing
KEY = lambda k: r'"?' + k + r'"?\s*:\s*["\']?'           # noqa: E731  the quotes around a key may be missing
TOLERANT_RE = re.compile(KEY("request") + r"(?P<request>.*?)" + SEP + KEY("response_a") + r"(?P<response_a>.*?)" + SEP
                         + KEY("response_b") + r"(?P<response_b>.*?)" + SEP + KEY("comparison") + r"(?P<comparison>.*?)" + SEP
                         + r'"?verdict"?\s*:\s*["\']?(?P<verdict>[AB])', re.S)

CARD = """# Multilingual preference pairs (128k) for reward modelling

`preferences.jsonl`: {rows:,} prompts in {nlang} languages, each with two model answers and a preference
label, one JSON object per line, in the same random order as `prompts.jsonl` (shuffle seed 20261113). Each
row is a conversation that ends in a user turn, two answers to it from models drawn at random from a pool of
17, and the preference between them: the share of the votes of three strong open LLM judges (a fourth on ties),
each judging both presentation orders, after two post-processing rules. Written by
`collect-preferences/export_jsonl.py`; the parquet files it reads keep every intermediate column.

## Fields

Every row has the same {nfields} fields in the same order: `messages`, `response_a`, `response_b`, `preference`,
then the other fields of the row of `prompts.jsonl` with the same `id`, unchanged and in its order: `id`,
`language`, `translation_status`, `translation_model`, `translated_from_language`, `category`, `task_type`,
`subject_domain`, `author_type`, `safety_label`, `locale_dependence`, `complexity`,
`explicit_constraint_types`, `request_summary`, `verifiable_instructions` and `source`. `messages` is the
conversation, `[{{"role", "content"}}]`, ending in the user turn to be answered; it and the other prompt fields
are described in `data/compiled/prompts.README.md`. {missing_prompts} prompts of `prompts.jsonl` have no row
here (see below).

### `response_a`, `response_b`

Two answers to the conversation. Which answer is `a` carries no information: the two were drawn
independently, and the labels are symmetric by construction (`a` is longer in {a_longer:.1f}% of the rows and
wins {a_wins:.1f}% of the label mass).

| Field | Meaning |
|---|---|
| `content` | The answer as the model wrote it, never edited. Empty in {empty:,} answers (the model produced nothing; the pair is labelled like any other). |
| `reasoning` | The model's hidden thinking before the answer, for `thinking` mode ({reasoning:,} answers; null when the mode is `direct` or the model left it empty). Not shown to the judges. |
| `model` | Hugging Face id of the model, and `model_revision` the commit of the weights. |
| `mode` | `direct` or `thinking` (the model's native thinking mode; stored apart in `reasoning`). |
| `finish_reason` | `stop`: the model ended its answer. `length`: it was cut at the token budget ({length:,} answers, {length_pct:.1f}%; 16,384 new tokens for `direct`, 32,768 for `thinking`, or the model's context if that is smaller). |
| `detected_language` | OpenLID-v3 label of the answer text (OpenLID's own code set, e.g. `nb`, `ekk`), with `detected_language_confidence`; null for {lid_null:,} answers that were empty or too short. A heuristic, for filtering wrong-language answers. |

### `preference`

| Field | Meaning |
|---|---|
| `prob_a_better` | **The label**: the probability that `response_a` is the better answer, in [0, 1]. It is the share of the valid votes for `a` over all votes of the pair (so a judge that changes its verdict with the presentation order contributes 0.5), set to exactly 0 or 1 where a `rule` applies. Values: {label_hist}. |
| `rule` | null for {rule_none:,} rows (the probability is the vote share). `think_tag` ({rule_think:,}): one answer contains `<think>` or `</think>` (reasoning written into the answer) and the other does not; that answer loses whatever the votes, which are still listed. `too_long` ({rule_long:,}): no judge could read the pair (the conversation with its two answers exceeds every judge's context, so it has no votes); the answer that ran to the token limit and is over four times longer than the other, which ended normally without looping, loses. All such long answers were read: every one is a loop, word salad or a drift into unrelated text. |
| `votes` | Every valid vote on the pair, in the order the judges were assigned, both presentation orders of each judge ({votes:,} votes in all, {votes_per_row:.2f} per row). `judge`: Hugging Face id. `shown_first`: which answer the judge saw as "Response A" (`a` or `b`; the other was "Response B"). `prefers`: `a` or `b`. `short_rationale`: the judge's `comparison` field, two to four English sentences on the decisive differences. `full_rationale`: the judge's whole answer, its four analysis fields and the verdict as paragraphs `Request: ...`, `Response A: ...`, `Response B: ...`, `Comparison: ...`, `Verdict: A` (or `B`), in the judge's own frame: "Response A" in both rationales is the answer named by `shown_first`. `full_rationale` is null for {no_full:,} votes whose JSON could not be parsed even leniently (their verdict was recovered from the text), `short_rationale` for those and for {no_short_only:,} more whose `comparison` was empty. A pair whose first six votes split evenly got a fourth judge, the next eligible substitute, whose two votes come last ({tiebreaks:,} rows: {tie_breakers}); {ties:,} rows ({ties_pct:.2f}%) remain at exactly 0.5, because the fourth judge's two votes went opposite ways or no eligible judge was left (374 pairs). |

## How the answers were made

Two answers per prompt, each from a (model, mode) drawn at random and independently (so {same_model_pct:.1f}% of
the rows have two answers from the same model), with weights that downsample models that often produce word
salad. Temperature 1.0 and nothing else; a model with both modes uses each half the time; the answer and the
thinking are stored as generated, never edited or filtered. Answers cut at the first pass's budget were
continued from where they stopped with the same model and sampling, so that long answers were not replaced by
short ones ({continued:,} answers, {continued_pct:.1f}%). Details: `collect-responses/README.md`.

| Model | Group | Modes | Answers |
|---|---|---|---|
{model_table}

Group: `prelude` are OpenEuroLLM Prelude 9B checkpoints (the models the reward model is for); `strong`, `mid`
and `weak` describe the other open models' quality for these languages. Three chat templates (Qwen2.5, Apertus,
Salamandra) inject their own identity as a system prompt; for these a system prompt of the same wording naming
Prelude was used instead, so that no answer is about a model name. No other prompt was ever added.

## How the labels were made

- **Judges.** Three judges on every pair, both presentation orders, with their native thinking mode on and
  the vendors' recommended sampling (never greedy). The pool is Gemma 4 31B, Qwen3.8-27B and gpt-oss-120b
  (reasoning effort high). A judge never judges a pair with an answer from its own family (self-preference);
  its place goes to the first eligible substitute in the order Qwen3.5-122B-A10B, Mistral Small 4, GLM-4.5-Air.
  A pair whose six votes split evenly gets the next eligible judge not yet used, in the same order. Prelude
  checkpoints are never judges, and the judges' licences (Apache-2.0, MIT) put no restriction on their outputs.

  | Judge | Revision | Role | Votes |
  |---|---|---|---|
{judge_table}

- **What the judges saw.** The conversation and the two answers verbatim, as "Response A" and "Response B":
  no model names, no hidden reasoning, no metadata. A single English system prompt (v3) asks for one overall
  preference: work out what the user needs, then assess both answers on criteria in order of priority, where a
  serious failure on a higher one outweighs any advantage on a lower one: harm (how to decline a request that
  should be declined), usability, language (answering in the user's language), correctness, instruction
  following (a wrongful refusal of a legitimate request loses to any acceptable answer but beats a wrong one),
  helpfulness and concision (of two equally good answers the shorter wins; moralising and padding count
  against). A system message and earlier assistant turns are context, not what is judged; position and
  claimed identity are to be ignored. The judge answers with a JSON object whose analysis fields (`request`,
  `response_a`, `response_b`, `comparison`) precede a forced `verdict` of "A" or "B"; `full_rationale` is that
  object. Prompt text: `collect-preferences/judge_prompt.py`.
- **Label.** The share of the valid votes for `a` over every vote of the pair, then the two rules above
  (`collect-preferences/build_labels.py`). Votes are binary by design: graded verdicts and token probabilities
  were not used.
- **Not labelled, not included.** 4 prompts whose two answers are both empty, and 70 pairs that no judge
  could read and the too-long rule does not cover. The other {rule_long} pairs no judge could read have a label
  from that rule.

## Properties and caveats

- **Agreement.** {unanimous_pct:.1f}% of the rows labelled by votes are unanimous (probability 0 or 1), {middle_pct:.1f}%
  are in the middle third [1/3, 2/3]. The labels were evaluated against Claude Opus 5.5 (adaptive thinking,
  effort high, the same prompt) only on the 1,008 pilot pairs, which are stratified (200 English, 17–50 per
  other language, safety and instruction-following slices boosted) and were also used to choose the prompt
  version and the judge pool: on the 945 of them where Opus is consistent across the two orders, the labels
  agree with Opus on 94.7% (an even split counting half; Brier score 0.046). That figure is in-sample and not
  an estimate for the whole dataset; the Opus votes are not in this file and were never used as labels.
- **Length.** The labels favour conciseness. Between equally strong responders, an answer twice as long as
  an equally good one wins 42.4% of the clean pairs (both answers at least 200 characters, neither cut, both
  in the prompt's language); on the pilot pairs 40.0%, against 39.2% for Opus. Length alone predicts the
  majority side of the label on 50.5% of the clean pairs. The substitutes lean longer than the pool judges
  (GLM-4.5-Air most) but cast {subst_pct:.0f}% of the votes. Full tables: `data/preferences/LENGTH_BIAS.txt`.
- **Quality of the answers.** Bad answers are wanted (the data trains a reward model): wrong-language
  answers, loops, word salad, refusals and empty answers are all in, labelled like any other. `finish_reason`,
  `detected_language` and the models' word-salad rates (`data/responses/STATS.txt`) help to filter.
- **Labels are model judgements**, not human annotations: the prompt fields were predicted by Gemma 4 31B
  (see `prompts.README.md`), the preferences by the judges above.
- **Rows at 0.5** carry no preference and can be dropped for pairwise training; rows labelled by a rule have
  a probability of exactly 0 or 1 whatever the votes were.

## Figures

`plots/`: the prompt distributions of these rows (`collect-prompts/compile/plot_stats.py`: languages,
composition, label fields, tasks and domains, prompt length, sources, allocation) and, from
`collect-preferences/plot_dataset.py`, `label_distribution`, `agreement` (unanimous / majority / tie per
language and task type), `model_win_rates` (label mass per responder model, thinking vs direct) and `length`
(answer length per model; label mass of the longer answer by length ratio). The numbers behind each figure are
in `plots/tables/`.

## Not included

Seeds and token counts of the answers, the judges' thinking (their raw outputs), the Opus silver votes of the
pilot, and the pipeline's internal columns. They are in `data/responses/responses.parquet`,
`data/preferences/votes/`, `data/preferences/pilot/v3/opus/` and the other parquet files named above.
"""


def render(obj):
    """the judge's JSON answer as labelled paragraphs, in the judge's own frame (Response A = shown first)"""
    parts = ["{}: {}".format(head, str(obj[key]).strip()) for key, head in JUDGE_KEYS if obj.get(key)]
    return "\n\n".join(parts) or None


def lenient(text):
    """the last JSON object with a verdict in text, allowing control characters inside strings"""
    dec = json.JSONDecoder(strict=False)
    for m in reversed(list(re.finditer(r"\{", text))):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("verdict") in ("A", "B"):
            return obj
    return None


def unescape(s):
    s = re.sub(r'\\([\\"/])', r"\1", s.replace("\\n", "\n").replace("\\t", "\t"))
    return s.strip().rstrip("\"'").strip()


def by_regex(pattern, text):
    ms = list(pattern.finditer(text))
    return {k: (v if k == "verdict" else unescape(v)) for k, v in ms[-1].groupdict().items()} if ms else None


def recover(final, raw, verdict):
    """(obj, stage) for a judge answer whose JSON did not parse in the run, or (None, 'unrecovered'). Tried on the
    answer part first, then on the whole output: a judge that quotes a leaked think tag in its analysis makes
    final_text split inside the JSON (the verdict regex of the run still saw the verdict)."""
    stages = (("strict=False", lambda t: lenient(t)),
              ("escape fix", lambda t: lenient(re.sub(r'\\(?![\\/"bfnrtu])', r"\\\\", t))),
              ("field regex", lambda t: by_regex(FIELDS_RE, t)),
              ("tolerant regex", lambda t: by_regex(TOLERANT_RE, t)))
    for where, text in (("", final), (" (whole output)", raw)):
        for name, fn in stages:
            try:
                obj = fn(text)
            except Exception:
                obj = None
            if obj is not None and obj.get("verdict") == verdict:
                return obj, name + where
    return None, "unrecovered"


def load_rationales(judges):
    """(pair id, judge, order) -> (short_rationale, full_rationale) of every valid vote, re-parsed from raw"""
    out, stat = {}, collections.Counter()
    files = sorted(glob.glob(os.path.join(PREF, "votes", "*", "*.parquet")))
    for f in files:
        t = pq.read_table(f, columns=["pair_id", "order", "judge", "status", "prefers", "verdict", "comparison", "raw"])
        t = t.filter(pc.is_in(t["status"], value_set=pa.array(["ok", "ok_regex"])))
        for r in t.to_pylist():
            if r["prefers"] not in ("x", "y"):
                continue
            final = final_text(r["raw"], judges[r["judge"]])
            if r["status"] == "ok":
                obj = extract(final)
                assert obj is not None and obj.get("verdict") == r["verdict"], (f, r["pair_id"], r["order"])
                stat["ok: comparison differs from the stored one"] += (obj.get("comparison") or None) != (r["comparison"] or None)
                stage = "ok"
            else:
                obj, stage = recover(final, r["raw"], r["verdict"])
            stat[stage] += 1
            short = (obj.get("comparison") if obj else None) or None
            out[(r["pair_id"], r["judge"], r["order"])] = (short.strip() if short else None, render(obj) if obj else None)
    assert not stat["ok: comparison differs from the stored one"], dict(stat)
    print("rationales of {:,} valid votes from {} files; parse: {}".format(len(out), len(files), dict(stat)), flush=True)
    return out


def response_out(r, lid, conf):
    return {
        "content": r["response"],
        "reasoning": r["reasoning"] or None,
        "model": r["model_hf"],
        "model_revision": r["model_revision"],
        "mode": MODE[r["mode"]],
        "finish_reason": r["finish_reason"],
        "detected_language": lid,
        "detected_language_confidence": conf if lid is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PREF, "preferences.jsonl"))
    args = ap.parse_args()
    judges = {j["name"]: j for j in yaml.safe_load(open(os.path.join(HERE, "judges.yaml"), encoding="utf-8"))["judges"]}
    hf = lambda name: judges[name]["hf"]

    prompts, order = {}, []
    with open(os.path.join(DATA, "compiled", "prompts.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            prompts[o["id"]] = o
            order.append(o["id"])
    assert tuple(prompts[order[0]]) == PROMPT_FIELDS, "prompts.jsonl fields changed: {}".format(list(prompts[order[0]]))
    labels = {r["id"]: r for r in pq.read_table(os.path.join(PREF, "labels.parquet")).to_pylist()}
    keep = [i for i in order if i in labels and labels[i]["label"] is not None]   # 4 prompts have no pair
    print("{:,} prompts, {:,} pairs, {:,} labelled".format(len(order), len(labels), len(keep)), flush=True)
    resp = collections.defaultdict(dict)
    t = pq.read_table(os.path.join(DATA, "responses", "responses.parquet"),
                      columns=["id", "slot", "model", "model_hf", "model_revision", "model_group", "mode", "reasoning",
                               "response", "finish_reason", "continued"])
    for r in t.to_pylist():
        resp[r["id"]][r["slot"]] = r
    del t
    pairs = {r["id"]: r for r in pq.read_table(os.path.join(PREF, "pairs.parquet"),
                                                columns=["id", "lid_x", "lid_conf_x", "lid_y", "lid_conf_y"]).to_pylist()}
    rationales = load_rationales(judges)

    cnt = collections.defaultdict(collections.Counter)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for i in keep:
            p, lab, rp = pairs[i], labels[i], resp[i]
            votes = []
            for v in lab["votes"]:
                key = (i, v["judge"], v["order"])
                assert key in rationales, "vote without a stored judgement: {}".format(key)
                short, full = rationales[key]
                votes.append({"judge": hf(v["judge"]), "shown_first": "a" if v["order"] == 0 else "b",
                              "prefers": SIDE[v["prefers"]], "short_rationale": short, "full_rationale": full})
                cnt["judge"][v["judge"]] += 1
                cnt["no_full"][full is None] += 1
                cnt["no_short_only"][full is not None and short is None] += 1
            tie = lab["judges"][3] if len(lab["judges"]) == 4 else None
            src = prompts[i]
            o = {"messages": src["messages"],
                 "response_a": response_out(rp[0], p["lid_x"], p["lid_conf_x"]),
                 "response_b": response_out(rp[1], p["lid_y"], p["lid_conf_y"]),
                 "preference": {"prob_a_better": lab["label"], "rule": lab["rule"], "votes": votes}}
            o.update((k, src[k]) for k in OUT_FIELDS[4:])
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            # statistics for the card
            cnt["language"][o["language"]] += 1
            cnt["rule"][lab["rule"]] += 1
            cnt["label"][round(lab["label"], 3)] += 1
            cnt["tie"][tie] += 1
            cnt["same_model"][rp[0]["model"] == rp[1]["model"]] += 1
            cnt["a_longer"][len(rp[0]["response"]) > len(rp[1]["response"])] += 1
            cnt["label_mass"]["a"] += lab["label"]
            if lab["rule"] is None:
                cnt["unanimous"][lab["label"] in (0.0, 1.0)] += 1
                cnt["middle"][1 / 3 - 1e-9 <= lab["label"] <= 2 / 3 + 1e-9] += 1
            for r in (rp[0], rp[1]):
                cnt["model"][(r["model_hf"], r["model_group"], r["model"])] += 1
                cnt["model_modes"][(r["model_hf"], r["mode"])] += 1
                cnt["finish"][r["finish_reason"]] += 1
                cnt["continued"][bool(r["continued"])] += 1
                cnt["empty"][not r["response"].strip()] += 1
                cnt["reasoning"][bool(r["reasoning"])] += 1
            cnt["lid_null"][p["lid_x"] is None] += 1
            cnt["lid_null"][p["lid_y"] is None] += 1
    del rationales

    # checks on the written file
    full = {r["id"]: r for r in pq.read_table(os.path.join(PREF, "pairs.parquet"),
                                               columns=["id", "response_x", "response_y"]).to_pylist()}
    ids, bad = [], collections.Counter()
    with open(tmp, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            i = o["id"]
            ids.append(i)
            lab, pr = labels[i], o["preference"]
            bad["fields"] += tuple(o) != OUT_FIELDS or tuple(pr) != ("prob_a_better", "rule", "votes")
            bad["prompt_fields"] += any(o[k] != prompts[i][k] for k in PROMPT_FIELDS)
            bad["last_turn"] += not (o["messages"][-1]["role"] == "user" and (o["messages"][-1]["content"] or "").strip())
            bad["answers"] += o["response_a"]["content"] != full[i]["response_x"] or o["response_b"]["content"] != full[i]["response_y"]
            bad["label"] += not (pr["prob_a_better"] is not None and 0.0 <= pr["prob_a_better"] <= 1.0 and pr["prob_a_better"] == lab["label"])
            bad["rule"] += pr["rule"] not in RULES or pr["rule"] != lab["rule"]
            bad["votes"] += len(pr["votes"]) != lab["n_votes"] or sum(v["prefers"] == "a" for v in pr["votes"]) != lab["votes_x"]
            bad["vote_values"] += any(v["shown_first"] not in ("a", "b") or v["prefers"] not in ("a", "b")
                                      or not all(isinstance(v[k], (str, type(None))) for k in ("short_rationale", "full_rationale"))
                                      for v in pr["votes"])
            if pr["rule"] is None and pr["votes"]:
                bad["vote_share"] += abs(pr["prob_a_better"] - sum(v["prefers"] == "a" for v in pr["votes"]) / len(pr["votes"])) > 1e-9
            if pr["rule"] == "think_tag":
                loser = o["response_b" if pr["prob_a_better"] == 1.0 else "response_a"]["content"]
                winner = o["response_a" if pr["prob_a_better"] == 1.0 else "response_b"]["content"]
                bad["think_tag"] += not (any(t in loser for t in THINK_TAGS) and not any(t in winner for t in THINK_TAGS))
            if pr["rule"] == "too_long":
                bad["too_long"] += pr["prob_a_better"] not in (0.0, 1.0) or bool(pr["votes"])
            for side in ("response_a", "response_b"):
                r = o[side]
                bad["response"] += r["mode"] not in MODE.values() or r["finish_reason"] not in ("stop", "length") or not r["model"]
    assert len(ids) == len(set(ids)) == len(keep) and ids == keep, "id mismatch"
    assert not sum(bad.values()), dict(bad)
    os.replace(tmp, args.out)

    # the card
    rows = len(ids)
    n_ans = 2 * rows
    fmt_hist = ", ".join("{} ({:,})".format(("{:g}".format(k) if k in (0.0, 0.5, 1.0) else "{:.3f}".format(k)), v)
                         for k, v in sorted(cnt["label"].items(), key=lambda kv: -kv[1])[:8])
    fmt_hist += " and {} more values".format(len(cnt["label"]) - 8) if len(cnt["label"]) > 8 else ""
    tb = cnt["tie"]
    tie_breakers = ", ".join("{} {:,}".format(hf(k), v) for k, v in tb.most_common() if k is not None)
    group_order = {"prelude": 0, "strong": 1, "mid": 2, "weak": 3}
    model_rows = []
    for (m, g, name), n in sorted(cnt["model"].items(), key=lambda kv: (group_order[kv[0][1]], -kv[1])):
        modes = ", ".join(MODE[k] for k in ("direct", "think") if cnt["model_modes"][(m, k)])
        model_rows.append("| `{}` | {} | {} | {:,} ({:.1f}%) |".format(m, g, modes, n, 100 * n / n_ans))
    pool = ("gemma-4-31B-it", "qwen3.8-27b", "gpt-oss-120b-bf16")
    subst = ("qwen3.5-122b-a10b", "mistral-small-4-119b", "glm-4.5-air")
    judge_rows = []
    for name in pool + subst:
        j = judges[name]
        role = "pool" if name in pool else "substitute {}, tie-breaker".format(subst.index(name) + 1)
        judge_rows.append("  | `{}` | `{}` | {} | {:,} |".format(j["hf"], j["revision"], role, cnt["judge"][name]))
    n_votes = sum(cnt["judge"].values())
    card = CARD.format(
        rows=rows, nlang=len(cnt["language"]), nfields=len(OUT_FIELDS), missing_prompts=len(order) - rows,
        a_longer=100 * cnt["a_longer"][True] / rows, a_wins=100 * cnt["label_mass"]["a"] / rows,
        empty=cnt["empty"][True], reasoning=cnt["reasoning"][True], length=cnt["finish"]["length"],
        length_pct=100 * cnt["finish"]["length"] / n_ans, lid_null=cnt["lid_null"][True], label_hist=fmt_hist,
        rule_none=cnt["rule"][None], rule_think=cnt["rule"]["think_tag"], rule_long=cnt["rule"]["too_long"],
        no_full=cnt["no_full"][True], no_short_only=cnt["no_short_only"][True],
        votes=n_votes, votes_per_row=n_votes / rows, tiebreaks=sum(v for k, v in tb.items() if k is not None),
        tie_breakers=tie_breakers, ties=cnt["label"][0.5], ties_pct=100 * cnt["label"][0.5] / rows,
        same_model_pct=100 * cnt["same_model"][True] / rows, continued=cnt["continued"][True],
        continued_pct=100 * cnt["continued"][True] / n_ans, model_table="\n".join(model_rows),
        judge_table="\n".join(judge_rows),
        unanimous_pct=100 * cnt["unanimous"][True] / max(1, sum(cnt["unanimous"].values())),
        middle_pct=100 * cnt["middle"][True] / max(1, sum(cnt["middle"].values())),
        subst_pct=100 * sum(cnt["judge"][s] for s in subst) / n_votes)
    card_path = os.path.join(os.path.dirname(args.out), "preferences.README.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card)
    print("{:,} rows, {:,} votes -> {} ({:.2f} GB); card -> {}".format(
        rows, n_votes, args.out, os.path.getsize(args.out) / 1e9, card_path))
    print("labels:", dict(cnt["rule"]), "| ties", cnt["label"][0.5], "| judges", dict(cnt["judge"]),
          "| full_rationale null", cnt["no_full"][True], "| short null beyond those", cnt["no_short_only"][True])


if __name__ == "__main__":
    main()
