"""Side-by-side chatbot arenas: compar:IA (French government arena, 4 repos),
AI-Arenaen (Danish compar:IA deployment) and LMArena (140k / 100k votes).

Common representation
---------------------
Every source row holds two side conversations A and B with the same user
messages; each side has its own (model's) assistant history. Each side is cut
into turns, one per user message (context = side conversation up to and
including that user message, reply = the assistant message right after it,
if any). For each turn:

* contexts of A and B identical (compared modulo surrounding whitespace)
  -> ONE record, responses = [reply A, reply B] (index 0 = A, 1 = B);
  source_id `<row id>:t<k>`. This is always the case for the first turn,
  except when the user text differs (a handful of rows).
* contexts differ (all later turns: each side has its own history)
  -> one record PER SIDE with that side's reply only (README rule 3);
  source_id `<row id>:t<k>:a|b`.

Hidden per-model system prompts: compar:IA gives a system prompt to some
models only. When the contexts differ ONLY by their system messages, the turn
is still treated as shared: the record context drops the system messages and
each response keeps its own prompt in response meta `system_prompt`. Identical
system prompts stay as the leading system message (README rule 5); per-side
records always keep their side's system message.

Turn indices k are 0-based (as compar:IA's own `turn` / `msg_rank`). Every
response has `model` set (compar:IA outputs of proprietary models are
restricted to evaluation use -> filter on the model name). For compar:IA
sources the response meta also carries `model_license` (license kind:
open-weights / proprietary) and `model_reuse`, looked up by name in the repo's
own models.json, else in the comparia-fr-arena download's models.json.
Response meta `side` is "a" or "b". Reasoning (`reasoning`,
`reasoning_content`, `<think>` blocks) is dropped -> `reasoning_dropped`;
`<think>` without a closing tag is left as is -> `think_unclosed`.

Preference types (responses index 0 = side A, 1 = side B)
---------------------------------------------------------
* pairwise               {"winner": 0|1|"tie"|"tie_good"|"tie_bad", "raw"}
    vote cast on exactly these two responses (single-turn battle, or a
    per-turn vote on the first turn).
* pairwise_conversation  same fields + "n_turns": the vote was cast after a
    multi-turn conversation; the record is its first (shared-context) turn.
    The later turns are per-side records without preference (vote in meta).
* side_vote              {"side": "a"|"b", "outcome": "win"|"loss"|"tie"|
    "tie_good"|"tie_bad", "raw"}: per-turn vote between two replies whose
    contexts differ (each side has its own history); the record holds one
    side, the other side is source_id with the other suffix.
* scores (reactions)     {"scores": [1|0], "scale": ..., "flags", "comment",
    "raw"}: thumbs up/down on a single message plus its qualifier flags.
Votes may carry per-side qualifiers as "labels" (list aligned with
responses, or merged into side_vote). Records of a voted row that carry no
preference keep the vote under meta["vote"]. Undecided votes ("idk") get no
preference (raw choice in meta).

Per source
----------
* comparia-conversations: every turn of every pair as above, no preference.
* comparia-votes: same turns; the conversation-level vote becomes pairwise
  (1-turn pair) or pairwise_conversation (multi-turn) on the first turn
  (side_vote if that turn is not shared). Votes whose chosen model is not in
  the pair (26 rows) get no preference (meta vote_error).
* comparia-reactions: one record per reaction (= one assistant message of one
  side): context = that side's conversation up to the message, responses =
  [message], preference scores.
* comparia-fr-arena, ai-arenaen: one source row = one turn of a comparison
  -> shared record with `choice` as pairwise (turn 0) or two side_vote
  records (later turns). Context from full_conversation_a/b, reply from
  response_a/b. ai-arenaen has the same schema as comparia-fr-arena.
* lmarena 140k: one row = one vote (evaluation_order) inside a session; turns
  of earlier evaluation orders are only in `full_conversation` (models were
  re-sampled after each vote but each side continues its own column). Side
  history = earlier entries' (user, model_side_x) + this vote's conversation;
  only this vote's turns are emitted (earlier ones belong to earlier rows),
  so for evaluation_order > 1 all records are per-side.
* lmarena 100k: classic arena format, one vote over the whole conversation.

Language: compar:IA `languages[0]` (list kept in meta) where the repo has it
(conversations, fr-arena, ai-arenaen; votes/reactions have no language
field -> null); LMArena `language` as given (140k: ISO codes, 100k: names).
"""
import json
import os
import re

from common import iter_parquet, norm_role, register, response

_THINK = re.compile(r"<think>.*?</think>", re.S)
FLAGS = ("useful", "complete", "creative", "clear_formatting", "incorrect",
         "superficial", "instructions_not_followed")
WINNER_INDEX = {"a": 0, "b": 1}


# ----------------------------------------------------------------- helpers

def _strip_think(text):
    """-> (text without <think> blocks, dropped?, unclosed <think>?)"""
    if "think>" not in text:
        return text, False, False
    out = _THINK.sub("", text)
    if "</think>" in out and "<think>" not in out:
        out = out.rsplit("</think>", 1)[1]  # opening tag was in the template
    if out != text:
        out = out.lstrip()
    return out, out != text, "<think>" in out


def _side(raw_msgs, text=lambda m: m.get("content")):
    """Normalize one side's messages -> (msgs, info); info[i] is the response
    meta for assistant message i."""
    msgs, info = [], []
    for m in raw_msgs or []:
        role = norm_role(m["role"])
        content = text(m)
        content = "" if content is None else str(content)
        extra = {}
        if role == "assistant":
            content, dropped, unclosed = _strip_think(content)
            if dropped or m.get("reasoning") or m.get("reasoning_content"):
                extra["reasoning_dropped"] = True
            if unclosed:
                extra["think_unclosed"] = True
            if (m.get("metadata") or {}).get("is_cached"):
                extra["is_cached"] = True
        msgs.append({"role": role, "content": content})
        info.append(extra)
    return msgs, info


def _reply(content, model, side, lic, **extra):
    meta = {"side": side}
    meta.update(lic.get(model) or {})
    meta.update(extra)
    return {"content": content, "model": model, "meta": meta}


def _turns(side_msgs, model, side, lic):
    """One (context, reply|None) per user message of a side conversation."""
    msgs, info = side_msgs
    out = []
    for i, m in enumerate(msgs):
        if m["role"] != "user":
            continue
        reply = None
        if i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
            reply = _reply(msgs[i + 1]["content"], model, side, lic, **info[i + 1])
        out.append((msgs[:i + 1], reply))
    return out


def _key(context):
    return tuple((m["role"], m["content"].strip()) for m in context)


def _shared(ctx_a, ctx_b):
    """Shared context of two sides -> (context, sys_a, sys_b) or None.
    sys_* are set when the sides differ only by their system prompts."""
    if _key(ctx_a) == _key(ctx_b):
        return ctx_a, None, None
    na = [m for m in ctx_a if m["role"] != "system"]
    nb = [m for m in ctx_b if m["role"] != "system"]
    if len(na) < len(ctx_a) or len(nb) < len(ctx_b):
        if _key(na) == _key(nb):
            sa = "\n".join(m["content"] for m in ctx_a if m["role"] == "system")
            sb = "\n".join(m["content"] for m in ctx_b if m["role"] == "system")
            return na, sa or None, sb or None
    return None


def _resp(r, system_prompt=None):
    meta = dict(r["meta"])
    if system_prompt:
        meta["system_prompt"] = system_prompt
    return response(r["content"], model=r["model"], **meta)


def _outcome(winner, side):
    if winner in WINNER_INDEX:
        return "win" if winner == side else "loss"
    return winner


def _emit_turn(ctx, base_id, k, ta, tb, meta, language, vote=None,
               vote_span=1, vote_first=0, subset="default"):
    """Yield the record(s) for turn k. ta/tb = (context, reply|None) or None.
    vote = {"winner": "a"|"b"|"tie"|"tie_good"|"tie_bad", "raw": ...,
    "labels": {"a": {...}, "b": {...}} (optional)} covering vote_span turns
    starting at turn vote_first."""
    meta = dict(meta, turn=k)

    def rec(sid, context, responses, pref, extra=None):
        if not context[-1]["content"].strip():
            ctx.skip("empty_user_turn")
            return None
        m = dict(meta, **(extra or {}))
        if vote is not None and pref is None:
            m["vote"] = {"winner": vote["winner"], "raw": vote["raw"]}
        return ctx.record(subset=subset, split="train", source_id=sid,
                          context=context, responses=responses,
                          preference=pref, language=language, meta=m)

    shared = _shared(ta[0], tb[0]) if ta and tb else None
    if shared is not None:
        context, sys_a, sys_b = shared
        ra, rb = ta[1], tb[1]
        responses = [_resp(r, s) for r, s in ((ra, sys_a), (rb, sys_b)) if r]
        pref = None
        if vote is not None and ra and rb and (vote_span == 1 or k == vote_first):
            w = vote["winner"]
            pref = {"type": "pairwise" if vote_span == 1 else "pairwise_conversation",
                    "winner": WINNER_INDEX.get(w, w), "raw": vote["raw"]}
            if vote_span > 1:
                pref["n_turns"] = vote_span
            if vote.get("labels"):
                pref["labels"] = [vote["labels"]["a"], vote["labels"]["b"]]
        r = rec("{}:t{}".format(base_id, k), context, responses, pref)
        if r is not None:
            yield r
        return
    both = bool(ta and tb and ta[1] and tb[1])
    for side, t in (("a", ta), ("b", tb)):
        if t is None:
            continue
        context, reply = t
        pref = None
        if vote is not None and vote_span == 1 and both:
            pref = {"type": "side_vote", "side": side,
                    "outcome": _outcome(vote["winner"], side), "raw": vote["raw"]}
            if vote.get("labels"):
                pref.update(vote["labels"][side])
        r = rec("{}:t{}:{}".format(base_id, k, side), context,
                [_resp(reply)] if reply else [], pref, {"side": side})
        if r is not None:
            yield r


def _emit_pair(ctx, base_id, turns_a, turns_b, meta, language, vote=None,
               start=0, subset="default"):
    """All turns >= start of a pair; a vote covers exactly these turns."""
    n = max(len(turns_a), len(turns_b))
    if n <= start:
        ctx.skip("no_user_turn")
        return
    for k in range(start, n):
        ta = turns_a[k] if k < len(turns_a) else None
        tb = turns_b[k] if k < len(turns_b) else None
        yield from _emit_turn(ctx, base_id, k, ta, tb, meta, language, vote,
                              vote_span=n - start, vote_first=start, subset=subset)


def _model_info(ctx):
    """human_id -> {"model_license", "model_reuse"} from compar:IA models.json."""
    for d in (ctx.raw_dir, os.path.join(os.path.dirname(ctx.raw_dir),
                                        "ministere-culture__comparia-fr-arena")):
        p = os.path.join(d, "models.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                models = json.load(f)
            out = {}
            for m in models:
                lic = m.get("license") or {}
                if m.get("human_id"):
                    out[m["human_id"]] = {"model_license": lic.get("kind"),
                                          "model_reuse": lic.get("reuse")}
            return out
    return {}


def _first(langs):
    return langs[0] if langs else None


def _unique(seen, sid):
    seen[sid] = seen.get(sid, 0) + 1
    return sid if seen[sid] == 1 else "{}#{}".format(sid, seen[sid])


# ------------------------------------------------------------ compar:IA

@register("ministere-culture/comparia-conversations")
def comparia_conversations(ctx):
    lic = _model_info(ctx)
    seen = {}
    for _, row in iter_parquet(ctx.files("conversations.parquet")):
        ma, mb = row["model_a_name"], row["model_b_name"]
        ta = _turns(_side(row["conversation_a"]), ma, "a", lic)
        tb = _turns(_side(row["conversation_b"]), mb, "b", lic)
        meta = {
            "id": row["id"],
            "conversation_pair_id": row["conversation_pair_id"],
            "conv_a_id": row["conv_a_id"], "conv_b_id": row["conv_b_id"],
            "session_hash": row["session_hash"], "visitor_id": row["visitor_id"],
            "timestamp": row["timestamp"], "model_a": ma, "model_b": mb,
            "conv_turns": row["conv_turns"], "mode": row["mode"],
            "custom_models_selection": row["custom_models_selection"],
            "languages": row["languages"], "categories": row["categories"],
            "keywords": row["keywords"],
        }
        yield from _emit_pair(ctx, _unique(seen, row["conversation_pair_id"]),
                              ta, tb, meta, _first(row["languages"]))


def _vote_labels(row):
    labels = {}
    for s in ("a", "b"):
        labels[s] = {
            "flags": [f for f in FLAGS if row.get("conv_{}_{}".format(f, s))],
            "comment": row.get("conv_comments_" + s) or None,
        }
    return labels


@register("ministere-culture/comparia-votes")
def comparia_votes(ctx):
    lic = _model_info(ctx)
    seen = {}
    for _, row in iter_parquet(ctx.files("votes.parquet")):
        ma, mb = row["model_a_name"], row["model_b_name"]
        ta = _turns(_side(row["conversation_a"]), ma, "a", lic)
        tb = _turns(_side(row["conversation_b"]), mb, "b", lic)
        chosen, equal = row["chosen_model_name"], row["both_equal"]
        raw = {"chosen_model_name": chosen, "both_equal": equal}
        meta = {
            "id": row["id"],
            "conversation_pair_id": row["conversation_pair_id"],
            "session_hash": row["session_hash"], "visitor_id": row["visitor_id"],
            "timestamp": row["timestamp"], "model_a": ma, "model_b": mb,
            "conv_turns": row["conv_turns"],
        }
        if equal:
            winner = "tie"
        elif chosen is not None and chosen == ma and chosen != mb:
            winner = "a"
        elif chosen is not None and chosen == mb and chosen != ma:
            winner = "b"
        else:
            winner = None
            meta["vote_error"] = "chosen_model_not_in_pair"
            meta["vote_raw"] = raw
        vote = None if winner is None else {
            "winner": winner, "raw": raw, "labels": _vote_labels(row)}
        yield from _emit_pair(ctx, _unique(seen, row["conversation_pair_id"]),
                              ta, tb, meta, None, vote)


def _locate(raw_conv, msgs, idx, content):
    """Index of the reacted assistant message -> (index, content_matches)."""
    want = (content or "").strip()

    def ok(i):
        return (0 <= i < len(msgs) and msgs[i]["role"] == "assistant"
                and (raw_conv[i]["content"] or "").strip() == want)
    if idx is not None:
        for i in (idx, idx + 1):  # msg_index sometimes skips a system message
            if ok(i):
                return i, True
    hits = [i for i in range(len(msgs)) if ok(i)]
    if hits:
        return min(hits, key=lambda i: abs(i - (idx or 0))), True
    if idx is not None and 0 <= idx < len(msgs) and msgs[idx]["role"] == "assistant":
        return idx, False
    return None, False


@register("ministere-culture/comparia-reactions")
def comparia_reactions(ctx):
    lic = _model_info(ctx)
    for _, row in iter_parquet(ctx.files("reactions.parquet")):
        side = (row["model_pos"] or "").lower()
        if side not in ("a", "b"):
            ctx.skip("unknown_model_pos")
            continue
        raw_conv = row["conversation_" + side] or []
        msgs, info = _side(raw_conv)
        j, matches = _locate(raw_conv, msgs, row["msg_index"], row["response_content"])
        if j is None:
            ctx.skip("reacted_message_not_found")
            continue
        users = [i for i in range(j) if msgs[i]["role"] == "user"]
        if not users or not msgs[users[-1]]["content"].strip():
            ctx.skip("no_user_turn")
            continue
        model = row["model_{}_name".format(side)]
        extra = dict(info[j])
        if not matches:
            extra["response_content_mismatch"] = True
        resp = _reply(msgs[j]["content"], model, side, lic, **extra)
        flags = [f for f in FLAGS if row.get(f)]
        raw = {k: row.get(k) for k in ("liked", "disliked", "comment") + FLAGS}
        if row["liked"] and not row["disliked"]:
            pref = {"type": "scores", "scores": [1]}
        elif row["disliked"] and not row["liked"]:
            pref = {"type": "scores", "scores": [0]}
        else:
            pref = None
        if pref is not None:
            pref.update({"scale": "thumbs: 1=up (liked), 0=down (disliked)",
                         "flags": flags, "comment": row["comment"] or None,
                         "raw": raw})
        yield ctx.record(
            subset="default", split="train", source_id=str(row["id"]),
            context=msgs[:users[-1] + 1], responses=[_resp(resp)],
            preference=pref, language=None,
            meta={
                "id": row["id"], "turn": len(users) - 1, "side": side,
                "conversation_pair_id": row["conversation_pair_id"],
                "conv_a_id": row["conv_a_id"], "conv_b_id": row["conv_b_id"],
                "refers_to_conv_id": row["refers_to_conv_id"],
                "question_id": row["question_id"],
                "msg_index": row["msg_index"], "msg_rank": row["msg_rank"],
                "session_hash": row["session_hash"], "visitor_id": row["visitor_id"],
                "timestamp": row["timestamp"],
                "model_a": row["model_a_name"], "model_b": row["model_b_name"],
                "conv_turns": row["conv_turns"],
                "current_conv_turn_when_reacting": row["current_conv_turn_when_reacting"],
                "vote": None if pref else raw,
            })


CHOICE = {"a_better": "a", "b_better": "b", "both_good": "tie_good",
          "both_bad": "tie_bad"}


@register("ministere-culture/comparia-fr-arena", "danish-foundation-models/ai-arenaen")
def comparia_turns(ctx):
    """One row per turn; see module docstring."""
    lic = _model_info(ctx)
    name = os.path.basename(ctx.raw_dir).split("__", 1)[1]
    for _, row in iter_parquet(ctx.files(name + ".parquet"), batch_size=512):
        k = row["turn"]
        md = row["metadata"] or {}
        sides = {}
        no_history = []
        for s in ("a", "b"):
            model = row["model_" + s]
            msgs, _ = _side(row["full_conversation_" + s])
            users = [i for i, m in enumerate(msgs) if m["role"] == "user"]
            turn_msgs, turn_info = _side(row["response_" + s])
            if k >= len(users):
                # full conversation shorter than the turn (not seen so far):
                # fall back to the per-turn exchange without history
                no_history.append(s)
                context = [m for m in turn_msgs if m["role"] == "user"][:1]
            else:
                context = msgs[:users[k] + 1]
            reply = None
            for m, inf in zip(turn_msgs, turn_info):
                if m["role"] == "assistant":
                    reply = _reply(m["content"], model, s, lic,
                                   tokens=md.get("tokens_" + s), **inf)
                    break
            sides[s] = (context, reply) if context else None
        choice = row["choice"]
        vote = None
        if choice in CHOICE:
            vote = {"winner": CHOICE[choice], "raw": choice}
        meta = {
            "response_id": row["response_id"],
            "comparison_id": row["comparison_id"],
            "timestamp": row["timestamp"],
            "model_a": row["model_a"], "model_b": row["model_b"],
            "choice": choice, "time_to_vote": md.get("time_to_vote"),
            "mode": md.get("mode"),
            "custom_models_selection": md.get("custom_models_selection"),
            "languages": md.get("languages"), "categories": md.get("categories"),
            "participation_terms_version": md.get("participation_terms_version"),
        }
        if no_history:
            meta["history_missing"] = no_history
        if not sides["a"] and not sides["b"]:
            ctx.skip("no_user_turn")
            continue
        yield from _emit_turn(ctx, row["comparison_id"], k, sides["a"], sides["b"],
                              meta, _first(md.get("languages")), vote)


# ------------------------------------------------------------- LMArena

def _parts_text(content):
    if isinstance(content, str) or content is None:
        return content
    texts = [p.get("text") or "" for p in content if p.get("type") == "text"]
    return "\n\n".join(t for t in texts if t)


LM_WINNER = {"model_a": "a", "model_b": "b", "tie": "tie", "both_bad": "tie_bad",
             "tie (bothbad)": "tie_bad"}


@register("lmarena-ai/arena-human-preference-140k")
def lmarena_140k(ctx):
    def text(m):
        return _parts_text(m.get("content"))

    for _, row in iter_parquet(ctx.files("data/train-*.parquet"), batch_size=512):
        conv = {"a": row["conversation_a"] or [], "b": row["conversation_b"] or []}
        n_cur = sum(1 for m in conv["a"] if m["role"] == "user")
        full = row["full_conversation"] or []
        prior = full[:max(len(full) - n_cur, 0)]
        turns = {}
        for s in ("a", "b"):
            hist = []
            for e in prior:
                if e.get("user"):
                    hist.append(e["user"])
                if e.get("model_side_" + s):
                    hist.append(e["model_side_" + s])
            n_prior = sum(1 for m in hist if m["role"] == "user")
            model = row["model_" + s]
            turns[s] = _turns(_side(hist + conv[s], text), model, s, {})
        w = LM_WINNER.get(row["winner"])
        vote = None if w is None else {"winner": w, "raw": row["winner"]}
        meta = {
            "id": row["id"],
            "evaluation_session_id": row["evaluation_session_id"],
            "evaluation_order": row["evaluation_order"],
            "timestamp": row["timestamp"],
            "model_a": row["model_a"], "model_b": row["model_b"],
            "n_prior_turns": n_prior, "is_code": row["is_code"],
            "category_tag": row["category_tag"],
        }
        yield from _emit_pair(ctx, row["id"], turns["a"], turns["b"], meta,
                              row["language"], vote, start=n_prior)


@register("lmarena-ai/arena-human-preference-100k")
def lmarena_100k(ctx):
    for _, row in iter_parquet(ctx.files("data/arena-explorer-preference-100k.parquet"),
                               batch_size=1024):
        ta = _turns(_side(row["conversation_a"]), row["model_a"], "a", {})
        tb = _turns(_side(row["conversation_b"]), row["model_b"], "b", {})
        w = LM_WINNER.get(row["winner"])
        vote = None if w is None else {"winner": w, "raw": row["winner"]}
        meta = {
            "question_id": row["question_id"], "judge_hash": row["judge_hash"],
            "tstamp": row["tstamp"], "turns": row["turn"], "anony": row["anony"],
            "model_a": row["model_a"], "model_b": row["model_b"],
            "is_code": row["is_code"], "is_refusal": row["is_refusal"],
            "dedup_tag": row["dedup_tag"], "category_tag": row["category_tag"],
        }
        yield from _emit_pair(ctx, row["question_id"], ta, tb, meta,
                              row["language"], vote)
