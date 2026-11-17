"""WildChat-1M and WildChat-4.8M (real ChatGPT users, ODC-BY).

One record per conversation: context = conversation up to the last user
turn, responses = [the logged final assistant reply]. The 1M release is
(nearly) a subset of 4.8M; cross-source dedup happens later.
"""
from common import iter_parquet, msg, register, response, split_last_turn


def _flagged(moderation):
    out = set()
    for m in moderation or []:
        for cat, v in ((m or {}).get("categories") or {}).items():
            if v:
                out.add(cat.replace("/", "_").replace("-", "_"))
    return sorted(out)


@register("allenai/WildChat-1M", "allenai/WildChat-4.8M")
def wildchat(ctx):
    seen = {}
    for path, row in iter_parquet(ctx.files("data/*.parquet")):
        conv = row["conversation"] or []
        messages = [msg(t["role"], t["content"]) for t in conv]
        context, reply = split_last_turn(messages)
        if context is None:
            ctx.skip("no_user_turn")
            continue
        user0 = next(t for t in conv if t["role"] == "user")
        # conversation_hash is not guaranteed unique; disambiguate repeats
        h = row["conversation_hash"]
        seen[h] = seen.get(h, 0) + 1
        sid = h if seen[h] == 1 else "{}#{}".format(h, seen[h])
        yield ctx.record(
            subset="default", split="train", source_id=sid,
            context=context,
            responses=[response(reply, model=row.get("model"))]
            if reply is not None else [],
            language=row.get("language"),
            meta={
                "model": row.get("model"),
                "timestamp": row.get("timestamp"),
                "turn": row.get("turn"),
                "toxic": row.get("toxic"),
                "redacted": row.get("redacted"),
                "country": user0.get("country"),
                "state": user0.get("state"),
                "hashed_ip": user0.get("hashed_ip"),
                "openai_flagged": _flagged(row.get("openai_moderation")),
            })
