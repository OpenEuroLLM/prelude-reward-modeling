"""Shared schema, readers and writer for the per-source extractors.

Unified record (one row = one prompt context):

  uid           str   "<source>::<source_id>", unique within the collection
  source        str   HF dataset id
  subset        str   config / file group inside the source
  split         str   train | validation | test  (as defined by the source;
                      validation/test rows are kept for hold-out, never
                      merged into the training pool)
  category      str   from datasets.txt (real_user, safety, ...)
  license       str   license from the download manifest
  source_id     str   id of the row in the source (unique within source)
  context       list<{role, content}>  conversation ending with a USER turn
                      (system/user/assistant); earlier assistant turns are
                      the source's own responses
  responses     list<{model, content, meta}>  existing assistant replies to
                      the last user turn (0..n); meta is a JSON string
  preference    str|null  JSON describing labels over `responses`, e.g.
                      {"type": "pairwise", "winner": "a"|"b"|"tie"|"both_bad"}
                      {"type": "chosen_rejected", "chosen": 0, "rejected": 1}
                      {"type": "ranking", "ranks": [0, 2, 1]}
                      {"type": "scores", ...}
  language      str|null  language label provided by the source (unverified)
  n_user_turns  int   number of user turns in `context`
  meta          str   JSON with source-specific fields (toxicity flags,
                      timestamps, user/session hashes, ...)
"""
import glob
import gzip
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

MESSAGE = pa.struct([("role", pa.string()), ("content", pa.string())])
RESPONSE = pa.struct([("model", pa.string()), ("content", pa.string()),
                      ("meta", pa.string())])
SCHEMA = pa.schema([
    ("uid", pa.string()),
    ("source", pa.string()),
    ("subset", pa.string()),
    ("split", pa.string()),
    ("category", pa.string()),
    ("license", pa.string()),
    ("source_id", pa.string()),
    ("context", pa.list_(MESSAGE)),
    ("responses", pa.list_(RESPONSE)),
    ("preference", pa.string()),
    ("language", pa.string()),
    ("n_user_turns", pa.int32()),
    ("meta", pa.string()),
])

ROLE_MAP = {
    "user": "user", "human": "user", "prompter": "user", "USER": "user",
    "Human": "user", "question": "user",
    "assistant": "assistant", "gpt": "assistant", "bot": "assistant",
    "model": "assistant", "chatbot": "assistant", "ASSISTANT": "assistant",
    "Assistant": "assistant", "answer": "assistant",
    "system": "system", "System": "system", "developer": "system",
}


def jdump(obj):
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False, default=str)


def norm_role(role):
    r = ROLE_MAP.get(role)
    if r is None:
        r = ROLE_MAP.get(str(role).strip().lower())
    if r is None:
        raise ValueError("unknown role: {!r}".format(role))
    return r


def msg(role, content):
    return {"role": norm_role(role), "content": "" if content is None else str(content)}


def norm_messages(messages, role_key="role", content_key="content"):
    return [msg(m[role_key], m[content_key]) for m in messages]


def split_last_turn(messages):
    """Split a normalized conversation into (context, final_reply).

    context ends with the last user message; final_reply is the assistant
    message that follows it (or None if the conversation ends on the user).
    Returns (None, None) if there is no user message.
    """
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"),
                    default=None)
    if last_user is None:
        return None, None
    context = messages[:last_user + 1]
    rest = [m for m in messages[last_user + 1:] if m["role"] == "assistant"]
    return context, (rest[0]["content"] if rest else None)


def response(content, model=None, **meta):
    return {"model": model, "content": "" if content is None else str(content),
            "meta": jdump(meta) if meta else None}


VALID_SPLITS = {"train", "validation", "test"}

# Extractors register themselves here: REGISTRY[hf_id] = function(ctx)
REGISTRY = {}


def register(*hf_ids):
    def deco(fn):
        for i in hf_ids:
            if i in REGISTRY:
                raise ValueError("duplicate extractor for " + i)
            REGISTRY[i] = fn
        return fn
    return deco


class Context:
    """Passed to every extractor. Use ctx.record(...) to build rows and
    ctx.skip(reason) to count dropped source rows."""

    def __init__(self, source, category, license, raw_dir):
        self.source = source
        self.category = category
        self.license = license
        self.raw_dir = raw_dir
        self.skipped = {}

    def skip(self, reason):
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def files(self, pattern):
        return files(self.raw_dir, pattern)

    def record(self, subset, split, source_id, context, responses=(),
               preference=None, language=None, meta=None):
        if split not in VALID_SPLITS:
            raise ValueError("split must be one of {}, got {!r}".format(
                VALID_SPLITS, split))
        if not context or context[-1]["role"] != "user":
            raise ValueError("context must end with a user turn")
        return {
            "uid": "{}::{}".format(self.source, source_id),
            "source": self.source,
            "subset": subset,
            "split": split,
            "category": self.category,
            "license": self.license,
            "source_id": str(source_id),
            "context": context,
            "responses": list(responses),
            "preference": jdump(preference),
            "language": language,
            "n_user_turns": sum(1 for m in context if m["role"] == "user"),
            "meta": jdump(meta),
        }


# ----------------------------------------------------------------- readers

def files(raw_dir, pattern):
    out = sorted(p for p in glob.glob(os.path.join(raw_dir, pattern), recursive=True)
                 if os.path.basename(p) != "_download_info.json")
    if not out:
        raise FileNotFoundError("no files match {} in {}".format(pattern, raw_dir))
    return out


def iter_parquet(paths, columns=None, batch_size=2048):
    for p in paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                yield p, row


def iter_jsonl(paths):
    for p in paths:
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield p, json.loads(line)


def iter_json(paths):
    for p in paths:
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8") as f:
            data = json.load(f)
        for row in (data if isinstance(data, list) else [data]):
            yield p, row


# ------------------------------------------------------------------ writer

class ShardWriter:
    """Writes records to <out_dir>/part-#####.parquet in fixed-size shards."""

    def __init__(self, out_dir, rows_per_shard=250_000, batch=5_000):
        self.out_dir = out_dir
        self.rows_per_shard = rows_per_shard
        self.batch = batch
        self.buf = []
        self.shard = 0
        self.in_shard = 0
        self.total = 0
        self.writer = None
        os.makedirs(out_dir, exist_ok=True)
        for old in glob.glob(os.path.join(out_dir, "part-*.parquet")):
            os.remove(old)

    def _open(self):
        path = os.path.join(self.out_dir, "part-{:05d}.parquet".format(self.shard))
        self.writer = pq.ParquetWriter(path, SCHEMA, compression="zstd")

    def _flush(self):
        if not self.buf:
            return
        if self.writer is None:
            self._open()
        self.writer.write_table(pa.Table.from_pylist(self.buf, schema=SCHEMA))
        self.in_shard += len(self.buf)
        self.total += len(self.buf)
        self.buf = []
        if self.in_shard >= self.rows_per_shard:
            self.writer.close()
            self.writer = None
            self.shard += 1
            self.in_shard = 0

    def write(self, rec):
        self.buf.append(rec)
        if len(self.buf) >= self.batch:
            self._flush()

    def close(self):
        self._flush()
        if self.writer is not None:
            self.writer.close()
