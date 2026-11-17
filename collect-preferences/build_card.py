#!/usr/bin/env python3
"""Build the Hugging Face dataset card, DATASET_CARD.md, from the GitHub README.

  python3 collect-preferences/build_card.py          # stdlib only; upload_hf.py runs it before uploading the card

The card is the front matter of collect-preferences/card_meta.yaml (Hub metadata: licence, languages, configs
with `preferences` as the default), then README.md with two changes: links into the repository
(`collect-prompts/`, `collect-preferences/judge_prompt.py`, ...) become GitHub URLs, since the Hub repo holds
only the data, the card and the figures; and lines marked `<!-- github-only -->` are dropped. Figure links
(figures/*.png) are left as they are: upload_hf.py puts the figures at the same paths in the Hub repo.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
README = os.path.join(ROOT, "README.md")
META = os.path.join(HERE, "card_meta.yaml")
CARD = os.path.join(ROOT, "DATASET_CARD.md")
GITHUB = "https://github.com/OpenEuroLLM/prelude-reward-modeling"
MARK = "<!-- github-only -->"
REPO_LINK = re.compile(r"\]\(((?:collect-prompts|collect-responses|collect-preferences)(?:/[^)\s]*)?)\)")


def github_url(path):
    path = path.rstrip("/")
    kind = "blob" if "." in os.path.basename(path) else "tree"
    return "{}/{}/main/{}".format(GITHUB, kind, path)


def build(readme=README, meta=META, out=CARD):
    with open(readme, encoding="utf-8") as f:
        lines = [l for l in f.read().split("\n") if MARK not in l]
    body = REPO_LINK.sub(lambda m: "]({})".format(github_url(m.group(1))), "\n".join(lines)).lstrip("\n")
    with open(meta, encoding="utf-8") as f:
        front = "\n".join(l for l in f.read().strip().split("\n") if not l.startswith("#"))
    with open(out, "w", encoding="utf-8") as f:
        f.write("---\n{}\n---\n\n{}".format(front, body))
    return out


if __name__ == "__main__":
    out = build()
    print("dataset card -> {} ({} lines)".format(out, sum(1 for _ in open(out, encoding="utf-8"))))
