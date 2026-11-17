#!/usr/bin/env python3
"""Per-language item quotas proportional to sqrt(HPLT 3.0 frequency).

  python3 compile/quotas.py [--total 128000] [--measure documents]

Unit: quotas count prompts/conversations; HPLT frequency is measured in
documents (the closest HPLT analogue of a conversation).

Stdlib only. Writes compile/quotas.tsv (code, hplt, frequency, share, quota).
Rounding: largest remainder, so quotas sum exactly to --total.
"""
import argparse
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip() and not l.startswith("#")]
    header = lines[0].split("\t")
    return [dict(zip(header, l.split("\t"))) for l in lines[1:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=128_000)
    ap.add_argument("--measure", default="documents",
                    choices=["bytes", "documents", "segments", "tokens_gemma3",
                             "characters"])
    ap.add_argument("--out", default=os.path.join(HERE, "quotas.tsv"))
    args = ap.parse_args()

    stats = {r["hplt_code"]: r for r in read_tsv(os.path.join(HERE, "hplt3_stats.tsv"))}
    langs = read_tsv(os.path.join(HERE, "languages.tsv"))
    freq = {l["code"]: float(stats[l["hplt"]][args.measure]) for l in langs}
    w = {c: math.sqrt(v) for c, v in freq.items()}
    z = sum(w.values())
    exact = {c: args.total * w[c] / z for c in w}
    quota = {c: int(math.floor(x)) for c, x in exact.items()}
    left = args.total - sum(quota.values())
    for c in sorted(exact, key=lambda c: exact[c] - quota[c], reverse=True)[:left]:
        quota[c] += 1

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# quotas for total={} measure={} (sqrt of HPLT 3.0 frequency)\n"
                .format(args.total, args.measure))
        f.write("code\thplt\tname\tfrequency\tshare\tquota\n")
        for l in sorted(langs, key=lambda l: -quota[l["code"]]):
            c = l["code"]
            f.write("{}\t{}\t{}\t{:.6g}\t{:.5f}\t{}\n".format(
                c, l["hplt"], l["name"], freq[c], quota[c] / args.total, quota[c]))
    print(open(args.out, encoding="utf-8").read())


if __name__ == "__main__":
    main()
