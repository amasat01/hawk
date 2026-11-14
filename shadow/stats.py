# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""stats.py <hawk corpus dir> <raptor corpus dir> <out CORPUS.md> <out slots txt>: statistics of the corpus, written by the script."""
import collections
import hashlib
import json
import os
import re
import statistics
import sys


def tag(v):
    if v is None:
        return "null"
    if v is True or v is False:
        return "bool"
    t = next(k for k in ("n", "i", "f", "s", "t", "l", "fs", "d", "r", "u") if k in v)
    return "r:" + v["r"] if t == "r" else t


def walks(d):
    for line in open(os.path.join(d, "walks.jsonl")):
        yield json.loads(line)


def dup_entries(nodes):
    h = []
    for nd in nodes:
        def sub(v):
            if isinstance(v, dict):
                if "n" in v and len(v) == 1:
                    return h[v["n"]]
                return {k: sub(x) for k, x in v.items()}
            if isinstance(v, list):
                return [sub(x) for x in v]
            return v
        h.append(hashlib.sha1(json.dumps([nd["c"], nd["m"], sub(nd["v"])], sort_keys=True).encode()).hexdigest())
    return len(h) - len(set(h))


def main(hd, rd, out, slots_out):
    s = json.load(open(os.path.join(hd, "SUMMARY.json")))
    sr = json.load(open(os.path.join(rd, "SUMMARY.json")))
    cls = collections.Counter()
    tags = collections.defaultdict(set)
    sizes = []
    dupw = 0
    for u in walks(hd):
        nodes = u["inputs"]["nodes"]
        sizes.append((len(nodes), u["tests"][0], u["key"][:12]))
        dupw += dup_entries(nodes) > 0
        for nd in nodes:
            cls[(nd["c"], "core" if nd["m"] == "raptor.ir.nodes" else nd["m"] + (" (dialect)" if "k" in nd else ""))] += 1
            for sl, v in nd["v"].items():
                tags[(nd["c"], sl)].add(tag(v))
    ns = sorted(x[0] for x in sizes)
    q = lambda p: ns[min(len(ns) - 1, int(p * len(ns)))]
    corecount = sum(c for (n, m), c in cls.items() if m == "core")
    dial = collections.Counter()
    for (n, m), c in cls.items():
        if m != "core":
            dial[m] += c
    alltags = collections.Counter(t for v in tags.values() for t in v)
    with open(slots_out, "w") as fh:
        for (c, sl), v in sorted(tags.items()):
            fh.write(f"{c}.{sl}: {' '.join(sorted(v))}\n")
    ssz = lambda p: f"{os.path.getsize(p) / 1e6:.1f} MB"
    L = ["# Walk corpus (written by stats.py)", "",
         f"- hawk full run: N={s['calls_N']} calls, U={s['unique_U']} unique walks, raised={s['raised_calls']} calls ({s['raised_unique']} unique), "
         f"unmirrorable={s['unmirrorable_calls']}; equal inputs gave equal results: {s['equal_inputs_equal_results']} ({s['findings_count']} findings)",
         f"- raptor suite: N={sr['calls_N']}, U={sr['unique_U']}, raised={sr['raised_calls']}, unmirrorable={sr['unmirrorable_calls']}",
         f"- unique walks where at least two distinct entries are structurally equal (same class, equal slot values, children compared deep): {dupw} of {s['unique_U']}",
         f"- walk sizes (node entries): median {statistics.median(ns)}, p90 {q(0.9)}, max {ns[-1]}",
         f"- node entries over unique walks: core {corecount}; dialect classes (made from declarations): {sum(dial.values())}", "",
         "## Node classes (entries over all unique walks)", "| class | origin | entries |", "|---|---|---|"]
    for (n, m), c in sorted(cls.items(), key=lambda x: -x[1])[:12]:
        L.append(f"| {n} | {m} | {c} |")
    L.append(f"({len(cls)} classes in all: {sum(1 for k in cls if k[1] == 'core')} core, {sum(1 for k in cls if k[1] != 'core')} dialect; per-class counts in the slot table)")
    L += ["", "## Value tags per slot", f"- the full table, one line per (class, slot): {slots_out}",
          "- tags: " + ", ".join(f"{t} ({c} slots)" for t, c in sorted(alltags.items())), "",
          "## 20 largest unique walks", "| nodes | key | first test |", "|---|---|---|"]
    for n, t, k in sorted(sizes, reverse=True)[:20]:
        L.append(f"| {n} | {k} | {t[:90]} |")
    open(out, "w").write("\n".join(L) + "\n")
    for d in (hd, rd):
        print(d, [(f, ssz(os.path.join(d, f))) for f in ("walks.jsonl",)], sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.startswith("w") and f.endswith(".jsonl") and f != "walks.jsonl") / 1e6, "MB raw")


main(*sys.argv[1:5])
