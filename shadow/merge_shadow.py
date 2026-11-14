# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""merge_shadow.py <SHADOW_DIR>: sums the shadow_<pid>.json files of a run."""
import glob
import json
import os
import sys

tot = {}
mism, exc, per = [], [], []
files = sorted(glob.glob(os.path.join(sys.argv[1], "shadow_*.json")))
for f in files:
    d = json.load(open(f))
    for k, v in d.items():
        if isinstance(v, dict):
            t = tot.setdefault(k, {})
            for a, b in v.items():
                t[a] = t.get(a, 0) + b
        elif isinstance(v, (int, float)) and k != "max_nodes" and k != "seconds_at_max_nodes":
            tot[k] = tot.get(k, 0) + v
    mism += d["mismatches"]
    exc += d["exception_samples"]
n, eq = tot.get("calls", 0), tot.get("equal", 0)
sk = tot.get("skipped", {})
print(f"shadow-equal {eq} / {n}   (files {len(files)})")
print(f"roundtrip-equal {tot.get('roundtrip_equal', 0)} / {tot.get('roundtrip_tried', 0)} (skipped: declared refs {tot.get('roundtrip_skipped_declared', 0)})")
print(f"not in this build: {sum(sk.values())} {dict(sk)}")
print(f"oracle-raised {tot.get('oracle_raised', 0)}: door refused too {tot.get('door_refused_too', 0)} (message equal {tot.get('door_message_equal', 0)}), door accepted {tot.get('door_accepted_oracle_raised', 0)}")
print(f"mismatches {tot.get('mismatch_count', 0)}; make-refused {tot.get('make_refused', {})}; unmirrorable {tot.get('unmirrorable', 0)}; shadow exceptions {tot.get('shadow_exceptions', 0)}")
print(f"seconds: oracle {tot.get('seconds_oracle', 0):.2f}, shadow {tot.get('seconds_shadow', 0):.2f}")
for m in mism[:5]:
    print("MISMATCH", json.dumps(m)[:400])
for e in exc[:5]:
    print("EXC", json.dumps(e)[:400])

# which walks the run saw: the distinct digests of the real calls, against a corpus (optional second argument: walks.jsonl)
seen = set()
for f in glob.glob(os.path.join(sys.argv[1], "digests_*.txt")):
    seen.update(x.strip() for x in open(f) if x.strip())
print(f"distinct digests seen by the shadow run: {len(seen)}")
if len(sys.argv) > 2:
    corpus = set()
    with open(sys.argv[2]) as fh:
        for line in fh:
            r = json.loads(line).get("result")
            if r and r.get("digest"):
                corpus.add(r["digest"])
    print(f"distinct digests in the corpus: {len(corpus)}; in the corpus and not seen: {len(corpus - seen)}; seen and not in the corpus: {len(seen - corpus)}")
