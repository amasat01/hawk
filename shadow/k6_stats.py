# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Counts over a kernel corpus (written by k6_plugin): python k6_stats.py <corpus dir> -> k6_stats.json in it, a summary on stdout.
Joins, shapes, memo hits, plain-digest check, sizes."""
import glob, json, os, sys, collections

d = sys.argv[1]
calls, units_of, ntext, joined, unjoined, badunit = {}, {}, 0, 0, collections.Counter(), 0
for f in glob.glob(os.path.join(d, "k*.jsonl")):
    for line in open(f):
        c = json.loads(line)
        calls[c["call"]] = c
        for u in c["units"]:
            units_of.setdefault(c["call"], set()).add(u["unit"])
unj_tests = collections.Counter()
for f in glob.glob(os.path.join(d, "t*.jsonl")):
    for line in open(f):
        # only the keys we need: avoid decoding the multi-MB record fully when it has no call
        r = json.loads(line)
        ntext += 1
        cid = r.get("call")
        if cid in calls:
            joined += 1
            if r.get("kunit") not in units_of.get(cid, ()):
                badunit += 1
        else:
            unj_tests[r.get("test")] += 1
unjoined = sum(unj_tests.values())

def kind_flags(u):
    return {"segment": u.get("segment") is not None, "active": u.get("active") is not None,
            "box": u.get("box") is not None, "one_step": bool(u.get("one_step")), "team": bool(u.get("team")),
            "warp": u.get("cls") == "WarpStage", "fused": bool(u.get("fused_hook")) and u.get("cls") not in ("TeamStage", "WarpStage")}

shapes = collections.Counter()           # unique units (by call-independent key: unit name + walk digest + cls)
seen = set()
nunits = ncalls = memo = errs = 0
chain_kernels = fused_kernels = seg_kernels = 0
plain_check = [0, 0, []]
for c in calls.values():
    ncalls += 1
    memo += c["memo_hit"]
    errs += "error" in c
    for k in c["kernels"]:
        if k.get("members") and len([m for m in k["members"] if "walk_digest" in m]) >= 2: chain_kernels += 1
        if k.get("fused_body"): fused_kernels += 1
        if k.get("segmented"): seg_kernels += 1
    sb = c.get("split")
    for u in c["units"]:
        nunits += 1
        key = (u["unit"], u["digest"], u["cls"], json.dumps(u.get("box")), json.dumps(u.get("active")), json.dumps(u.get("segment")))
        if key in seen: continue
        seen.add(key)
        fl = kind_flags(u)
        placed = u.get("stage_block") is not None and u["cls"] == "Kernel"
        if u.get("stage_block") is not None: fl["staged"] = True
        for n, v in fl.items():
            if v: shapes[n] += 1
        if not any(fl.values()) and u.get("stage_block") is None and u["cls"] == "Kernel":
            shapes["plain"] += 1
            # plain: one input kernel of the same name and walk digest
            ks = [k for k in c["kernels"] if k["name"] == u["unit"]]
            if len(c["units"]) == len(c["kernels"]) and ks and not u.get("record_error"):
                plain_check[0] += 1
                if ks[0]["walk_digest"] == u["digest"]: plain_check[1] += 1
                else: plain_check[2].append((c["call"], u["unit"]))
out = {"calls": ncalls, "units": nunits, "unique_units": len(seen), "memo_hit_calls": memo, "calls_with_error": errs,
       "text_records": ntext, "joined": joined, "unjoined": unjoined, "joined_unit_missing": badunit,
       "unjoined_top_tests": unj_tests.most_common(8), "shapes_unique_units": dict(shapes),
       "kernel_inputs": {"chain": chain_kernels, "fused_body": fused_kernels, "segmented": seg_kernels},
       "plain_digest_check": {"checked": plain_check[0], "equal": plain_check[1], "differ": plain_check[2][:5]},
       "bytes": {"k": sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, "k*.jsonl"))),
                 "t": sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, "t*.jsonl")))}}
json.dump(out, open(os.path.join(d, "k6_stats.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
