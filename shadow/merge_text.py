# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""merge_text.py [--apply-exclusions] <TEXT_DIR> [<earlier TEXT_DIR>]: merges t<pid>.jsonl into texts.jsonl (unique inputs) and SUMMARY.json. Equal inputs must give equal outputs.
With an earlier directory it also prints which unit keys were added and removed per target (both files keyed by unit_key)."""
import collections
import glob
import hashlib
import json
import os
import re
import sys

from text_plugin import load_excluded


def canon(j):
    return json.dumps(j, sort_keys=True, separators=(",", ":"))


def tp_shape(args):
    return "(" + ",".join(next(iter(x)) if isinstance(x, dict) and x else str(type(x).__name__) for x in args) + ")"


def tp_tag(a):
    if "error" in a:
        return "error"
    r = a["result"]
    return next(iter(r)) if isinstance(r, dict) and r else str(type(r).__name__)


def nodes_in(a):
    out = []

    def walk(v):
        if isinstance(v, dict):
            if set(v) == {"n"}:
                out.append(v["n"])
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(a["args"])
    walk(a["kwargs"])
    walk(a.get("result"))
    return out


# Fields of a record that are the run's own bookkeeping and read by no writer: the digest of the primal unit (it hashes the cache path),
# the deploy bundle's source tuple (digest and cache path) and the address of the loaded entry.
_LEFT_OUT = ("primal_unit", "_hawk_source", "host_entry")


# a test's temporary directory (pytest's own, or the pinned base temp dir of a run) up to the worker folder
_TMP = re.compile(r"/[^\"]*?(?:pytest-\d+|basetemp)/popen-gw\d+")


# the run's own folder name (the snapshot name) in a path under snap/
_RUN = re.compile(r"/snap/[^/\"]+/")


def scrub(v):
    if isinstance(v, dict):
        return {k: scrub(x) for k, x in v.items() if k not in _LEFT_OUT}
    return [scrub(x) for x in v] if isinstance(v, list) else v


def unit_key(tg, inputs):
    """The unit's key: its inputs without the run's bookkeeping fields, and with the temporary directory of a test (run, worker) made equal."""
    return hashlib.sha256(_RUN.sub("/snap/<run>/", _TMP.sub("<tmp>", canon([tg, scrub(inputs)]))).encode()).hexdigest()


FACT_NAMES = {"aether.wings.base.rtstore:select_weight"}       # a patched writer function whose answers `facts` records: not a reason to exclude


def has_walk(v):
    """True when the encoded inputs hold a Walk record."""
    if isinstance(v, dict):
        return v.get("r") == "Walk" or any(has_walk(x) for x in v.values())
    return isinstance(v, list) and any(has_walk(x) for x in v)


def apply_exclusions(d):
    """Without the raw files: rewrite texts.jsonl so that every unit named in EXCLUDED_UNITS.tsv carries its `exclude` reason."""
    ex, n, tmp = load_excluded(d), 0, os.path.join(d, "texts.jsonl.new")
    with open(tmp, "w") as out:
        for line in open(os.path.join(d, "texts.jsonl")):
            u = json.loads(line)
            if u["key"] in ex and not u.get("exclude"):
                u["exclude"] = ex[u["key"]]
                n += 1
                line = json.dumps(u, sort_keys=True, separators=(",", ":")) + "\n"
            out.write(line)
    os.replace(tmp, os.path.join(d, "texts.jsonl"))
    print("exclusions applied", n)


def main(d, earlier=None):
    uniq, calls, findings = {}, {}, []
    n = raised = unm = 0
    per, seams = {}, collections.Counter()
    idx = {}
    wo = {"units_with_walks": 0, "exempt": 0, "declared": 0, "force": 0, "digests": set(), "two_sets": set()}
    inn = collections.Counter()
    patched = set()
    ck, callables = collections.Counter(), collections.Counter()
    excluded = load_excluded(d)
    for f in sorted(x for x in glob.glob(os.path.join(d, "t*.jsonl")) if re.fullmatch(r"t\d+\.jsonl", os.path.basename(x))):
        for line in open(f):
            rec = json.loads(line)
            n += 1
            tg = rec["target"]
            p = per.setdefault(tg, {"calls": 0, "unique": 0, "raised": 0, "unmirrorable": 0, "texts": set(), "lines": 0})
            p["calls"] += 1
            seams[rec["seam"]] += 1
            p["raised"] += "error" in rec
            raised += "error" in rec
            p["unmirrorable"] += bool(rec["unmirrorable"])
            unm += bool(rec["unmirrorable"])
            key = unit_key(tg, rec["inputs"])
            outcome = canon([rec.get("output"), rec.get("error")])
            u = uniq.get(key)
            if u is None:
                ex = None
                rest = [x for x in rec.get("patched", []) if x not in FACT_NAMES]
                if rest:
                    ex = "patched render: " + ", ".join(rest[:3])
                elif key in excluded:
                    ex = excluded[key]
                elif tg in ("fast", "host2", "optb", "toy"):
                    ex = "test-defined backend: not in the registry, not replayable"
                elif "demo_aether_spellings" in line:
                    ex = "spelling module created by the test at run time"
                u = uniq[key] = {"exclude": ex, "analyses": rec.get("analyses", []), "analysis_nested": rec.get("analysis_nested", {}), "node_sites": rec.get("node_sites", {}), "analysis_extra": rec.get("analysis_extra"),"key": key, "target": tg, "unit": rec["unit"], "seam": rec["seam"], "calls": 1, "tests": [rec["test"]], "inputs": rec["inputs"],
                             "output": rec.get("output"), "error": rec.get("error"), "texts": rec.get("texts", []), "unknown": rec["unknown"],
                             "unmirrorable": rec["unmirrorable"], "closures": rec.get("closures", []), "walk_opts": rec.get("walk_opts", {}), "walk_opts_used": rec.get("walk_opts_used", {}), "walk_props": rec.get("walk_props", {}), "outside_calls": rec.get("outside_calls", 0), "placement": rec.get("placement"), "patched": rec.get("patched"), "facts": rec.get("facts"),
                             "kept": "patched only in a machine fact whose answers are recorded in `facts`" if rec.get("patched") and not rest else None, "inner": rec.get("inner"), "_o": outcome}
                p["unique"] += 1
                if ex:
                    p["excluded"] = p.get("excluded", 0) + 1
                for dg, sets in rec.get("walk_opts", {}).items():
                    wo["digests"].add(dg)
                    if len(sets) > 1:
                        wo["two_sets"].add(dg)
                    for s_ in sets:
                        o = s_["opts"]
                        if o != "default":
                            wo["exempt"] += bool(o["exempt"].get("l"))
                            wo["declared"] += bool(o["declared"].get("l"))
                            wo["force"] += o["force"] is not None
                if rec.get("walk_opts"):
                    wo["units_with_walks"] += 1
                inn[(tg, "none" if rec.get("inner") is None else "inputs" if rec["inner"] == "inputs" else "recorded")] += 1
                if isinstance(rec.get("inner"), dict) and "texts" in rec["inner"]:
                    a_ = {x[0]: x[1] for x in rec["inner"]["texts"]}
                    b_ = {x[0]: x[1] for x in rec.get("texts", [])}
                    inn[(tg, "outer text equals inner text" if a_.get("text") == b_.get("text") else "outer text differs from inner")] += 1
                ck["units_with_walk"] += has_walk(rec["inputs"])
                ck["units_with_walk_props"] += bool(rec.get("walk_props"))
                ck["units_with_walk_but_no_walk_props"] += has_walk(rec["inputs"]) and not rec.get("walk_props")
                ck["walks_opts_unknown_of_two_sets"] += sum(1 for dg, o in rec.get("walk_opts_used", {}).items() if o is None and len(rec.get("walk_opts", {}).get(dg, [])) > 1)
                ck["walks_with_two_sets"] += sum(1 for dg in rec.get("walk_opts_used", {}) if len(rec.get("walk_opts", {}).get(dg, [])) > 1)
                ck["outside_calls_attached"] += rec.get("outside_calls", 0)
                ck["units_with_placement"] += bool(rec.get("placement"))
                ck["units_patched"] += bool(rec.get("patched"))
                ck["units_with_facts"] += bool(rec.get("facts"))
                ck["units_patched_kept"] += bool(rec.get("patched")) and not rest
                for a in rec.get("analyses", []):
                    ck["analyses_outside"] += bool(a.get("outside"))
                    ck["generator_answers"] += bool(a.get("gen"))
                    ck["generator_answers_not_a_list"] += bool(a.get("gen")) and not (isinstance(a.get("result"), dict) and "l" in a["result"])
                    ck["answers_unrecorded_object"] += tp_tag(a) == "u"
                    ck["closure_calls_recorded"] += bool(a.get("closure"))
                    if tp_tag(a) in ("closure", "bm", "fn") and not a.get("closure"):
                        callables[a["fn"] + " -> " + tp_tag(a)] += a["n"]
                for a in rec.get("analyses", []):
                    i = idx.setdefault(a["fn"], {"calls": 0, "units": 0, "targets": collections.Counter(), "arg_shapes": set(), "result_tags": collections.Counter(), "nodes_made_in_render": 0, "callers": collections.Counter(), "varies": 0})
                    i["calls"] += a["n"]
                    i["units"] += 1
                    i["targets"][tg] += 1
                    i["arg_shapes"].add(tp_shape(a["args"]))
                    i["result_tags"][tp_tag(a)] += 1
                    i["callers"][a["caller"]] += a["n"]
                    i["varies"] += a["varies"]
                    n0 = (rec.get("analysis_extra") or {}).get("offsets", [10**9])[0]
                    i["nodes_made_in_render"] += int(any(x >= n0 for x in nodes_in(a)))
                for fld, sha, lines in rec.get("texts", []):
                    if fld == "text":
                        p["texts"].add(sha)
                        p["lines"] += lines
            else:
                u["calls"] += 1
                if [x for x in rec.get("patched", []) if x not in FACT_NAMES] and not u["exclude"]:
                    u["exclude"] = "patched render: " + ", ".join([x for x in rec["patched"] if x not in FACT_NAMES][:3])
                if rec["test"] not in u["tests"] and len(u["tests"]) < 3:
                    u["tests"].append(rec["test"])
                if u["_o"] != outcome:
                    patched.add(key)
                    findings.append({"key": key, "unit": rec["unit"], "target": tg, "test": rec["test"], "first_test": u["tests"][0],
                                     "texts_now": rec.get("texts"), "texts_first": u["texts"]})
    with open(os.path.join(d, "texts.jsonl"), "w") as fh:
        for k, u in uniq.items():
            if k in patched and not u["exclude"]:
                u["exclude"] = "patched render: equal inputs gave different texts (tests/test_finish_own_sample.py)"
                per[u["target"]]["excluded"] = per[u["target"]].get("excluded", 0) + 1
            u.pop("_o")
            fh.write(json.dumps(u, sort_keys=True, separators=(",", ":")) + "\n")
    summ = {"checks": dict(ck), "callable_answers": dict(callables), "walk_opts": {**{k: (len(v) if isinstance(v, set) else v) for k, v in wo.items()}, "two_sets_digests": sorted(wo["two_sets"])[:10]}, "inner": {f"{a}|{b}": n for (a, b), n in inn.items()},
            "calls": n, "unique_units": len(uniq), "raised_calls": raised, "unmirrorable_calls": unm, "calls_per_seam": dict(seams),
            "per_target": {t: {**{k: v for k, v in p.items() if k != "texts"}, "distinct_texts": len(p["texts"])} for t, p in per.items()},
            "equal_inputs_equal_outputs": not findings, "findings_count": len(findings), "findings": findings[:20]}
    json.dump({f: {**{k: (dict(v) if isinstance(v, collections.Counter) else sorted(v) if isinstance(v, set) else v) for k, v in i.items()}} for f, i in idx.items()}, open(os.path.join(d, "analyses_index.json"), "w"), indent=1)
    json.dump(summ, open(os.path.join(d, "SUMMARY.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "findings"}))
    if earlier:
        old = {}
        for line in open(os.path.join(earlier, "texts.jsonl")):
            x = json.loads(line)
            old[unit_key(x["target"], x["inputs"])] = x["target"]
        new = {k: u["target"] for k, u in uniq.items()}
        for tg in sorted(set(old.values()) | set(new.values())):
            o, n_ = {k for k, t in old.items() if t == tg}, {k for k, t in new.items() if t == tg}
            print(f"unit keys [{tg}]: earlier {len(o)}, now {len(n_)}, added {len(n_ - o)}, removed {len(o - n_)}")


if sys.argv[1] == "--apply-exclusions":
    apply_exclusions(sys.argv[2])
else:
    main(*sys.argv[1:3])
