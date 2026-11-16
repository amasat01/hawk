# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Compare two emit_record dirs. Exit 0 when the emitted text is the same.

usage: emit_compare.py <base_dir> <new_dir> [--kind src|blob|all] [--ignore <noise.json>] [--files f1,f2]
Verdict: the set of distinct emitted texts over the selected files, plus per-test outcomes. --ignore names a JSON list of
test files whose blob rows differ between two runs of the SAME tree (A/A noise).
--write-noise <path> writes the differing files there (use on an A/A pair).
"""
import collections
import functools
import glob
import hashlib
import re
import gzip
import json
import sys


# The unit digest covers the compiled image, whose bytes differ per link on macOS; it is not emitted text.
# The two include roots are absolute paths of the tree the run happened in (a product finding: sidecars are not
# relocatable); masked so a tree in another directory compares equal.
# A development version string (two wheels of different commits) is masked too.
_MASK = re.compile(r'("primal_unit": ")[0-9a-f]{64}|("(?:aether|eagle)_include": ")[^"]*'
                   r'|()\d+\.\d+\.\d+\.dev\d+\+g[0-9a-f]{6,}(?:\.d\d{8})?')


@functools.lru_cache(maxsize=None)
def _norm(d, sha):
    text = gzip.open(f"{d}/blobs/{sha}.gz", "rt").read()
    if "primal_unit" not in text and "_include" not in text and ".dev" not in text:
        return sha
    return hashlib.sha256(_MASK.sub(lambda m: next((g for g in m.groups() if g), "") + "*", text).encode()).hexdigest()


# Compiler commands carry the cache shard folder and a temp name with the process id: run noise, not flags.
_ARGV = re.compile(r"<CACHE>/[0-9a-f]{2}/|\.tmp\.\d+\.")


@functools.lru_cache(maxsize=None)
def _norm_argv(d, sha):
    text = gzip.open(f"{d}/blobs/{sha}.gz", "rt").read()
    return hashlib.sha256(_ARGV.sub(lambda m: "<CACHE>/" if m.group().startswith("<") else ".tmp.", text).encode()).hexdigest()


def load(d, kind):
    out = collections.defaultdict(collections.Counter)
    tags = {}
    for path in glob.glob(d + "/rec_*.json"):
        for f, rows in json.load(open(path)).items():
            for k, sha, tag in rows:
                if kind in ("all", k):
                    if k in ("blob", "src"):
                        sha, tag = _norm(d, sha), (sha if k == "blob" else tag)
                    if k == "argv":
                        sha, tag = _norm_argv(d, sha), f"{tag} [{sha[:12]}]"
                    out[f][(k, sha)] += 1
                    tags[sha] = tag
    return out, tags


def main(argv):
    base, new = argv[1], argv[2]
    opt = dict(zip(argv[3::2], argv[4::2]))
    kind = opt.get("--kind", "src")
    ignore = set(json.load(open(opt["--ignore"]))) if "--ignore" in opt else set()
    only = set(opt["--files"].split(",")) if "--files" in opt else None
    a, ta = load(base, kind)
    b, tb = load(new, kind)
    # Verdict: the SET of distinct emitted texts over the selected files. Which test file a text is first
    # emitted under (and how often) moves with xdist scheduling and in-process memos, so it is not compared.
    files = sorted(set(a) | set(b)) if only is None else sorted(only)
    sa = {(key, f) for f in files if f not in ignore for key in a.get(f, ())}
    sb = {(key, f) for f in files if f not in ignore for key in b.get(f, ())}
    da, db = {k for k, _ in sa}, {k for k, _ in sb}
    bad = []
    for label, only_keys, rows, tags, d in (("-", da - db, sa, ta, base), ("+", db - da, sb, tb, new)):
        for key in sorted(only_keys):
            f = next(f for k, f in rows if k == key)
            bad.append((f, key))
            print(f"DIFF {label} {f} {key[0]} {tags.get(key[1])} {d}/blobs/{tags.get(key[1]) if key[0] == 'blob' else key[1]}.gz")
    print(f"kind={kind} files={len(files)} distinct(base)={len(da)} distinct(new)={len(db)} ignored={len(ignore)} "
          f"only_base={len(da - db)} only_new={len(db - da)}")
    oa, ob = ({k: v for q in glob.glob(d + "/out_*.json") for k, v in json.load(open(q)).items()} for d in (base, new))
    if only is not None:
        oa, ob = ({k: v for k, v in o.items() if k.split("::")[0] in only} for o in (oa, ob))
    moved = sorted(k for k in set(oa) | set(ob) if oa.get(k) != ob.get(k))
    tally = collections.Counter(oa.values())
    print(f"outcomes(base)={dict(tally)} changed={len(moved)}")
    for k in moved:
        print(f"OUTCOME {k}: {oa.get(k)} -> {ob.get(k)}")
    bad += [(k,) for k in moved]
    # Failure REASONS (exception class of the first failing phase): a test red in both runs for different
    # reasons is a change. Present only in snapshots taken with the upgraded recorder.
    ra, rb = ({k: v for q in glob.glob(d + "/reasons_*.json") for k, v in json.load(open(q)).items()} for d in (base, new))
    if ra and rb:
        if only is not None:
            ra, rb = ({k: v for k, v in r.items() if k.split("::")[0] in only} for r in (ra, rb))
        differ = sorted(k for k in set(ra) & set(rb) if ra[k] != rb[k])
        print(f"reasons(base)={len(ra)} changed={len(differ)} classes(base)={dict(collections.Counter(ra.values()).most_common(6))}")
        for (x, y), n in collections.Counter((ra[k], rb[k]) for k in differ).most_common():
            print(f"REASON-CLASS {n} x {x} -> {y}")
        for k in differ[:40]:
            print(f"REASON {k}: {ra[k]} -> {rb[k]}")
        bad += [(k,) for k in differ]
    # DETAILS (first line of the message, masked, and the file:line that raised): present only in snapshots taken
    # with the second recorder upgrade. Reported, and part of the verdict.
    xa, xb = ({k: v for q in glob.glob(d + "/details_*.json") for k, v in json.load(open(q)).items()} for d in (base, new))
    if xa and xb:
        if only is not None:
            xa, xb = ({k: v for k, v in x.items() if k.split("::")[0] in only} for x in (xa, xb))
        both = sorted(set(xa) & set(xb))
        msg = [k for k in both if xa[k][0] != xb[k][0]]
        loc = [k for k in both if xa[k][0] == xb[k][0] and xa[k][1] != xb[k][1]]
        print(f"details(both red)={len(both)} message changed={len(msg)} same message, other line={len(loc)}")
        for (x, y), n in collections.Counter((xa[k][0][:110], xb[k][0][:110]) for k in msg).most_common(25):
            print(f"MESSAGE {n} x {x!r} -> {y!r}")
        for (x, y), n in collections.Counter((xa[k][1], xb[k][1]) for k in loc).most_common(15):
            print(f"LINE {n} x {x} -> {y}")
        bad += [(k,) for k in msg]
    # Attribution (which test file emits a text) is NOT part of the verdict; shown so a jump is visible
    # (two runs of the same tree differ by a handful of pairs).
    print(f"attribution: (text, file) pairs only in base={len(sa - sb)} only in new={len(sb - sa)}")
    if "--write-noise" in opt:
        json.dump(sorted({f for f, *_ in bad if f.startswith("tests/") and "::" not in f}), open(opt["--write-noise"], "w"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
