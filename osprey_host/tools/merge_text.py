# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""merge_text.py [--apply-exclusions] <TEXT_DIR> [<earlier TEXT_DIR>]: merges t<pid>.jsonl into texts.jsonl (unique inputs) and SUMMARY.json. Equal inputs must give equal outputs.
With an earlier directory it also prints which unit keys were added and removed per target (both files keyed by unit_key)."""
import codecs
import collections
import glob
import hashlib
import json
import os
import re
import sys
import tempfile

from text_plugin import load_excluded


_DECODER = json.JSONDecoder()
_OPENS_WITH = '{"analyses":['
_PIECE = 1 << 24


class Line:
    """One line of a recorded file: its bytes when it is short, its place in the file when it is long. A recorded unit is a JSON object whose
    `analyses` array is the large part of it (the keys are sorted, but a key written before it, such as `_hawk_source`, may come first): the
    line is read field by field, from pieces, and the analyses one element at a time, so that a line of any size is never parsed whole."""

    def __init__(self, data=None, path=None, start=0, size=0):
        self.data, self.path, self.start, self.size, self.has_analyses = data, path, start, size, False

    def pieces(self):
        """The bytes of the line, one piece at a time."""
        if self.data is not None:
            yield self.data
            return
        with open(self.path, "rb") as f:
            f.seek(self.start)
            left = self.size
            while left:
                piece = f.read(min(_PIECE, left))
                left -= len(piece)
                yield piece

    def __contains__(self, needle):
        keep, find = b"", needle.encode()
        for piece in self.pieces():
            if find in keep + piece:
                return True
            keep = piece[1 - len(find) :]
        return False

    def write(self, out):
        """Writes the line to the text file `out`."""
        decode = codecs.getincrementaldecoder("utf-8")().decode
        for piece in self.pieces():
            out.write(decode(piece))

    def fields(self):
        """Walks the top-level object: yields (None, an analysis) for each element of `analyses` and (key, value) for every other field."""
        texts = map(codecs.getincrementaldecoder("utf-8")().decode, self.pieces())
        buf, pos = next(texts, ""), 0

        def more():
            """Appends pieces to the unread text, at least as much as it holds (a value cut by the end of the text is read again from a text twice as long)."""
            nonlocal buf, pos
            got = []
            while sum(map(len, got)) < max(1, len(buf) - pos):
                piece = next(texts, None)
                if piece is None:
                    break
                got.append(piece)
            if not got:
                raise ValueError("a recorded line ends inside a value")
            buf, pos = buf[pos:] + "".join(got), 0

        def peek():
            """The next character that is not white space."""
            nonlocal pos
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n":
                    pos += 1
                if pos < len(buf):
                    return buf[pos]
                more()

        def value():
            """The JSON value at the position, which moves past it."""
            nonlocal pos
            peek()
            while True:
                try:
                    item, end = _DECODER.raw_decode(buf, pos)
                    if end < len(buf):
                        pos = end
                        return item
                except json.JSONDecodeError:
                    pass
                more()

        if peek() != "{":
            raise ValueError("a recorded line is not an object")
        pos += 1
        while peek() != "}":
            key = value()
            if peek() != ":":
                raise ValueError("a recorded line has a key without a value")
            pos += 1
            if key == "analyses" and peek() == "[":
                self.has_analyses = True
                pos += 1
                while peek() != "]":
                    yield None, value()
                    if peek() == ",":
                        pos += 1
                pos += 1
            else:
                yield key, value()
            if peek() == ",":
                pos += 1

    def analyses(self):
        """The analyses, one value at a time."""
        return (a for key, a in self.fields() if key is None)

    def write_analyses(self, out):
        """Writes `{"analyses":[...` up to the end of the array, in the form the merged files hold them."""
        out.write(_OPENS_WITH)
        for n, a in enumerate(self.analyses()):
            out.write(("," if n else "") + canon(a))
        out.write("]")


def split_record(line):
    """(the record without its analyses, a function that reads its analyses one by one, a function that writes them to a file or None when
    the record has none)."""
    head = {key: v for key, v in line.fields() if key is not None}
    return head, line.analyses, line.write_analyses if line.has_analyses else None


def read_lines(path):
    """The lines of a file as `Line`s (a line longer than a piece is measured and left in the file)."""
    with open(path, "rb") as f:
        while head := f.readline(_PIECE):
            if head.endswith(b"\n") or len(head) < _PIECE:
                yield Line(head)
                continue
            start, size = f.tell() - len(head), len(head)
            while True:
                block = f.read(_PIECE)
                end = block.find(b"\n")
                size += len(block) if end < 0 else end + 1
                if end >= 0 or not block:
                    break
            f.seek(start + size)
            yield Line(path=path, start=start, size=size)


def copy_spooled(spool, start, size, out):
    """Writes `size` bytes of the spool file, from `start`, to the text file `out` (the spooled text is ASCII)."""
    spool.seek(start)
    while size:
        piece = spool.read(min(size, 1 << 24))
        out.write(piece.decode())
        size -= len(piece)


# The fields of a merged unit that are held in the spool file, not in memory, until the unit is written.
_SPOOLED = ("analyses", "analysis_extra", "analysis_nested", "closures", "error", "facts", "inner", "inputs", "node_sites", "output", "patched", "placement", "walk_opts", "walk_opts_used", "walk_props")


def spool_value(spool, v):
    """Writes the canonical text of `v` to the spool and returns (where it starts, its size)."""
    start = spool.tell()
    spool.write(canon(v).encode())
    return start, spool.tell() - start


def write_unit(fh, u, spool):
    """Writes a merged unit as one line: the keys sorted, the spooled fields copied from the spool."""
    fh.write("{")
    for n, k in enumerate(sorted(u)):
        fh.write(("," if n else "") + json.dumps(k) + ":")
        if k in _SPOOLED:
            copy_spooled(spool, *u[k], fh)
        else:
            fh.write(canon(u[k]))
    fh.write("}\n")


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
        for line in read_lines(os.path.join(d, "texts.jsonl")):
            u, _, write_head = split_record(line)
            if u["key"] in ex and not u.get("exclude"):
                u["exclude"] = ex[u["key"]]
                n += 1
                text = json.dumps(u, sort_keys=True, separators=(",", ":")) + "\n"
                if write_head:
                    write_head(out)
                    text = "," + text[1:]
                out.write(text)
            else:
                line.write(out)
            del line, write_head
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
    spool = tempfile.TemporaryFile(dir=d)    # the analyses of the unique units, in the form they are written in
    for f in sorted(x for x in glob.glob(os.path.join(d, "t*.jsonl")) if re.fullmatch(r"t\d+\.jsonl", os.path.basename(x))):
        for line in read_lines(f):
            rec, analyses, _ = split_record(line)
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
            outcome = hashlib.sha256(canon([rec.get("output"), rec.get("error")]).encode()).digest()    # equal outcomes, compared by digest
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
                u = uniq[key] = {"exclude": ex, "analyses": None, "analysis_nested": rec.get("analysis_nested", {}), "node_sites": rec.get("node_sites", {}), "analysis_extra": rec.get("analysis_extra"),"key": key, "target": tg, "unit": rec["unit"], "seam": rec["seam"], "calls": 1, "tests": [rec["test"]], "inputs": rec["inputs"],
                             "output": rec.get("output"), "error": rec.get("error"), "texts": rec.get("texts", []), "unknown": rec["unknown"],
                             "unmirrorable": rec["unmirrorable"], "closures": rec.get("closures", []), "walk_opts": rec.get("walk_opts", {}), "walk_opts_used": rec.get("walk_opts_used", {}), "walk_props": rec.get("walk_props", {}), "outside_calls": rec.get("outside_calls", 0), "placement": rec.get("placement"), "patched": rec.get("patched"), "facts": rec.get("facts"),
                             "kept": "patched only in a machine fact whose answers are recorded in `facts`" if rec.get("patched") and not rest else None, "inner": rec.get("inner"), "_o": outcome}
                for name in _SPOOLED[1:]:
                    u[name] = spool_value(spool, u[name])
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
                off = spool.tell()
                spool.write(b"[")
                for j, a in enumerate(analyses()):
                    spool.write(("," + canon(a) if j else canon(a)).encode())
                    ck["analyses_outside"] += bool(a.get("outside"))
                    ck["generator_answers"] += bool(a.get("gen"))
                    ck["generator_answers_not_a_list"] += bool(a.get("gen")) and not (isinstance(a.get("result"), dict) and "l" in a["result"])
                    ck["answers_unrecorded_object"] += tp_tag(a) == "u"
                    ck["closure_calls_recorded"] += bool(a.get("closure"))
                    if tp_tag(a) in ("closure", "bm", "fn") and not a.get("closure"):
                        callables[a["fn"] + " -> " + tp_tag(a)] += a["n"]
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
                spool.write(b"]")
                u["analyses"] = (off, spool.tell() - off)
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
            del line, analyses, _    # the next line is read while this one would still be held
    with open(os.path.join(d, "texts.jsonl"), "w") as fh:
        for k, u in uniq.items():
            if k in patched and not u["exclude"]:
                u["exclude"] = "patched render: equal inputs gave different texts (tests/test_finish_own_sample.py)"
                per[u["target"]]["excluded"] = per[u["target"]].get("excluded", 0) + 1
            u.pop("_o")
            write_unit(fh, u, spool)
    summ = {"checks": dict(ck), "callable_answers": dict(callables), "walk_opts": {**{k: (len(v) if isinstance(v, set) else v) for k, v in wo.items()}, "two_sets_digests": sorted(wo["two_sets"])[:10]}, "inner": {f"{a}|{b}": n for (a, b), n in inn.items()},
            "calls": n, "unique_units": len(uniq), "raised_calls": raised, "unmirrorable_calls": unm, "calls_per_seam": dict(seams),
            "per_target": {t: {**{k: v for k, v in p.items() if k != "texts"}, "distinct_texts": len(p["texts"])} for t, p in per.items()},
            "equal_inputs_equal_outputs": not findings, "findings_count": len(findings), "findings": findings[:20]}
    json.dump({f: {**{k: (dict(v) if isinstance(v, collections.Counter) else sorted(v) if isinstance(v, set) else v) for k, v in i.items()}} for f, i in idx.items()}, open(os.path.join(d, "analyses_index.json"), "w"), indent=1)
    json.dump(summ, open(os.path.join(d, "SUMMARY.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "findings"}))
    if earlier:
        old = {}
        for line in read_lines(os.path.join(earlier, "texts.jsonl")):
            x, _, _ = split_record(line)
            old[unit_key(x["target"], x["inputs"])] = x["target"]
            del line, _
        new = {k: u["target"] for k, u in uniq.items()}
        for tg in sorted(set(old.values()) | set(new.values())):
            o, n_ = {k for k, t in old.items() if t == tg}, {k for k, t in new.items() if t == tg}
            print(f"unit keys [{tg}]: earlier {len(o)}, now {len(n_)}, added {len(n_ - o)}, removed {len(o - n_)}")


if sys.argv[1] == "--apply-exclusions":
    apply_exclusions(sys.argv[2])
else:
    main(*sys.argv[1:3])
