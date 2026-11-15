# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""rebuild_text.py <TEXT_DIR>: in a fresh process rebuild the inputs of every unique render call from texts.jsonl ALONE, call today's render
seam and compare every output with the recorded one. Prints `text-rebuild-equal K / U` per target and the first differences."""
import collections
import difflib
import hashlib
import importlib
import json
import os
import sys
import traceback

import shadow_plugin as sp
import text_plugin as tp
from raptor.ir.nodes import Node

sp._NODE[0] = Node
ns = {}
exec(compile(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rebuild.py")).read().replace("\nmain(sys.argv[1])", "\n"), "rebuild", "exec"), ns)


def make(cls):
    try:
        return object.__new__(cls)
    except TypeError:
        return cls.__new__(cls)


def base_new(base, cls):
    return {"list": list, "dict": dict, "set": set}[base].__new__(cls)


class Dec:
    def __init__(self, inp):
        self.inp = inp
        self.nodes, self.objs, self.sent = [], [], {}
        kinds = self.kinds = ns["make_kinds"](inp)
        for o in inp["objs"]:
            self.objs.append(make(self.cls(o["m"], o["c"])))
        for nd in inp["nodes"]:
            n = object.__new__(kinds[nd["k"]] if "k" in nd else ns["klass"](nd["m"], nd["c"]))
            for s, x in nd["v"].items():
                object.__setattr__(n, s, self.val(x))
            self.nodes.append(n)
        for o, e in zip(self.objs, inp["objs"]):
            for k, x in e["a"].items():
                object.__setattr__(o, k, self.val(x))

    @staticmethod
    def cls(m, q):
        return tp._lookup("%s:%s" % (m, q)) or ns["klass"](m, q)

    def val(self, v):
        if v is None or v is True or v is False:
            return v
        k = next(k for k in ("n", "i", "f", "s", "t", "l", "fs", "d", "r", "u", "b", "set", "enum", "path", "mp", "fn", "bm", "o", "lock", "sub", "inert", "closure", "backend", "sentinel", "idmap", "proxy") if k in v)
        x = v[k]
        if k == "n":
            return self.nodes[x]
        if k == "idmap":
            return {id(self.nodes[i]): self.val(y) for i, y in x}
        if k == "proxy":
            from hawk.artifact.bundle import _ActiveUnit
            return _ActiveUnit(self.val(v["kernel"]), self.val(v["walk"]))
        if k == "sentinel":
            return self.sent.setdefault(x, object())
        if k == "backend":
            from hawk import backends
            return backends.need(x)
        if k == "inert":
            return None
        if k == "closure":
            def stub(*a, **k):
                raise RuntimeError("recorded closure %s was called by the render" % x)
            stub.__module__, _, stub.__qualname__ = x.partition(":")      # it encodes again as the closure it stands for
            return stub
        if k == "sub":
            cls, of, base = tp._lookup(x), self.val(v["of"]), v["base"]
            if base == "tuple":
                o = tuple.__new__(cls, of)
            elif base == "frozenset":
                o = frozenset.__new__(cls, of)
            else:
                o = base_new(base, cls)
                {"list": list.extend, "dict": dict.update, "set": set.update}[base](o, of)
            for a, b in v["attrs"].items():
                object.__setattr__(o, a, self.val(b))
            return o
        if k == "lock":
            import threading
            return threading.Lock()
        if k == "o":
            return self.objs[x]
        if k == "i":
            return int(x)
        if k == "f":
            return float.fromhex(v["x"])
        if k == "s":
            return x
        if k == "t":
            return tuple(self.val(i) for i in x)
        if k == "l":
            return [self.val(i) for i in x]
        if k == "fs":
            return frozenset(self.val(i) for i in x)
        if k == "set":
            return {self.val(i) for i in x}
        if k == "d":
            return {self.val(a): self.val(b) for a, b in x}
        if k == "mp":
            import types
            return types.MappingProxyType(self.val(x))
        if k == "b":
            return bytes.fromhex(x)
        if k == "enum":
            return tp._lookup(x)[v["name"]]
        if k == "path":
            return tp._lookup(v["cls"])(x)
        if k == "fn":
            return tp._lookup(x)
        if k == "bm":
            return getattr(self.val(x), v["name"])
        if k == "r":
            cls = ns["klass"](v["m"], v["r"], list(v["v"]))
            vals = {f: self.val(i) for f, i in v["v"].items()}
            if issubclass(cls, tuple):
                return tuple.__new__(cls, [vals[f] for f in cls._fields])
            o = object.__new__(cls)
            if "kt" in v:
                if not hasattr(cls, "__key_text__"):
                    cls.__key_text__ = lambda self: self._kt      # a stand-in answers with the recorded key text; a class with its own method keeps it
                if cls.__key_text__.__name__ == "<lambda>":
                    object.__setattr__(o, "_kt", v["kt"])
            for f, i in vals.items():
                object.__setattr__(o, f, i)
            return o
        raise ValueError("tag " + k)


def _w_lt():
    from aether.wings.base import wrapper as _w
    return _w.literal_table


def inner_main(d):
    """Rebuild each unit from its inner arguments alone (the arguments of render_source) and compare text and body with the inner result."""
    from aether.wings.base import wrapper as _w
    ok, total, bad = collections.Counter(), collections.Counter(), []
    for line in open(os.path.join(d, "texts.jsonl")):
        u = json.loads(line)
        inner = u.get("inner")
        if inner is None or u["seam"].endswith("literal_table"):
            continue
        tg = u["target"]
        direct = inner == "inputs"
        if direct:
            inner = u["inputs"]
        if "encode_error" in inner or "args" not in inner or u.get("exclude") or u["key"] in tp.load_excluded(d):
            continue
        total[tg] += 1
        try:
            dec = Dec(inner)
            res = _w.render_source(*dec.val(inner["args"]), **dec.val(inner["kwargs"]))
            want = {x[0]: x[1] for x in (u["texts"] if direct else inner.get("texts", []))}
            got = {f: hashlib.sha256(getattr(res, f).encode()).hexdigest() for f in ("text", "body")}
            if all(want.get(f) == got[f] for f in ("text", "body")):
                ok[tg] += 1
            else:
                bad.append((tg, u["unit"], "differs", [f for f in ("text", "body") if want.get(f) != got[f]]))
        except BaseException as e:
            err = u.get("error") if direct else inner.get("error")      # a direct call of the inner function keeps its refusal on the unit
            if err and err["cls"] == type(e).__name__:
                ok[tg] += 1
            else:
                fr = traceback.extract_tb(e.__traceback__)[-1]
                bad.append((tg, u["unit"], "failed", "%s: %s [raised at %s:%d; recorded test %s]" % (type(e).__name__, str(e)[:120], os.path.relpath(fr.filename, "/").split("staging/")[-1], fr.lineno, u["tests"][0][:80])))
    for tg in sorted(total):
        print(f"inner-rebuild-equal {ok[tg]} / {total[tg]}   [{tg}]")
    for b in bad:
        print("INNER MISMATCH", *b)
    print("inner mismatches", len(bad))


def seeded(dec, inp):
    """An encoder holding the node, object and kind tables the recording had when the render began."""
    e = tp.TextEnc()
    e.nodes, e.keep, e.index = [None] * len(dec.nodes), list(dec.nodes), {id(n): i for i, n in enumerate(dec.nodes)}
    e.objs, e.hold, e.oidx = [None] * len(dec.objs), list(dec.objs), {id(o): i for i, o in enumerate(dec.objs)}
    e.kinds, e.kidx = [None] * len(dec.kinds), {c: i for i, c in enumerate(dec.kinds)}
    e.sent = {id(o): k for k, o in dec.sent.items()}
    return e


def serve_facts(dec, u):
    """A unit that recorded machine facts (`facts`): the functions that read them answer from the record for equal arguments (the real function
    otherwise), as the compiled writers will be given them; returns the function that puts the real ones back."""
    undo = []
    for mn, names in tp._FACTS_OF.items():
        mod = importlib.import_module(mn)
        for n in names:
            entries = (u.get("facts") or {}).get(n)
            if not entries:
                continue
            e = seeded(dec, u["inputs"])
            empty = tp._ser(e.val({}))
            table = {tp._ser(x[0]) + "|" + (tp._ser(x[2]) if len(x) > 2 else empty): dec.val(x[1]) for x in entries}
            real = getattr(mod, n)

            def served(*a, _t=table, _e=e, _r=real, **k):
                key = tp._ser(_e.val(list(a))) + "|" + tp._ser(_e.val(k))
                return _t[key] if key in _t else _r(*a, **k)
            setattr(mod, n, served)
            undo.append((mod, n, real))

    def restore():
        for mod, n, real in undo:
            setattr(mod, n, real)
    return restore


def recorded_calls(u):
    out = {}
    for a in u.get("analyses", []):
        rs = json.dumps({"error": a["error"]} if "error" in a else {"result": a["result"]}, sort_keys=True, separators=(",", ":"))
        out[a["fn"] + "|" + tp._ser(a["args"]) + "|" + tp._ser(a["kwargs"])] = {**a, "_rs": rs}
    return out


def main(d, analyses=False):
    from hawk import backends
    real_need = backends.need
    if analyses:
        tp.install_analysis_hooks()
    atally, adiffs, unreplayed = collections.defaultdict(collections.Counter), [], collections.Counter()
    miss_at, never_at, near = collections.Counter(), collections.Counter(), []
    excluded = tp.load_excluded(d)
    xok, xtotal = collections.Counter(), collections.Counter()
    real_lt = _w_lt()
    from aether.wings.base import wrapper as _w
    real_rs = _w.render_source
    ok, total, bad = collections.Counter(), collections.Counter(), []
    for line in open(os.path.join(d, "texts.jsonl")):
        u = json.loads(line)
        tg = u["target"]
        ex = u.get("exclude") or excluded.get(u["key"])
        okc = xok if ex else ok
        if ex:
            xtotal[tg] += 1
        else:
            total[tg] += 1
        if u["unmirrorable"] and "encode_error" in u["inputs"]:
            bad.append((tg, u["unit"], "unencodable inputs", None))
            continue
        try:
            dec = Dec(u["inputs"])
            direct = u["seam"].endswith(("render_source", "literal_table"))
            if u["seam"].endswith("literal_table"):
                args, kwargs = [dec.val(u["inputs"]["source"])], {}
            elif direct:
                args, kwargs = dec.val(u["inputs"]["args"]), dec.val(u["inputs"]["kwargs"])
            else:
                lowered, options = dec.val(u["inputs"]["lowered"]), dec.val(u["inputs"]["options"])
            if "self" in u["inputs"]:        # a backend a test made itself: the registry is told about it for this call
                row = dec.val(u["inputs"]["self"])
                backends.need = lambda i, _r=row, _n=real_need: _r if i == _r.id else _n(i)
            else:
                backends.need = real_need
                row = None if direct else real_need(tg)
            restore = serve_facts(dec, u)
            actx = tp._analysis_begin(seeded(dec, u["inputs"]), None, "check", recorded_calls(u)) if analyses else None
            try:
                res = (real_lt if u["seam"].endswith("literal_table") else real_rs)(*args, **kwargs) if direct else row.render(lowered, options)
            except BaseException as e:
                got, want = ("error", type(e).__name__, str(e)), ("error", u["error"]["cls"], u["error"]["msg"]) if u["error"] else None
                if got == want:
                    okc[tg] += 1
                else:
                    bad.append((tg, u["unit"], "refusal differs", [str(got)[:300], str(want)[:300]]))
                continue
            finally:
                restore()
                if actx is not None:
                    tp._analysis_end(actx, None)
                    for q, c in actx.tally.items():
                        atally[q].update(c)
                        if c["missing"]:
                            miss_at[(q, tg, u["unit"])] += c["missing"]
                    adiffs.extend(actx.diffs[:3])
                    near.extend((u["unit"],) + x for x in actx.near)
                    for kk, e in actx.recorded.items():
                        if kk not in actx.seen_keys and e.get("outside", 0) < e["n"]:      # a call made outside the render is not a render's to repeat
                            unreplayed[e["fn"]] += 1
                            never_at[(e["fn"], tg, u["unit"])] += 1
            if u["error"]:
                bad.append((tg, u["unit"], "recorded a refusal, rebuild returned", None))
                continue
            out = tp.TextEnc()
            val = out.val(res)
            diffs = []
            for fld, sha, _n in u["texts"]:
                now = getattr(res, fld)
                if hashlib.sha256(now.encode()).hexdigest() != sha:
                    was = u["output"]["value"]["v"][fld]["s"]
                    diffs.append("".join(list(difflib.unified_diff(was.splitlines(True), now.splitlines(True), "recorded " + fld, "rebuilt " + fld, n=1))[:40]))
            if not diffs and tp.sp._key({"value": val, "nodes": out.nodes, "objs": out.objs}) != tp.sp._key({k: u["output"][k] for k in ("value", "nodes", "objs")}):
                diffs.append("texts equal; the rest of the output (report, flags) differs")
            if diffs:
                bad.append((tg, u["unit"], "text differs", diffs[0]))
            else:
                okc[tg] += 1
        except BaseException as e:
            bad.append((tg, u["unit"], "rebuild failed", "%s: %s" % (type(e).__name__, str(e)[:300])))
    with open(os.path.join(d, "rebuild_mismatches.txt"), "w") as fh:
        for b in bad:
            fh.write(" | ".join([b[0], str(b[1]), b[2], str(b[3])[:240].replace("\n", " ")]) + "\n")
    for tg in sorted(xtotal):
        print(f"excluded (not a fair oracle) text-rebuild-equal {xok[tg]} / {xtotal[tg]}   [{tg}]")
    if analyses:
        tot = collections.Counter()
        for q in sorted(atally):
            c = atally[q]
            n = c["equal"] + c["differs"] + c["missing"]
            tot.update(c)
            print(f"analysis-rebuild-equal {c['equal']} / {n}   {q}" + (f"   differs {c['differs']} missing {c['missing']}" if c["differs"] or c["missing"] else ""))
        print(f"analysis-rebuild-equal TOTAL {tot['equal']} / {tot['equal'] + tot['differs'] + tot['missing']} (differs {tot['differs']}, missing {tot['missing']}, recorded calls never made by the replay {sum(unreplayed.values())}, encode errors {tot['encode_error']})")
        for (q, tg, unit), n in miss_at.most_common(12):
            print(f"ANALYSIS MISSING {q} [{tg}] unit {unit}: {n}")
        for q, n in unreplayed.most_common():
            print(f"ANALYSIS NEVER MADE {q}: {n} in {sum(1 for k in never_at if k[0] == q)} units")
        for (q, tg, unit), n in never_at.most_common(8):
            print(f"ANALYSIS NEVER MADE unit {q} [{tg}] {unit}: {n}")
        for x in near[:4]:
            print("ANALYSIS MISSING PARTS FROM THE CLOSEST RECORDED CALL", x[0], x[1], "at char", x[2], "|", x[3], "|", x[4])
        for q, why, x in adiffs[:5]:
            print("ANALYSIS DIFF", q, why, x)
    for tg in sorted(total):
        print(f"text-rebuild-equal {ok[tg]} / {total[tg]}   [{tg}]")
    for b in bad[:5]:
        print("MISMATCH", b[0], b[1], b[2]); print(b[3] if b[3] else "")
    print("mismatches", len(bad))


def memo_free(v):
    """The value without the cache a Loop fills when its body is first keyed (`_body_key`): it is no part of what a pass returned."""
    if isinstance(v, dict):
        return {k: memo_free(x) for k, x in v.items() if k != "_body_key"}
    return [memo_free(x) for x in v] if isinstance(v, list) else v


def placement_main(d):
    """Call the old placement pass again with each recorded call's arguments and compare its returned sinks and planes with the recorded result."""
    from raptor.wings.base import wide as _wide
    ok, total, bad = 0, 0, []
    for line in open(os.path.join(d, "texts.jsonl")):
        u = json.loads(line)
        if not u.get("placement"):
            continue
        ex = u.get("analysis_extra") or {}
        inp = dict(u["inputs"])
        if ex.get("offsets") != [len(inp["nodes"]), len(inp["objs"]), len(inp["kinds"])]:
            total += len(u["placement"])
            bad.append((u["unit"], "tables do not line up"))
            continue
        inp["nodes"], inp["objs"], inp["kinds"] = inp["nodes"] + ex["nodes"], inp["objs"] + ex["objs"], inp["kinds"] + ex["kinds"]
        for p in u["placement"]:
            total += 1
            try:
                dec = Dec(inp)
                got, want = _wide.place(*dec.val(p["args"]), **dec.val(p["kwargs"])), dec.val(p["result"])
                keys = []
                for res in (got, want):
                    e = tp.TextEnc()
                    keys.append(json.dumps(memo_free({"v": e.val(res), "nodes": e.nodes, "objs": e.objs}), sort_keys=True, separators=(",", ":")))
                if keys[0] == keys[1]:
                    ok += 1
                else:
                    i = len(os.path.commonprefix(keys))
                    bad.append((u["unit"], "result differs at char %d: rebuilt %s | recorded %s" % (i, keys[0][max(0, i - 60):i + 80], keys[1][max(0, i - 60):i + 80])))
            except BaseException as e:
                bad.append((u["unit"], "failed %s: %s" % (type(e).__name__, str(e)[:160])))
    print(f"placement-rebuild-equal {ok} / {total}")
    for b in bad[:5]:
        print("PLACEMENT MISMATCH", *b)


if "--placement" in sys.argv:
    placement_main([a for a in sys.argv[1:] if not a.startswith("--")][0])
elif "--inner" in sys.argv:
    inner_main([a for a in sys.argv[1:] if not a.startswith("--")][0])
else:
    main([a for a in sys.argv[1:] if not a.startswith("--")][0], "--analyses" in sys.argv)
