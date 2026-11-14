# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""rebuild.py <corpus dir>: rebuild every unique walk in a fresh process FROM walks.jsonl ALONE, call raptor.ir.walk.canonical,
compare digest, arg_spec, access, order length with the record. Needs raptor (and its wings) importable."""
import importlib
import json
import os
import sys

from raptor.ir import walk as _walk


_standin = {}


def klass(m, c, fields=()):
    try:
        return getattr(importlib.import_module(m), c)
    except ImportError:
        # a record of a module outside raptor and its wings (hawk is not importable here): a stand-in with the recorded name and fields
        if (m, c) not in _standin:
            import dataclasses
            k = dataclasses.make_dataclass(c, [(f, object) for f in fields], frozen=True)
            k.__module__ = m
            _standin[(m, c)] = k
        return _standin[(m, c)]


def slots(cls):
    out = []
    for b in reversed(cls.__mro__):
        s = b.__dict__.get("__slots__", ())
        out.extend([s] if isinstance(s, str) else s)
    return out


def dec(v, nodes):
    if v is None or v is True or v is False:
        return v
    tag = next(k for k in ("n", "i", "f", "s", "t", "l", "fs", "d", "r", "u") if k in v)
    x = v[tag]
    if tag == "n":
        return nodes[x]
    if tag == "i":
        return int(x)
    if tag == "f":
        return float.fromhex(v["x"])
    if tag == "s":
        return x
    if tag == "t":
        return tuple(dec(i, nodes) for i in x)
    if tag == "l":
        return [dec(i, nodes) for i in x]
    if tag == "fs":
        return frozenset(dec(i, nodes) for i in x)
    if tag == "d":
        return {dec(k, nodes): dec(i, nodes) for k, i in x}
    if tag == "r":
        cls = klass(v["m"], v["r"], list(v["v"]))
        vals = {f: dec(i, nodes) for f, i in v["v"].items()}
        if issubclass(cls, tuple):
            return tuple.__new__(cls, [vals[f] for f in cls._fields])
        o = object.__new__(cls)
        if "kt" in v:
            cls.__key_text__ = lambda self: self._kt      # a stand-in answers with the recorded key text
            object.__setattr__(o, "_kt", v["kt"])
        for f, i in vals.items():
            object.__setattr__(o, f, i)
        return o
    raise ValueError(f"tag {tag}")


def make_kinds(inp):
    """One class per recorded declaration chain, made by raptor's own class builder from the recorded declaration."""
    from raptor.ir import kinds
    reg = kinds.Registry()

    def one(chain, i):
        decl = dec(chain[i]["decl"], [])
        if decl.name not in reg.classes:
            if i + 1 < len(chain):
                one(chain, i + 1)
            reg.kinds[decl.name] = decl
            reg.classes[decl.name] = kinds._build_class(reg, decl, chain[i]["root"])
        return reg.classes[decl.name]
    return [one(chain, 0) for chain in inp.get("kinds", [])]


def build(inp):
    nodes = []
    made = make_kinds(inp)
    for nd in inp["nodes"]:
        o = object.__new__(made[nd["k"]] if "k" in nd else klass(nd["m"], nd["c"]))
        for s, x in nd["v"].items():
            object.__setattr__(o, s, dec(x, nodes))
        nodes.append(o)
    return ([dec(s, nodes) for s in inp["sinks"]], dec(inp["declared"], nodes),
            dec(inp["force"], nodes), dec(inp["exempt"], nodes))


def main(d):
    ok = total = 0
    bad = []
    for line in open(os.path.join(d, "walks.jsonl")):
        u = json.loads(line)
        total += 1
        try:
            sinks, declared, force, exempt = build(u["inputs"])
            try:
                w = _walk.canonical(sinks, declared, force_access=force, exempt_planes=exempt)
            except BaseException as e:
                got = ("error", type(e).__name__, str(e))
                want = ("error", u["error"]["cls"], u["error"]["msg"]) if u["error"] else None
                (ok := ok + 1) if got == want else bad.append((u["key"][:12], "refusal differs", got, want))
                continue
            if u["error"]:
                bad.append((u["key"][:12], "recorded a refusal, rebuild returned", w.digest))
                continue
            r = u["result"]
            probs = []
            if w.digest != r["digest"]:
                probs.append(f"digest {w.digest} != {r['digest']}")
            if [list(a) for a in w.arg_spec] != [list(dec(a, [])) for a in r["arg_spec"]["l"]]:
                probs.append("arg_spec")
            if w.access.cls != dec(r["access"], []):
                probs.append(f"access {w.access.cls} != {dec(r['access'], [])}")
            if len(w.order) != r["n_order"] or [type(n).__name__ for n in w.order] != r["order_kinds"]:
                probs.append("order")
            if probs:
                bad.append((u["key"][:12], probs))
            else:
                ok += 1
        except BaseException as e:
            bad.append((u["key"][:12], "rebuild failed", type(e).__name__, str(e)[:200]))
    print(f"rebuild-equal {ok} / {total}")
    for b in bad[:5]:
        print("MISMATCH", b)
    print("mismatches", len(bad))


main(sys.argv[1])
