# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""pytest plugin: record every call of raptor._core.canonical (inputs, result or refusal) into $SHADOW_DIR/w<pid>.jsonl.

Use: SHADOW_MODE=record SHADOW_DIR=<dir> PYTHONPATH=<this dir>:<raptor> pytest -p shadow_plugin ...
A node is recorded generically: class name, class module, value of every set slot (state, not constructor arguments).
Values use a closed grammar (see enc); anything else goes to the record's `unknown` list and marks it unmirrorable.
Standard library only.
"""
import dataclasses
import json
import os
import sys

_MODE = os.environ.get("SHADOW_MODE")
_DIR = os.environ.get("SHADOW_DIR")
_state = {"test": "collection", "seq": 0, "fh": None, "pid": None}
_NODE = [None]


def _slots(cls):
    out = []
    for b in reversed(cls.__mro__):
        s = b.__dict__.get("__slots__", ())
        out.extend([s] if isinstance(s, str) else s)
    seen = []
    for x in out:
        if x not in seen:
            seen.append(x)
    return seen


def _is_record(v):
    t = type(v)
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return True
    if isinstance(v, tuple) and hasattr(t, "_fields"):
        return True
    mod = getattr(t, "__module__", "") or ""
    return mod.split(".")[0] in ("raptor", "hawk", "aether") and hasattr(t, "__slots__") and not isinstance(v, _NODE[0])


def _fields(v):
    t = type(v)
    if dataclasses.is_dataclass(v):
        return [f.name for f in dataclasses.fields(v)]
    if hasattr(t, "_fields"):
        return list(t._fields)
    return _slots(t)


def _kids(v, out):
    """Nodes held directly by a value (no recursion through nodes)."""
    if isinstance(v, _NODE[0]):
        out.append(v)
    elif type(v) in (tuple, list, frozenset):
        for x in v:
            _kids(x, out)
    elif type(v) is dict:
        for k, x in v.items():
            _kids(k, out)
            _kids(x, out)
    elif v is None or type(v) in (int, float, str, bool):
        return
    elif _is_record(v):
        for f in _fields(v):
            if hasattr(v, f):
                _kids(getattr(v, f), out)


def _key(j):
    return json.dumps(j, sort_keys=True, separators=(",", ":"))


class _Enc:
    def __init__(self):
        self.nodes = []        # encoded node records
        self.index = {}        # id(node) -> position
        self.keep = []
        self.kinds = []        # declared (dialect) kinds: per class, its declarations most-derived first
        self.kidx = {}         # class -> index in kinds
        self.unknown = []

    def val(self, v):
        t = type(v)
        if v is None:
            return None
        if t is bool:
            return v
        if t is int:
            return {"i": str(v)}
        if t is float:
            return {"f": repr(v), "x": v.hex()}
        if t is str:
            return {"s": v}
        if isinstance(v, _NODE[0]):
            return {"n": self.index[id(v)]}
        if t is tuple:
            return {"t": [self.val(x) for x in v]}
        if t is list:
            return {"l": [self.val(x) for x in v]}
        if t is frozenset:
            return {"fs": sorted((self.val(x) for x in v), key=_key)}
        if t is dict:
            return {"d": sorted(([self.val(k), self.val(x)] for k, x in v.items()), key=_key)}
        if _is_record(v):
            out = {"r": t.__name__, "m": t.__module__,
                   "v": {f: self.val(getattr(v, f)) for f in _fields(v) if hasattr(v, f)}}
            if hasattr(t, "__key_text__") and t.__module__.split(".")[0] != "raptor":
                out["kt"] = v.__key_text__()       # the key text a record outside raptor contributes is computed by its own method
            return out
        self.unknown.append(t.__module__ + "." + t.__qualname__)
        return {"u": t.__qualname__}

    def kind_of(self, cls):
        """A class made at run time from a declaration cannot be imported: record its declarations (class to base)."""
        i = self.kidx.get(cls)
        if i is None:
            chain = [{"decl": self.val(b.__dict__["decl"]), "root": b.decl_root} for b in cls.__mro__ if "decl" in b.__dict__]
            i = self.kidx[cls] = len(self.kinds)
            self.kinds.append(chain)
        return i

    def dag(self, roots):
        """Iterative post-order over every node reachable from roots (fields included)."""
        stack = []
        for r in reversed(roots):
            _kids(r, kids := [])
            stack.extend((k, False) for k in reversed(kids))
        while stack:
            n, done = stack.pop()
            if id(n) in self.index:
                continue
            if done:
                self.index[id(n)] = len(self.nodes)
                self.keep.append(n)
                vals = {}
                for s in _slots(type(n)):
                    try:
                        x = object.__getattribute__(n, s)
                    except AttributeError:
                        continue
                    vals[s] = self.val(x)
                ent = {"c": type(n).__name__, "m": type(n).__module__, "v": vals}
                if "decl" in type(n).__dict__:
                    ent["k"] = self.kind_of(type(n))
                self.nodes.append(ent)
                continue
            stack.append((n, True))
            kids = []
            for s in _slots(type(n)):
                try:
                    _kids(object.__getattribute__(n, s), kids)
                except AttributeError:
                    pass
            stack.extend((k, False) for k in reversed(kids) if id(k) not in self.index)


def _record(sinks, declared, force, exempt, call):
    enc = _Enc()
    sinks = list(sinks)
    enc.dag(sinks + list(declared))
    inputs = {"nodes": enc.nodes, "kinds": enc.kinds, "sinks": [enc.val(s) for s in sinks], "declared": enc.val(list(declared)),
              "force": enc.val(force), "exempt": enc.val(list(exempt))}
    rec = {"inputs": inputs}
    err = None
    res = None
    try:
        res = call()
    except BaseException as e:
        err = e
    try:
        if err is not None:
            rec["error"] = {"cls": type(err).__name__, "msg": str(err)}
        else:
            order, leaves, arg_spec, slot_of, slot_types, cls, quantities, digest, dispatches, prior, finish = res
            pre = None
            try:
                from raptor import _core
                pre = _ORIG[0](sinks)[3]  # walk preimage text (node part)
                pre = pre if isinstance(pre, str) else repr(pre)
            except BaseException:
                pre = None
            rec["result"] = {
                "digest": digest, "arg_spec": enc.val(list(arg_spec)), "access": enc.val(cls),
                "leaves": [[s.role, s.name, str(s.ttype)] for s in leaves],
                "quantities": enc.val(dict(quantities)), "dispatches": enc.val(list(dispatches)),
                "prior": enc.val(prior), "finish": enc.val(finish), "n_order": len(order),
                "order_kinds": [type(n).__name__ for n in order]}
            rec["preimage"] = pre
    except BaseException as e:      # a recording failure never changes the call
        rec["record_error"] = repr(e)
    rec["unknown"] = sorted(set(enc.unknown))
    rec["unmirrorable"] = bool(enc.unknown) or "record_error" in rec
    return rec, res, err


_ORIG = [None, None]


def _wrapper(sinks, declared, force, env, exempt):
    orig = _ORIG[1]
    if _DIR is None or _MODE != "record":
        return orig(sinks, declared, force, env, exempt)
    rec, res, err = _record(sinks, declared, force, exempt, lambda: orig(sinks, declared, force, env, exempt))
    try:
        pid = os.getpid()
        if _state["pid"] != pid:
            os.makedirs(_DIR, exist_ok=True)
            _state["fh"] = open(os.path.join(_DIR, f"w{pid}.jsonl"), "a")
            _state["pid"] = pid
        _state["seq"] += 1
        rec["seq"] = _state["seq"]
        rec["pid"] = pid
        rec["test"] = _state["test"]
        _state["fh"].write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
        _state["fh"].flush()
    except BaseException:
        pass
    if err is not None:
        raise err
    return res


_SH = {"calls": 0, "equal": 0, "roundtrip_equal": 0, "roundtrip_tried": 0, "roundtrip_skipped_declared": 0, "skipped": {}, "oracle_raised": 0,
       "door_refused_too": 0, "door_message_equal": 0, "door_accepted_oracle_raised": 0, "make_refused": {}, "mismatch_count": 0,
       "mismatches": [], "shadow_exceptions": 0, "exception_samples": [], "unmirrorable": 0, "seconds_shadow": 0.0, "seconds_oracle": 0.0,
       "max_nodes": 0, "seconds_at_max_nodes": 0.0}
_EXT = [None, False]


def _ext():
    if _EXT[0] is None and not _EXT[1]:
        _EXT[1] = True
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.environ.get("SHADOW_EXT_DIR", os.path.join(base, "osprey", "build")))
        import _osprey
        _EXT[0] = _osprey
    return _EXT[0]


def _shadow_write():
    try:
        os.makedirs(_DIR, exist_ok=True)
        with open(os.path.join(_DIR, f"shadow_{os.getpid()}.json"), "w") as fh:
            json.dump(_SH, fh, sort_keys=True)
    except BaseException:
        pass


_SEEN = set()


def _seen_digest(res):
    """Append each new digest the real call gave to digests_<pid>.txt (which walks this run saw)."""
    if res is None or res[7] in _SEEN:
        return
    _SEEN.add(res[7])
    try:
        os.makedirs(_DIR, exist_ok=True)
        with open(os.path.join(_DIR, f"digests_{os.getpid()}.txt"), "a") as fh:
            fh.write(res[7] + "\n")
    except BaseException:
        pass


def _bump(table, key):
    table[key] = table.get(key, 0) + 1


def _mirror_call(sinks, declared, force, exempt, res, err):
    """Mirror one call into new contexts through the binding and compare with what the real call gave."""
    import shadow_mirror as sm
    ext = _ext()
    enc = _Enc()
    enc.dag(list(sinks) + list(declared))
    inp = {"nodes": enc.nodes, "kinds": enc.kinds, "sinks": [enc.val(s) for s in sinks], "declared": enc.val(list(declared)),
           "force": enc.val(force), "exempt": enc.val(list(exempt))}
    if enc.unknown:
        _SH["unmirrorable"] += 1
        return
    c = ext.Ctx()
    ms = sm.manifests(inp)
    sm.declare(c, ms)
    try:
        nodes = sm.mirror(c, inp)
    except ValueError as e:
        m = str(e)
        if m.startswith("not in this build: "):
            _bump(_SH["skipped"], m[len("not in this build: "):])
        else:
            _bump(_SH["make_refused"], m[:200])
        return
    h = list(range(len(nodes)))
    sink_nodes = [nodes[s["n"]] for s in inp["sinks"]]
    dtext, etext = sm.val(inp["declared"], h), sm.val(inp["exempt"], h)
    fo = force if isinstance(force, str) else None
    try:
        w = c.canonical(sink_nodes, dtext, fo, etext)
    except ValueError as e:
        m = str(e)
        if err is not None:
            _SH["oracle_raised"] += 1
            _SH["door_refused_too"] += 1
            _SH["door_message_equal"] += m == str(err)
        elif m.startswith("not in this build: "):
            _bump(_SH["skipped"], m[len("not in this build: "):])
        else:
            _bad(inp, "door refused: " + m, None)
        return
    if err is not None:
        _SH["oracle_raised"] += 1
        _SH["door_accepted_oracle_raised"] += 1
        _bad(inp, "oracle raised %s, door accepted" % type(err).__name__, None)
        return
    order, leaves, arg_spec, slot_of, slot_types, cls, quantities, digest, dispatches, prior, finish = res
    bad = []
    dg = c.digest(w)
    if dg != digest:
        bad.append("digest")
    if c.walk_field(w, "arg_spec") != sm.val(enc.val(tuple(arg_spec)), h):
        bad.append("arg_spec")
    if c.walk_field(w, "access") != sm.val(enc.val(cls), h):
        bad.append("access")
    if bad:
        pre = None
        if "digest" in bad:
            try:
                want, got = _ORIG[0](list(sinks))[3].splitlines(), c.walk_field(w, "preimage").splitlines()
                i = next((i for i in range(max(len(want), len(got))) if (want[i:i + 1] != got[i:i + 1])), 0)
                pre = "line %d: today %r door %r" % (i, want[i:i + 1], got[i:i + 1])
            except BaseException as e:
                pre = "preimage unavailable: %r" % (e,)
        _bad(inp, "+".join(bad), (digest, dg, pre))
        return
    _SH["equal"] += 1
    if "#" in dtext:
        _SH["roundtrip_skipped_declared"] += 1
        return
    _SH["roundtrip_tried"] += 1
    c2 = ext.Ctx()
    sm.declare(c2, ms)
    roots = c2.load(c.dump(sink_nodes))
    if c2.digest(c2.canonical(list(roots), dtext, fo, etext)) == dg:
        _SH["roundtrip_equal"] += 1
    else:
        _bad(inp, "roundtrip digest", (digest, dg, None))


def _bad(inp, what, extra):
    _SH["mismatch_count"] += 1
    if len(_SH["mismatches"]) < 20:
        _SH["mismatches"].append({"test": _state["test"], "what": what, "detail": extra})


def _shadow_wrapper(sinks, declared, force, env, exempt):
    import time
    t0 = time.perf_counter()
    res = err = None
    try:
        res = _ORIG[1](sinks, declared, force, env, exempt)
    except BaseException as e:
        err = e
    t1 = time.perf_counter()
    try:
        _SH["calls"] += 1
        _seen_digest(res)
        _mirror_call(sinks, declared, force, exempt, res, err)
    except BaseException as e:      # the shadow never changes a test's outcome
        _SH["shadow_exceptions"] += 1
        if len(_SH["exception_samples"]) < 5:
            _SH["exception_samples"].append({"test": _state["test"], "error": repr(e)[:300]})
    t2 = time.perf_counter()
    _SH["seconds_oracle"] += t1 - t0
    _SH["seconds_shadow"] += t2 - t1
    if _SH["calls"] % 25 == 0:
        _shadow_write()
    if err is not None:
        raise err
    return res


def _install():
    if _ORIG[1] is not None or _MODE not in ("record", "shadow") or not _DIR:
        return
    from raptor import _core
    from raptor.ir.nodes import Node
    _NODE[0] = Node
    _ORIG[0] = _core.walk
    _ORIG[1] = _core.canonical
    _core.canonical = _wrapper if _MODE == "record" else _shadow_wrapper
    if _MODE == "shadow":
        import atexit
        atexit.register(_shadow_write)


_install()


def pytest_runtest_protocol(item, nextitem):
    _state["test"] = item.nodeid
    return None


def pytest_runtest_logfinish(nodeid, location):
    _state["test"] = "collection"
    if _MODE == "shadow" and _SH["calls"]:
        _shadow_write()


def pytest_unconfigure(config):
    if _MODE == "shadow" and _SH["calls"]:
        _shadow_write()
