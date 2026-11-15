# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""pytest plugin: record every call of a render seam (inputs, outputs) into $TEXT_DIR/t<pid>.jsonl.

Use: TEXT_MODE=record TEXT_DIR=<dir> pytest -p text_plugin ...   (shadow_plugin.py must be importable: it holds the node encoder)
Seams: hawk.backends.Backend.render (the host and CUDA targets) and ospreys.c.backend.C.render (the C target).
Each call becomes one JSON line: target, unit name, every input (objects as class, module and state; nodes as in the walk corpus; callables
by name), the outputs verbatim with their sha256, or the exception. A value outside the grammar goes to `unknown` and marks the record
`unmirrorable`. Standard library only."""
import collections
import enum
import functools
import hashlib
import json
import os
import pathlib
import sys
import threading
import types

import shadow_plugin as sp

_MODE = os.environ.get("TEXT_MODE")
_DIR = os.environ.get("TEXT_DIR")
_state = {"test": "collection", "seq": 0, "fh": None, "pid": None}
_ANALYSES = os.environ.get("TEXT_ANALYSES", "1") != "0"
_SCALARS = (str, int, bool, float, type(None))
_SIMPLE = (type(None), bool, int, float, str, tuple, list, dict, frozenset)


class TextEnc(sp._Enc):
    """The walk corpus' encoder, extended with the values a render call holds besides nodes."""

    def __init__(self):
        super().__init__()
        self.objs, self.oidx, self.hold, self.closures = [], {}, [], []
        self.memo, self.sent = {}, {}
        self.walks = {}                        # digest -> the live Walk the unit holds (its properties are recorded per walk)

    def val(self, v):
        t = type(v)
        if isinstance(v, sp._NODE[0]):
            if id(v) not in self.index:
                self.dag([v])
            return {"n": self.index[id(v)]}
        if t is object:
            return {"sentinel": self.sent.setdefault(id(v), len(self.sent))}      # a bare object(): one per index in a record
        if t is tuple or (t not in _SIMPLE and sp._is_record(v)):
            m = self.memo.get(id(v))           # immutable values are encoded once per unit (the same encoded object is returned)
            if m is not None:
                return m[1]
            r = self._val(v)
            self.memo[id(v)] = (v, r)
            if t.__name__ == "Walk" and isinstance(getattr(v, "digest", None), str):
                self.walks[v.digest] = v
            return r
        return self._val(v)

    def _val(self, v):
        t = type(v)
        if (t is dict or t is types.MappingProxyType) and v and all(type(x) is int and x in self.index for x in v):
            # a map keyed by the address of nodes (id(node)): addresses are no part of a value, the keys become node references
            return {"idmap": sorted(([self.index[x], self.val(y)] for x, y in v.items()), key=lambda p: p[0])}
        if t not in _SIMPLE and _is_row(v):
            return {"backend": v.id}
        if t.__name__ == "_ActiveUnit" and t.__module__ == "hawk.artifact.bundle":
            from hawk.artifact.stages import TeamStage
            if isinstance(v._kernel, TeamStage):       # a team stage seen through its active-set walk: the stage keeps its own type
                return {"proxy": "hawk.artifact.bundle:_ActiveUnit", "kernel": self.val(v._kernel), "walk": self.val(v.walk)}
        if t not in _SIMPLE:
            ref = _singleton(v)
            if ref:
                return {"fn": ref}
        if t in _SIMPLE or sp._is_record(v):
            return super().val(v)
        if isinstance(v, enum.Enum):
            return {"enum": "%s:%s" % (t.__module__, t.__qualname__), "name": v.name}
        if t.__module__.split(".")[0] == "ctypes" or t.__module__ == "hawk._core" or (isinstance(v, type) and v.__module__.split(".")[0] == "ctypes"):
            return {"inert": "%s.%s" % (getattr(v, "__module__", t.__module__), getattr(v, "__qualname__", t.__qualname__))}     # a loaded library: no part of a render
        for base in (tuple, list, dict, set, frozenset):
            if isinstance(v, base):
                # a subclass (read-only facts, a tagged sink tuple): the content as the base type, then its attributes
                st = {**{k: getattr(v, k) for k in sp._slots(t) if hasattr(v, k)}, **getattr(v, "__dict__", {})}
                of = self.val(dict(v) if base is dict else sorted(v, key=repr) if base in (set, frozenset) else list(v))
                return {"sub": "%s:%s" % (t.__module__, t.__qualname__), "base": base.__name__, "of": of, "attrs": {k: self.val(x) for k, x in sorted(st.items())}}
        if isinstance(v, (str, int, float, bytes)):
            self.unknown.append("subclass " + t.__module__ + "." + t.__qualname__)
            return {"u": t.__qualname__}
        if t is type(threading.Lock()):
            return {"lock": 1}      # a lock carries no value: the rebuild makes a new one
        if t is bytes:
            return {"b": v.hex()}
        if t is set:
            return {"set": sorted((self.val(x) for x in v), key=sp._key)}
        if isinstance(v, enum.Enum):
            return {"enum": "%s:%s" % (t.__module__, t.__qualname__), "name": v.name}
        if isinstance(v, pathlib.PurePath):
            return {"path": str(v), "cls": "%s:%s" % (t.__module__, t.__qualname__)}
        if t is types.MappingProxyType:
            return {"mp": self.val(dict(v))}
        if isinstance(v, type) or isinstance(v, types.FunctionType) or isinstance(v, types.BuiltinFunctionType):
            q = getattr(v, "__qualname__", "")
            ref = "%s:%s" % (getattr(v, "__module__", ""), q)
            if "<" not in q and _lookup(ref) is v:
                return {"fn": ref}
            self.closures.append(ref)      # a closure cannot travel: the rebuild puts a stub that raises if the render calls it
            return {"closure": ref}
        if isinstance(v, types.MethodType):
            return {"bm": self.val(v.__self__), "name": v.__func__.__name__}
        if t.__module__ == "builtins" or t.__module__.split(".")[0] in ("functools", "re", "threading", "weakref", "collections"):
            self.unknown.append(t.__module__ + "." + t.__qualname__)
            return {"u": t.__qualname__}
        return self.obj(v)

    def obj(self, v):
        i = self.oidx.get(id(v))
        if i is None:
            t = type(v)
            state = {}
            for s in sp._slots(t):
                if hasattr(v, s):
                    state[s] = getattr(v, s)
            state.update(getattr(v, "__dict__", {}))
            if not state and not sp._slots(t) and "__dict__" not in dir(v):
                self.unknown.append("opaque " + t.__module__ + "." + t.__qualname__)
            i = self.oidx[id(v)] = len(self.objs)
            self.hold.append(v)
            self.objs.append(None)
            self.objs[i] = {"c": t.__qualname__, "m": t.__module__, "a": {k: self.val(x) for k, x in sorted(state.items())}}
        return {"o": i}


_SING = {}


def _is_row(v):
    """True for a backend row the registry holds (it is found again by its id)."""
    try:
        from hawk import backends
        return isinstance(v, backends.Backend) and backends.need(v.id) is v
    except BaseException:
        return False


def _singleton(v):
    """`module:NAME` when v is the very object a module-level name of its class's module holds (a render may compare by identity)."""
    t = type(v)
    mod = sys.modules.get(t.__module__)
    if mod is None or t.__module__ == "builtins":
        return None
    m = _SING.get(t)
    if m is None:
        m = _SING[t] = {}
        for n, x in list(vars(mod).items()):
            if type(x) is t and not n.startswith("__"):
                m.setdefault(id(x), "%s:%s" % (t.__module__, n))
    return m.get(id(v))


def _lookup(ref):
    import importlib
    mod, _, q = ref.partition(":")
    try:
        o = importlib.import_module(mod)
        for p in q.split("."):
            o = getattr(o, p)
        return o
    except BaseException:
        return None


def _texts(res):
    """(field, sha256, text) for every str field of a result record."""
    out = []
    for f in getattr(res, "__dataclass_fields__", {}):
        x = getattr(res, f)
        if isinstance(x, str):
            out.append([f, hashlib.sha256(x.encode()).hexdigest(), len(x.splitlines())])
    return out


def _write(rec):
    try:
        pid = os.getpid()
        if _state["pid"] != pid:
            os.makedirs(_DIR, exist_ok=True)
            _state["fh"] = open(os.path.join(_DIR, f"t{pid}.jsonl"), "a")
            _state["pid"] = pid
        _state["seq"] += 1
        rec["seq"], rec["pid"], rec["test"] = _state["seq"], pid, _state["test"]
        _state["fh"].write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
        _state["fh"].flush()
    except BaseException:
        pass



# ---------------------------------------------------------------- the analysis record
# Every call from aether's writer code (aether.wings.*) and the cpu backend into raptor.wings.*, raptor.ir.* or raptor.rules.* that happens
# during a recorded render is recorded with its arguments and its result, through the unit's own encoder. The functions are wrapped where they
# live and in every module that imported them by name (a wrapper outside a render costs one call and one test).
_CALLEE_ROOTS = ("raptor.wings", "raptor.ir", "raptor.rules")
_ORIGINALS = {}
# Calls the writers' inputs depend on that are also made outside a render (while the kernel is traced); they are kept per test and
# attached to every unit of that test whose inputs hold an object they were called with.
_PLACE = "raptor.wings.base.wide:place"       # the placement pass: run by hawk's split before the render seam; its calls go to the units rendered from the stage
_OUTSIDE = {"raptor.wings.cpu.vectorise:lane_form_access", "raptor.wings.base.cost:is_library_call", "raptor.wings.base.split:compensated_target"}


def _raising(e):
    raise e
    yield


def _snap(v, depth=4):
    """The arguments of a call as they are when it is made: containers are copied (a caller may keep changing a set or list it passed);
    nodes and records are immutable and are kept."""
    if depth and type(v) in (set, list, dict, tuple):
        if type(v) is dict:
            return {x: _snap(y, depth - 1) for x, y in v.items()}
        if type(v) is tuple:
            return tuple(_snap(x, depth - 1) for x in v) if any(type(x) in (set, list, dict, tuple) for x in v) else v
        return type(v)(_snap(x, depth - 1) for x in v)
    return v


def _pend(qual, fn, a, k):
    keep_a, keep_k = _snap(a), _snap(k)
    err = res = None
    try:
        res = fn(*a, **k)
    except BaseException as e:
        err = e
    p = _state.setdefault("pending", [])
    if len(p) < 20000:
        p.append((_state["test"], qual, keep_a, keep_k, res, err))
    if err is not None:
        raise err
    return res


def _related(enc, a, k):
    stack, n = list(a) + list(k.values()), 0
    while stack and n < 300:
        x = stack.pop()
        n += 1
        if type(x) in (tuple, list):
            stack.extend(x[:60])
        elif id(x) in enc.index or id(x) in enc.oidx or id(x) in enc.memo:
            return True
    return False


def _attach_placement(c, rec):
    """The placement calls of this test whose returned sinks are held by this unit: sinks and facts given, sinks and planes returned, as nodes."""
    enc, out = c.enc, []
    for t, qual, a, k, res, err in _state.get("pending", ()):
        if qual != _PLACE or t != _state["test"] or err is not None or len(out) >= 20:
            continue
        if not (isinstance(res, tuple) and res and isinstance(res[0], tuple) and any(id(x) in enc.index for x in res[0])):
            continue
        try:
            out.append({"args": enc.val(list(a)), "kwargs": enc.val(k), "result": enc.val(res)})
        except BaseException:
            continue
    if out:
        rec["placement"] = out


def _attach_outside(c, rec):
    """The pending calls of this test that hold an object of this unit's inputs, recorded as the unit's own (marked `outside`)."""
    enc, made = c.enc, 0
    for t, qual, a, k, res, err in _state.get("pending", ()):
        if qual == _PLACE or t != _state["test"] or not _related(enc, a, k):
            continue
        try:
            ea, ek = enc.val(list(a)), enc.val(k)
            er = {"error": {"cls": type(err).__name__, "msg": str(err)}} if err is not None else {"result": enc.val(res)}
            rs = json.dumps(er, sort_keys=True, separators=(",", ":"))
        except BaseException:
            continue
        key = qual + "|" + _ser(ea) + "|" + _ser(ek)
        e = c.calls.get(key)
        if e is None:
            c.calls[key] = {"fn": qual, "caller": "outside the render", "args": ea, "kwargs": ek, **er, "n": 1, "varies": 0, "unknown": [], "_rs": rs, "outside": 1}
        else:
            e["n"] += 1
            e["outside"] = e.get("outside", 0) + 1
        made += 1
    rec["outside_calls"] = made


def _ser(v, cache={}):
    """JSON text of an encoded value, cached by the identity of memoized encodings."""
    c = cache.get(id(v))
    if c is None or c[0] is not v:
        c = cache[id(v)] = (v, json.dumps(v, sort_keys=True, separators=(",", ":")))
    return c[1]


class AnalysisCtx:
    def __init__(self, enc, mode="record", recorded=None):
        self.enc, self.mode, self.recorded = enc, mode, recorded or {}
        self.depth = 0
        self.calls, self.nested, self.sites = {}, collections.Counter(), collections.Counter()
        self.n0, self.o0, self.k0 = len(enc.nodes), len(enc.objs), len(enc.kinds)
        self.tally = collections.defaultdict(lambda: collections.Counter())
        self.diffs, self.seen_keys = [], set()
        self.fast, self.alive, self.near = {}, [], []
        self.facts = {}


def _writer_side(name):
    return name.startswith("aether.wings") or name == "ospreys.cpu.backend"


def _account(c, qual, key, caller, ea, ek, er, rs, unk, extra):
    """One analysis call: recorded under its key, or (check mode) compared with the recorded answer."""
    if c.mode == "record":
        e = c.calls.get(key)
        if e is None:
            c.calls[key] = {"fn": qual, "caller": caller, "args": ea, "kwargs": ek, **er, "n": 1, "varies": 0, "unknown": sorted(set(unk)), "_rs": rs, **extra}
        else:
            e["n"] += 1
            e["varies"] += e["_rs"] != rs
    else:                                            # check: the live call must find its recorded answer
        e = c.recorded.get(key)
        if e is None:
            c.tally[qual]["missing"] += 1
            if len(c.diffs) < 5:
                c.diffs.append((qual, "no recorded call with these arguments", key[:200]))
            if len(c.near) < 2:        # where the live arguments part from the closest recorded call of the same callee
                best = max((r for r in c.recorded if r.startswith(qual + "|")), key=lambda r: len(os.path.commonprefix([r, key])), default=None)
                if best is not None:
                    i = len(os.path.commonprefix([best, key]))
                    c.near.append((qual, i, "live " + key[max(0, i - 70):i + 90], "recorded " + best[max(0, i - 70):i + 90]))
        elif e["_rs"] == rs:
            c.tally[qual]["equal"] += 1
            c.seen_keys.add(key)
        else:
            c.tally[qual]["differs"] += 1
            c.seen_keys.add(key)
            if len(c.diffs) < 5:
                c.diffs.append((qual, "answer differs", "recorded %s | live %s" % (e["_rs"][:200], rs[:200])))


def _closure_proxy(c, parent_key, qual, fn):
    """A closure an analysis returned, as a function that records each call the writers make to it (same arguments, same answers); `made_by` names the call that returned it."""
    made = hashlib.sha1(parent_key.encode()).hexdigest()[:16]

    @functools.wraps(fn)
    def call(*a, **k):
        if _state.get("actx") is not c or c.depth:
            return fn(*a, **k)
        enc = c.enc
        u0, c0 = len(enc.unknown), len(enc.closures)
        try:
            ea, ek = enc.val(list(a)), enc.val({**k, "made_by": made})
            key = qual + "()|" + _ser(ea) + "|" + _ser(ek)
        except BaseException:
            return fn(*a, **k)
        c.depth += 1
        err = res = None
        try:
            res = fn(*a, **k)
        except BaseException as e:
            err = e
        finally:
            c.depth -= 1
        try:
            er = {"error": {"cls": type(err).__name__, "msg": str(err)}} if err is not None else {"result": enc.val(res)}
            rs = json.dumps(er, sort_keys=True, separators=(",", ":"))
        except BaseException:
            er, rs = {"result": {"u": "unencodable"}}, "unencodable"
        unk = enc.unknown[u0:]
        del enc.unknown[u0:]
        del enc.closures[c0:]
        _account(c, qual + "()", key, "closure", ea, ek, er, rs, unk, {"closure": 1})
        if err is not None:
            raise err
        return res
    return call


def _analysis_wrapper(qual, fn):
    def call(*a, **k):
        c = _state.get("actx")
        if c is None:
            if (qual in _OUTSIDE or qual == _PLACE) and _MODE == "record" and _state["test"] != "collection":
                return _pend(qual, fn, a, k)
            return fn(*a, **k)
        if c.depth:
            c.nested[qual] += 1
            return fn(*a, **k)
        if qual not in _OUTSIDE and not _writer_side(sys._getframe(1).f_globals.get("__name__", "")):
            return fn(*a, **k)
        enc = c.enc
        u0, c0 = len(enc.unknown), len(enc.closures)
        try:                    # the same objects asked the same thing again: the key is known without encoding them
            fk = (qual, tuple(x if type(x) in _SCALARS else id(x) for x in a), tuple(sorted((kk, x if type(x) in _SCALARS else id(x)) for kk, x in k.items())))
            hash(fk)
        except BaseException:
            fk = None
        key = c.fast.get(fk) if fk is not None else None
        ea = ek = None
        if key is None:
            try:
                ea, ek = enc.val(list(a)), enc.val(k)
                key = qual + "|" + _ser(ea) + "|" + _ser(ek)
            except BaseException as e:
                c.tally[qual]["encode_error"] += 1
                return fn(*a, **k)
            if fk is not None:
                c.fast[fk] = key
                c.alive.append((a, k))
        c.depth += 1
        err = res = shown = None
        gen = False
        try:
            res = shown = fn(*a, **k)
            if type(res) is types.GeneratorType:      # an answer that is a generator is recorded as the list it yields; the caller still gets a generator
                gen = True
                shown = list(res)
                res = (x for x in shown)
        except BaseException as e:
            err = e
            if gen:                       # the generator failed while drained: the caller meets the same error when it iterates
                res = _raising(e)
        finally:
            c.depth -= 1
        try:
            er = {"error": {"cls": type(err).__name__, "msg": str(err)}} if err is not None else {"result": enc.val(shown)}
            rs = _ser(er) if False else json.dumps(er, sort_keys=True, separators=(",", ":"))
        except BaseException:
            er, rs = {"result": {"u": "unencodable"}}, "unencodable"
        unk = enc.unknown[u0:]
        del enc.unknown[u0:]
        del enc.closures[c0:]
        _account(c, qual, key, sys._getframe(1).f_globals.get("__name__", ""), ea, ek, er, rs, unk, {"gen": 1} if gen else {})
        if err is None and type(shown) in (types.FunctionType, types.MethodType, functools.partial) and (type(shown) is not types.FunctionType or "<locals>" in shown.__qualname__):
            res = _closure_proxy(c, key, qual, res)        # what a returned closure answers when the writers call it is recorded too
        if err is not None and not gen:
            raise err
        return res
    call.__wrapped__ = fn
    call.__name__ = getattr(fn, "__name__", "call")
    call.__qualname__ = getattr(fn, "__qualname__", "call")
    call.__doc__ = fn.__doc__
    return call


def install_analysis_hooks():
    import importlib
    import pkgutil
    for root in _CALLEE_ROOTS:
        try:
            pk = importlib.import_module(root)
            for m in pkgutil.walk_packages(pk.__path__, root + "."):
                try:
                    importlib.import_module(m.name)
                except BaseException:
                    pass
        except BaseException:
            pass
    wrappers = {}
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith(_CALLEE_ROOTS):
            continue
        for n, f in list(vars(mod).items()):
            if type(f) is types.FunctionType and f.__module__ == name and not n.startswith("_") and id(f) not in wrappers:
                wrappers[id(f)] = (f, _analysis_wrapper(name + ":" + f.__qualname__, f))
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        for n, f in list(vars(mod).items()):
            w = wrappers.get(id(f)) if type(f) is types.FunctionType else None
            if w is not None and w[0] is f:
                try:
                    setattr(mod, n, w[1])
                except BaseException:
                    pass
    # the node constructor: where nodes are made during a render (the writer side building IR)
    try:
        from raptor.ir import nodes as _n
        orig = _n._construct if hasattr(_n, "_construct") else None
        if orig is not None:
            def construct(*a, **k):
                c = _state.get("actx")
                if c is not None and c.mode == "record":
                    f = sys._getframe(1)
                    while f is not None and (f.f_globals.get("__name__", "").startswith(("raptor.ir", "raptor._")) or f.f_code.co_filename.startswith("<")):
                        f = f.f_back
                    if f is not None:
                        c.sites["%s:%d" % (f.f_globals.get("__name__", "?"), f.f_lineno)] += 1
                return orig(*a, **k)
            _n._construct = construct
    except BaseException:
        pass


def _analysis_begin(enc, rec, mode="record", recorded=None):
    c = AnalysisCtx(enc, mode, recorded)
    _state["actx"] = c
    return c


def _analysis_end(c, rec):
    _state["actx"] = None
    if c.mode != "record":
        return
    enc = c.enc
    rec["analyses"] = [{k: v for k, v in e.items() if k != "_rs"} for e in c.calls.values()]
    if c.facts:
        rec["facts"] = {q.split(":")[1]: [list(x) for x in d.values()] for q, d in c.facts.items()}
    rec["analysis_nested"] = dict(c.nested)
    rec["node_sites"] = dict(c.sites)
    rec["analysis_extra"] = {"nodes": enc.nodes[c.n0:], "objs": enc.objs[c.o0:], "kinds": enc.kinds[c.k0:], "offsets": [c.n0, c.o0, c.k0]}
    rec["analysis_unknown"] = sum(1 for e in c.calls.values() if e["unknown"])



_OPTS = {}


def _opts_wrapper(orig):
    """raptor._core.canonical for the whole session: the options each result digest was made with (declared slots, force, exempt planes)."""
    def canonical(sinks, declared, force, env, exempt):
        res = orig(sinks, declared, force, env, exempt)
        try:
            if declared or force is not None or exempt:
                e = TextEnc()
                o = {"declared": e.val(list(declared)), "force": e.val(force), "exempt": e.val(list(exempt))}
                k = json.dumps(o, sort_keys=True, separators=(",", ":"))
            else:
                o = k = "default"
            _state["last_opt"] = k
            d = _OPTS.setdefault(res[7], {})
            if k in d:
                d[k]["n"] += 1
            else:
                d[k] = {"opts": o, "n": 1}
        except BaseException:
            pass
        return res
    canonical.__wrapped__ = orig
    return canonical


def _walk_props(enc):
    """For each Walk the unit holds, keyed by digest: every property of the Walk class (the derived facts the writers read), encoded."""
    out = {}
    for d, w in enc.walks.items():
        props = {}
        for n, p in sorted(vars(type(w)).items()):
            if isinstance(p, property):
                try:
                    props[n] = {"result": enc.val(getattr(w, n))}
                except BaseException as e:
                    props[n] = {"error": {"cls": type(e).__name__, "msg": str(e)}}
        out[d] = props
    return out


_WALKSET = {}
def load_excluded(d):
    """{unit key: reason} for this recording from corpus/EXCLUDED_UNITS.tsv (columns: recording, unit key, reason, test)."""
    out, path = {}, os.path.join(os.path.dirname(os.path.abspath(d)), "EXCLUDED_UNITS.tsv")
    if os.path.exists(path):
        for line in open(path):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3 and f[0] == os.path.basename(os.path.abspath(d)):
                out[f[1]] = f[2]
    return out


# Machine facts a writer function reads at render time (a hardware card): every call during a render is recorded with its answer, so a replay
# can serve the answer instead of reading the card; a test that replaces such a function is recorded the same way.
_FACTS_OF = {"aether.wings.base.rtstore": ("select_weight",)}


def _fact_recorder(qual, fn):
    def call(*a, **k):
        r = fn(*a, **k)
        c = _state.get("actx")
        if c is not None and c.mode == "record":
            enc = c.enc
            u0, c0 = len(enc.unknown), len(enc.closures)
            try:
                ea, ek, er = enc.val(list(a)), enc.val(k), enc.val(r)
                entry = [ea, er] + ([ek] if k else [])
                c.facts.setdefault(qual, {}).setdefault(_ser(ea) + "|" + _ser(ek) + "|" + json.dumps(er, sort_keys=True, separators=(",", ":")), entry)
            except BaseException:
                pass
            del enc.unknown[u0:]
            del enc.closures[c0:]
        return r
    call._fact_rec = True
    call.__wrapped__ = fn
    call.__name__ = getattr(fn, "__name__", "call")
    return call


class _FactModule(types.ModuleType):
    """A module whose recorded functions stay recorded when a test replaces them."""

    def __setattr__(self, n, v):
        if n in _FACTS_OF.get(self.__name__, ()) and callable(v) and not getattr(v, "_fact_rec", False):
            v = _fact_recorder("%s:%s" % (self.__name__, n), v)
        super().__setattr__(n, v)


def importlib_import(name):
    import importlib
    return importlib.import_module(name)


def _install_facts():
    for mn, names in _FACTS_OF.items():
        mod = importlib_import(mn)
        for n in names:
            f = getattr(mod, n)
            _patch_everywhere(f, _fact_recorder("%s:%s" % (mn, n), f))
        mod.__class__ = _FactModule


_SNAP, _SNAPPED = [], set()


def _snapshot_new():
    """The functions and methods of the writers' modules (aether wings, raptor wings), as they are when first seen: (dict holding it, name, object, label)."""
    for mn, mod in list(sys.modules.items()):
        if mod is None or mn in _SNAPPED or not mn.startswith(("aether.wings", "raptor.wings")):
            continue
        _SNAPPED.add(mn)
        for n, x in list(vars(mod).items()):
            if type(x) is types.FunctionType:
                _SNAP.append((vars(mod), n, x, "%s:%s" % (mn, n)))
            elif isinstance(x, type) and x.__module__ == mn:
                for cn, cx in list(x.__dict__.items()):
                    if isinstance(cx, (types.FunctionType, staticmethod, classmethod, property)):
                        _SNAP.append((x.__dict__, cn, cx, "%s:%s.%s" % (mn, x.__qualname__, cn)))


def _patched():
    """Names of the writers' functions that are not the objects first seen: a test replaced them (a patched render is no fair oracle)."""
    _snapshot_new()
    return sorted({lab for d, n, x, lab in _SNAP if d.get(n) is not x})


def _walk_wrapper(orig):
    """raptor.ir.walk.canonical: remembers, per Walk made, the option set its digest was made with."""
    def canonical(*a, **k):
        _state["last_opt"] = None
        w = orig(*a, **k)
        _WALKSET[id(w)] = (w, _state.get("last_opt"))
        return w
    canonical.__wrapped__ = orig
    return canonical


def _used_opts(w):
    """The option set (`default` or declared/force/exempt) the given Walk was made with; None when the Walk was made outside this test."""
    e = _WALKSET.get(id(w))
    if e is None or e[0] is not w or e[1] is None:
        return None
    d = _OPTS.get(w.digest, {}).get(e[1])
    return None if d is None else d["opts"]


def _walk_opts(inp):
    """For each Walk record held by a unit's inputs, keyed by its digest: the option sets canonical() was called with to make it."""
    out, seen = {}, set()
    stack = [inp.get(k) for k in ("lowered", "options", "args", "kwargs", "source", "self")] + [o["a"] for o in inp.get("objs", []) if o]
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            if id(v) in seen:
                continue
            seen.add(id(v))
            if v.get("r") == "Walk":
                d = (v["v"].get("digest") or {}).get("s")
                if d in _OPTS:
                    out[d] = list(_OPTS[d].values())
            stack.extend(v.values())
        elif isinstance(v, list):
            if id(v) in seen:
                continue
            seen.add(id(v))
            stack.extend(v)
    return out


def _wrap_source(orig):
    """The common function `render_source`, recorded only when it is called directly (the per-target entry records the calls it makes)."""
    def render_source(*args, **kwargs):
        if _state.get("in_render") and _MODE == "record" and _state.get("unit") is not None and "inner" not in _state["unit"]:
            # the call the per-target entry makes: its arguments are the one form the compiled entry takes (own node table)
            u = _state["unit"]
            e = TextEnc()
            try:
                u["inner"] = {"args": e.val(list(args)), "kwargs": e.val(kwargs), "nodes": e.nodes, "kinds": e.kinds, "objs": e.objs}
            except BaseException as x:
                u["inner"] = {"encode_error": repr(x)}
            try:
                res = orig(*args, **kwargs)
            except BaseException as x:
                u["inner"]["error"] = {"cls": type(x).__name__, "msg": str(x)}
                raise
            u["inner"]["texts"] = _texts(res)
            u["inner"]["output"] = {f: getattr(res, f) for f in ("text", "body")}
            return res
        if _state.get("in_render") or _MODE != "record":
            return orig(*args, **kwargs)
        target = args[3] if len(args) > 3 else kwargs.get("target")
        rec = {"seam": "aether.wings.base.wrapper.render_source", "target": getattr(target, "id", "?"), "unit": args[0] if args else kwargs.get("name")}
        enc = TextEnc()
        try:
            inputs = {"args": enc.val(list(args)), "kwargs": enc.val(kwargs)}
            inputs.update({"nodes": list(enc.nodes), "kinds": list(enc.kinds), "objs": list(enc.objs)})
        except BaseException as e:
            inputs = {"encode_error": repr(e)}
            enc.unknown.append("encode_error")
        rec["inputs"] = inputs
        return _finish(rec, enc, lambda: orig(*args, **kwargs))
    render_source.__wrapped__ = orig
    return render_source


def _finish(rec, enc, call):
    err = res = None
    try:
        rec["walk_opts"] = _walk_opts(rec["inputs"])
    except BaseException:
        rec["walk_opts"] = {}
    if rec["seam"].endswith("render_source"):
        rec["inner"] = "inputs"                    # recorded directly at the inner seam: its inputs are the inner arguments
    actx = _analysis_begin(enc, rec) if _ANALYSES else None
    if actx is not None:
        _state["actx"] = None
        try:
            rec["walk_props"] = _walk_props(enc)
            _attach_outside(actx, rec)
            _attach_placement(actx, rec)
        finally:
            _state["actx"] = actx
    rec["walk_opts_used"] = {d: _used_opts(w) for d, w in enc.walks.items()}
    pt = _patched()
    if pt:
        rec["patched"] = pt
    try:
        res = call()
    except BaseException as e:
        err = e
    if actx is not None:
        _analysis_end(actx, rec)
    try:
        if err is not None:
            rec["error"] = {"cls": type(err).__name__, "msg": str(err)}
        else:
            out = TextEnc()
            rec["output"] = {"value": out.val(res), "nodes": out.nodes, "kinds": out.kinds, "objs": out.objs}
            rec["texts"] = _texts(res)
            enc.unknown += out.unknown
            enc.closures += out.closures
    except BaseException as e:
        rec["record_error"] = repr(e)
    rec["closures"] = sorted(set(enc.closures))
    rec["unknown"] = sorted(set(enc.unknown))
    rec["unmirrorable"] = bool(enc.unknown) or "record_error" in rec
    _write(rec)
    if err is not None:
        raise err
    return res


def _wrap_literal(orig):
    """`literal_table`, recorded only when it is called directly (the CUDA backend's own call is inside a recorded seam)."""
    def literal_table(source):
        if _state.get("in_render") or _MODE != "record":
            return orig(source)
        rec = {"seam": "aether.wings.base.wrapper.literal_table", "target": getattr(source, "target", "?"), "unit": getattr(source, "entry", None)}
        enc = TextEnc()
        try:
            inputs = {"source": enc.val(source), "args": enc.val([source]), "kwargs": enc.val({})}      # the call is literal_table(source): the one argument is the unit's source
            inputs.update({"nodes": list(enc.nodes), "kinds": list(enc.kinds), "objs": list(enc.objs)})
        except BaseException as e:
            inputs = {"encode_error": repr(e)}
            enc.unknown.append("encode_error")
        rec["inputs"] = inputs
        return _finish(rec, enc, lambda: orig(source))
    literal_table.__wrapped__ = orig
    return literal_table


def _patch_everywhere(orig, new):
    for m in list(sys.modules.values()):
        try:
            for n, x in list(vars(m).items()):
                if x is orig:
                    setattr(m, n, new)
        except BaseException:
            pass


def _wrap(orig, seam):
    def render(self, lowered, options):
        rec = {"seam": seam, "target": self.id, "unit": getattr(lowered, "name", None)}
        enc = TextEnc()
        try:
            inputs = {"lowered": enc.val(lowered), "options": enc.val(dict(options))}
            try:
                from hawk import backends as _b
                known = _b.need(self.id) is self
            except BaseException:
                known = False
            if not known:           # a backend that is not a registered row (a test's own): its state travels too
                inputs["self"] = enc.val(self)
            inputs.update({"nodes": list(enc.nodes), "kinds": list(enc.kinds), "objs": list(enc.objs)})
        except BaseException as e:
            inputs = {"encode_error": repr(e)}
            enc.unknown.append("encode_error")
        rec["inputs"] = inputs
        _state["in_render"] = True
        _state["unit"] = rec
        try:
            return _finish(rec, enc, lambda: orig(self, lowered, options))
        finally:
            _state["in_render"] = False
            _state["unit"] = None
    render.__wrapped__ = orig
    return render


def pytest_configure(config):
    if _MODE != "record" or not _DIR:
        return
    from raptor.ir.nodes import Node
    sp._NODE[0] = Node
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    from hawk import backends
    backends.Backend.render = _wrap(backends.Backend.render, "hawk.backends.Backend.render")
    from aether.wings.base import wrapper as _w
    _w.render_source = _wrap_source(_w.render_source)
    from raptor import _core as _rc
    _rc.canonical = _opts_wrapper(_rc.canonical)
    from raptor.ir import walk as _wk
    _patch_everywhere(_wk.canonical, _walk_wrapper(_wk.canonical))
    lt = _w.literal_table
    _patch_everywhere(lt, _wrap_literal(lt))
    _install_facts()
    if _ANALYSES:
        install_analysis_hooks()
    try:
        from ospreys.c import backend as cb
        cb.C.render = _wrap(cb.C.render, "ospreys.c.backend.C.render")
    except ImportError:
        pass


def pytest_runtest_protocol(item, nextitem):
    _state["test"] = item.nodeid
    _state["pending"] = []
    _WALKSET.clear()
    return None


def pytest_runtest_logfinish(nodeid, location):
    _state["test"] = "collection"
