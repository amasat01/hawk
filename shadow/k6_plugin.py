# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""pytest plugin: the kernel corpus (slice L0). One pass records, per build call, the kernel BEFORE lowering, the units the lowering
produced, and (through text_plugin, which must be loaded too) every render-seam record tagged with the call id and the unit name.

Use: TEXT_MODE=record TEXT_DIR=<dir> SHADOW_MODE=off pytest -p shadow_plugin -p text_plugin -p k6_plugin ...
Also b<pid>.jsonl: one line per device launch's block resolution (eagle.launch._resolved_block), after a header record naming the join key.
Files in $TEXT_DIR: k<pid>.jsonl (one line per build call, written when the call ends) beside text_plugin's t<pid>.jsonl, whose records
gain `call` (the build-call id, `<pid>.<n>`) and `kunit` (the unit being lowered when the render ran); `call` is null outside a build call.
Hooks (wrapping, from here, at session start; hawk's tree is not edited): build_bundle, build, emit_source (open a call), _emitted,
_expand_segmented (the kernels, kind-resolved), _expand_split (the resolved spec and the stage blocks), lower (one unit), _draft (its sidecar),
_note (memo hits). The unit memo is emptied before every test."""
import contextvars
import inspect
import json
import os
import sys
import hashlib
import re
import threading

import shadow_plugin as sp
import text_plugin as tp

_ON = tp._MODE == "record" and bool(tp._DIR)
_CUR = contextvars.ContextVar("k6_call", default=None)
_seq = [0]
_fh = {"pid": None, "fh": None}
_WOPT = {}          # id(Walk) -> (Walk, declared, force_access, exempt_planes): the options canonical() made it with


def _enc_val(c, v):
    try:
        return c["enc"].val(v)
    except BaseException as e:
        c["enc"].unknown.append("encode:" + type(e).__name__)
        return {"u": type(v).__name__}


def _wopts(k):
    w = k.walk
    e = _WOPT.get(id(w))
    if e is not None and e[0] is w:
        return e[1], e[2], e[3], "recorded"
    from hawk.trace.kernel import _declared
    try:        # a kernel traced before the plugin was installed (a conftest import): rebuild its walk the way Kernel does; the digest must agree
        from raptor.wings.base import cells as _cw
        _tl.last = None
        w2 = _cw.walk(k.sinks, _declared(getattr(k, "planes", {}) or {}))
        if getattr(_tl, "last", None) is not None and w2.digest == w.digest:
            _WOPT[id(w)] = (w, *_tl.last)
            return (*_tl.last, "recomputed")
    except BaseException:
        pass
    return (), None, (), "missing"


def _roots(k, out, depth=0):
    out.extend(list(k.sinks))
    out.extend(_wopts(k)[0])
    one = getattr(k, "one_step", None)
    if one is not None and depth < 3:
        _roots(one, out, depth + 1)
    for m in getattr(k, "members", None) or ():
        if hasattr(m, "sinks") and hasattr(m, "walk") and depth < 3:
            _roots(m, out, depth + 1)


def _member(c, k, depth=0):
    declared, force, exempt, how = _wopts(k)
    r = {"cls": type(k).__name__, "name": k.name, "walk_digest": k.walk.digest,
         "sinks": [_enc_val(c, s) for s in k.sinks], "declared": _enc_val(c, list(declared)), "force": _enc_val(c, force),
         "exempt": _enc_val(c, list(exempt)), "opts": how}
    sk = k.sinks
    if type(sk).__name__ == "Derived":
        r["derived"] = {a: _enc_val(c, getattr(sk, a, None)) for a in ("primal", "kind", "wrt", "primal_unit", "jump", "_fused_steps")}
    return r


def _kernel(c, k, kind=None):
    r = _member(c, k)
    r.update({"kind": _enc_val(c, kind if kind is not None else getattr(k, "kind", None)), "own_kind": _enc_val(c, getattr(k, "kind", None)),
              "group": getattr(k, "group", 1), "lane_space": _enc_val(c, getattr(k, "lane_space", None)),
              "steps": _enc_val(c, getattr(k, "steps", None)), "planes": _enc_val(c, getattr(k, "planes", None)),
              "time": getattr(k, "time", None), "time_step": getattr(k, "time_step", None), "jump": _enc_val(c, getattr(k, "jump", None)),
              "source_digest": getattr(k, "source_digest", None), "fused_body": hasattr(k, "fused_body"),
              "segmented": any(d.strategy == "segmented" for d in k.walk.dispatches)})
    if type(k.sinks).__name__ == "Derived" and not getattr(k, "planes", None):     # a derivative's planes are derived from its primal's at build time
        try:
            from hawk.trace.kernel import derived_axes
            r["derived_planes"] = _enc_val(c, derived_axes(k.walk, getattr(k.sinks, "source", None)))
        except BaseException as e:
            c["enc"].unknown.append("derived_planes:" + type(e).__name__)
    one = getattr(k, "one_step", None)
    r["one_step"] = None if one is None else _member(c, one)
    mem = getattr(k, "members", None)
    if mem is not None:
        r["members"] = [_member(c, m) if hasattr(m, "sinks") and hasattr(m, "walk") else {"opaque": type(m).__name__} for m in mem]
    return r


def _kernels_in(c, kernels, kinds, stage):
    roots = []
    for k in kernels:
        try:
            _roots(k, roots)
        except BaseException:
            pass
    try:
        c["enc"].dag(roots)
    except BaseException as e:
        c["enc"].unknown.append("dag:" + type(e).__name__)
    for k, kd in zip(kernels, kinds):
        try:
            rec = _kernel(c, k, kd)
        except BaseException as e:
            rec = {"name": getattr(k, "name", None), "record_error": repr(e)}
        rec["stage"] = stage
        c["kernels"].append(rec)


def _new(entry, a, k, orig):
    _seq[0] += 1
    c = {"rec": "call", "call": f"{os.getpid()}.{_seq[0]}", "test": tp._state["test"], "entry": entry, "enc": tp.TextEnc(),
         "kernels": [], "units": [], "memo_hit": False, "cur": None}
    try:
        bound = inspect.signature(orig).bind_partial(*a, **k).arguments
    except BaseException:
        bound = {}
    c["args"] = {n: _enc_val(c, v) for n, v in bound.items() if n not in ("kernels", "kernel", "directory", "cache_dir", "layout_sizes_override")}
    c["explicit_kind"] = bound.get("kind") is not None
    c["exports_override"] = _enc_val(c, bound.get("layout_sizes_override"))
    try:
        from hawk.trace import steps as _steps
        c["auto_k_max"] = _steps.AUTO_K_MAX          # the auto steps cap in force when the call starts
    except BaseException as e:
        c["enc"].unknown.append("auto_k_max:" + type(e).__name__)
    c["fusion_rows"] = []                             # the fusion row each chain cut of the call used (a test may set its own)
    c["patched"] = _patched_now()
    c["box_env"] = os.environ.get("HAWK_BOX")
    return c


def _flush(c):
    enc = c.pop("enc")
    c.pop("cur", None)
    for u in c.get("units", ()):
        _KCALL[u.get("unit")] = c["call"]
    c.update({"nodes": enc.nodes, "kinds": enc.kinds, "objs": enc.objs, "unknown": sorted(set(enc.unknown)), "pid": os.getpid()})
    try:
        pid = os.getpid()
        if _fh["pid"] != pid:
            os.makedirs(tp._DIR, exist_ok=True)
            _fh["fh"], _fh["pid"] = open(os.path.join(tp._DIR, f"k{pid}.jsonl"), "a"), pid
        _fh["fh"].write(json.dumps(c, sort_keys=True, separators=(",", ":"), default=str) + "\n")
        _fh["fh"].flush()
    except BaseException:
        pass


def _close_orphan():
    """A call opened by a direct lower() (a test that lowers and renders itself) lasts until the next call or the end of the test."""
    c = _CUR.get()
    if c is not None and c.get("orphan"):
        _CUR.set(None)
        _flush(c)


def _entry(orig, entry, kernels_of):
    def call(*a, **k):
        if _CUR.get() is not None and not _CUR.get().get("orphan"):      # build() calls _emitted: one call, one id
            return orig(*a, **k)
        _close_orphan()
        c = _new(entry, a, k, orig)
        tok = _CUR.set(c)
        try:
            return orig(*a, **k)
        except BaseException as e:
            c["error"] = {"cls": type(e).__name__, "msg": str(e)[:400]}
            raise
        finally:
            _CUR.reset(tok)
            _flush(c)
    call.__wrapped__ = orig
    return call


def _step(orig, hook):
    def call(*a, **k):
        c = _CUR.get()
        if c is not None:
            try:
                hook(c, a, k)
            except BaseException as e:
                c["enc"].unknown.append("hook:" + type(e).__name__)
        return orig(*a, **k)
    call.__wrapped__ = orig
    return call


def _h_segmented(c, a, k):
    kernels, names, kinds = a[:3]
    if not any(r["stage"] == "expand_segmented" for r in c["kernels"]):
        _kernels_in(c, kernels, kinds, "expand_segmented")


def _h_emitted(c, a, k):
    kernel, name, mode, kind = a[:4]
    _kernels_in(c, [kernel], [kind], "emitted")
    c["emitted"] = {"name": name, "mode": mode, "targets": _enc_val(c, a[4])}


def _wrap_split(orig):
    def call(kernels, names, kinds, **k):
        c = _CUR.get()
        out = orig(kernels, names, kinds, **k)
        if c is not None:
            try:
                c["spec"] = _enc_val(c, k.get("spec"))
                smode = k.get("smode")
                c["split_in"] = {"names": list(names), "smode": getattr(smode, "id", None), "device": k.get("device"), "device_only": k.get("device_only")}
                c["split_out_names"] = list(out[1])
                c["split"] = _enc_val(c, out[3])
            except BaseException as e:
                c["enc"].unknown.append("split:" + type(e).__name__)
        return out
    call.__wrapped__ = orig
    return call


def _wrap_segmented(orig):
    def call(kernels, names, kinds):
        c = _CUR.get()
        if c is not None:
            try:
                _h_segmented(c, (kernels, names, kinds), {})
            except BaseException as e:
                c["enc"].unknown.append("seg:" + type(e).__name__)
        out = orig(kernels, names, kinds)
        if c is not None:
            c["segmented_out_names"] = list(out[1])
        return out
    call.__wrapped__ = orig
    return call


def _wrap_lower(orig):
    def call(kernel, name, smode, kind, targets, spec):
        c = _CUR.get()
        if c is None:
            _seq[0] += 1
            c = {"rec": "call", "call": f"{os.getpid()}.{_seq[0]}", "test": tp._state["test"], "entry": "lower", "orphan": True, "enc": tp.TextEnc(),
                 "kernels": [], "units": [], "memo_hit": False, "cur": None, "args": {}, "explicit_kind": kind is not None,
                 "exports_override": None, "patched": _patched_now(), "box_env": os.environ.get("HAWK_BOX")}
            _CUR.set(c)
            try:
                _kernels_in(c, [kernel], [kind], "lower")
            except BaseException as e:
                c["enc"].unknown.append("lower:" + type(e).__name__)
        u = {"unit": name, "digest": None, "cls": type(kernel).__name__, "smode": getattr(smode, "id", None),
             "targets": list(targets) if isinstance(targets, (list, tuple)) else _enc_val(c, targets)}
        try:
            u["digest"] = kernel.walk.digest
            u["group"] = getattr(kernel, "group", 1)
            u["team"] = getattr(kernel, "team", None) is not None
            u["one_step"] = getattr(kernel, "one_step", None) is not None
            u["members"] = len(getattr(kernel, "members", None) or ())
            if "spec" not in c:
                c["spec"] = _enc_val(c, spec)
        except BaseException as e:
            u["record_error"] = repr(e)
        u["box_env"] = os.environ.get("HAWK_BOX")         # "0" = the old lowering was told to build without a box
        c["units"].append(u)
        c["cur"] = name
        try:
            lowered = orig(kernel, name, smode, kind, targets, spec)
        except BaseException as e:
            u["error"] = {"cls": type(e).__name__, "msg": str(e)[:400]}
            raise
        try:
            u["segment"] = _enc_val(c, lowered.segment)
            u["active"] = _enc_val(c, lowered.active)
            u["box"] = _enc_val(c, lowered.box)
            u["rows"] = lowered.rows
            u["fused_hook"] = lowered.body is not None or hasattr(kernel, "fused_body")
            u["kind"] = _enc_val(c, lowered.kind)
        except BaseException as e:
            u["record_error"] = repr(e)
        return lowered
    call.__wrapped__ = orig
    return call


def _wrap_draft(orig):
    def call(kernel, lowered, sources, options, *, targets, derivative, split=None):
        c = _CUR.get()
        d = orig(kernel, lowered, sources, options, targets=targets, derivative=derivative, split=split)
        if c is not None:
            try:
                u = next((x for x in reversed(c["units"]) if x["unit"] == d.name and "sidecar" not in x), None)
                if u is not None:
                    u["sidecar"] = d.sidecar
                    u["files"] = [[f.name, f.digest] for f in d.files]
                    u["stage_block"] = None if split is None else _enc_val(c, split)
                    u["derivative"] = _enc_val(c, derivative)
            except BaseException as e:
                c["enc"].unknown.append("draft:" + type(e).__name__)
        return d
    call.__wrapped__ = orig
    return call


def _wrap_note(orig):
    def call(kind, message):
        c = _CUR.get()
        if c is not None and kind == "memo_hits":
            c["memo_hit"] = True
        return orig(kind, message)
    call.__wrapped__ = orig
    return call


_tl = threading.local()


def _wrap_cuts(orig):
    def cuts(members, card, scalar_type, samples):
        c = _CUR.get()
        if c is not None:
            try:
                c["fusion_rows"].append(_enc_val(c, dict(card)))
            except BaseException as e:
                c["enc"].unknown.append("fusion_rows:" + type(e).__name__)
        return orig(members, card, scalar_type, samples)
    cuts.__wrapped__ = orig
    return cuts


def _walk_opts_wrapper(orig):
    def canonical(sinks, declared=(), *, force_access=None, exempt_planes=()):
        w = orig(sinks, declared, force_access=force_access, exempt_planes=exempt_planes)
        opts = (tuple(declared), force_access, tuple(exempt_planes))
        _tl.last = opts
        _WOPT[id(w)] = (w, *opts)
        return w
    canonical.__wrapped__ = orig
    return canonical


def _cells_walk_wrapper(orig):
    """The kernel's walk is `replace(found, dispatches=...)` of the LAST canonical() call (a new object): it takes that call's options."""
    def walk(sinks, declared=(), **kw):
        _tl.last = None
        w = orig(sinks, declared, **kw)
        if getattr(_tl, "last", None) is not None:
            _WOPT[id(w)] = (w, *_tl.last)
        return w
    walk.__wrapped__ = orig
    return walk


# the unit key of the per-unit corpus, as shadow/merge_text.py computes it (merge_text runs its main on import, so the few lines are repeated)
_LEFT_OUT = ("primal_unit", "_hawk_source", "host_entry")
_TMP = re.compile(r"/[^\"]*?(?:pytest-\d+|basetemp)/popen-gw\d+")
_RUN = re.compile(r"/snap/[^/\"]+/")


def _scrub(v):
    if isinstance(v, dict):
        return {k: _scrub(x) for k, x in v.items() if k not in _LEFT_OUT}
    return [_scrub(x) for x in v] if isinstance(v, list) else v


def _unit_key(tg, inputs):
    text = json.dumps([tg, _scrub(inputs)], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(_RUN.sub("/snap/<run>/", _TMP.sub("<tmp>", text)).encode()).hexdigest()


def _select_weight_wrapper(orig):
    def select_weight(*a, **k):
        r = orig(*a, **k)
        c = _CUR.get()
        if c is not None:
            c.setdefault("select_weight", []).append({"args": _enc_val(c, list(a)), "kwargs": _enc_val(c, k), "result": _enc_val(c, r)})
        return r
    select_weight.__wrapped__ = orig
    return select_weight


def _seams():
    """The functions whose replacement by a test makes an old render no oracle: name -> object."""
    import aether.wings.base.render as _r
    import aether.wings.base.wrapper as _w
    from hawk import backends
    from hawk.artifact import bundle as b
    out = {"hawk.backends.Backend.render": backends.Backend.__dict__.get("render"), "aether.wings.base.wrapper.render_source": _w.render_source,
           "aether.wings.base.wrapper.literal_table": _w.literal_table, "aether.wings.base.render.render_body": _r.render_body,
           "hawk.artifact.bundle._emit": b._emit, "hawk.artifact.bundle.lower": b.lower, "hawk.artifact.bundle._draft": b._draft}
    try:
        from ospreys.c import backend as cb
        out["ospreys.c.backend.C.render"] = cb.C.__dict__.get("render")
    except ImportError:
        pass
    return out


_SEAM0 = {}


def _patched_now():
    try:
        if not _SEAM0:
            _SEAM0.update(_seams())
        now = _seams()
        names = [n for n, o in _SEAM0.items() if now.get(n) is not o]
        names += ["writer:" + n for n in tp._patched()]
        return sorted(set(names))
    except BaseException as e:
        return ["check_error:" + type(e).__name__]


def _text_write(orig):
    def write(rec):
        c = _CUR.get()
        rec["call"] = None if c is None else c["call"]
        rec["kunit"] = None if c is None else c["cur"]
        try:        # the key of the per-unit corpus (merge_text.unit_key), so a replay can look the unit up in the accepted list
            if rec.get("target") == "host" and "inputs" in rec:
                rec["ukey"] = _unit_key("host", rec["inputs"])
        except BaseException:
            rec["ukey"] = None
        return orig(rec)
    return write


def _patch(orig, new):
    """Module variables (text_plugin's helper) and the namespaces hawk's wing views copied the function into when they loaded."""
    import gc
    tp._patch_everywhere(orig, new)
    for o in gc.get_objects():
        try:
            if type(o).__name__ == "_View" and o.__module__ == "hawk.wings" and o._ns is not None:
                for n, x in list(vars(o._ns).items()):
                    if x is orig:
                        setattr(o._ns, n, new)
        except BaseException:
            pass


def _wrap_plugin_entry(orig):
    """The compile format of a call's device image ("ptx", "cubin", "fatbin"), as the bundle's manifest states it: read off
    the compiled bytes by the build, so only a machine that compiles records it. A call whose plugins disagree keeps the first
    format and notes the disagreement."""
    def call(*a, **k):
        c = _CUR.get()
        if c is not None:
            try:
                bound = inspect.signature(orig).bind(*a, **k)
                bound.apply_defaults()
                fmt = str(bound.arguments["fmt"]).lower()
                if c.get("image_format") is None:
                    c["image_format"] = fmt
                elif c["image_format"] != fmt:
                    c["enc"].unknown.append("image_format:" + fmt)
            except BaseException as e:
                c["enc"].unknown.append("image_format:" + type(e).__name__)
        return orig(*a, **k)
    call.__wrapped__ = orig
    return call


_bh = {"pid": None, "fh": None}
_KCALL = {}         # unit/kernel name -> id of the last build call of THIS test that made it (cleared at each test start)
_SYMS = {}          # raw function handle -> kernel symbol, filled where eagle loads the function (_core.module_function)
_B_HEADER = {"rec": "header", "join": ["call", "kernel"], "call": "build-call id (<pid>.<n>) of the k-file line, null outside a build call",
             "kernel": "kernel symbol as loaded by _core.module_function (null when unknown)",
             "call_from": "last build call of this test that made this kernel",
             "note": "'passed' is absent when the caller passed no block; 'declared' is null when absent; 'resolved' is the block returned"}


def _b_write(rec):
    pid = os.getpid()
    if _bh["pid"] != pid:
        os.makedirs(tp._DIR, exist_ok=True)
        _bh["fh"], _bh["pid"] = open(os.path.join(tp._DIR, f"b{pid}.jsonl"), "a"), pid
        _bh["fh"].write(json.dumps(_B_HEADER, sort_keys=True, separators=(",", ":")) + "\n")
    _bh["fh"].write(json.dumps(rec, sort_keys=True, separators=(",", ":"), default=str) + "\n")
    _bh["fh"].flush()


def _wrap_resolved_block(orig):
    def _resolved_block(fn, n, block, device=-1, declared=None):
        out = orig(fn, n, block, device, declared)
        try:
            c = _CUR.get()
            sym = _SYMS.get(int(fn)) if isinstance(fn, int) else None
            rec = {"call": _KCALL.get(sym) if c is None else c["call"], "test": tp._state["test"], "kernel": sym,
                   "n": n, "declared": declared, "resolved": out}
            if block is not None:
                rec["passed"] = block
            _b_write(rec)
        except BaseException:
            pass
        return out
    _resolved_block.__wrapped__ = orig
    return _resolved_block


def _wrap_module_function(orig):
    def module_function(module, name, *a, **k):
        h = orig(module, name, *a, **k)
        try:
            _SYMS[int(h)] = name
        except BaseException:
            pass
        return h
    module_function.__wrapped__ = orig
    return module_function


def _patch_eagle_blocks():
    import importlib
    try:
        el = sys.modules.get("eagle.launch") or importlib.import_module("eagle.launch")
        core = importlib.import_module("eagle._core")
    except BaseException:
        return
    orig = el._resolved_block
    new = _wrap_resolved_block(orig)
    tp._patch_everywhere(orig, new)       # the module's own name and every module that bound it directly
    core.module_function = _wrap_module_function(core.module_function)


def pytest_configure(config):
    if not _ON:
        return
    _patch_eagle_blocks()
    from raptor.ir import walk as _wk
    from hawk.artifact import bundle as b
    _patch(_wk.canonical, _walk_opts_wrapper(_wk.canonical))
    from raptor.wings.base import cells as _cw
    _patch(_cw.walk, _cells_walk_wrapper(_cw.walk))
    from aether.wings.base import rtstore as _rt
    _patch(_rt.select_weight, _select_weight_wrapper(_rt.select_weight))
    for name, entry in (("build_bundle", "build_bundle"), ("build", "build"), ("emit_source", "emit_source")):
        orig = getattr(b, name)
        tp._patch_everywhere(orig, _entry(orig, entry, None))
    b._emitted = _step(b._emitted, _h_emitted)
    b._expand_segmented = _wrap_segmented(b._expand_segmented)
    b._expand_split = _wrap_split(b._expand_split)
    b.lower = _wrap_lower(b.lower)
    b._draft = _wrap_draft(b._draft)
    b._note = _wrap_note(b._note)
    from hawk.passes import fusion as _fusion
    _fusion.cuts = _wrap_cuts(_fusion.cuts)
    from hawk.artifact import manifest as _manifest
    _manifest.plugin_entry = _wrap_plugin_entry(_manifest.plugin_entry)
    tp._write = _text_write(tp._write)


def pytest_runtest_logfinish(nodeid, location):
    if _ON:
        _close_orphan()


def pytest_runtest_protocol(item, nextitem):
    if _ON:
        _KCALL.clear()
        if not _SEAM0:
            try:
                _SEAM0.update(_seams())
            except BaseException:
                pass
        try:
            from hawk.artifact import bundle as b
            b._UNIT_MEMO.clear()
        except BaseException:
            pass
    return None
