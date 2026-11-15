# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""pytest plugin: renders every host unit of a test run a second time with the compiled host
library, reached only through the
generated binding, at the seam `hawk.backends.Backend.render`.

HOST_MODE selects what happens with the library's answer:
  off     nothing is changed (the baseline run);
  shadow  the old render always runs and its result is returned; the library's text is compared and
  counted;
  kroute  kernel-level route (kroute.py): the library lowers the whole kernel for every target of
          the build call and the build takes its text for each unit and target; a kernel the
          library does not answer as the old lowering produced it keeps the old text, counted by
          reason;
  route   the old render runs, and when the library answers, the unit returned to the build is made
  of the library's parts; a unit
          the library refuses by name is returned as the old render made it. The unit key's emission
          term is the library's identity.
HOST_LIB is the path of the library, HOST_DIR the folder the counters of each worker are written to
(merged by merge_host.py).

This module also owns the conversion of one recorded call into the arguments of the binding
(`prepare`, `contract`), so a recorded
unit can be rendered the same way (a test, a replay). The conversion below is the one description of
the contract in Python.
Standard library only; the live part imports the old build lazily.
"""

import atexit
import dataclasses
import json
import math
import os
import statistics
import sys
import time

import shadow_mirror as sm

_MODE = os.environ.get("HOST_MODE", "off")
_LIB = os.environ.get("HOST_LIB")
_DIR = os.environ.get("HOST_DIR")


# ---------------------------------------------------------------- one call as the binding takes it


def _elements(value):
    """The items of a recorded sequence value (a tuple or list, or a subclass recorded as {"base",
    "of"}).
    """
    if value is None:
        return []
    for key in ("t", "l", "of"):
        if key in value:
            inner = value[key]
            return (
                _elements(inner)
                if isinstance(inner, dict) and ("t" in inner or "l" in inner)
                else list(inner)
            )
    return []


def _text(value):
    """The text of a tagged string, or None."""
    return value.get("s") if isinstance(value, dict) else None


def contract(value, nodes):
    """A tagged value of the recorder as plain JSON data: records become objects with "$type", node
    references {"$node": position},
    module-level references {"$ref": "module:name"}, sequences arrays, maps with text keys objects
    and other maps {"$map": [[k, v]]}.
    `nodes` is the number of nodes of the unit; a reference outside it becomes -1.
    """
    if value is None or isinstance(value, bool):
        return value
    if not isinstance(value, dict):
        return None
    if "i" in value:
        return int(value["i"])
    if "f" in value:
        number = float(value["f"]) if _is_number(value["f"]) else math.nan
        return number if math.isfinite(number) else {"$float": value["f"]}
    if "s" in value:
        return value["s"]
    for key in ("t", "l", "fs"):
        if key in value:
            return [contract(item, nodes) for item in value[key]]
    if "mp" in value:
        return contract(value["mp"], nodes)
    if "d" in value:
        pairs = value["d"]
        if all(isinstance(pair[0], dict) and "s" in pair[0] for pair in pairs):
            return {pair[0]["s"]: contract(pair[1], nodes) for pair in pairs}
        return {"$map": [[contract(key, nodes), contract(item, nodes)] for key, item in pairs]}
    if not value:
        return {}
    if "n" in value:
        return {"$node": value["n"] if 0 <= value["n"] < nodes else -1}
    if "fn" in value:
        return {"$ref": value["fn"]}
    if "r" in value:
        out = {"$type": value["r"]}
        out.update({name: contract(item, nodes) for name, item in value.get("v", {}).items()})
        return out
    return {name: contract(item, nodes) for name, item in value.items()}


def _is_number(text):
    """True when the text parses as a number."""
    try:
        float(text)
        return True
    except ValueError:
        return False


def _kernel_object(low, inputs):
    """A kernel the recorder could not encode is an object of the record: its attributes are the
    kernel's fields; a wrapper object
    (an active-set unit) holds the kernel under `_kernel` and its own walk. Returns a Kernel record,
    or None.
    """
    index = (low.get("kernel") or {}).get("o")
    if index is None:
        return None
    attrs = inputs["objs"][index]["a"]
    inner = (attrs.get("_kernel") or {}).get("v")
    if inner is not None:
        fields = {name: item for name, item in inner.items() if name != "walk"}
        if "walk" in attrs:
            fields["walk"] = attrs["walk"]
    else:
        fields = dict(attrs)
    return {"r": "Kernel", "v": fields}


class Unit:
    """One host call prepared for the binding: the dump of its nodes, the dialect declarations, the
    walk's inputs and the contract.
    """

    def __init__(self, inputs, opts, facts=None):
        """Prepare the unit from the recorder's encoding, the walk's options and the facts."""
        low = (inputs.get("lowered") or {}).get("v")
        if low is None:
            raise ValueError("no lowered record in the call")
        self.count = len(inputs["nodes"])
        kernel_record = _kernel_object(low, inputs)
        kernel = (kernel_record or low["kernel"])["v"]
        walk = kernel["walk"]["v"]
        self.digest = _text(walk.get("digest"))
        self.sinks = [int(item["n"]) for item in _elements(kernel.get("sinks"))]
        self.declared, self.force, self.exempt = self._walk_inputs(walk, opts)
        body = {"$type": "BuiltLowered"}
        for name, item in low.items():
            body[name] = contract(
                kernel_record if name == "kernel" and kernel_record else item, self.count
            )
        body["options"] = contract(inputs.get("options"), self.count)
        if facts:
            # machine facts the old writer read from inside itself, as recorded:
            # {function: [[args, answer]]}
            body["facts"] = facts
        self.contract = json.dumps(body, separators=(",", ":"))
        self.dialects = "".join(
            manifest + "\n" + key + "\n" for manifest, key in sm.manifests(inputs)
        )
        self.inputs = inputs

    def _walk_inputs(self, walk, opts):
        """The walk's declared slots, forced access and exempt planes as `canonical` takes them:
        from the options the walk was made
        with, or, when they are not known, the walk's own leaves declared and no exempt plane.
        """
        identity = list(range(self.count))
        if opts is None:
            leaves = walk.get("leaves") or {}
            slots = [sm.val(item, identity) for item in _elements(leaves)]
            return "[" + ", ".join(slots) + "]", None, "[]"
        if isinstance(opts, str):
            return "()", None, "()"
        return (
            sm.val(opts["declared"], identity),
            _text(opts.get("force")),
            sm.val(opts["exempt"], identity),
        )


def render(ctx, library, unit, timing=None):
    """The four parts (text, body, entry name, meta as data) of a unit rendered by the library
    `library` opened on `ctx`.
    Raises ValueError with the stage-tagged message of the library or the door. A dict `timing`
    receives the seconds of the
    mirror and dump ("mirror_dump").
    """
    started = time.perf_counter()
    mirror = _osprey().Ctx()
    sm.declare(mirror, sm.manifests(unit.inputs))
    nodes = sm.mirror(mirror, unit.inputs)
    dump = mirror.dump(nodes)
    if timing is not None:
        timing["mirror_dump"] = time.perf_counter() - started
    out = ctx.backend_render(
        library,
        dump,
        unit.dialects,
        unit.declared,
        unit.force,
        unit.exempt,
        unit.sinks,
        unit.contract,
    )
    text, body, entry, meta = out
    return text, body, entry, json.loads(meta)


def _osprey():
    import _osprey

    return _osprey


# ---------------------------------------------------------------- the live plugin

_state = {"test": "collection", "lib": None, "ctx": None, "identity": None, "wrote": False}
_counts = {
    "calls": 0,
    "identical": 0,
    "different": 0,
    "crossing": 0,
    "routed": 0,
    "not_routed": 0,
    "not_host": 0,
    "raised": 0,
    "refused": {},
    "costs": {},
    "not_host_why": {},
    "differences": [],
    "crossings": [],
    "key_example": None,
}
_LIST_LIMIT = 40


def _bump(table, key):
    """Adds one to table[key]."""
    table[key] = table.get(key, 0) + 1


def _first_difference(old, new):
    """The first line two texts differ at, as (line number, old line, new line)."""
    old_lines, new_lines = old.splitlines(), new.splitlines()
    for at in range(max(len(old_lines), len(new_lines))):
        left = old_lines[at] if at < len(old_lines) else None
        right = new_lines[at] if at < len(new_lines) else None
        if left != right:
            return at + 1, left, right
    return 0, None, None


def _note_difference(name, old, parts):
    """Keeps the unit, the test and the first differing line of a unit whose library text differs
    from the old one.
    """
    text, body, entry, meta = parts
    field = next(
        (
            f
            for f, a, b in (
                ("text", old.text, text),
                ("body", old.body, body),
                ("entry", old.entry, entry),
                ("mode", old.mode, meta["mode"]),
                ("vectorised", old.vectorised, meta["vectorised"]),
                ("box", old.box, meta.get("box")),
            )
            if a != b
        ),
        "?",
    )
    at, left, right = _first_difference(old.text, text)
    if len(_counts["differences"]) < _LIST_LIMIT:
        _counts["differences"].append(
            {
                "unit": name,
                "test": _state["test"],
                "field": field,
                "line": at,
                "old": left,
                "new": right,
            }
        )
        if field in ("box", "mode", "vectorised"):
            _counts["differences"][-1]["values"] = [
                repr(getattr(old, field)),
                repr(meta.get(field)),
            ]


def _equal(old, parts):
    """True when the old unit and the library's parts agree on text, body, entry, mode, vectorised
    and box.
    """
    text, body, entry, meta = parts
    return (old.text, old.body, old.entry, old.mode, old.vectorised, old.box) == (
        text,
        body,
        entry,
        meta["mode"],
        meta["vectorised"],
        meta.get("box"),
    )


def _walk_options(encoder, unit_digest):
    """The options the live walk of this digest was made with (declared slots, force, exempt
    planes), or None when not seen.
    """
    import text_plugin as tp

    walk = encoder.walks.get(unit_digest)
    used = tp._used_opts(walk) if walk is not None else None
    if used is None:
        known = tp._OPTS.get(unit_digest, {})
        used = next(iter(known.values()))["opts"] if known else None
    return used


def _encode(backend, lowered, options):
    """The recorder's encoding of the seam's inputs, or None when the call cannot be encoded whole
    (it is then not a unit of this test).
    """
    import text_plugin as tp

    encoder = tp.TextEnc()
    try:
        inputs = {"lowered": encoder.val(lowered), "options": encoder.val(dict(options))}
        inputs.update(
            {"nodes": list(encoder.nodes), "kinds": list(encoder.kinds), "objs": list(encoder.objs)}
        )
    except BaseException:
        return None, encoder
    return (None if encoder.unknown else inputs), encoder


def _facts_begin(encoder):
    """Starts recording the machine facts the old render reads (calls of the recorded functions), as
    the recorder does.
    """
    import text_plugin as tp

    tp._state["actx"] = tp.AnalysisCtx(encoder, "record")


def _facts_end():
    """Stops recording; the facts as the recorder's `facts` field: {function: [[args, answer(,
    kwargs)]]}.
    """
    import text_plugin as tp

    actx, tp._state["actx"] = tp._state.get("actx"), None
    if actx is None or not actx.facts:
        return None
    return {
        qual.split(":")[1]: [list(entry) for entry in calls.values()]
        for qual, calls in actx.facts.items()
    }


def _library_parts(inputs, encoder, lowered, facts):
    """Renders the call with the library: (unit, parts) or raises. Stage-tagged ValueError for a
    refusal or a crossing error.
    """
    if inputs is None:
        raise LookupError("excluded: the call holds a value the encoder cannot carry")
    low = inputs["lowered"]["v"]
    kernel = (_kernel_object(low, inputs) or low["kernel"])["v"]
    digest = _text(kernel["walk"]["v"].get("digest"))
    unit = Unit(inputs, _walk_options(encoder, digest), facts)
    return unit, render(_state["ctx"], _state["lib"], unit)


def _with_box(parts, old):
    """The parts with the thread box rebuilt as the old record: the library returns it as data
    (axis, T, stride).
    """
    text, body, entry, meta = parts
    given = meta.get("box")
    box = type(old.box)(**given) if isinstance(given, dict) and old.box is not None else given
    return text, body, entry, {**meta, "box": box}


def _routed(old, parts):
    """The unit the build gets in route mode: the old unit with the library's text, body, entry,
    mode and vectorised flag.
    """
    text, body, entry, meta = parts
    return dataclasses.replace(
        old,
        text=text,
        body=body,
        entry=entry,
        target=meta["target"],
        mode=meta["mode"],
        vectorised=meta["vectorised"],
        box=meta["box"],
    )


_COST_UNITS = set(filter(None, os.environ.get("HOST_COST_UNITS", "").split(",")))


def _median_ms(call):
    """Milliseconds of one call: the median of five, after one warm-up call, all in this process."""
    call()
    times = []
    for _ in range(5):
        started = time.perf_counter()
        call()
        times.append((time.perf_counter() - started) * 1000)
    return statistics.median(times)


def _measure(orig, backend, lowered, options, name, unit, old):
    """Cost mode: for the first call of each named unit the library writes, the old render's time,
    the library path's time
    (mirror, dump, load, canonical, render) and the crossing part of it (the same path with a render
    that refuses at once, so
    only mirror, dump, load and canonical are paid).
    """
    if name in _counts["costs"] or (_COST_UNITS and name not in _COST_UNITS):
        return
    refusing = Unit.__new__(Unit)
    refusing.__dict__.update(unit.__dict__, contract="{}")

    def crossing():
        try:
            render(_state["ctx"], _state["lib"], refusing)
        except ValueError:
            pass

    _counts["costs"][name] = {
        "lines": len(old.text.splitlines()),
        "test": _state["test"],
        "old_ms": _median_ms(lambda: orig(backend, lowered, options)),
        "new_ms": _median_ms(lambda: render(_state["ctx"], _state["lib"], unit)),
        "crossing_ms": _median_ms(crossing),
    }


def _wrap(orig):
    """The seam: the old render, then the library's, then counting (and in route mode the swap)."""

    def render_seam(self, lowered, options):
        _counts["calls"] += 1
        if _MODE == "off" or self.id != "host" or _state["lib"] is None:
            _counts["not_host"] += 1
            _bump(_counts["not_host_why"], "other target" if self.id != "host" else "mode off")
            return orig(self, lowered, options)
        inputs, encoder = _encode(self, lowered, options)
        _facts_begin(encoder)
        try:
            old = orig(self, lowered, options)
        except BaseException:
            _facts_end()
            _counts["raised"] += 1
            raise
        facts = _facts_end()
        name = getattr(lowered, "name", "?")
        try:
            unit, parts = _library_parts(inputs, encoder, lowered, facts)
            parts = _with_box(parts, old)
        except LookupError as why:
            _counts["not_host"] += 1
            _bump(_counts["not_host_why"], str(why))
            return old
        except ValueError as why:
            message = str(why)
            if message.startswith("render: "):
                _bump(_counts["refused"], message[len("render: ") :][:100])
            else:
                _crossing(name, message)
            _counts["not_routed"] += 1
            return old
        except BaseException as why:
            _crossing(name, "%s: %s" % (type(why).__name__, why))
            _counts["not_routed"] += 1
            return old
        if _equal(old, parts):
            _counts["identical"] += 1
            if _MODE == "cost":
                _measure(orig, self, lowered, options, name, unit, old)
        else:
            _counts["different"] += 1
            _note_difference(name, old, parts)
        if _MODE != "route":
            return old
        try:
            routed = _routed(old, parts)
        except (ValueError, TypeError) as why:
            _crossing(name, str(why))
            _counts["not_routed"] += 1
            return old
        _counts["routed"] += 1
        return routed

    render_seam.__wrapped__ = orig
    return render_seam


def _crossing(name, message):
    """Counts a failure of the crossing itself (open, load, canonical, or a Python error in the
    plugin) and keeps its first ones.
    """
    _counts["crossing"] += 1
    if len(_counts["crossings"]) < _LIST_LIMIT:
        _counts["crossings"].append(
            {"unit": name, "test": _state["test"], "message": message[:300]}
        )


def _write_counts():
    """Writes this worker's counters to HOST_DIR/h<pid>.json (called at the end of the session and
    at exit).
    """
    if not _DIR or _MODE == "off":
        return
    os.makedirs(_DIR, exist_ok=True)
    with open(os.path.join(_DIR, "h%d.json" % os.getpid()), "w") as handle:
        json.dump(
            {"mode": _MODE, "identity": _state["identity"], **_counts}, handle, sort_keys=True
        )


def _patch_unit_key(unit_cache):
    """Route mode: the emission term of a host unit's key is the library's identity; the first key
    made is kept as the example.
    """
    original_digest, original_key = unit_cache.emission_digest, unit_cache._unit_key
    import hashlib

    def emission_digest(targets, isa=None):
        if all(t == "host" for t in targets):
            return _state["identity"]
        rest = original_digest(targets, isa)
        return hashlib.sha256((rest + ":" + _state["identity"]).encode()).hexdigest()

    def unit_key(*args, **kwargs):
        key = original_key(*args, **kwargs)
        if _counts["key_example"] is None and all(t == "host" for t in kwargs.get("targets", ())):
            _counts["key_example"] = {"key": key, "emission_term": _state["identity"]}
        return key

    unit_cache.emission_digest, unit_cache._unit_key = emission_digest, unit_key
    from hawk.artifact import bundle

    bundle._unit_key = unit_key  # the bundle holds its own reference to the function


def pytest_configure(config):
    """Open the library, install the recorder hooks and the seam wrapper (not in mode off)."""
    if _MODE == "off" or not _LIB:
        return
    import shadow_plugin as sp
    import text_plugin as tp
    from raptor.ir.nodes import Node

    sp._NODE[0] = Node
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    from raptor import _core as core

    core.canonical = tp._opts_wrapper(core.canonical)
    from raptor.ir import walk as walk_module

    tp._patch_everywhere(walk_module.canonical, tp._walk_wrapper(walk_module.canonical))
    if _MODE == "kroute":
        import kroute

        if not hasattr(config, "workerinput"):
            kroute.clear()
        kroute.install(_LIB)
        return
    ctx = _osprey().Ctx()
    _state["ctx"], _state["lib"] = ctx, ctx.backend_open(_LIB)
    _state["identity"] = ctx.backend_identity(_state["lib"])
    tp._install_facts()
    from hawk import backends

    backends.Backend.render = _wrap(backends.Backend.render)
    if _MODE == "route":
        from hawk.artifact import unit_cache

        _patch_unit_key(unit_cache)
    atexit.register(_write_counts)


def pytest_runtest_protocol(item, nextitem):
    """Note the running test for the lists of differences and crossing errors."""
    _state["test"] = item.nodeid
    if _MODE == "kroute" and _LIB:
        import kroute

        kroute.on_test(item.nodeid)
    tp = sys.modules.get("text_plugin")
    if tp is not None:
        tp._WALKSET.clear()
    return None


def pytest_sessionfinish(session, exitstatus):
    """Write this worker's counters."""
    _write_counts()
    if _MODE == "kroute" and _LIB:
        import kroute

        kroute.write()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Kernel route mode: the merged counters of every worker, printed and written (controller)."""
    if _MODE == "kroute" and _DIR and not hasattr(config, "workerinput"):
        import kroute

        terminalreporter.write_line("")
        for line in kroute.report(_DIR):
            terminalreporter.write_line(line)
