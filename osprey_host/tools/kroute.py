# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Kernel-level route mode of the host plugin (HOST_MODE=kroute).

For every build call of a test run the kernel is captured before lowering (the recorder's hooks,
reused), its IR is mirrored into a library context, ONE request is built with every target of the
call, and the library's `lower` answers the units and their texts. The old build then runs as it is
and, at the render seam, hands the compile step the library's text for each unit and target. A
kernel is routed whole or not at all: the answer's units must match the old units by name and walk
digest, and every served target must have a text; otherwise the kernel keeps the old text and the
call is counted by reason. Everything after the text (compile, sidecar, publish, load, run) stays
the old code.

Counters of each worker go to HOST_DIR/k<pid>.json; the controller merges them, prints the summary
and writes HOST_DIR/kroute_counters.json. KROUTE_RAISE=1 re-raises an error inside this module
instead of counting it. Needs the recorder modules (k6_plugin, text_plugin, shadow_mirror) on the
path.
"""

import atexit
import contextlib
import dataclasses
import glob
import json
import os
import re
import sys

import shadow_mirror as sm

_RAISE = os.environ.get("KROUTE_RAISE") == "1"
_DIR = os.environ.get("HOST_DIR")
_LIST_LIMIT = 300
_COMPILERS = (
    "c++ || Apple clang version 21.0.0 (clang-2100.0.123.102) || 0",
    "g++ || g++ (GCC) 13.2.0 || 0",
)
# the recorder keeps no identity of the device compiler: the router states one release line
# and counts the calls that needed it
_DEVICE_COMPILER = "nvcc || Cuda compilation tools, release 12.6, V12.6.77 || 0"
_SERVED = {"host": "Host", "cuda": "Device"}
_NAMES = {
    "fold_form": ("auto", "static", "rolled"),
    "wide_form": ("auto", "thread", "team"),
    "linalg_form": ("auto", "team", "thread"),
}
_STAGES = ("expand_segmented", "emitted")


class Unbuildable(Exception):
    """The request cannot be built from what the recorder captured."""


_S = {"lib": None, "ctx": None, "k6": None, "facts": {}, "accepted": None, "facts_dir": None}
_C = {
    "build_calls": 0,
    "calls_with_units": 0,
    "kernels": 0,
    "kernels_routed": 0,
    "kernels_incomplete": 0,
    "kernels_incomplete_after_error": 0,
    "units_routed": 0,
    "texts_taken": {},
    "texts_old": {},
    "texts_old_why": {},
    "texts_identical": 0,
    "texts_different": 0,
    "texts_different_accepted": 0,
    "unclaimed_old_units": 0,
    "adapted": {},
    "fallbacks": {},
    "fallback_examples": [],
    "crossing": 0,
    "device_line_stated": 0,
    "differences": [],
    "crossings": [],
}


def _bump(table, key, by=1):
    """Adds `by` to table[key]."""
    table[key] = table.get(key, 0) + by


def _fall(reason, kernel=None, **more):
    """Counts a kernel that keeps its old text, by reason, and keeps the first examples."""
    _bump(_C["fallbacks"], reason)
    if len(_C["fallback_examples"]) < _LIST_LIMIT:
        _C["fallback_examples"].append(
            {"reason": reason, "kernel": kernel, "test": _test(), **more}
        )


def _crossing(where, why):
    """Counts an error inside this module (a tool bug); re-raised under KROUTE_RAISE."""
    _C["crossing"] += 1
    if len(_C["crossings"]) < _LIST_LIMIT:
        _C["crossings"].append(
            {"where": where, "test": _test(), "error": "%s: %s" % (type(why).__name__, why)}
        )
    if _RAISE:
        raise why


def _test():
    """The id of the running test."""
    import text_plugin as tp

    return tp._state.get("test")


# ------------------------------------------------------------------ the recorder's values


def _rec(j):
    """The `v` of a tagged record, or the value."""
    return j.get("v", j) if isinstance(j, dict) else j


def _text(j):
    """The text of a tagged string, or None."""
    return j.get("s") if isinstance(j, dict) else None


def _num(j):
    """The integer of a tagged integer, or None."""
    return int(j["i"]) if isinstance(j, dict) and "i" in j else None


def _items(j):
    """The items of a tagged list or tuple."""
    return (j.get("l") if "l" in j else j.get("t", [])) if isinstance(j, dict) else []


def _entry(mapping, key):
    """The value under a text key of a tagged dict, seen through a frozen wrapper."""
    while isinstance(mapping, dict) and ("of" in mapping or "mp" in mapping):
        mapping = mapping.get("of", mapping.get("mp"))
    for pair in (mapping or {}).get("d", []) if isinstance(mapping, dict) else []:
        if _text(pair[0]) == key:
            return pair[1]
    return None


def _ref(j):
    """The module-level reference a recorded value names."""
    return j.get("fn") if isinstance(j, dict) else None


def _camel(name):
    """`per_sample` as `PerSample`: the name the request spells a member by."""
    return "".join(part.capitalize() for part in name.split("_"))


def _kind_of(kind):
    """The request's kind from the recorded one."""

    def plain(guard):
        return {"guard": guard, "active_set": False, "lanes": None, "sink": "Plain"}

    if kind is None:
        return plain("Default")
    if _ref(kind) is not None:
        if _ref(kind) == "hawk.ext:DEFAULT_KIND":
            return plain("Default")
        raise Unbuildable("kernel kind " + _ref(kind))
    kind = _rec(kind)
    guard = kind.get("guard")
    if guard is None:
        raise Unbuildable("kind without a guard")
    active = False
    if _ref(guard) == "hawk.ext:DEFAULT_GUARD":
        out = "Default"
    elif _ref(guard) == "hawk.ext:DATA_ONLY":
        out = "DataOnly"
    elif _ref(guard) is not None:
        raise Unbuildable("guard " + _ref(guard))
    else:
        g = _rec(guard)
        masks = [m for m in map(_text, _items(g.get("masks"))) if m is not None]
        if not masks and _text(g.get("mask")) is not None:
            masks = [_text(g["mask"])]
        out = {"Masks": masks}
        active = g.get("active_set") is True
    sink = kind.get("sink")
    if sink is None:
        raise Unbuildable("kind without a sink")
    if _ref(sink) == "hawk.ext:PLAIN":
        policy = "Plain"
    else:
        s = _rec(sink)
        plane = _text(s.get("compensation"))
        if plane is None:
            raise Unbuildable("a sink policy of another form")
        output = _rec(kind["output"]) if kind.get("output") else None
        body = {"plane": plane}
        slot = _text(output.get("slot")) if output else None
        if slot is not None:
            body["output_slot"] = slot
        body["output_names"] = [
            n for n in map(_text, _items(output.get("names") if output else None)) if n
        ]
        policy = {"Compensated": body}
    lanes = None
    lane_set = kind.get("lanes")
    if lane_set is not None:
        ls = _rec(lane_set)
        lanes = {
            "bits": 64 if _num(ls.get("width")) == 64 else 32,
            "prefix": ls.get("prefix") is True,
        }
    return {"guard": out, "active_set": active, "lanes": lanes, "sink": policy}


def _declared(slot):
    """A declared slot of the record."""
    slot = _rec(slot)
    ttype = _rec(slot.get("ttype") or {})
    role, dtype = _text(slot.get("role")), _text(ttype.get("dtype"))
    if role is None or dtype is None or _text(slot.get("name")) is None:
        raise Unbuildable("a declared slot")
    out = {
        "role": _camel(role),
        "name": _text(slot["name"]),
        "dtype": _camel(dtype),
        "shape": [n for n in map(_num, _items(ttype.get("shape"))) if n is not None],
    }
    if _text(ttype.get("tag")) is not None:
        out["tag"] = _text(ttype["tag"])
    return out


def _member(name, record):
    """A kernel body: its sinks (node handles) and the walk's inputs."""
    out = {
        "name": name,
        "sinks": [s["n"] for s in record.get("sinks") or [] if isinstance(s, dict) and "n" in s],
        "declared": [_declared(s) for s in _items(record.get("declared"))],
        "exempt": [e for e in map(_text, _items(record.get("exempt"))) if e is not None],
    }
    if _text(record.get("force")) is not None:
        out["force"] = _camel(_text(record["force"]))
    return out


# The members below mirror the replay tool's converters (src/bin/kreplay.rs `steps_of`, `fusion_of`,
# `lane_space_of`, `sink_axes_of`, `shared_of`, `broadcast_of`, `recorded_image_format`): the replay
# is in Rust and this plugin in Python, so each is written twice.
_AUTO_STEPS_CAP = 64


def _seq(j):
    """The items of a recorded list, tagged or plain."""
    return j if isinstance(j, list) else _items(j)


def _axes(pairs):
    """The axes of a recorded tuple of `(label, extent)` pairs; an entry may lack either."""
    out = []
    for axis in _seq(pairs):
        pair = _seq(axis)
        one = {}
        label = _text(pair[0]) if pair else None
        extent = _num(pair[1]) if len(pair) > 1 else None
        if label is not None:
            one["label"] = label
        if extent is not None:
            one["extent"] = extent
        out.append(one)
    return out


def _declared_planes(kernel, key):
    """The planes the kernel record declares under `key`, as (name, declaration record)."""
    mapping = kernel.get(key)
    while isinstance(mapping, dict) and ("of" in mapping or "mp" in mapping):
        mapping = mapping.get("of", mapping.get("mp"))
    out = []
    for pair in mapping.get("d", []) if isinstance(mapping, dict) else []:
        name = _text(pair[0])
        if name is not None and len(pair) > 1:
            out.append((name, _rec(pair[1])))
    return out


def _steps_members(kernel, call):
    """The `steps` (and the cap of an auto count) a kernel record declares."""
    steps = kernel.get("steps")
    if _num(steps) is not None:
        return {"steps": _num(steps)}
    if _text(steps) == "auto":
        cap = call.get("auto_k_max")
        return {"steps": "auto", "steps_max": cap if isinstance(cap, int) else _AUTO_STEPS_CAP}
    return {}


def _number(j):
    """A recorded number: a whole number or a float, None for null."""
    if _num(j) is not None:
        return _num(j)
    try:
        return float(j["f"]) if isinstance(j, dict) and "f" in j else None
    except ValueError:
        return None


def _fusion(call):
    """The fusion row the call's last chain cut used, when the recorder stored it.

    The row names the five numbers of the device and instruction set documents; a number the
    recorder did not store is None, as the documents state it.
    """
    rows = _seq(call.get("fusion_rows"))
    if not rows:
        return None
    row = rows[-1]
    base, file = _num(_entry(row, "R0")), _num(_entry(row, "R_file"))
    if base is None or file is None:
        return None
    out = {"regs_base": base, "regs_file": file}
    for key, field in (
        ("launch_us", "L_us"),
        ("plane_us_per_byte", "c_plane_us_per_B"),
        ("local_over_plane", "c_local_over_plane"),
    ):
        out[key] = _number(_entry(row, field))
    return out


def _fusion_cards(call, device):
    """The documents a call is built under: the shipped ones, but for the recorded fusion row.

    A chain cut depends on the fusion row of its build. When the recording holds a row that is not
    the shipped one, the document that carries the row (the device document for a build with a
    device target, else the instruction set document of the host's vector instruction set) is
    stated with the recorded row in place of the shipped one, as the replay tool does. The same
    documents go to the facts call and to the request. Returns {document name: text}, empty when
    the shipped row is the recorded one.
    """
    row = _fusion(call)
    if row is None:
        return {}
    if device:
        name, place = "cuda.json", ("fusion",)
    else:
        machine = json.loads(_facts(call, "host", {}))
        name = {"Arm64": "arm.json", "X86_64": "x86.json"}.get(machine.get("isa"))
        if name is None or not machine.get("vector_isa"):
            raise Unbuildable("a host without an instruction set card")
        place = ("fusion_host", machine["vector_isa"])
    document = json.loads(_S["ctx"].backend_card(_S["lib"], _S["facts_dir"], name))
    holder = document
    for part in place[:-1]:
        holder = holder[part]
    if holder[place[-1]] == row:
        return {}
    holder[place[-1]] = row
    return {name: json.dumps(document)}


def _payload(plane):
    """The per-sample component of a recorded plane: its declared extents, empty for a scalar."""
    ttype = _rec(plane.get("ttype") or {})
    return [n for n in map(_num, _items(ttype.get("shape"))) if n is not None]


def _part_members(kernel, call):
    """The per-kernel members of a request besides its body, from the kernel record."""
    out = _steps_members(kernel, call)
    lanes = kernel.get("lane_space")
    out["lane_space"] = _axes(lanes) if lanes is not None else [{}]
    planes = _declared_planes(kernel, "planes")
    sinks = [(n, p) for n, p in planes if _text(p.get("form")) == "accum"]
    sinks += _declared_planes(kernel, "derived_planes")
    axes = [
        {"name": n, "payload": _payload(p), "axes": _axes(p.get("dims"))}
        for n, p in sinks
        if _axes(p.get("dims"))
    ]
    if axes:
        out["sink_axes"] = axes
    shared = [n for n, p in planes if p.get("shared") and _seq(p["shared"])]
    if shared:
        out["shared"] = shared
    broadcast = []
    for pair in _seq(kernel.get("broadcast")):
        pair = _seq(pair)
        name = pair[0] if pair and isinstance(pair[0], str) else _text(pair[0]) if pair else None
        if name is not None and len(pair) > 1:
            broadcast.append({"plane": name, "axes": [f is True for f in _seq(pair[1])]})
    if broadcast:
        out["broadcast"] = broadcast
    return out


def _settings(spec, split):
    """The request's settings from the spec; a call with no split never chose a warp per lane."""
    s = _rec(spec["settings"])
    out = {}
    for key, field in (
        ("fold_form", "form"),
        ("wide_form", "wide_form"),
        ("linalg_form", "linalg_form"),
    ):
        value = _text(s.get(field)) or "auto"
        if value not in _NAMES[key]:
            raise Unbuildable("%s %s" % (key, value))
        out[key] = _camel(value)
    if not split:
        out["wide_form"] = "Thread"
    if s.get("route_to_ordered") is True:
        out["route_to_ordered"] = True
    bounds = _items(s.get("launch_bounds"))
    if len(bounds) == 2:
        out["launch_bounds"] = [_num(bounds[0]) or 0, _num(bounds[1]) or 0]
    if _num(s.get("fold_k")):
        out["fold_k"] = _num(s["fold_k"])
    out["warps"] = _num(s.get("warps")) or 1
    out["team_threads"] = _num(s.get("team_threads")) or 0
    return out


def _option(call, name):
    """A build option the call states, as a keyword or as a `cuda.<name>` entry of its options."""
    args = call["args"]
    return args.get(name) or _entry(args.get("options"), "cuda." + name)


def _own_card(spec, device):
    """Whether the build ran under a card of the test's own (the request carries no card)."""
    spec = _rec(spec)
    if device:
        cards = _entry(spec.get("by_target", spec), "cuda")
        rows = _entry(_entry(cards, "cards") or {}, "thread_box") if cards else None
        return bool(_items(_entry(rows, "rows") if rows else None))
    form = _entry(spec["host"], "form") if spec.get("host") else None
    width = _num(_entry(form, "lazy_leaf_width")) if form else None
    return width is not None and width != 32


def _host_stated(spec):
    """(stated machine JSON, whether the compiler is a GCC) of the host the recorded spec states."""
    machine = _rec(spec["machine"])
    mp = (machine.get("facts") or {}).get("mp")
    if mp is None:
        raise Unbuildable("a machine without facts")

    def get(key):
        return _entry(mp, key)

    isa = _text(machine.get("isa"))
    profile = _text(get("host_profile"))
    vector = _text(get("vector_isa"))
    gcc = get("arm64_gcc") is True or (
        isa == "x86_64" and profile is not None and "vector" in profile
    )
    stated = {"headers": "roots"}
    for key, value in (("isa", isa), ("vector_isa", vector), ("host_profile", profile)):
        if value is not None:
            stated[key] = value
    l1, l2 = _num(get("l1d_cache_bytes")), _num(get("l2_cache_bytes"))
    if l1 and l2:
        stated.update(l1d_bytes=l1, l2_bytes=l2, cache_source=_text(get("cache_source")) or "")
    return stated, gcc


def _facts(call, target, cards):
    """The facts answer for the target: the host the spec states, the device arch.

    `cards` are the documents the build is stated under (see `_fusion_cards`).
    """
    spec = _rec(call["spec"])
    if target == "host":
        stated, gcc = _host_stated(spec)
        compiler = _COMPILERS[gcc]
    else:
        compiler = _DEVICE_COMPILER
        _C["device_line_stated"] += 1
        arch = (
            _text(_rec(spec["machine"]).get("arch"))
            or _text(_option(call, "device_arch"))
            or _text(_option(call, "arch"))
        )
        stated = {"device_arch": arch} if arch else {}
    if cards:
        stated["cards"] = cards
    key = (target, compiler, json.dumps(stated, sort_keys=True))
    if key not in _S["facts"]:
        try:
            _S["facts"][key] = _S["ctx"].backend_facts(
                _S["lib"], _S["facts_dir"], target, compiler, json.dumps(stated)
            )
        except ValueError as why:
            raise Unbuildable("facts: %s" % str(why)[:100]) from why
    return _S["facts"][key]


def _is_chain(kernel):
    """Whether the kernel is a chain (or fused lanes) of member kernels."""
    members = kernel.get("members") or []
    return bool(members) and isinstance(members[0], dict) and "sinks" in members[0]


def _request(call, kernel, targets, built):
    """The request of one recorded kernel for the targets, as the replay builds it from a record."""
    name = kernel.get("name")
    if not name:
        raise Unbuildable("a kernel without a name")
    if kernel.get("record_error"):
        raise Unbuildable("kernel record: " + kernel["record_error"][:80])
    adapted = []
    bare = name.lstrip("_")
    given = _text((call.get("args") or {}).get("name"))
    if name not in built and given in built and len(call["kernels"]) == 1:
        name, adapted = given, ["build name"]
    elif name not in built and bare in built:
        name, adapted = bare, ["name as built"]
    split = call.get("split_in")
    if not split:
        adapted.append("wide form")
    spec = _rec(call["spec"])
    if any(_own_card(call["spec"], t == "cuda") for t in targets):
        raise Unbuildable("own card")
    cards = _fusion_cards(call, "cuda" in targets)
    scalar = {"float64": "Float64", "float32": "Float32", "banded": "Banded"}.get(call["smode"])
    if scalar is None:
        raise Unbuildable("scalar mode " + str(call["smode"]))
    if _is_chain(kernel):
        members = [_member(m.get("name") or "", m) for m in kernel["members"]]
        body = {"Fused" if kernel.get("fused_body") else "Chain": members}
    else:
        body = {"Plain": _member(name, kernel)}
    one = kernel.get("one_step")
    plane_map = kernel.get("planes")
    while isinstance(plane_map, dict) and ("of" in plane_map or "mp" in plane_map):
        plane_map = plane_map.get("of", plane_map.get("mp"))
    planes = (
        [t for t in (_text(p[0]) for p in (plane_map or {}).get("d", [])) if t is not None]
        if isinstance(plane_map, dict)
        else []
    )
    settings = _settings(spec, split)
    sizes = [_num(s) for s in _items(call.get("exports_override"))]
    if len(sizes) == 5 and all(s is not None and s >= 0 for s in sizes):
        settings["exports"] = sizes
    if _option(call, "literal_table") is True:
        settings["literal_table"] = True
    request = {
        "v": 1,
        "name": name,
        "scalar": scalar,
        "targets": [_SERVED[t] for t in targets],
        "kind": _kind_of(kernel.get("kind")),
        "body": body,
        "group": kernel.get("group") or 1,
        "settings": settings,
        "facts": {_SERVED[t]: _facts(call, t, cards) for t in targets},
    }
    if one:
        request["one_step"] = _member(_text(one.get("name")) or "", one)
    if planes:
        request["planes"] = planes
    request.update(_part_members(kernel, call))
    if cards:
        request["cards"] = cards
    if "cuda" in targets:
        # the format the recorded build asked for; a recording that kept none gets a binary for one
        # architecture
        request["image_format"] = _camel(
            call.get("image_format") if call.get("image_format") in ("ptx", "fatbin") else "cubin"
        )
    return name, request, adapted


# ------------------------------------------------------------------ the plan of one call


def _owned(kernel, unit):
    """Whether an old unit name belongs to the kernel: its name, stages and segment branches."""
    if unit == kernel or unit.startswith(kernel + "__k") or unit == kernel + "__cls":
        return True
    rest = unit[len(kernel) + 1 :] if unit.startswith(kernel + "_") else ""
    return rest.isdigit()


def _unit(u):
    """The name and the digest of an answered unit: they are in its meta record."""
    return u["meta"]["name"], u["meta"]["digest"]


def _reason(why):
    """A refusal of the library as a counter key: its kind and the first words of its reason."""
    kind = type(why).__name__
    return "refused by the library (%s): %s" % (kind, re.sub(r"\d+", "N", str(why))[:90])


def _plain(value):
    """The recorder's value as JSON data (tuples become lists)."""
    return json.loads(json.dumps(value, default=str))


def _plan(c, kernel, name, smode, targets):
    """The plan of the call at its first unit: the texts of each routed kernel's units; the rest is
    counted by reason.
    """
    old = c.get("old_units") or [(name, kernel.walk.digest)]
    served = [t for t in dict.fromkeys(targets) if t in _SERVED]
    plan = {"units": {}, "kernels": {}}
    records = [k for k in c["kernels"] if k.get("stage") in _STAGES]
    _C["calls_with_units"] += 1
    if not served or not records:
        _bump(
            _C["fallbacks"],
            "no served target" if not served else "no kernel record",
            len(records) or 1,
        )
        _C["kernels"] += len(records)
        return plan
    call = {
        key: _plain(c.get(key))
        for key in (
            "kernels",
            "spec",
            "args",
            "split_in",
            "exports_override",
            "image_format",
            "auto_k_max",
            "fusion_rows",
        )
    }
    call["smode"] = smode.id
    if call["spec"] is None:
        raise Unbuildable("no spec")
    enc = c["enc"]
    inputs = {"nodes": list(enc.nodes), "kinds": list(enc.kinds), "objs": list(enc.objs)}
    mirror = _osprey().Ctx()
    sm.declare(mirror, sm.manifests(inputs))
    dump = mirror.dump(sm.mirror(mirror, inputs))
    dialects = "".join(m + "\n" + k + "\n" for m, k in sm.manifests(inputs))
    built = {n for n, _ in old}
    claimed = set()
    # a call can hold kernels whose names look like another kernel's stages:
    # an old unit named as a kernel of the call is that kernel's
    kernels = {record.get("name") for record in call["kernels"]}
    for record, raw in zip(call["kernels"], c["kernels"]):
        if raw.get("stage") not in _STAGES:
            continue
        _C["kernels"] += 1
        try:
            if c.get("patched"):
                raise Unbuildable("the test replaces the old renderer")
            if os.environ.get("HAWK_BOX") == "0":
                raise Unbuildable("the build is told to write no thread box")
            kname, request, adapted = _request(call, record, served, built)
        except Unbuildable as why:
            _fall("request not buildable: %s" % why, record.get("name"))
            continue
        try:
            answer = json.loads(
                _S["ctx"].backend_lower(_S["lib"], dump, dialects, json.dumps(request))
            )
        except (ValueError, RuntimeError) as why:
            _fall(_reason(why), kname, full=str(why)[:300])
            continue
        # a split kernel's answer also holds the unit that lists its stages and has no entry of its
        # own, and a segmented group with a dynamic selector also holds the fused unit that sweeps
        # its stale samples: the old build made neither
        extra = (kname + "_source", kname + "__sweep")
        units = [
            u for u in answer["units"] if not (_unit(u)[0] in extra and _unit(u)[0] not in built)
        ]
        mine = [n for n, _ in old if _owned(kname, n) and (n == kname or n not in kernels)]
        digests = dict(old)
        problem = None
        if sorted(mine) != sorted(_unit(u)[0] for u in units):
            problem = "unit mismatch: count or name"
        elif any(digests.get(_unit(u)[0]) != _unit(u)[1] for u in units):
            problem = "unit mismatch: digest"
        elif any({t["target"] for t in u["texts"]} != {_SERVED[t] for t in served} for u in units):
            problem = "unit mismatch: targets"
        if problem:
            _fall(problem, kname, old=old, answer=[(_unit(u)[0], _unit(u)[1][:12]) for u in units])
            continue
        _C["kernels_routed"] += 1
        _C["units_routed"] += len(units)
        for tag in adapted:
            _bump(_C["adapted"], tag)
        plan["kernels"][kname] = {
            "units": [_unit(u)[0] for u in units],
            "taken": set(),
            "served": served,
        }
        for u in units:
            claimed.add(_unit(u)[0])
            plan["units"][_unit(u)[0]] = {
                "kernel": kname,
                "adapted": adapted,
                "texts": {t["target"]: t for t in u["texts"]},
            }
    _C["unclaimed_old_units"] += len({n for n, _ in old} - claimed)
    return plan


def _osprey():
    import _osprey

    return _osprey


# ------------------------------------------------------------------ the seams


def _first_difference(old, new):
    """The first line two texts differ at: (line number, old line, new line)."""
    a, b = old.splitlines(), new.splitlines()
    for at in range(max(len(a), len(b))):
        left, right = a[at] if at < len(a) else None, b[at] if at < len(b) else None
        if left != right:
            return at + 1, left, right
    return 0, None, None


def _accepted():
    """The unit names on the accepted list."""
    if _S["accepted"] is None:
        names = set()
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "corpus_accepted.tsv")
        with contextlib.suppress(OSError), open(path) as handle:
            names = {
                line.split("\t")[1] for line in handle if not line.startswith("#") and "\t" in line
            }
        _S["accepted"] = names
    return _S["accepted"]


def _box_class():
    """The class of a unit's thread box, when the old unit has none to take it from."""
    from hawk import wings

    return wings.view("raptor", "base", "box").Box


def _swap(old, entry, side, name, target):
    """The old unit with the library's text, body, entry, mode, vectorised flag and box."""
    text = entry["texts"][_SERVED[side]]
    given = text.get("box")
    box = given
    if isinstance(given, dict):
        box = (type(old.box) if old.box is not None else _box_class())(**given)
    new = dataclasses.replace(
        old,
        text=text["text"],
        body=text["body"],
        entry=text["entry"],
        mode=text["mode"],
        vectorised=text["vectorised"],
        box=box,
    )
    fields = ("text", "body", "entry", "mode", "vectorised", "box")
    field = next((f for f in fields if getattr(old, f) != getattr(new, f)), None)
    _bump(_C["texts_taken"], side)
    if field is None:
        _C["texts_identical"] += 1
    else:
        _C["texts_different"] += 1
        accepted = name in _accepted()
        _C["texts_different_accepted"] += accepted
        if len(_C["differences"]) < _LIST_LIMIT:
            at, left, right = _first_difference(old.text, new.text)
            _C["differences"].append(
                {
                    "unit": name,
                    "target": side,
                    "test": _test(),
                    "field": field,
                    "line": at,
                    "old": (left or "")[:200] if left is not None else None,
                    "new": (right or "")[:200] if right is not None else None,
                    "accepted": accepted,
                    "adapted": entry["adapted"],
                }
            )
    return new


def _wrap_render(orig):
    """Backend.render: the old render, then the library's text for a routed unit."""

    def render_seam(self, lowered, options):
        old = orig(self, lowered, options)
        k6 = _S["k6"]
        c = k6._CUR.get()
        plan = None if c is None or c.get("orphan") else c.get("kplan")
        entry = None if plan is None else plan["units"].get(lowered.name)
        if entry is None or self.id not in _SERVED or _SERVED[self.id] not in entry["texts"]:
            _bump(_C["texts_old"], self.id)
            why = (
                "no plan"
                if plan is None
                else "target not served"
                if self.id not in _SERVED
                else "kernel kept its old text"
            )
            _bump(_C["texts_old_why"], "%s/%s" % (self.id, why))
            return old
        try:
            new = _swap(old, entry, self.id, lowered.name, self.id)
        except Exception as why:
            _crossing("swap", why)
            _bump(_C["texts_old"], self.id)
            return old
        plan["kernels"][entry["kernel"]]["taken"].add((lowered.name, self.id))
        return new

    render_seam.__wrapped__ = orig
    return render_seam


def _wrap_lower(orig):
    """bundle.lower: at the first unit of a build call, plan the whole call."""

    def lower(kernel, name, smode, kind, targets, spec):
        c = _S["k6"]._CUR.get()
        if c is not None and not c.get("orphan") and "kplan" not in c:
            c["kplan"] = {"units": {}, "kernels": {}}
            try:
                c["kplan"] = _plan(c, kernel, name, smode, tuple(targets))
            except Unbuildable as why:
                _bump(_C["fallbacks"], "request not buildable: %s" % why)
            except Exception as why:
                _crossing("plan", why)
        return orig(kernel, name, smode, kind, targets, spec)

    lower.__wrapped__ = orig
    return lower


def _wrap_unit_key(orig):
    """bundle._unit_key: the names and walk digests of every unit the old lowering will produce."""

    def unit_key(names, digests, *a, **k):
        c = _S["k6"]._CUR.get()
        if c is not None:
            c["old_units"] = list(zip(names, digests))
        return orig(names, digests, *a, **k)

    unit_key.__wrapped__ = orig
    return unit_key


def _flush(c):
    """End of a build call: counts it, and the kernels whose texts were only partly taken."""
    if c.get("orphan"):
        return
    _C["build_calls"] += 1
    plan = c.get("kplan")
    for info in (plan or {"kernels": {}})["kernels"].values():
        want = len(info["units"]) * len(info["served"])
        if info["taken"] and len(info["taken"]) != want:
            _C["kernels_incomplete_after_error" if c.get("error") else "kernels_incomplete"] += 1


def install(lib_path):
    """Opens the library and installs the recorder's hooks and the kernel route."""
    import k6_plugin as k6
    import text_plugin as tp
    from hawk import backends
    from hawk.artifact import bundle

    here = os.path.dirname(os.path.abspath(lib_path))
    _S["facts_dir"] = next(
        (
            os.path.abspath(facts)
            for facts in (os.path.join(here, "facts"),)
            if os.path.isdir(facts)
        ),
        None,
    )
    _S["k6"] = k6
    _S["ctx"] = _osprey().Ctx()
    _S["lib"] = _S["ctx"].backend_open(lib_path)
    k6._ON = True
    bundle.lower = _wrap_lower(bundle.lower)
    bundle._unit_key = _wrap_unit_key(bundle._unit_key)
    k6.pytest_configure(None)
    k6._flush = _flush
    backends.Backend.render = _wrap_render(backends.Backend.render)
    tp._state["test"] = "collection"
    atexit.register(write)


def clear():
    """Removes the counters of an earlier run in HOST_DIR (the controller does it at the start)."""
    for path in glob.glob(os.path.join(_DIR or "", "k[0-9]*.json")) + glob.glob(
        os.path.join(_DIR or "", "kroute_counters.json")
    ):
        os.remove(path)


def on_test(nodeid):
    """Called at the start of each test."""
    import text_plugin as tp

    tp._state["test"] = nodeid
    if _S["k6"] is not None and not _S["k6"]._SEAM0:
        _S["k6"]._patched_now()


def write():
    """Writes this worker's counters."""
    if not _DIR or _S["k6"] is None:
        return
    os.makedirs(_DIR, exist_ok=True)
    with open(os.path.join(_DIR, "k%d.json" % os.getpid()), "w") as handle:
        json.dump(_C, handle, sort_keys=True)


# ------------------------------------------------------------------ the merged counters


def merge(folder):
    """The sum of the workers' counters in the folder."""
    total = None
    for path in sorted(glob.glob(os.path.join(folder, "k[0-9]*.json"))):
        with open(path) as handle:
            part = json.load(handle)
        if total is None:
            total = json.loads(json.dumps(part))
            total["workers"] = 1
            continue
        total["workers"] += 1
        for key, value in part.items():
            if isinstance(value, int):
                total[key] += value
            elif isinstance(value, dict):
                for sub, n in value.items():
                    total[key][sub] = total[key].get(sub, 0) + n
            else:
                total[key] = (total[key] + value)[:_LIST_LIMIT]
    return total


def summary(total):
    """The counter lines of a merged total."""
    if total is None:
        return ["kroute: no counters"]
    fallbacks = sum(total["fallbacks"].values())
    lines = [
        "kroute: %d workers, %d build calls (%d with units), %d kernels: routed whole %d, "
        "fell back %d, incomplete %d (after a build error %d); "
        "units routed %d, unclaimed old units %d"
        % (
            total["workers"],
            total["build_calls"],
            total["calls_with_units"],
            total["kernels"],
            total["kernels_routed"],
            fallbacks,
            total["kernels_incomplete"],
            total["kernels_incomplete_after_error"],
            total["units_routed"],
            total["unclaimed_old_units"],
        ),
        "kroute: texts taken from the library %s, left old %s %s; identical %d, different %d "
        "(on the accepted list %d); "
        "adapted requests %s; crossing errors %d; device facts asked with the stated 12.6 line %d"
        % (
            total["texts_taken"],
            total["texts_old"],
            total["texts_old_why"],
            total["texts_identical"],
            total["texts_different"],
            total["texts_different_accepted"],
            total["adapted"],
            total["crossing"],
            total["device_line_stated"],
        ),
    ]
    lines += [
        "kroute: fallback %5d  %s" % (n, why)
        for why, n in sorted(total["fallbacks"].items(), key=lambda i: -i[1])
    ]
    lines += [
        "kroute: FALLBACK %s" % json.dumps(d, default=str) for d in total["fallback_examples"][:6]
    ]
    lines += ["kroute: DIFFERENT %s" % json.dumps(d) for d in total["differences"][:10]]
    lines += ["kroute: CROSSING %s" % json.dumps(d) for d in total["crossings"][:10]]
    return lines


def report(folder):
    """Merges, writes HOST_DIR/kroute_counters.json and returns the lines to print."""
    total = merge(folder)
    if total is not None:
        with open(os.path.join(folder, "kroute_counters.json"), "w") as handle:
            json.dump(total, handle, sort_keys=True, indent=1)
    return summary(total)


if __name__ == "__main__":
    print("\n".join(report(sys.argv[1])))
