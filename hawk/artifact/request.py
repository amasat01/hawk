# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The request one kernel sends to the lowering library, built from the kernel, the settings of the build and the target rows.

Every member is read from the objects the build holds; nothing is guessed from a record and nothing is retried. The members the
library defaults (a group of one, no planes, settings at their defaults) are left out, as the
library's own writer leaves them out. :func:`sinks_of` lists the nodes a request names; their positions in the dump come from
:func:`hawk.artifact.bridge.dump`."""

from __future__ import annotations

import functools
import json
import os
import platform
import sys
from pathlib import Path

from raptor.ir.nodes import LEAF_ROLES
from raptor.ir.types import Slot
from raptor.wings.base.exchange import ToolchainAbsent

from .. import wings as _wings
from .._errors import HawkError
from ..compile import toolchain as _tc
from ..ext import PLAIN, DEFAULT_GUARD, DATA_ONLY, DEFAULT_KIND
from ..trace import steps as _steps
from ..trace.decl import Plane
from ..targets import TARGETS
from ..trace.kernel import _declared, derived_axes

_cells = _wings.view("raptor", "base", "cells")

#: The targets the library serves: the id of a row, the name the request spells it by.
SERVED = {"host": "Host", "cuda": "Device"}
_SCALARS = {"float64": "Float64", "float32": "Float32", "banded": "Banded"}


def _camel(word: str) -> str:
    """``per_sample`` as ``PerSample``: the way the request spells a name of a closed set."""
    return "".join(part.capitalize() for part in word.split("_"))


# ------------------------------------------------------------------ the library


@functools.lru_cache(maxsize=None)
def _library_file() -> str:
    suffix = "dylib" if sys.platform == "darwin" else "so"
    beside = Path(__file__).resolve().parents[2] / "osprey" / f"libosprey_host.{suffix}"
    path = str(beside) if beside.is_file() else os.environ.get("OSPREY_HOST_LIB")
    if not path:
        raise HawkError("the lowering library is not found: put it at osprey/ beside the hawk package, or name its file in OSPREY_HOST_LIB")
    return path


@functools.lru_cache(maxsize=None)
def library() -> tuple:
    """``(binding module, open library, facts folder)``: the one place the lowering library and its data folder are found. The
    library is the file ``osprey/libosprey_host.<dylib or so>`` beside the hawk package when it exists, else the file named by
    ``OSPREY_HOST_LIB``; the data folder is ``facts`` beside the library, or two levels above it."""
    import _osprey

    path = _library_file()
    here = Path(path).resolve().parent
    facts = next((d for d in (here / "facts", here.parent.parent / "facts") if d.is_dir()), None)
    if facts is None:
        raise HawkError(f"the lowering library's data folder is not found beside {path}")
    return _osprey, _osprey.Ctx().backend_open(path), str(facts)


@functools.lru_cache(maxsize=None)
def library_identity() -> str:
    """The digest of the library file: a unit key names the library that wrote the unit."""
    import hashlib

    return hashlib.sha256(Path(_library_file()).read_bytes()).hexdigest()


@functools.lru_cache(maxsize=None)
def _facts(target: str, compiler: str, stated: str) -> str:
    """The library's facts answer for a target under a compiler and a stated machine, asked once per process for each."""
    module, lib, folder = library()
    return module.Ctx().backend_facts(lib, folder, target, compiler, stated)


def _macro(flag: str):
    """The macro a ``-D<name>`` or ``-D<name>=<value>`` flag defines (``None``: the flag defines none)."""
    return flag[2:].split("=", 1)[0] if flag.startswith("-D") else None


def compiling_host() -> tuple:
    """``(facts, line)`` of the machine and the compiler that COMPILE a host build: this machine's facts as its backend states them
    (instruction set, vector set, profile in effect, caches) and the identity of the compiler the driver runs, in the library's
    three-part form. They are the owner's facts of the compile, whatever machine a spec's texts are written for
    (:func:`_host_line`, :func:`_stated_host`)."""
    host = TARGETS["host"]
    return host.machine_facts(), _identity(_tc.compiler_identity(host.compiler()))


def host_build(facts: dict, line: str, options, witness=None, caller_flags=()) -> dict:
    """The library's instructions for the host compile of this build, the ``host_build`` record of its facts answer: the argv
    templates of a unit's compile, of the set text's compile and of the link. The question is about the machine and the compiler
    that run the compile (``facts`` and ``line``, from :func:`compiling_host`), never about the machine a spec's texts are written
    for: a flag of another machine does not compile here. It carries the host profile the build asks for, when it asks for one, and
    the optimisation level in effect.

    The masked-call flag is in the template only for a verdict of the witness. ``witness`` (``None``: there is none) is asked for the
    verdict when the library's templates for a fixed compiler and for an unproven one differ: it is called with the template of the
    unproven one and returns ``fixed``, ``miscompile`` or ``unrun``, which the question then states. A macro the caller sets
    itself wins without a probe: when one of ``caller_flags`` (the build's own ``-D`` flags) defines a macro that the fixed
    template's extra flags define, the witness is not run and no verdict is stated. Which macro that is comes from the library's
    two templates, never from a name written here."""
    stated = _stated(facts["isa"], facts)
    if options.get("host_profile"):
        stated["host_profile"] = options["host_profile"]
    stated["opt_level"] = _tc.opt_level(options.get("opt_level") or None)

    def ask(verdict=None) -> dict:
        said = {**stated, **({} if verdict is None else {"masked_call_witness": verdict})}
        return json.loads(_facts("host", line, json.dumps(said)))["host_build"]

    try:
        build = ask()
        # what a fixed compiler adds to the unit compile, and the macros those flags define
        added = [flag for flag in ask("fixed")["compile_unit"] if flag not in build["compile_unit"]]
        theirs = {_macro(flag) for flag in added} - {None}
        mine = {_macro(str(flag)) for flag in caller_flags} - {None}
        if witness is not None and added and not theirs & mine:
            build = ask(witness(build["compile_unit"]))
        return build
    except library()[0].BadRequest as why:
        # a profile (asked for in the options or the environment, else the machine's own) that this machine's compiler does not build:
        # the machine lacks the toolchain for it
        raise ToolchainAbsent(f"host profile {stated.get('host_profile')!r}: {why}", toolchain="host-profile") from why


def set_text(units: list, exports=None) -> str:
    """The C++ text of a set of units built into one library: the self-check block once and the library record over ``units``
    (their ``meta`` records in the answers, in the order they are linked). ``exports`` are the five layout sizes to state in
    place of the ones the headers give, when the build is given them."""
    module, lib, _ = library()
    asked = {"v": 1, "units": units, **({} if exports is None else {"exports": [int(n) for n in exports]})}
    try:
        return module.Ctx().backend_set_text(lib, json.dumps(asked))
    except ValueError as why:
        raise HawkError(f"the lowering library refuses the set of units {[u['name'] for u in units]}: {why}") from why


def lower(request: dict, dump: str, dialects: str) -> dict:
    """The library's answer to ``request`` over the nodes of ``dump``, as parsed JSON: one context for the call, made and dropped
    here."""
    module, lib, folder = library()
    try:
        return json.loads(module.Ctx().backend_lower(lib, folder, dump, dialects, json.dumps(request)))
    except ValueError as why:
        raise HawkError(f"the lowering library refuses {request['name']!r}: {why}") from why


# ------------------------------------------------------------------ the compiler and the machine


def _identity(line: str) -> str:
    """The compile driver's ``<path>||<version line>||<digest>`` as the three-part line the library reads."""
    return " || ".join(line.split("||"))


@functools.lru_cache(maxsize=None)
def _this_isa() -> str:
    return "arm64" if platform.machine().lower() in ("aarch64", "arm64") else "x86_64"


#: The compiler lines of a machine no compiler can be asked about, by what is missing. They stand in until the library can answer
#: the facts of a stated compiler family with no compiler present; nowhere else is a compiler line made up.
_NO_COMPILER = {
    "another x86_64 host": "g++||g++ (GCC) 13.2.0||0",
    "no device toolchain": "nvcc||Cuda compilation tools, release 12.6, V12.6.77||0",
}

def _host_line(spec, options) -> str:
    """The compiler identity a host unit is written under: the compiler the compile driver runs here; but when the spec names
    another instruction set than this machine's and a vector-math profile, which an x86-64 host takes only with GCC, the GCC line of
    that machine."""
    machine = spec.machine
    if machine.isa != _this_isa() and machine.isa == "x86_64" and machine.facts.get("host_profile") not in machine.facts.get("exact_host_profiles", ()):
        return _identity(_NO_COMPILER["another x86_64 host"])
    return _identity(_tc.compiler_identity(TARGETS["host"].compiler()))


def _compile_plan(target: str, spec, options):
    """The plan the compile driver will run for the target of this build."""
    from ..compile import CompileOptions, drivers
    row = TARGETS[target]
    return drivers.plan_for(
        CompileOptions(target=target, mode=spec.settings.scalar, arch=options.get("device_arch", ""),
                       host_profile="", opt_level=options.get("opt_level") or "",
                       settings=spec.for_target(row).settings))


def _device_line(identity: str) -> str:
    """The device compiler's identity in the library's three-part form. The first version line of
    the device compiler driver names no release, and the in-process compiler names its release, not
    a path: it is written out here."""
    if "||" not in identity:
        from ..compile import nvrtc

        major, minor = nvrtc.nvrtc_version()
        digest = identity.partition("cccl:")[2]
        identity = f"nvrtc||Cuda compilation tools, release {major}.{minor}, V{major}.{minor}.0||{digest}"
    return _identity(identity)


def device_build(identity: str, *, arch: str, release: str | None, driver: str | None,
                 level: str) -> dict:
    """The library's instructions for the device compile of a build, the ``device_build`` record of
    its facts answer: the form of image the build makes (``image_format``), the argv template of the
    one compile that makes it (``compile``) and the recipe that embeds the image file in the set
    library (``embed``).

    hawk states, as the facts of the machine and the compiler that compile: ``identity`` (the
    compile driver's identity of the device compiler), its ``arch``, its ``release``
    (``major.minor``; ``None``: the identity names it), the ``driver`` release the device driver
    supports (``major.minor``; ``None``: unknown, which the library answers with the form that needs
    no driver) and the optimisation ``level``. The library states the form, every flag and the
    texts; hawk fills the places."""
    stated = {"device_arch": arch, "opt_level": level}
    if release:
        stated["device_release"] = release
    if driver:
        stated["driver_release"] = driver
    return json.loads(_facts("cuda", _device_line(identity), json.dumps(stated)))["device_build"]


def _device(spec, options) -> tuple:
    """``(stated device machine, compiler line, image format)`` of the device compile of this build. The stated machine holds the
    architecture when one is asked for and the compiler's release as its compile plan states it: the first version line of the
    device compiler driver names no release, so the line alone cannot tell the library which compiler it is."""
    arch = spec.machine.arch or options.get("device_arch") or ""
    stated = _under_cards({"device_arch": arch} if arch else {}, spec)
    try:
        plan = _compile_plan("cuda", spec, options)
    except ToolchainAbsent:
        return stated, _identity(_NO_COMPILER["no device toolchain"]), "cubin"
    if plan.get("release"):
        stated["device_release"] = plan["release"]
    return stated, _device_line(plan["identity"]), plan["meta"]["device_target"]


def _stated_host(spec) -> dict:
    """The machine a host unit's TEXT is written for (the spec's) and where its compile finds the headers, as the library's facts
    entry takes them."""
    return _under_cards(_stated(spec.machine.isa, spec.machine.facts), spec)


def _under_cards(stated: dict, spec) -> dict:
    """``stated`` with the documents the build states (:func:`hawk.targets.build_spec`), so the facts answer is made from them; the
    request carries the same documents, and the library refuses an answer made under others."""
    return {**stated, "cards": dict(spec.cards)} if spec.cards else stated


def _stated(isa: str, facts) -> dict:
    """``isa`` and its ``facts`` (a machine's, as :class:`hawk.targets.Machine` holds them) and where a compile finds the headers, as
    the library's facts entry takes them."""
    from ..compile import drivers

    stated = {"isa": isa, "vector_isa": facts.get("vector_isa"), "host_profile": facts.get("host_profile")}
    stated = {key: value for key, value in stated.items() if value is not None}
    stated["headers"] = drivers.header_shape()
    l1, l2 = facts.get("l1d_cache_bytes"), facts.get("l2_cache_bytes")
    if l1 and l2:
        stated.update(l1d_bytes=l1, l2_bytes=l2, cache_source=facts.get("cache_source") or "")
    return stated


# ------------------------------------------------------------------ the members


def _slot(slot) -> dict:
    out = {"role": _camel(slot.role), "name": slot.name, "dtype": _camel(slot.ttype.dtype), "shape": list(slot.ttype.shape)}
    if slot.ttype.tag is not None:
        out["tag"] = slot.ttype.tag
    return out


def _mixes(body) -> tuple:
    """The branch-mix plane of each ``auto`` dispatch the body's walk binds."""
    found = ((("lookup", _cells.mix_name(info.name))) for info in body.walk.dispatches)
    return tuple(Slot(role, name, body.walk.slot_types[role, name]) for role, name in found if (role, name) in body.walk.slot_types)


def _member(body, name: str, position: dict) -> dict:
    """A body of the request: the nodes it writes and how its walk is taken. A kernel declares its terminated planes, the
    companion plane of each compensated sum and the branch-mix plane of each ``auto`` dispatch; a body with no planes of its own
    declares the slots its walk binds."""
    planes = getattr(body, "planes", None)
    if planes is not None:
        pairs = _cells.companion_pairs(body.sinks)
        declared = [_slot(slot) for slot in _declared(planes)]
        declared += [{**_slot(slot), "compensates": row.name} for row, slot in pairs]     # the sum row the plane compensates
        declared += [_slot(slot) for slot in _mixes(body)]
    else:
        declared = [_slot(Slot(role, slot, body.walk.slot_types[role, slot])) for role, slot in body.walk.arg_spec if role in LEAF_ROLES]
    return {
        "name": name,
        "sinks": [position[id(sink)] for sink in body.sinks],
        "declared": declared,
        "exempt": [_cells.FUSED_STEPS_PLANE],
    }


def _kind(kind) -> dict:
    """The kind of the kernel as the request states it; a policy of another form is refused."""
    guard = kind.guard
    if guard is DEFAULT_GUARD:
        out = "Default"
    elif guard is DATA_ONLY:
        out = "DataOnly"
    else:
        out = {"Masks": list(guard.names)}
    if kind.sink is PLAIN or (kind.sink.id == "plain" and kind.sink.compensation is None):
        sink = "Plain"
    elif kind.sink.compensation is None:
        raise HawkError(f"kind {kind.slug!r}: a sink policy of another form ({kind.sink.id!r}) is not built")
    else:
        sink = {"Compensated": {"plane": kind.sink.compensation, "output_names": list(kind.output.names),
                                **({} if kind.output.slot is None else {"output_slot": kind.output.slot})}}
    return {"guard": out, "active_set": bool(guard.active_set), "sink": sink,
            "lanes": None if kind.lanes is None else {"bits": kind.lanes.bits, "prefix": kind.lanes.prefix}}


def _axes(pairs) -> list:
    return [{**({} if label is None else {"label": label}), **({} if extent is None else {"extent": extent})} for label, extent, *_ in pairs]


def _settings(spec, options, targets) -> dict:
    """The settings of the build that differ from the library's defaults."""
    s = spec.settings
    out = {}
    for key, value in (("fold_form", s.form), ("wide_form", s.wide_form), ("linalg_form", s.linalg_form)):
        if value != "auto":
            out[key] = _camel(value)
    if s.route_to_ordered:
        out["route_to_ordered"] = True
    if s.launch_bounds is not None:
        out["launch_bounds"] = [int(n) for n in s.launch_bounds]
    if s.fold_k:
        out["fold_k"] = s.fold_k
    if s.warps != 1:
        out["warps"] = s.warps
    if s.team_threads:
        out["team_threads"] = s.team_threads
    sizes = options.get("layout_sizes_override")
    if sizes is not None:
        out["exports"] = [int(n) for n in sizes]
    if any(TARGETS[t].own_options(options).get("literal_table") for t in targets):
        out["literal_table"] = True
    return out


def sinks_of(kernel) -> list:
    """Every node the kernel's request can name or the lowering reads: the sinks of the kernel, of its single step and of its
    members, and of theirs in turn."""
    out = list(kernel.sinks)
    one = getattr(kernel, "one_step", None)
    if one is not None:
        out += sinks_of(one)
    for member in getattr(kernel, "members", None) or ():
        if hasattr(member, "sinks") and hasattr(member, "walk"):
            out += sinks_of(member)
    return out


def request(kernel, *, name: str, scalar: str, kind, targets, spec, position: dict, options: dict, exports_block: bool = True) -> dict:
    """The request of ``kernel`` for ``targets`` as a dict: ``name`` the unit name wanted, ``scalar`` the scalar mode's id, ``kind``
    the kind the kernel is built under, ``spec`` the build spec with the settings resolved, ``position`` the handle of each node
    (:func:`hawk.artifact.bridge.dump`), ``options`` the build inputs (:func:`hawk.artifact.bundle._options`). The texts carry the
    self-check block of a library unless ``exports_block`` is false, which a build asks: its units are linked with the text of the
    set (:func:`set_text`), which carries the block once."""
    for target in targets:
        if target not in SERVED:
            raise HawkError(f"target {target!r} is refused: a back end registered from Python is not built in this version; the build writes {tuple(SERVED)} only")
    kind = kind if kind is not None else DEFAULT_KIND
    members = getattr(kernel, "members", None)
    if members and hasattr(members[0], "sinks"):
        body = {"Chain": [_member(m, m.name, position) for m in members]}
    else:
        body = {"Plain": _member(kernel, name, position)}
    out = {"v": 1, "name": name, "scalar": _SCALARS[scalar], "targets": [SERVED[t] for t in targets], "kind": _kind(kind), "body": body}
    facts = {}
    for target in targets:
        if target == "host":
            line = _host_line(spec, options)
            facts["Host"] = _facts("host", line, json.dumps(_stated_host(spec)))
        else:
            stated, line, image = _device(spec, options)
            facts["Device"] = _facts("cuda", line, json.dumps(stated))
            out["image_format"] = _camel(image)
    one = getattr(kernel, "one_step", None)
    if one is not None:
        out["one_step"] = _member(one, one.name, position)
    if getattr(kernel, "group", 1) != 1:
        out["group"] = kernel.group
    planes = getattr(kernel, "planes", None) or {}
    if planes:
        out["planes"] = list(planes)
    settings = _settings(spec, options, targets)
    if settings:
        out["settings"] = settings
    steps = getattr(kernel, "steps", None)
    if isinstance(steps, int) and steps >= 1:
        out["steps"] = steps
    elif steps == "auto":
        out["steps"], out["steps_max"] = "auto", _steps.AUTO_K_MAX
    lanes = _axes(getattr(kernel, "lane_space", ((None, None),)))   # the default the sidecar writer takes
    if lanes:
        out["lane_space"] = lanes
    sinks = [(n, p) for n, p in planes.items() if isinstance(p, Plane) and p.form == "accum"]
    if type(kernel.sinks).__name__ == "Derived" and not planes:
        sinks += derived_axes(kernel.walk, getattr(kernel.sinks, "source", None)).items()
    # A sink with named axes is bound (payload, then the named axes): the payload is the plane's own per-sample component,
    # which only the declaration knows (empty for a scalar plane).
    axes = [{"name": n, "axes": _axes(p.dims), "payload": list(p.ttype.shape)} for n, p in sinks if p.dims]
    if axes:
        out["sink_axes"] = axes
    shared = [n for n, p in planes.items() if isinstance(p, Plane) and p.shared]
    if shared:
        out["shared"] = shared
    bound = {name for _, name in kernel.walk.arg_spec}
    broadcast = [{"plane": n, "axes": [flag is True for flag in pattern]}
                 for plane, pattern in (getattr(kernel, "broadcast", None) or {}).items()
                 for n in (plane, "dot_" + plane) if n in bound]
    if broadcast:
        out["broadcast"] = broadcast
    out["facts"] = facts
    if spec.cards:
        out["cards"] = dict(spec.cards)
    if not exports_block:
        out["exports_block"] = False
    return out
