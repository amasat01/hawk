# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The v2 sidecar: the FULL field set, every block generated from ``Walk``.

Every slot-naming block below is a projection of
:attr:`hawk.ir.walk.Walk.arg_spec` and :attr:`~hawk.ir.walk.Walk.slot_types`
— nothing re-enumerates the walk's leaves. The set is: the shared base
(``format``, ``schema_version``, ``pattern``, ``aether_abi``,
``scalar_type``, ``kernel``, ``params``/``params_schema``, ``per_sample``,
``vector_inputs``, ``terminated_readonly``, ``host_entry``, ``derivative``)
plus the pattern-specific blocks every consumer actually iterates —
``arg_spec`` (the load-bearing key both C++ registries and
``eagle.plan._pack_args`` walk), ``buffers``, and the ``mutables`` dtype
declarations ``eagle.roles.classify_arg`` resolves a ``mutable`` role by
(without them a ``Mutable[Vector[3]]`` packs as a 32-byte handle where the
kernel expects a 40-byte mirror) — plus the execution axis, the resolved
header roots and ``Walk.digest``, which live here because the manifest's
key order is contractual.

``arg_widths`` covers input planes too, and two fields — ``arg_dtypes``,
``arg_shapes`` — close the rest: a v2 consumer otherwise has no way to
check a bound plane's shape against what the body was compiled for, nor to
tell an integer-valued handle from a Real one.

``vec_widths`` is the schema's per-name vector width (``arg_widths`` restricted to
the vector inputs), so a host can allocate ``(W, N)`` buffers.

``arg_widths`` stays INT-ONLY and NEVER ``None`` — a load-bearing
constraint. ``eagle.plan._declared_widths`` reads a plugin's
``arg_widths`` through an unconditional ``int(v)`` on every value,
including straight off a deployed sidecar with no HAWK projection in
between; a ``None`` for a runtime-length role (``lookup``, ``wide_in``,
``wide_out``, ``accum_out``) crashes that path with a ``TypeError``. So
``arg_widths`` widens to cover every statically-width-known role, but a
runtime-length role stays OMITTED — an absent key, not a ``None`` value,
which every existing reader already treats as "no declared width".

The runtime-length fact still deserves a positive declaration, not
silence a reader infers from a role name. Two new fields carry it, both
complete over every plane-bound slot, and neither read by any existing
consumer:

* ``arg_shapes`` — the same flat width ``arg_widths`` carries for a
  statically-shaped role, ``None`` for a runtime-length one; kept
  separate so completing it can never poison a caller that assumes every
  ``arg_widths`` value is an int.
* ``arg_dtypes`` — the plane's wire dtype, for every role including the
  runtime-length ones: the compiled ``scalar_type`` for an ``f64``/``f32``
  leaf (the compiled MODE decides the wire width, not the IR's own static
  ``f64``), always ``int64`` for an ``i32``/``i64`` index leaf (HAWK's
  emitted entry spells every index integer ``Int`` regardless of declared
  width), ``int32``/``uint32`` for an integer plane (the reserved 4-byte
  planes: ``active_map`` is ``int32`` (``int64`` for a 64-bit lane set); the counters ``finished_count``,
  ``active_count`` and an ``auto`` dispatch's ``<group>_mix`` are
  ``uint32``), and ``bool`` for a ``Terminated`` mask.

All three fields are additive, so no ``schema_version`` bump accompanies
them; :func:`hawk.artifact.plan_view` projects all three straight through.
"""

from __future__ import annotations

from .. import _contracts
from ..compile.toolchain import _strict_aether_include, _strict_eagle_include
from ..ir import HawkError
from ..ir.segment import class_name, mix_name, stale_name
from ..targets import ENTRY_KINDS
from ..types import INT_PLANE_DTYPES


def _root_or_sealed_marker(resolver) -> str:
    """The provenance value for one header root — the real resolved root
    when one exists, else a SEALED marker naming the payload it was built
    from (a wheel install carries no C++ header tree, so calling
    ``resolver`` unconditionally would refuse even after a successful
    compile). Re-raises when neither exists: that combination means the
    kernel could not have been compiled at all."""
    try:
        return resolver()
    except HawkError:
        from ..compile import payload as _payload
        return f"sealed:{_payload.current_payload().digest}"

#: The ``params`` block's own wire version: decl-carrying ``{name, dtype}``
#: entries. HAWK never writes the v1 bare-name shape — an integer uniform
#: bound through a ``double`` is wrong from 2^53 up and produces a
#: plausible number.
PARAMS_SCHEMA = 2

#: A writable slot's declared dtype, by rank: the context
#: :func:`eagle.roles.classify_arg` resolves a ``mutable`` role's ABI shape by.
MUTABLE_DTYPES = {0: "scalar", 1: "vector", 2: "matrix"}


def _names(walk, role) -> list:
    return [name for r, name in walk.arg_spec if r == role]


def _mutable_dtype(ttype) -> str:
    if not ttype.shape and ttype.dtype in ("i32", "i64"):
        return "int"
    if ttype.dtype in INT_PLANE_DTYPES:
        return ttype.dtype
    return MUTABLE_DTYPES[len(ttype.shape)]


def _mutables(walk) -> list:
    out = []
    for role, name in walk.arg_spec:
        if role not in ("mutable", "out"):
            continue
        t = walk.slot_types[(role, name)]
        entry = {"name": name, "dtype": _mutable_dtype(t),
                 "width": _width(t), "default": None}
        if len(t.shape) == 2:
            entry["shape"] = list(t.shape)
        if name in walk.prior_reads:
            # The body reads this plane's launch-start value through
            # `.prior`: the host must preserve it between launches, never
            # treat it as scratch.
            entry["prior"] = True
        out.append(entry)
    return out


#: The local memory one thread may hold, per CUDA (all targeted archs), in bytes.
LOCAL_LIMIT_B = 512 * 1024


def _launch(lane_space, group: int, box=None, block: int | None = None,
            index_set: dict | None = None, local: tuple | None = None) -> dict:
    """The ``launch.lane_space`` block: the launch shape (row-major, the last axis
    fastest, only the first extent may be ``None``: known at run time) and the
    compaction ``group``, a lane count. With a thread ``box``
    (:class:`hawk.emit.box.Box`), ``launch.box`` = ``{axis, T, stride}``: a
    launch of ``n`` lanes runs ``ceil(n / (T * stride)) * stride`` threads
    (:func:`hawk.emit.box.threads`), and ``base``/``count`` count threads.
    ``block`` is the exact threads per block an emit that fixed one (a staged
    tile) needs; a launch uses it verbatim. ``index_set`` is the record of a kernel
    declaring ``lanes=IndexSet()``: ``launch.index_set`` = ``{"width": bits}`` (the map word, the host
    bind's cue to range- and duplicate-check the caller's ``active_map``), or, for the prefix form
    (``IndexSet(prefix=True)``), ``{"width": 32, "prefix": true, "count": "active_count"}``: no map, the
    named one-element ``uint32`` plane holds how many leading lanes run. Absent for every other kernel.
    ``local`` is ``(bytes, source)`` of a device unit: ``launch.local_bytes`` is the local memory one thread
    holds (``source`` ``launch.local_source``: ``"ptxas"``, the stack frame of a CUBIN, or ``"ptx"``, the
    depots the PTX declares, a ceiling), ``launch.local_limit`` the device's per-thread limit. A launch of
    ``n`` threads needs about ``n * local_bytes`` of device memory beside the planes. No threads-per-block
    cap follows from it (hawk emits no ``__launch_bounds__``)."""
    axes = [{"label": label, "extent": extent} for label, extent in lane_space]
    out = {"lane_space": {"axes": axes, "group": int(group)}}
    if box is not None:
        out["box"] = box.sidecar()
    if index_set is not None:
        out["index_set"] = dict(index_set)
    if local is not None:
        out["local_bytes"], out["local_source"], out["local_limit"] = int(local[0]), local[1], LOCAL_LIMIT_B
    if block is not None:
        if not isinstance(block, int) or isinstance(block, bool) or block < 1:
            raise ValueError(f"launch.block must be an integer >= 1, got {block!r}")
        out["block"] = block
    return out


def _sorted_inputs(walk, planes) -> list:
    """The bound inputs declared ``Sorted[Index]``."""
    return [name for _role, name in walk.arg_spec if getattr(planes.get(name), "sorted", False)]


def _index_planes(walk) -> list:
    """The bound input planes declared ``Index`` (32-bit on the wire): a runtime may take a
    wider integer array for one when every value fits ``int32``."""
    return [name for role, name in walk.arg_spec if walk.slot_types[(role, name)].tag == "index"]


def _plane_axes(walk, planes) -> dict:
    """``{name: {payload, axes}}`` for each bound sink declared with named axes:
    it is bound ``(*payload, *axes)``."""
    out = {}
    for role, name in walk.arg_spec:
        dims = getattr(planes.get(name), "dims", ())
        if dims and role in _SINK_ROLES:
            out[name] = {"payload": list(planes[name].ttype.shape),
                         "axes": [{"label": label, "extent": size} for label, size, _s in dims]}
    return out


def _width(ttype) -> int:
    w = 1
    for e in ttype.shape:
        w *= int(e)
    return w


def _params(walk) -> list:
    return [{"name": name,
             "dtype": ("int" if walk.slot_types[(role, name)].dtype
                       in ("i32", "i64", *INT_PLANE_DTYPES) else "float")}
            for role, name in walk.arg_spec if role == "uniform"]


#: Roles bound to a plane whose leading (component) extent is a static
#: fact of the compiled body — the flat product of the slot's shape, 1 for
#: a rank-0 plane. The trailing (sample-count) axis is never declared
#: here, since it's a runtime quantity by construction.
_SAMPLE_SHAPED_ROLES = ("per_sample", "vec_in", "mat_in", "terminated", "mutable",
                        "out")

#: Roles bound to a plane whose own LENGTH is a runtime quantity nothing
#: in the walk knows. Omitted from ``arg_widths`` (never a guessed number,
#: never ``None`` either — see the module docstring); declared ``None`` in
#: ``arg_shapes`` instead.
_RUNTIME_LENGTH_ROLES = ("lookup", "wide_in", "wide_out", "accum_out")

#: The roles a body writes.
_SINK_ROLES = ("mutable", "out", "wide_out", "accum_out")

#: Every role that binds a memory plane at all — the domain ``arg_widths``
#: and ``arg_dtypes`` are complete over. ``uniform``/``nsamples`` are
#: excluded: neither carries a plane, so neither has a width or dtype.
_PLANE_ROLES = _SAMPLE_SHAPED_ROLES + _RUNTIME_LENGTH_ROLES

#: The wire dtype an ``i32``/``i64`` or ``bool`` leaf carries, fixed
#: regardless of the compiled scalar mode; ``None`` for ``f64``/``f32``
#: marks "look at ``scalar_type`` instead" — see :func:`_wire_dtype`.
_FIXED_WIRE_DTYPE = {"f64": None, "f32": None, "i32": "int64", "i64": "int64",
                     "int32": "int32", "uint32": "uint32",
                     "bool": "bool"}


def _wire_dtype(ttype, scalar_type: str) -> str:
    """The WIRE dtype string one plane-bound slot actually carries.

    An ``f64``/``f32`` leaf is not reported at its own IR dtype: every
    payload constructor bakes ``f64`` in statically, but it's the
    artifact's compiled ``scalar_type`` that decides the wire's byte width
    (a ``mode="float32"`` build carries ``float32`` planes end to end
    regardless). An ``i32``/``i64`` leaf is reported ``int64``
    unconditionally, since HAWK's emitted entry spells every index
    integer ``Int`` (64-bit) regardless of the IR's declared width; an
    ``int32``/``uint32`` plane (the reserved 4-byte counters among them) is
    reported at its own 4-byte dtype.
    ``bool`` passes straight through: the one dtype that never widens."""
    fixed = _FIXED_WIRE_DTYPE[ttype.dtype]
    return fixed if fixed is not None else scalar_type


def _arg_widths(walk) -> dict:
    """The per-slot width block, widened to every statically-shaped
    plane-bound role, but still int-only and silent (an absent key, never
    ``None``) about a runtime-length role. See the module docstring for
    why; :func:`_arg_shapes` is where that fact is declared instead."""
    return {name: _width(walk.slot_types[(role, name)])
            for role, name in walk.arg_spec if role in _SAMPLE_SHAPED_ROLES}


def _arg_shapes(walk) -> dict:
    """The complete per-slot shape block: :func:`_arg_widths` plus an
    explicit ``None`` for every runtime-length role. Kept as its own key
    so completing it can never poison a caller that assumes every
    ``arg_widths`` value is an ``int``."""
    out = dict(_arg_widths(walk))
    for role, name in walk.arg_spec:
        if role in _RUNTIME_LENGTH_ROLES:
            out[name] = None
    return out


def _arg_dtypes(walk, scalar_type: str) -> dict:
    """The per-slot wire dtype block — every plane-bound role, inputs
    included, so a consumer can tell an integer-valued handle from a Real
    one without guessing."""
    return {name: _wire_dtype(walk.slot_types[(role, name)], scalar_type)
            for role, name in walk.arg_spec if role in _PLANE_ROLES}


def _dispatches(infos) -> list:
    """The per-kernel dispatch listing: ``{name, K, policy,
    static_selector, selector_reads}`` (the kernel-scoped classification and
    the selector's read slots a launching runtime re-scopes to a whole
    step), straight off
    an iterable of :class:`~hawk.ir.walk.DispatchInfo` — ordinarily
    :attr:`~hawk.ir.walk.Walk.dispatches` itself. A SEGMENTED unit's walk
    carries no Dispatch node at all (the per-unit split already replaced
    it), so :func:`sidecar_meta`'s ``dispatch_override`` supplies the
    original group's record instead, K ``units`` names included."""
    out = []
    for d in infos:
        entry = {"name": d.name, "K": d.K, "policy": d.policy,
                 "static_selector": d.static_selector,
                 "selector_reads": [list(pair) for pair in d.selector_reads]}
        if d.policy == "auto":
            entry["mix"] = mix_name(d.name)
        if d.policy == "segmented":
            entry["class"], entry["stale"] = class_name(d.name), stale_name(d.name)
        if d.units:
            entry["units"] = list(d.units)
        out.append(entry)
    return out


def _finish(walk) -> dict:
    """The optional ``finish`` key of a kernel that finishes its own
    sample: the mask it marks, the reserved counter plane it counts newly
    finished samples into, and the steps one launch takes. Absent means
    the kernel never finishes. A ``steps="auto"`` kernel also records
    ``steps_max``, the most steps one launch can take."""
    finish = getattr(walk, "finish", None)
    if finish is None:
        return {}
    from ..ir.nodes import FINISHED_PLANE

    mask, steps = finish
    out = {"mask": mask, "counter": FINISHED_PLANE, "steps": steps}
    if steps == "auto":
        from ..trace.steps import AUTO_K_MAX

        out["steps_max"] = AUTO_K_MAX
    return {"finish": out}


def _gather_mats(walk, broadcast: dict | None) -> dict:
    """``{compact Matrix plane: the lane axes it varies along (one bool per axis of
    the lane space), or None when the kernel is not a broadcast variant}``: what the
    binder checks its column count against. A tangent plane takes its input's pattern."""
    from ..diff.transform import TANGENT_PREFIX

    held = broadcast or {}
    out = {}
    for name in walk.gather_mats:
        base = name[len(TANGENT_PREFIX) + 1:] if name.startswith(TANGENT_PREFIX + "_") else name
        kept = held.get(name) or held.get(base)
        out[name] = None if kept is None else [bool(k) for k in kept]
    return out


def sidecar_meta(name: str, walk, *, fmt: str, scalar_type: str,
                 exec_targets=ENTRY_KINDS, host_entry: str | None = None,
                 derivative: dict | None = None, lanes=None,
                 lane_space=((None, None),), group: int = 1, box=None, block: int | None = None,
                 index_set: dict | None = None,
                 planes: dict | None = None, broadcast: dict | None = None,
                 dispatch_override=None, step_ops: int | None = None,
                 fp64_ops_ir: int | None = None,
                 entry_counts_finished: bool = False,
                 ptxas: dict | None = None, local: tuple | None = None) -> dict:
    """The full v2 sidecar for one kernel.

    ``buffers`` is emitted present-and-empty: a HAWK ``lookup`` plane's
    length is a runtime quantity, so it carries no compile-time ``count``
    and a fabricated one would be a lie a loader would allocate against.
    The plane is still declared where every consumer reads it —
    ``arg_spec``, role ``lookup``.

    ``lanes`` is present only for a fused entry: one
    :class:`~hawk.types.LaneMeta` record per lane, so a consumer that
    composed the group can tell, from the deployed artifact alone, which
    lanes are inside it and whether any moved since.

    ``arg_shapes``/``arg_dtypes`` are complete over every plane-bound
    slot, inputs included — the width (or ``None`` for a runtime-length
    role) and wire dtype the body was compiled for. ``arg_widths`` is
    their int-only, backward-compatible subset (see the module docstring).

    ``lane_space`` and ``group`` are the kernel's launch shape and compaction
    unit, written as ``launch.lane_space`` for every kernel (one unlabelled
    runtime axis and group 1 when none is declared). ``block`` is the threads
    per block of an emit that fixed one, written as ``launch.block``. ``planes`` are the kernel's
    declarations: ``sorted_inputs`` names the bound ``Sorted[Index]`` inputs, ``index_planes`` the ``Index`` inputs
    (32-bit on the wire; a wider integer array binds when every value fits) and
    ``plane_axes`` the sinks declared with named axes, each only when present.

    ``dispatch_override`` (the per-unit split) replaces ``walk.dispatches``
    as the ``dispatch`` block's source when given, since a segmented
    unit's own walk carries no Dispatch node.

    ``step_ops`` (one step's operation count) is present only for a kernel
    whose body steps; the launching runtime reads it, the body does not.
    ``fp64_ops_ir`` (one step's 64-bit floating-point operations, counted
    in the IR) sits beside it: the cost of a step the runtime's chunk
    length reads, not an instruction count -- the compiler fuses and shares
    operations, so the machine issues fewer.
    ``entry_counts_finished`` is present (``true``) when the kernel's fast
    entries count samples finished on entry into the finish counter, so the
    runtime zeroes the counter instead of counting the mask before a run.
    ``ptxas`` (present when the device compile ran ptxas, i.e. built a CUBIN)
    is ``{kernel: {regs, spill_store_B, spill_load_B, stack_B}}``, recorded for a
    reader of the artifact; nothing in the build acts on it."""
    planes = planes or {}
    sorted_in, axes = _sorted_inputs(walk, planes), _plane_axes(walk, planes)
    index_in = _index_planes(walk)
    meta = {
        "format": fmt,
        "schema_version": _contracts.MAX_SCHEMA_VERSION,
        "pattern": "pure",
        "aether_abi": _contracts.AETHER_ABI_VERSION,
        "exec_targets": list(exec_targets),
        "exec_access": walk.access.cls,
        # exec_op is required iff exec_access == 'mapreduce', forbidden
        # otherwise; sits beside its sibling keys, not appended.
        **({"exec_op": walk.access.op} if walk.access.op is not None else {}),
        "scalar_type": scalar_type,
        "kernel": name,
        "params": _params(walk),
        "params_schema": PARAMS_SCHEMA,
        "per_sample": _names(walk, "per_sample"),
        "vector_inputs": _names(walk, "vec_in"),
        "matrix_inputs": _names(walk, "mat_in"),
        "mat_shapes": {n: list(walk.slot_types[("mat_in", n)].shape)
                       for n in _names(walk, "mat_in")},
        "mutables": _mutables(walk),
        "arg_widths": (widths := _arg_widths(walk)),
        "vec_widths": {n: widths[n] for n in _names(walk, "vec_in")},
        "arg_shapes": _arg_shapes(walk),
        "arg_dtypes": _arg_dtypes(walk, scalar_type),
        "wide_inputs": _names(walk, "wide_in"),
        "wide_outputs": _names(walk, "wide_out"),
        "accum_outputs": _names(walk, "accum_out"),
        "dispatch": _dispatches(
            dispatch_override if dispatch_override is not None else walk.dispatches),
        "buffers": [],
        "arg_spec": [list(pair) for pair in walk.arg_spec],
        "terminated_readonly": getattr(walk, "finish", None) is None,
        **_finish(walk),
        **({"param_planes": list(walk.param_planes)} if walk.param_planes else {}),
        # every reduction over samples, with its op: the rows it writes (one
        # per 32 samples) are the runtime's scratch, finished into the cell
        **({"cells": walk.cells} if walk.cells else {}),
        # the companion planes of the compensated keyed sums: the runtime
        # makes them with the bind and the caller never passes them
        **({"companions": walk.companions} if walk.companions else {}),
        # an adjoint plane is its lookup plane's size, not the lane count
        **({"adjoint_like": walk.adjoint_like} if walk.adjoint_like else {}),
        # a Matrix read only at lane indices is the caller's compact plane: its
        # column count is not the lane count (a broadcast input)
        **({"gather_mats": _gather_mats(walk, broadcast)} if walk.gather_mats else {}),
        # one step's operation count, for a kernel whose body steps (absent
        # otherwise): read by the launching runtime, never by the body
        **({"step_ops": step_ops} if step_ops is not None else {}),
        **({"fp64_ops_ir": fp64_ops_ir} if fp64_ops_ir is not None else {}),
        **({"entry_counts_finished": True} if entry_counts_finished else {}),
        **({"ptxas": ptxas} if ptxas else {}),

        "launch": _launch(lane_space, group, box, block, index_set, local),
        **({"sorted_inputs": sorted_in} if sorted_in else {}),
        **({"index_planes": index_in} if index_in else {}),
        **({"plane_axes": axes} if axes else {}),
        "digest": walk.digest,
        "quantities": {p: list(span.wires) for p, span in walk.quantities.items()},
        "aether_include": _root_or_sealed_marker(_strict_aether_include),
        "eagle_include": _root_or_sealed_marker(_strict_eagle_include),
    }
    if lanes:
        meta["lanes"] = [{"name": lane.name,
                          "slots": [list(pair) for pair in lane.slots],
                          "digest": lane.digest} for lane in lanes]
    if host_entry is not None:
        meta["host_entry"] = host_entry
    if derivative is not None:
        meta["derivative"] = derivative
    return meta
