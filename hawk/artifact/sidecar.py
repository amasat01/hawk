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
header roots and the unit ``digest`` (the library's, of the lowered unit), which live here because the manifest's
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
* ``arg_mirrors`` — ``{slot name: "GRefMirror" | "ScalarHandle" | "value"}``, the by-value
  shape the entry takes each argument in, as the lowering answer states it for the slot; a
  runner binds from it and refuses a sidecar that lacks it.
* ``arg_dtypes`` — the slot's wire dtype, for EVERY slot (uniform slots and the
  runtime-length planes included): the compiled ``scalar_type`` for an ``f64``/``f32``
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

import re
from types import MappingProxyType, SimpleNamespace

from .. import _contracts, _ir
from ..targets import ENTRY_KINDS


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
    if ttype.dtype in _ir.INT_PLANE_DTYPES:
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


def _launch(lane_space, group: int, box=None, block: int | None = None,
            index_set: dict | None = None, measured: dict | None = None, wide: dict | None = None) -> dict:
    """The ``launch.lane_space`` block: the launch shape (row-major, the last axis
    fastest, only the first extent may be ``None``: known at run time) and the
    compaction ``group``, a lane count. With a thread ``box``
    (a record with ``sidecar()``), ``launch.box`` = ``{axis, T, stride}``: a
    launch of ``n`` lanes runs ``ceil(n / (T * stride)) * stride`` threads
    , and ``base``/``count`` count threads.
    ``block`` is the exact threads per block an emit that fixed one (a staged
    tile) needs; a launch uses it verbatim. ``index_set`` is the record of a kernel
    declaring ``lanes=IndexSet()``: ``launch.index_set`` = ``{"width": bits}`` (the map word, the host
    bind's cue to range- and duplicate-check the caller's ``active_map``), or, for the prefix form
    (``IndexSet(prefix=True)``), ``{"width": 32, "prefix": true, "count": "active_count"}``: no map, the
    named one-element ``uint32`` plane holds how many leading lanes run. Absent for every other kernel.
    ``measured`` is the library's own record of the unit's launch needs (the local memory one thread holds, say), stored as given.
    ``wide`` is the placement of the kernel's wide values (the library's placement): ``launch.wide`` =
    ``{plane: {form, plane, layout, width}}``, one record per plane a wide value lives in, and ``class``, the device feature class whose thresholds placed them (a reader that does not know
    the key ignores it; a scratch plane is also in the sidecar's ``team_planes``, which sizes it). Absent when no value is placed."""
    axes = [{"label": label, "extent": extent} for label, extent in lane_space]
    out = {"lane_space": {"axes": axes, "group": int(group)}}
    if wide:
        out["wide"] = {name: (rec if name == "class" else dict(rec)) for name, rec in wide.items()}
    if box is not None:
        out["box"] = box.sidecar()
    if index_set is not None:
        out["index_set"] = dict(index_set)
    if measured:
        out.update(measured)
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


def _plane_meta(walk, planes) -> dict:
    """``{name: {key: value}}`` of the opaque notes (``Annotated[T, Meta(...)]``) on bound planes."""
    return {name: dict(planes[name].meta) for _role, name in walk.arg_spec
            if getattr(planes.get(name), "meta", ())}


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
                       in ("i32", "i64", *_ir.INT_PLANE_DTYPES) else "float")}
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

#: The wire dtype an ``i32``/``i64`` or ``bool`` leaf carries, fixed
#: regardless of the compiled scalar mode; ``None`` for ``f64``/``f32``
#: marks "look at ``scalar_type`` instead" — see :func:`_wire_dtype`.
_FIXED_WIRE_DTYPE = {"f64": None, "f32": None, "i32": "int64", "i64": "int64",
                     "int32": "int32", "uint32": "uint32",
                     "bool": "bool",
                     # the 16-byte partial sum of a banded unit: one width whatever the scalar mode
                     "band_partial": "band_partial"}


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


def _slot_shapes(walk) -> dict:
    """The static shape of every slot the unit binds, as lists of ints: ``[]`` for a scalar, ``[n]`` for a vector, ``[r, c]`` for a
    matrix. A consumer that sizes a plane before a build reads it here; ``arg_spec`` stays the pairs that bind them."""
    return {name: [int(e) for e in walk.slot_types[(role, name)].shape] for role, name in walk.arg_spec}


def _arg_dtypes(walk, scalar_type: str) -> dict:
    """The per-slot wire dtype block — EVERY slot the unit binds, uniform and lookup slots included, from the
    same slot record as :func:`_slot_shapes`, so a consumer can tell an integer-valued handle from a Real one without guessing."""
    return {name: _wire_dtype(walk.slot_types[(role, name)], scalar_type) for role, name in walk.arg_spec}


def _branch_cost(record: dict) -> dict:
    """One branch's operation counts as the library states them; ``stores`` is negative when no sink takes the branch's value directly."""
    return {**record, "stores": -1.0 if record["stores"] is None else record["stores"]}


def _joined_names(walk) -> dict:
    """``{plane: {"group": ..., "wire": ...}}`` of every plane whose name joins a quantity's namespace to a wire name (a plane
    of a derived kernel), so that a reader never parses the separator."""
    out = {}
    for _role, name in walk.arg_spec:
        group, sep, wire = name.rpartition(_ir.NAMESPACE_SEP)
        if sep:
            out[name] = {"group": group, "wire": wire}
    return out


def _dispatches(infos) -> list:
    """The per-kernel dispatch listing: ``{name, K, strategy,
    static_selector, selector_reads}`` (the kernel-scoped classification and
    the selector's read slots a launching runtime re-scopes to a whole
    step), straight off the dispatch records of :func:`_unit_walk`. A SEGMENTED unit's own walk
    carries no Dispatch node (the per-unit split replaced it), which is why the records come from the kernel's walk."""
    out = []
    for d in infos:
        entry = {"name": d.name, "K": d.K, "strategy": d.strategy, "static_selector": d.static_selector,
                 "selector_reads": [list(pair) for pair in d.selector_reads]}
        if d.strategy == "auto":
            entry["mix"] = d.mix
            entry["cost"] = {"branches": [_branch_cost(cost) for cost in d.branch_costs]}
        if d.strategy == "segmented":
            planes = _ir.group_plane_names(d.name)
            entry["class"], entry["stale"], entry["offsets"] = planes.cls, planes.stale, planes.offsets
            if d.classify is not None:
                entry["classify"] = d.classify
        if d.units:
            entry["units"] = list(d.units)
        out.append(entry)
    return out


def _finish(walk, steps) -> dict:
    """The optional ``finish`` key of a kernel that finishes its own
    sample: the mask it marks, the reserved counter plane it counts newly
    finished samples into, and the steps one launch takes. Absent means
    the kernel never finishes. A ``steps="auto"`` kernel also records
    ``steps_max``, the most steps one launch can take."""
    finish = getattr(walk, "finish", None)
    if finish is None:
        return {}
    out = {"mask": finish, "counter": _ir.FINISHED_PLANE, "steps": steps}
    if steps == "auto":
        from ..trace.steps import AUTO_K_MAX

        out["steps_max"] = AUTO_K_MAX
    return {"finish": out}


def _gather_mats(walk, broadcast: dict | None) -> dict:
    """``{compact Matrix plane: the lane axes it varies along (one bool per axis of
    the lane space), or None when the kernel is not a broadcast twin}``: what the
    binder checks its column count against. A tangent plane takes its input's pattern."""
    from ..diff.transform import TANGENT_PREFIX

    held = broadcast or {}
    out = {}
    for name in walk.gather_mats:
        base = name[len(TANGENT_PREFIX) + 1:] if name.startswith(TANGENT_PREFIX + "_") else name
        kept = held.get(name) or held.get(base)
        out[name] = None if kept is None else [bool(k) for k in kept]
    return out


def sidecar_meta(name: str, walk, *, fmt: str, scalar_type: str, steps: int | str = 1,
                 exec_targets=ENTRY_KINDS, host_entry: str | None = None,
                 derivative: dict | None = None, lanes=None,
                 lane_space=((None, None),), group: int = 1, box=None, block: int | None = None,
                 index_set: dict | None = None,
                 planes: dict | None = None, broadcast: dict | None = None,
                 step_ops: int | None = None,
                 fp64_ops_ir: int | None = None,
                 entry_counts_finished: bool = False,
                 measured: dict | None = None, tool: dict | None = None, stated: dict | None = None,
                 wide: dict | None = None) -> dict:
    """The full v2 sidecar for one kernel.

    ``buffers`` is emitted present-and-empty: a HAWK ``lookup`` plane's
    length is a runtime quantity, so it carries no compile-time ``count``
    and a fabricated one would be a lie a loader would allocate against.
    The plane is still declared where every consumer reads it —
    ``arg_spec``, role ``lookup``.

    ``lanes`` is present only for a fused entry: one
    record per lane, so a consumer that
    composed the group can tell, from the deployed artifact alone, which
    lanes are inside it and whether any moved since.

    ``arg_shapes`` is complete over every plane-bound slot, inputs included (the width, or ``None`` for a runtime-length
    role), and ``arg_dtypes`` over every slot, uniforms included — the wire dtype the body was compiled for. ``arg_widths`` is
    their int-only, backward-compatible subset (see the module docstring).

    ``lane_space`` and ``group`` are the kernel's launch shape and compaction
    unit, written as ``launch.lane_space`` for every kernel (one unlabelled
    runtime axis and group 1 when none is declared). ``block`` is the threads
    per block of an emit that fixed one, written as ``launch.block``. ``planes`` are the kernel's
    declarations: ``sorted_inputs`` names the bound ``Sorted[Index]`` inputs, ``index_planes`` the ``Index`` inputs
    (32-bit on the wire; a wider integer array binds when every value fits) and
    ``plane_axes`` the sinks declared with named axes, and ``unused_planes`` the
    declared planes the body never touches (pruned from the compiled kernel), ``plane_meta`` the opaque notes of an annotated plane, and ``param_shapes`` the natural
    shape of each ``Param[Vector/Matrix/Table]`` plane, each only when present.

    ``step_ops`` (one step's operation count) is present only for a kernel
    whose body steps; the launching runtime reads it, the body does not.
    ``fp64_ops_ir`` (one step's 64-bit floating-point operations, counted
    in the IR) sits beside it: the cost of a step the runtime's chunk
    length reads, not an instruction count -- the compiler fuses and shares
    operations, so the machine issues fewer.
    ``entry_counts_finished`` is present (``true``) when the kernel's fast
    entries count samples finished on entry into the finish counter, so the
    runtime zeroes the counter instead of counting the mask before a run.
    ``measured`` is the library's record of the unit's launch needs, written into ``launch`` as given. ``tool`` is
    ``{OSPREY name: the figures its build gave}`` and ``stated`` the machine record each OSPREY stated, both written as given
    for a reader of the artifact; nothing in hawk acts on them."""
    planes = planes or {}
    sorted_in, axes = _sorted_inputs(walk, planes), _plane_axes(walk, planes)
    index_in = _index_planes(walk)
    bound = {n for _role, n in walk.arg_spec} | set(walk.companions or ()) | set(walk.cells or ())
    unused = [n for n, decl in planes.items()
              if n not in bound and not getattr(decl, "grid", None)]
    shared = {n: list(decl.shared) for n, decl in planes.items()
              if getattr(decl, "shared", ()) and n in bound}
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
        "slot_shapes": _slot_shapes(walk),
        "wide_inputs": _names(walk, "wide_in"),
        "wide_outputs": _names(walk, "wide_out"),
        "accum_outputs": _names(walk, "accum_out"),
        "dispatch": _dispatches(walk.dispatches),
        "planes_written": list(walk.planes_written),
        **({"plane_names": names} if (names := _joined_names(walk)) else {}),
        "buffers": [],
        "arg_spec": [list(pair) for pair in walk.arg_spec],
        "arg_mirrors": dict(walk.mirrors),
        "terminated_readonly": getattr(walk, "finish", None) is None,
        **_finish(walk, steps),
        **({"param_planes": list(walk.param_planes)} if walk.param_planes else {}),
        # the reserved planes of the kernel's own, by name: the runtime binds them and the caller never passes them
        **({"steps_plane": _ir.FUSED_STEPS_PLANE} if ("lookup", _ir.FUSED_STEPS_PLANE) in walk.arg_spec else {}),
        **({"cell_plane": _ir.CELL_PLANE} if ("per_sample", _ir.CELL_PLANE) in walk.arg_spec else {}),
        # every reduction over samples, with its op: the rows it writes (one
        # per 32 samples) are the runtime's scratch, finished into the cell
        **({"cells": walk.cells} if walk.cells else {}),
        # the companion planes of the compensated keyed sums: the runtime
        # makes them with the bind and the caller never passes them
        **({"companions": walk.companions} if walk.companions else {}),
        # a reverse derivative's gradient planes, each with the input it is the gradient of
        **({"gradient_of": walk.gradient_of} if walk.gradient_of else {}),
        # the gradients accumulated at lookup indices: an adjoint plane is its lookup plane's size, not the lane count
        **({"adjoint_like": walk.adjoint_like} if walk.adjoint_like else {}),
        # a reverse derivative's cotangent plane: the forward output it seeds
        **({"cotangent_of": walk.cotangent_of} if walk.cotangent_of else {}),
        # a step's next-value plane: the carried plane it holds the value of after the step
        **({"next_of": walk.next_of} if walk.next_of else {}),
        # a Matrix read only at lane indices is the caller's compact plane: its
        # column count is not the lane count (a broadcast input)
        **({"gather_mats": _gather_mats(walk, broadcast)} if walk.gather_mats else {}),
        # one step's operation count, for a kernel whose body steps (absent
        # otherwise): read by the launching runtime, never by the body
        **({"step_ops": step_ops} if step_ops is not None else {}),
        **({"fp64_ops_ir": fp64_ops_ir} if fp64_ops_ir is not None else {}),
        **({"entry_counts_finished": True} if entry_counts_finished else {}),
        **({"tool": tool} if tool else {}),
        **({"stated": stated} if stated else {}),
        "launch": _launch(lane_space, group, box, block, index_set, measured, wide),
        **({"sorted_inputs": sorted_in} if sorted_in else {}),
        **({"index_planes": index_in} if index_in else {}),
        **({"plane_axes": axes} if axes else {}),
        **({"plane_meta": notes} if (notes := _plane_meta(walk, planes)) else {}),
        # one value shared by every lane (``Param[Vector[W]]``, ``Param[Matrix[R,C]]``,
        # ``Param[Table[...]]``): its natural shape, bound flat and row-major
        **({"param_shapes": shared} if shared else {}),
        # declared but never read or written: the compiled kernel has no slot for them
        **({"unused_planes": unused} if unused else {}),
        "digest": walk.digest,
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


# -- The sidecar of a unit the lowering library answered. ------------------------------------------------------------------- #
#: The IR dtype each wire dtype of a unit record stands for (a ``Real`` follows the scalar mode).
#: The element type a unit record states for a slot (the record's closed set, every member) -> the dtype the sidecar code
#: above works with. A name missing here is a KeyError on the first record that carries it.
_RECORD_DTYPES = MappingProxyType({"Real": "f64", "Int64": "i64", "Int32": "int32", "Uint32": "uint32", "Bool": "bool",
                                   "BandPartial": "band_partial"})
_SLOT_PRIOR, _SLOT_PARAM_PLANE, _SLOT_GATHER = 1, 2, 4
_FINISH_ENTRY_COUNTS = 1


def _snake(word: str) -> str:
    return re.sub(r"(?<!^)([A-Z])", r"_\1", word).lower()


def _is_classify_record(dispatch: dict) -> bool:
    """Whether ``dispatch`` is the classify unit's own record of its dispatch: a segmented record with no classify unit, no branches and
    no sweep, whose class slot is set. The old sidecar format has no place for it, so it goes with that format."""
    return (dispatch["strategy"] == "Segmented" and dispatch["classify"] is None and not dispatch["branches"]
            and dispatch["sweep"] is None and dispatch["class_slot"] is not None)


def _unit_walk(unit: dict, planes: dict, kernel, family: list) -> SimpleNamespace:
    """What :func:`sidecar_meta` reads off a walk, rebuilt from the record of a unit and the declarations of its kernel."""
    meta = unit["meta"]
    body = meta["body"]
    slots = body["slots"]
    names = [s["name"] for s in slots]
    arg_spec = tuple((_snake(s["role"]), s["name"]) for s in slots)
    held_slots = _ir.slot_of(kernel.walk)
    # the record has dtype and shape but not the tag (an index plane's): the kernel's own walk states it
    slot_types = {(_snake(s["role"]), s["name"]): _ir.TensorType(
                      tuple(s["shape"]), _RECORD_DTYPES[s["dtype"]],
                      _ir.slot_type(kernel.walk, _snake(s["role"]), s["name"]).tag if (_snake(s["role"]), s["name"]) in held_slots else None)
                  for s in slots}
    reads = {n for r, n in arg_spec if r in ("lookup", "wide_in")}
    finish = meta.get("finish")
    held = {group.name: (index, group) for index, group in enumerate(_ir.dispatch_groups(kernel.walk))}
    mixes = _ir.mix_slots(kernel.walk)
    listed = [d for d in body["dispatches"] if not _is_classify_record(d)]
    cotangent_of = {c: out for c, out in kernel.cotangents.items() if c in names}
    # a step derivative reads its seeds, not the next-value planes: those are named by the seeds' targets
    next_of = {n: carried for n, carried in kernel.next_values.items() if n in names or n in cotangent_of.values()}

    return SimpleNamespace(
        arg_spec=arg_spec, slot_types=slot_types, digest=meta["digest"],
        mirrors={s["name"]: s["mirror"] for s in slots},
        planes_written=tuple(sorted(s["name"] for s in slots if s["written"])),
        access=SimpleNamespace(cls=_snake(body["access"]), op=None),
        cells={c["name"]: {"op": _snake(c["op"]), "columns": c["columns"]} for c in body["cells"]},
        companions={names[c["slot"]]: names[c["target_slot"]] for c in body["companions"]},
        gradient_of={g: x for g, x in kernel.gradients.items() if g in names},
        adjoint_like={g: x for g, x in kernel.gradients.items() if ("accum_out", g) in arg_spec and x in reads},
        cotangent_of=cotangent_of,
        next_of=next_of,
        gather_mats=tuple(sorted(s["name"] for s in slots if s["flags"] & _SLOT_GATHER)),
        param_planes=tuple(s["name"] for s in slots if s["flags"] & _SLOT_PARAM_PLANE),
        prior_reads=frozenset(s["name"] for s in slots if s["flags"] & _SLOT_PRIOR),
        finish=None if finish is None else names[finish["mask_slot"]],
        dispatches=tuple(SimpleNamespace(name=d["name"], K=held[d["name"]][1].branches, strategy=_snake(d["strategy"]),
                                         static_selector=_ir.static_selector(kernel.walk, held[d["name"]][0]), mix=mixes.get(d["name"]),
                                         selector_reads=_ir.selector_reads(kernel.walk, held[d["name"]][0]),
                                         branch_costs=d["branch_costs"],
                                         units=tuple(ref["name"] for ref in d["branches"]),       # the record lists a segmented group's branch units in order
                                         classify=None if d["classify"] is None else d["classify"]["name"])
                         for d in listed),
    )


class _RecordBox:
    """A thread box as the answer states it, written as the sidecar spells a box."""

    def __init__(self, record: dict) -> None:
        self.record = record

    def sidecar(self) -> dict:
        return dict(self.record)


def _keys_record(keyed: dict) -> dict:
    """The keys of a block sum as the sidecar spells a keyed plane."""
    axes = [[label, None if extent < 0 else extent] for label, extent in zip(keyed["axis_labels"], keyed["axis_extents"])]
    return {"width": keyed["width"], "axes": axes, "payload": list(keyed["payload"])}


def _team_record(team: dict, source_body: dict, stage_slots: list) -> dict:
    """The sidecar's ``team`` of a team stage."""
    out = {"threads": team["threads"]}
    if team["lanes"] > 1:
        out["lanes"] = team["lanes"]
    if team["warps"] > 1:
        out["warps"] = team["warps"]
    grid = _snake(team["grid"])
    if grid == "keys":
        out["grid"] = "keys"
        if team["groups"] > 1:
            out["groups"] = team["groups"]
        out["keyed"] = _keys_record(source_body["keyed"][team["keyed"]])
    elif grid in ("chunks", "targets"):
        out["grid"], out["of"] = grid, stage_slots[team["of_slot"]]["name"]
        if grid == "chunks":
            out["status"] = source_body["team_planes"][team["status_plane"]]["name"]
    return out


def _split_block(source: dict, family: list) -> dict:
    """What the first stage of a cut kernel carries: every stage, every cell, the scratch planes and the keyed planes, as the
    entry-less unit that lists the stages records them."""
    body = source["meta"]["body"]
    cells = {c["name"]: {"op": _snake(c["op"]), "columns": c["columns"], **({"internal": True} if c["internal"] else {})} for c in body["cells"]}
    stages = []
    for st in source["meta"]["stages"]:
        record = {"kernel": st["unit"]["name"], "cells_in": list(st["cells_in"]), "cells_out": list(st["cells_out"])}
        if st["team"] is not None:
            record["team"] = _team_record(st["team"], body, body["slots"])
        stages.append(record)
    out = {"cells": cells, "stages": stages}
    scratch = {}
    for p in body["team_planes"]:
        row = {"dtype": {"Real": "real"}.get(p["dtype"], _snake(p["dtype"])), "per_lane": p["per_lane"]}
        if _snake(p["layout"]) == "plane_major":
            row["layout"] = "plane_major"
        unit = _snake(p["unit"])
        if unit == "key":
            row["keys"] = _keys_record(body["keyed"][p["keyed"]])
        elif unit == "target":
            row["per_target"] = body["slots"][p["target_slot"]]["name"] if p["target_slot"] is not None else None
        elif unit == "once":
            row["once"] = True
        if p["fill_promised"]:
            row["fill"] = p["fill"]
        scratch[p["name"]] = row
    if scratch:
        out["team_planes"] = scratch
    used = {p["keyed"] for p in body["team_planes"] if p["keyed"] is not None} | {st["team"]["keyed"] for st in source["meta"]["stages"] if st["team"] and st["team"]["keyed"] is not None}
    keyed = {k["name"]: _keys_record(k) for i, k in enumerate(body["keyed"]) if i not in used}
    if keyed:
        out["keyed"] = keyed
    return out


def _counted(finish: dict | None, key: str) -> int | None:
    """A count the library states in the finish record; absent or negative is not counted."""
    value = None if finish is None else finish.get(key)
    return None if value is None or value < 0 else int(value)


def sidecar_of_unit(kernel, unit: dict, *, lead_device: bool, fmt: str, scalar_type: str, exec_targets, family: list, steps=None,
                    derivative: dict | None = None, block: int | None = None, measured: dict | None = None, tool: dict | None = None,
                    stated: dict | None = None, records: dict | None = None) -> dict:
    """The sidecar of one unit the lowering library answered. ``unit`` is the unit's record, ``lead_device`` whether the lead target is the device (its thread box is the sidecar's), ``kernel`` the kernel the unit belongs to (its declarations, broadcast pattern, lanes and
    steps), ``family`` the records of every unit the answer holds for the kernel (the entry-less unit that lists the stages of a cut
    kernel, and the branch units of a segmented one, are among them)."""
    meta = unit["meta"]
    body, launch = meta["body"], meta["launch"]
    from ..trace.kernel import derived_axes

    planes = getattr(kernel, "planes", None) or derived_axes(kernel.walk, getattr(kernel.sinks, "source", None), kernel.gradients)
    # the unit that lists the stages this unit is one of: the kernel's entry-less source unit or its first stage; each branch
    # unit of a segmented split lists itself alone, so a family may hold several lists
    source = next((u for u in family if any(st["unit"]["name"] == meta["name"] for st in u["meta"]["stages"])), None)
    stages = source["meta"]["stages"] if source is not None else []
    at = next((i for i, st in enumerate(stages) if st["unit"]["name"] == unit["meta"]["name"]), None)
    segmented = any(d["strategy"] == "Segmented" for d in body["dispatches"])     # a branch unit or the classify unit, by its own record
    if (at and stages[at]["team"] is not None) or segmented:
        planes = {}                 # a team stage after the first, and a unit of a segmented split, declare none of the kernel's planes
    walk = _unit_walk(unit, planes, kernel, family)
    names = [s["name"] for s in body["slots"]]
    labels, extents = launch["lane_labels"], launch["lane_extents"]
    lane_space = tuple(zip(labels, [None if e < 0 else e for e in extents])) or ((None, None),)
    index_set = None
    if launch["index_count_slot"] is not None:
        index_set = ({"width": 32, "prefix": True, "count": names[launch["index_count_slot"]]} if launch["index_map_slot"] is None
                     else {"width": 64 if body["slots"][launch["index_map_slot"]]["dtype"] == "Int64" else 32})
    finish = meta.get("finish")
    box = launch["device_box"] if lead_device else launch["host_box"]
    out = sidecar_meta(
        meta["name"], walk, steps=getattr(kernel, "steps", 1) if steps is None else steps, fmt=fmt, scalar_type=scalar_type,
        exec_targets=exec_targets, host_entry=meta["host_entry"], derivative=derivative, lanes=getattr(kernel, "lanes", None),
        lane_space=lane_space, group=launch["group"], box=None if box is None else _RecordBox({"axis": box["axis"], "T": box["t"], "stride": box["stride"]}),
        block=block, index_set=index_set, planes=planes, broadcast=getattr(kernel, "broadcast", None),
        step_ops=_counted(finish, "step_ops"), wide={p["name"]: {"layout": "warp_tiled"} for p in body["team_planes"] if p["wide"] == "WarpTiled"} or None, fp64_ops_ir=_counted(finish, "fp64_ops_ir"),
        entry_counts_finished=bool(finish and finish["flags"] & _FINISH_ENTRY_COUNTS), measured=measured, tool=tool, stated=stated)
    # the records of the lowering answer the runtime reads, kept as given: the steps table and the dispatch costs; the
    # contractions the family's units are, each with the route word the library chose (``contractions`` below)
    for key, member in (("steps_reverse_table", "steps_reverse"), ("dispatch_costs", "dispatch_costs")):
        if records and records.get(member) is not None:
            out[key] = records[member]
    from ..diff.tape import refine_plane_names

    refine = refine_plane_names(kernel)
    if refine is not None:
        out["refine_planes"] = refine
    if records and records.get("index_bytes") is not None:
        out["index_bytes"] = records["index_bytes"]
    contractions = [{"unit": u["meta"]["name"], **u["contraction"]} for u in family if u.get("contraction")]
    if contractions:
        out["contractions"] = contractions
    if stages and stages[0]["unit"]["name"] == meta["name"]:
        out.update(_split_block(source, family))
    if body["cell_row"]:
        out["cell_row"] = body["cell_row"]
    # The chunk cap belongs to an ordered scatter: the unit of the family that has one states it, and the
    # family's other units carry the same number; a family with no ordered scatter states none.
    chunk_cap = max(u["meta"]["body"]["scatter_chunks"] for u in family)
    if chunk_cap:
        out["scatter_chunks"] = chunk_cap
    return out
