# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The launch shape a kernel declares (``hawk.Grid[...]``), read at the host bind door.

The sidecar's ``launch.lane_space`` is ``{"axes": [{"label", "extent"}, ...],
"group"}``: the FIRST extent may be ``None`` (runtime-sized), the later ones are
static, and the LAST axis varies fastest. A kernel that declares no ``Grid`` has
one unlabelled runtime axis, and none of this applies to it.

A ``Grid`` kernel takes its runtime extent BY LABEL (``k(x, channel=64)``) or from
a lane-shaped per-sample plane: ``head + lane shape``, which C-contiguous IS the
flat plane, so it binds as a view through the buffer protocol (no copy, no
numpy). ``head`` is the component axis of a ``Vector[w]`` plane; the rank and the
declared head tell the lane form from the flat ``(w, N)`` one. ``n_samples=`` and
``n=`` are the flat kernels' totals and are refused here, naming the label.
"""

from __future__ import annotations

import math
from math import prod

from . import _broadcast
from .ir import HawkError


def lane_axes(sidecar: dict) -> list | None:
    """``[(label, extent|None), ...]`` of a kernel that declares a ``Grid``, else ``None``."""
    space = (sidecar.get("launch") or {}).get("lane_space")
    if not space or space["axes"][0]["label"] is None:
        return None
    return [(ax["label"], ax["extent"]) for ax in space["axes"]]


def gather_columns(sidecar: dict, name: str, n: int):
    """The column count the compact Matrix plane ``name`` must have at ``n`` lanes
    (the product of the extents of the axes it varies along, a runtime one being
    ``n`` over the static ones); ``None`` when the sidecar records no pattern."""
    kept = (sidecar.get("gather_mats") or {}).get(name)
    axes = lane_axes(sidecar)
    if kept is None or axes is None:
        return None
    static = math.prod(e for _a, e in axes if e is not None)
    return math.prod((n // static if e is None else e) for (_a, e), k in zip(axes, kept) if k)


def spelling(axes) -> str:
    return "Grid[" + ", ".join(f'"{a}"' if e is None else f'"{a}":{e}' for a, e in axes) + "]"


def free_labels(sidecar: dict) -> set:
    """The labels of the sinks' runtime axes that no lane axis shares: each is its
    own extent, given as ``<label>=N`` or by the plane passed."""
    lane = {a for a, _e in lane_axes(sidecar) or ()}
    return {ax["label"] for spec in (sidecar.get("plane_axes") or {}).values()
            for ax in spec["axes"] if ax["extent"] is None and ax["label"] not in lane}


def output_shape(sidecar: dict, name: str, width: int, n: int, extents: dict | None = None) -> tuple:
    """The shape to allocate for the output plane ``name`` of ``n`` lanes and
    per-lane ``width``: lane-shaped for a ``Grid`` kernel (``(width, *lane
    shape)``, the component axis only when ``width > 1``), the named-axis
    ``(*payload, *axes)`` for a sink declared with axes (a reduction's output:
    the payload, then the kept lane axes; a runtime axis no lane axis shares
    takes its extent from ``extents[label]``), else the flat ``(width, n)`` / ``(n,)``."""
    declared = (sidecar.get("plane_axes") or {}).get(name)
    axes = lane_axes(sidecar)
    if declared is not None:
        # a kept runtime axis (a reduction's output) is as long as the lane axis
        runtime = n // prod(e for _a, e in axes[1:]) if axes else n
        free = free_labels(sidecar)

        def extent(ax):
            if ax["extent"] is not None:
                return ax["extent"]
            if ax["label"] not in free:
                return runtime
            if (extents or {}).get(ax["label"]) is None:
                raise HawkError(f"hawk.runtime.run: {name!r} has a runtime axis "
                                f"{ax['label']!r}; pass {ax['label']}=<extent of {ax['label']!r}>")
            return extents[ax["label"]]

        return tuple(declared["payload"]) + tuple(extent(ax) for ax in declared["axes"])
    head = (width,) if width > 1 else ()
    if axes is None:
        return head + (n,)
    static = tuple(e for _a, e in axes[1:])
    return head + (n // prod(static),) + static


def keyed_size(sidecar: dict, spec: dict, n: int) -> int:
    """The elements of a keyed plane the runtime owns between the stages of a
    split kernel: ``spec["width"]`` payload entries over the kept lane axes
    (a runtime kept axis is as long as the lane axis)."""
    axes = lane_axes(sidecar)
    runtime = n // prod(e for _a, e in axes[1:]) if axes else n
    return spec["width"] * prod(runtime if e is None else e for _a, e in spec["axes"])


def key_count(sidecar: dict, spec: dict, n: int) -> int:
    """The keys of a block sum over ``n`` lanes (``spec`` is a team stage's ``keyed``, or a team
    plane's ``keys``): the kept lane axes' extents, a runtime axis as long as the lane axis allows."""
    return keyed_size(sidecar, spec, n) // spec["width"]


#: The most lane chunks an ordered ``hawk.scatter`` adds in (a chunk is one thread's run of lanes).
SCATTER_CHUNKS = 256


def scatter_chunks(n: int, targets: int) -> int:
    """The chunks an ordered scatter of ``n`` lanes into ``targets`` cells adds in:
    ``clamp(n // targets, 1, 256)``, so the partial sums (``chunks * targets`` cells) stay O(n)."""
    return min(SCATTER_CHUNKS, max(1, n // max(targets, 1)))


def target_size(arrays: dict, name: str) -> int:
    """Elements of the bound plane ``name``, the cells an ordered scatter adds into."""
    size = getattr(arrays[name], "size", None)
    return int(size() if callable(size) else size)


def team_grid(sidecar: dict, team: dict, n: int, arrays: dict) -> int:
    """The blocks a team stage launches over ``n`` lanes: a lane (no ``grid``), a key of a block sum
    (``"keys"``), a lane chunk (``"chunks"``) or a target cell (``"targets"``) of an ordered scatter
    into the plane ``team["of"]``."""
    kind = team.get("grid", "lanes")
    if kind == "keys":
        return key_count(sidecar, team["keyed"], n)
    if kind == "chunks":
        return scatter_chunks(n, target_size(arrays, team["of"]))
    if kind == "targets":
        return target_size(arrays, team["of"])
    return n


def team_units(sidecar: dict, spec: dict, n: int, arrays: dict) -> int:
    """Elements of a team plane per ``per_lane``: lanes, keys, one (``once``), or (``per_target``: the
    plane an ordered scatter adds into) ``chunks * targets`` cells."""
    if spec.get("once"):
        return 1
    if "per_target" in spec:
        size = target_size(arrays, spec["per_target"])
        return scatter_chunks(n, size) * size
    return n if "keys" not in spec else key_count(sidecar, spec["keys"], n)


def check_scatter_status(kernel: str, team: dict, status, cells: int) -> None:
    """Refuse, by name, a run whose ordered scatter into ``team["of"]`` met a key outside ``[0, cells)`` that
    was not the skip key -1 (``status`` is the plane ``team["status"]`` after the run: flags from element 1)."""
    if any(int(f) for f in status[1:]):
        raise HawkError(
            f"{kernel}: hawk.scatter({team['of']}, policy='ordered') got a key outside [0, {cells}) "
            f"that is not the skip key -1; those adds were dropped, the rest are in {team['of']}")


def check_total(sidecar: dict, n: int) -> None:
    """Refuse a flat lane count ``n`` that is not a whole number of rows of a
    ``Grid``'s static axes."""
    axes = lane_axes(sidecar)
    if axes is None:
        return
    row = prod(e for _a, e in axes[1:])
    if n % row if axes[0][1] is None else n != row * axes[0][1]:
        raise HawkError(
            f"hawk.runtime.run: {n} lanes do not fit {spelling(axes)} "
            + (f"(a whole number of rows of {row})" if axes[0][1] is None
               else f"({row * axes[0][1]} lanes)"))


#: The largest dense extent a 32-bit map word can address.
INT32_MAX = 2**31 - 1


def index_width(extent: int, *, scalar_type: str = "float64", forced: int | str = "auto",
                map_bits: int | None = None) -> int:
    """The map word, 32 or 64 bits, of a lane set over ``extent`` dense lanes.

    ``"auto"`` is 32 and promotes to 64 only past ``2**31 - 1`` lanes (a pair
    grid of 65537 items); ``forced`` 32 or 64 is taken as given. ``map_bits``
    is the width of the map the caller holds, when it holds one: a 64-bit map
    over a 32-bit extent is refused rather than narrowed. A ``float32`` kernel
    over a 64-bit extent is refused: its lane id cannot be held in a ``float``."""
    if forced not in ("auto", 32, 64):
        raise HawkError(f"IndexSet width is 'auto', 32 or 64, got {forced!r}")
    wide = extent > INT32_MAX
    if wide and scalar_type == "float32":
        raise HawkError(
            f"an index set over {extent} lanes needs 64-bit lane ids, which a float32 "
            "kernel cannot hold (a pair kernel's ids are exact only below 2**31): "
            "build the float64 unit")
    if wide and forced == 32:
        raise HawkError(f"IndexSet(width=32) cannot address {extent} lanes (more than "
                        f"{INT32_MAX}); use width=64 or leave it 'auto'")
    bits = 64 if wide or forced == 64 else 32
    if map_bits == 64 and bits == 32:
        raise HawkError(
            f"the index map is int64, but the dense extent {extent} fits 32 bits: cast "
            "the map to int32, or build the unit with IndexSet(width=64)")
    return bits


def check_index_set(sidecar: dict, arrays: dict, extent: int | None) -> None:
    """The bind checks of a kernel that declares ``lanes=IndexSet()``: the map
    holds ``active_count[0]`` lanes, each in ``[0, extent)`` and none twice, and
    its word is the unit's. Refuses naming the FIRST offending position. A
    kernel without a lane set is returned as it came."""
    spec = (sidecar.get("launch") or {}).get("index_set")
    lanes, live = arrays.get("active_map"), arrays.get(spec.get("count", "active_count") if spec else "active_count")
    if spec is None or live is None or (lanes is None and not spec.get("prefix")):
        return
    where = f"kernel {sidecar.get('kernel', '?')!r}: the index set"
    if spec.get("prefix"):
        held = int(memoryview(live).cast("B").cast("I")[0])
        if extent is not None and held > extent:
            raise HawkError(f"{where}: {spec['count']} is {held}, but the kernel is bound at {extent} "
                            "lanes; a count above the bound lane count needs a rebind")
        return
    mv, held = memoryview(lanes), int(memoryview(live).cast("B").cast("I")[0])
    if mv.itemsize * 8 != spec["width"]:
        raise HawkError(
            f"{where} is a {spec['width']}-bit unit, but active_map holds "
            f"{mv.itemsize * 8}-bit words: "
            + ("cast the map to int32, or build the unit with IndexSet(width=64)"
               if mv.itemsize == 8 else "pass an int64 map (the unit reads 64-bit words)"))
    if extent is not None and extent > INT32_MAX and sidecar.get("scalar_type") == "float32":
        index_width(extent, scalar_type="float32")
    if held > mv.nbytes // mv.itemsize:
        raise HawkError(f"{where}: active_count is {held}, but active_map holds only "
                        f"{mv.nbytes // mv.itemsize} entries")
    first: dict = {}
    for position, lane in enumerate(mv.cast("B").cast(mv.format)[:held].tolist()):
        if lane < 0 or (extent is not None and lane >= extent):
            raise HawkError(f"{where}: active_map[{position}] = {lane} is outside the "
                            f"dense lanes [0, {extent})")
        if lane in first:
            raise HawkError(f"{where}: active_map[{position}] = {lane} repeats "
                            f"active_map[{first[lane]}]; each lane may be listed once")
        first[lane] = position


def _flat_extent(arg_spec, arrays: dict, per_sample_roles) -> int | None:
    """The lane count a flat kernel's per-sample planes agree on, when one is bound."""
    for role, name in arg_spec:
        if role in per_sample_roles and role != "mat_in" and name in arrays:
            try:
                return int(memoryview(arrays[name]).shape[-1])
            except TypeError:
                return None
    return None


def bind(sidecar: dict, arg_spec, arrays: dict, n_samples, per_sample_roles) -> tuple:
    """``(arrays, n_samples)`` for a ``Grid`` kernel: the extent by label consumed,
    lane-shaped planes flattened, the total lane count settled. A kernel with no
    ``Grid`` is returned as it came. A kernel declaring ``lanes=IndexSet()`` has its
    map checked against the lane count (:func:`check_index_set`)."""
    out, total = _bind_grid(sidecar, arg_spec, arrays, n_samples, per_sample_roles)
    if (sidecar.get("launch") or {}).get("index_set") is not None:
        check_index_set(sidecar, out, total if total is not None else n_samples
                        if n_samples is not None
                        else _flat_extent(arg_spec, out, per_sample_roles))
    return out, total


def _bind_grid(sidecar: dict, arg_spec, arrays: dict, n_samples, per_sample_roles) -> tuple:
    axes = lane_axes(sidecar)
    if axes is None:
        return arrays, n_samples
    first, runtime = axes[0][0], axes[0][1] is None
    static = tuple(e for _a, e in axes[1:])
    slots = {name for _r, name in arg_spec}
    out = dict(arrays)
    where = f"hawk.runtime.run: {spelling(axes)}"
    if n_samples is not None or ("n" in out and "n" not in slots):
        out.pop("n", None)
        raise HawkError(
            f"{where} takes its extent by label, not as a total: pass "
            + (f"{first}=<extent of {first!r}>" if runtime else "no extent (every axis is static)")
            + " instead of n=/n_samples=")
    extent = None
    for label, fixed in axes:
        if label in out and label not in slots:
            if not (runtime and label == first):
                raise HawkError(
                    f"{where}: the axis {label!r} is "
                    + (f"static ({fixed}); drop {label}=" if fixed is not None
                       else "the runtime one"))
            extent = out.pop(label)
            if isinstance(extent, bool) or not isinstance(extent, int) or extent < 1:
                raise HawkError(f"{where}: {first}= is the extent of the first axis, "
                                f"a positive int; got {extent!r}")
    shapes = sidecar.get("arg_shapes", {})
    named = sidecar.get("plane_axes") or {}
    mats = sidecar.get("mat_shapes") or {}
    heads = {name: tuple(mats[name]) if role == "mat_in" and name in mats
             else (int(shapes.get(name) or 1),) if int(shapes.get(name) or 1) > 1 else ()
             for role, name in arg_spec if name in out and name not in named}
    known = _broadcast.runtime_extent(
        [(_broadcast.shape_of(out[n]), h) for n, h in heads.items()], axes, extent)
    seen = None
    for role, name in arg_spec:
        if name in heads:
            shape = _broadcast.shape_of(out[name])
            kept = _broadcast.of(name, role, out[name], heads[name], axes, known, where, False)
            if kept is not None:
                raise HawkError(f"{where}: " + _broadcast.dense_only(
                    name, shape, kept, heads[name], axes))
        if role not in per_sample_roles or role == "mat_in" or name not in out or len(axes) < 2:
            continue
        try:
            mv = memoryview(out[name])
        except TypeError:
            continue                                  # a marked array: the layout rules read it
        width = int(shapes.get(name) or 1)
        head = (width,) if width > 1 else ()
        got = tuple(mv.shape)
        if len(got) != len(head) + len(axes):
            continue                                  # a flat plane: the layout rules read it
        dims = got[len(head):]
        declared = tuple(e for _a, e in axes)
        if got[:len(head)] != head or (dims[1:] != declared[1:] if runtime else dims != declared):
            want = "(" + ", ".join(map(str, head + ((first, *static) if runtime else declared))) + ")"
            raise HawkError(
                f"{where}: argument {name!r} has shape {got}, which contradicts the "
                f"declaration; expected {want} (the static extents are part of the kernel"
                + (f", only {first!r} is the caller's)" if runtime else ")"))
        if not mv.c_contiguous:
            raise HawkError(
                f"{where}: lane-shaped argument {name!r} (shape {got}) must be C-contiguous "
                f"to bind without a copy — pass np.ascontiguousarray({name})")
        if seen is not None and dims != seen:
            raise HawkError(f"{where}: lane-shaped arguments disagree on the lane shape: "
                            f"{seen} vs {dims} ({name!r})")
        seen = dims
        flat = mv.cast("B").cast(mv.format, shape=list(head) + [prod(dims)]) if head \
            else mv.cast("B").cast(mv.format)
        out[name] = flat
    total = None if seen is None else prod(seen)
    if extent is not None:
        want = extent * prod(static)
        if total is not None and total != want:
            raise HawkError(f"{where}: {first}={extent} gives {want} lanes, but the "
                            f"lane-shaped planes bound hold {total}")
        total = want
    elif total is None and not runtime:
        total = prod(e for _a, e in axes)             # every axis static: no extent to pass
    elif total is None and not any(r in per_sample_roles and nm in out for r, nm in arg_spec):
        raise HawkError(
            f"{where} has a runtime first axis {first!r} and this launch binds no "
            f"per-sample plane to read it from; pass {first}=<extent of {first!r}> "
            "(a Table's or buffer's length is never the lane count)")
    return out, total
