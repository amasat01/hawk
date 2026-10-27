# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A read-only per-lane input of a ``Grid`` kernel may be given broadcast.

Its shape is any that broadcasts, right-aligned, against ``(head, *lane shape)``
(``head`` is a ``Vector[w]``'s component axis, which stays exact): a scalar, axes
of size 1, fewer axes. NumPy's rule. The input binds as the CALLER's buffer,
compact and never copied: the pattern (which lane axes it varies along) picks a
variant of the kernel in which the per-lane read is a lookup at those axes
(``Table["a", "c"]``, or a flat table when the runtime axis is kept) or, when it
varies along none, a uniform. One variant per pattern, memoised on the kernel;
the exact shape runs the kernel untouched.

Both hawk doors classify with :func:`kept_axes`. :func:`route` is
``Kernel.__call__``'s: it picks the variant and compacts the arguments.
``hawk.runtime.run`` binds a loaded artifact, which is one kernel's, so it
refuses a broadcast shape with the same words and names the variant to build.
Written, accumulated and mask planes are exact: only a plane the kernel reads
broadcasts."""

from __future__ import annotations

from math import prod

from .ir import HawkError

#: The roles of a read-only per-lane input; the ones that broadcast.
READ_ROLES = ("per_sample", "vec_in", "mat_in")
#: The roles of a per-lane plane that is written or masked: exact.
EXACT_ROLES = ("mutable", "out", "terminated")


def _head(ttype, role: str) -> tuple:
    """The component axes of a per-lane slot, which stay exact: a Vector's ``(w,)``,
    a Matrix's ``(rows, cols)``."""
    return ttype.shape[:2] if role == "mat_in" else ttype.shape[:1]


def shape_of(value) -> tuple | None:
    """``value``'s shape: ``()`` for a Python number, ``None`` when it has none
    (a marked or raw buffer, which the layout rules read)."""
    if isinstance(value, (bool, int, float)):
        return ()
    shape = getattr(value, "shape", None)
    return None if shape is None else tuple(shape)


def lane_text(head: tuple, axes) -> str:
    """The ``(head, *lane shape)`` a plane is checked against, as ``(a, 64, 64)``."""
    lane = tuple(label if extent is None else extent for label, extent in axes)
    return "(" + ", ".join(map(str, head + lane)) + ")"


def runtime_extent(shaped, axes, label_extent=None):
    """The extent of the runtime first axis: the label's, else that of the first
    plane given in the full lane shape (``shaped``: ``(shape, head)`` pairs)
    whose first lane axis is not 1; ``None`` when nothing says."""
    if label_extent is not None or axes[0][1] is not None:
        return label_extent
    for shape, head in shaped:
        if shape is not None and shape[:len(head)] == head \
                and len(shape) == len(head) + len(axes) and shape[len(head)] != 1:
            return shape[len(head)]
    return None


def kept_axes(shape: tuple, head: tuple, axes, extent=None, strict: bool = True) -> tuple | None:
    """Which lane axes a plane of ``shape`` varies along (``True`` per axis), or
    ``None`` when it is the exact form (lane-shaped, or the flat plane). Raises
    ``ValueError`` for a shape that does not broadcast, and for a head that is
    not the declared one the exact form is returned for the layout rules. Unless
    ``strict``, a runtime axis that is not ``extent`` is left to the label's own
    check."""
    if shape[:len(head)] != head:
        return None
    lane = shape[len(head):]
    static = tuple(e for _l, e in axes[1:])
    if len(lane) == 1 and len(axes) > 1 and (
            lane[0] >= (extent or 1) * prod(static) or lane[0] not in (1, axes[-1][1])):
        return None                       # the flat plane: the layout rules read its length
    if len(lane) > len(axes):
        raise ValueError
    padded = (1,) * (len(axes) - len(lane)) + lane
    kept = []
    for i, (d, (_label, fixed)) in enumerate(zip(padded, axes)):
        if fixed is None:
            if strict and d not in (1, extent) and extent is not None:
                raise ValueError
            kept.append(d != 1 or extent in (None, 1))
        else:
            if d not in (1, fixed):
                raise ValueError
            kept.append(d == fixed)
    return None if len(lane) == len(axes) and all(kept) else tuple(kept)


def of(name: str, role: str, value, head: tuple, axes, extent, where: str,
       strict: bool = True) -> tuple | None:
    """The kept axes of the plane ``name`` bound to ``value``, ``None`` for the
    exact form. Refuses a shape that does not broadcast and any broadcast shape
    on a plane that is not a read-only per-lane input."""
    shape = shape_of(value)
    if shape is None or role not in READ_ROLES + EXACT_ROLES:
        return None
    try:
        kept = kept_axes(shape, head, axes, extent, strict)
    except ValueError:
        if role in EXACT_ROLES:
            return None                   # not a broadcast: the exact-shape rules say why
        raise HawkError(
            f"{where}: argument {name!r} has shape {shape}, which does not broadcast "
            f"against {lane_text(head, axes)}: a read-only per-lane input may be a "
            "scalar, have axes of size 1, or have fewer axes (right-aligned, as in NumPy)"
        ) from None
    if kept is not None and role in EXACT_ROLES:
        raise HawkError(
            f"{where}: argument {name!r} has shape {shape}, but it is "
            f"{'written' if role in ('mutable', 'out') else 'bound whole'} per lane and "
            f"takes the exact shape {lane_text(head, axes)}; only a read-only "
            "per-lane input broadcasts")
    return kept


def pattern_text(kept: tuple, axes) -> str:
    """A pattern as the axes it varies along: ``[a, c]``, ``[]`` for none."""
    return "[" + ", ".join(label for (label, _e), k in zip(axes, kept) if k) + "]"


def dense_only(name: str, shape: tuple, kept: tuple, head: tuple, axes) -> str:
    """The refusal of a loaded artifact given a broadcast shape: it reads the
    plane in the exact lane shape."""
    return (f"argument {name!r} has shape {shape}, which broadcasts against "
            f"{lane_text(head, axes)} along {pattern_text(kept, axes)} only; this "
            "artifact reads it per lane. Call the kernel (k(...)), which binds the "
            "broadcast as given, or build and load the variant "
            "hawk._broadcast.variant(k, {name: kept})")


def check_rebind(name: str, held: tuple, got: tuple | None, axes) -> None:
    """Refuse binding ``name`` as a pattern other than the one a plan was bound
    with (``got`` is ``None`` for the exact shape): the plan reads the caller's
    buffer at the first one's strides."""
    if held != got:
        now = "the exact lane shape" if got is None else f"varying along {pattern_text(got, axes)}"
        raise HawkError(
            f"argument {name!r} was bound varying along {pattern_text(held, axes)}; "
            f"it cannot be rebound {now}. Bind afresh for the new pattern")


def guard_rebind(kernel, name: str, bound, value) -> None:
    """The check of a plan rebinding plane ``name`` of the broadcast ``kernel``
    (a variant): a rebound plane keeps the shape it was bound with."""
    old, new = shape_of(bound), shape_of(value)
    if old == new or new is None:
        return
    dims = getattr(kernel.planes.get(name), "dims", ())
    head = (dims[0][1],) if dims and dims[0][0].startswith("component") else ()
    try:
        got = kept_axes(new, head, kernel.lane_space, extent=0, strict=False)   # size 1 is not the runtime extent
    except ValueError:
        got = None
    check_rebind(name, kernel.broadcast[name], got, kernel.lane_space)
    raise HawkError(f"argument {name!r} was bound with shape {old}; it cannot be "
                    f"rebound with shape {new}")


# -- the variant: the kernel with the per-lane read of a plane rewritten ----- #


def variant(kernel, patterns: dict, reverse: tuple = ()):
    """``kernel`` with each plane of ``patterns`` (``{name: kept axes}``) read at
    the axes it varies along; the memoised variant of this pattern. A plane named
    in ``reverse`` (to differentiate with respect to) that varies along no axis
    is read from a one-cell table rather than as a uniform, so its gradient is
    the sum over every lane."""
    key = (tuple(sorted(patterns.items())), tuple(sorted(reverse)))
    with kernel._lock:
        held = kernel._variants.get(key)
        if held is None:
            held = kernel._variants[key] = _build(kernel, dict(patterns), reverse)
    return held


def _build(kernel, patterns: dict, reverse: tuple):
    from .diff.tape import _reachable
    from .ir.nodes import At
    from .trace.kernel import Kernel
    from .trace.steps import default_step, default_steps
    from .ir import Leaf

    step = default_step(kernel)
    if step is not kernel and step.name != kernel.name:
        raise HawkError(f"{kernel.name!r} takes several steps per launch; broadcast "
                        "inputs bind to a plain or automatic kernel")
    nodes = _reachable(step.sinks)
    crossed = {id(n.operands[0]) for n in nodes if isinstance(n, At)}
    leaves: dict = {}
    for n in nodes:
        if isinstance(n, Leaf) and n.role in READ_ROLES and n.name in patterns:
            if id(n) in crossed:
                raise HawkError(
                    f"{kernel.name!r} reads {n.name!r} at another lane (.at), so it "
                    "needs the exact lane shape")
            leaves.setdefault(n.name, []).append(n)
    planes, mapping = dict(step.planes), {}
    for name, found in leaves.items():
        if len(found[0].ttype.shape) > 1:       # a Matrix: the caller's compact plane, read at its index
            read = At(found[0], _index(patterns[name], step.lane_space), found[0].ttype)
        else:
            planes[name], read = _read(name, found[0].ttype, patterns[name], step.lane_space,
                                        name in reverse)
        mapping.update({id(leaf): read for leaf in found})
    out = Kernel(step.name, _swap(step.sinks, mapping), planes, kind=step.kind,
                 source_digest=step.source_digest, returns=step.returns)
    out.time, out.time_step, out.jump, out.group = step.time, step.time_step, step.jump, step.group
    out.broadcast = {name: patterns[name] for name in leaves}
    if step is kernel:
        return out
    fused = default_steps(out)
    fused.time, fused.time_step, fused.jump, fused.group = (
        step.time, step.time_step, step.jump, kernel.group)
    fused.broadcast = out.broadcast
    return fused


def _index(kept: tuple, axes):
    """The flat index of a lane in the compact plane of pattern ``kept``: row-major
    over the kept axes, the last fastest (a runtime axis is never in the C), ``0``
    when it varies along none."""
    from .ir import Const
    from .trace.lane_axes import LaneAxes
    from .types import TensorType

    held = [i for i, k in enumerate(kept) if k]
    if not held:
        return Const(0, TensorType((), "i32"))
    lane, flat, stride = LaneAxes(axes), None, 1
    for i in reversed(held):
        term = lane[i] * stride
        flat = term if flat is None else flat + term
        stride *= axes[i][1] or 1
    return flat.node


def _read(name: str, ttype, kept: tuple, axes, cell: bool) -> tuple:
    """``(declaration, node)``: the plane ``name`` read where it varies, the
    caller's compact buffer: a uniform when it varies along no lane axis (a
    one-cell table when ``cell``), else a lookup. A flat table when it varies along the runtime axis (a runtime
    extent is never in the C), else a table named by its kept axes."""
    from .ir import Leaf, make
    from .trace.decl import Plane, _axes
    from .trace.lane_axes import LaneAxes
    from .trace.value import TableRef
    from .types import TensorType

    width = ttype.shape[0] if ttype.shape else 0
    element = TensorType((), ttype.dtype)
    held = [i for i, k in enumerate(kept) if k]
    if not held and not width:
        if cell:
            return Plane("table", element), TableRef(name, element).at(0).node
        return Plane("uniform", ttype), Leaf("uniform", "uniform", name, ttype)
    lane = LaneAxes(axes)
    if kept[0] and axes[0][1] is None:
        if width:
            raise HawkError(
                f"{name!r} is a Vector given broadcast but varying along the runtime "
                f"axis {axes[0][0]!r}; pass it in the exact lane shape")
        flat, stride = None, 1
        for i in reversed(held):
            term = lane[i] * stride
            flat = term if flat is None else flat + term
            stride *= axes[i][1] or 1
        return Plane("table", element), TableRef(name, element).at(flat).node
    labels = [axes[i][0] for i in held]
    comp = "component"
    while comp in labels:
        comp += "_"
    spans = ([slice(comp, width)] if width else []) + [slice(axes[i][0], axes[i][1]) for i in held]
    dims = _axes("Table", tuple(spans))
    table = TableRef(name, element, dims=dims)
    at = {axes[i][0]: lane[i] for i in held}
    if not width:
        return Plane("table", element, dims=dims), table.at(**at).node
    parts = tuple(table.at(**at, **{comp: k}).node for k in range(width))
    return Plane("table", element, dims=dims), make("vec", parts)


def _swap(sinks, mapping: dict) -> tuple:
    """``sinks`` with the nodes in ``mapping`` (by ``id``) replaced, in value and key."""
    from .diff.tape import _resink
    from .ir import AccumWrite
    from .ir.walk import substitute

    roots = substitute(tuple(o for s in sinks for o in s.operands), mapping)
    out, at = [], 0
    for s in sinks:
        part = roots[at:at + len(s.operands)]
        at += len(s.operands)
        out.append(AccumWrite(s.name, part[0], part[1] if len(part) > 1 else None, s.ttype, s.policy)
                   if isinstance(s, AccumWrite) else _resink(s, part[0]))
    return tuple(out)


# -- Kernel.__call__'s door -------------------------------------------------- #


def route(kernel, args: tuple, kwargs: dict) -> tuple:
    """``(kernel to run, args, kwargs)`` for a call: a kernel that declares no
    ``Grid``, or a call whose planes are all the exact shape, is returned as it
    came; otherwise the variant of its pattern, with each broadcast plane
    compacted to the caller's buffer."""
    axes = kernel.lane_space
    if axes[0][0] is None:
        return kernel, args, kwargs
    source = getattr(kernel.sinks, "source", None)
    if source is not None and hasattr(source, "lane_space"):
        return _route_derived(kernel, source, args, kwargs)
    order = [n for n, p in kernel.planes.items() if not getattr(p, "grid", None)]
    given = dict(zip(order, args))
    if given.keys() & kwargs.keys():
        return kernel, args, kwargs
    given.update(kwargs)
    roles = {n: r for r, n in kernel.arg_spec}
    types = kernel.walk.slot_types
    where = f"{kernel.name}(...)"
    heads = {n: _head(types[(r, n)], r) for n, r in roles.items() if r in READ_ROLES + EXACT_ROLES}
    named = {n for n in given if getattr(kernel.planes.get(n), "dims", ())}
    shaped = [(shape_of(given[n]), heads[n]) for n in heads if n in given and n not in named]
    extent = runtime_extent(shaped, axes, kwargs.get(axes[0][0]))
    patterns = {}
    for name, head in heads.items():
        if name in given and name not in named:
            kept = of(name, roles[name], given[name], head, axes, extent, where)
            if kept is not None:
                patterns[name] = kept
    if not patterns:
        return kernel, args, kwargs
    run = variant(kernel, patterns)
    return run, (), {n: compact(n, v, patterns[n], heads[n], where) if n in patterns else v
                     for n, v in given.items()}


def _route_derived(kernel, source, args: tuple, kwargs: dict) -> tuple:
    """:func:`route` for a derivative (``vjp``/``jvp``) of ``source``: broadcast
    inputs are those of ``source``, so the variant is the derivative OF the
    primal's variant (the reverse scatters into the compact shape, summing the
    axes the input does not vary along, compensated). A tangent takes its
    input's pattern; an omitted gradient is allocated in the input's own shape."""
    from .diff.transform import ADJOINT_PREFIX, TANGENT_PREFIX

    derived, axes, where = kernel.sinks, kernel.lane_space, f"{kernel.name}(...)"
    if args:
        return kernel, args, kwargs
    roles = {n: r for r, n in source.arg_spec}
    types = source.walk.slot_types
    heads = {n: _head(types[(r, n)], r) for n, r in roles.items() if r in READ_ROLES + EXACT_ROLES}
    shaped = [(shape_of(kwargs[n]), h) for n, h in heads.items() if n in kwargs]
    extent = runtime_extent(shaped, axes, kwargs.get(axes[0][0]))
    patterns = {}
    for name, head in heads.items():
        if name in kwargs:
            kept = of(name, roles[name], kwargs[name], head, axes, extent, where)
            if kept is not None:
                patterns[name] = kept
    if not patterns:
        return kernel, args, kwargs
    reverse = tuple(patterns) if derived.kind == "vjp" else ()
    key = ("derived", tuple(sorted(patterns.items())))
    with kernel._lock:
        run = kernel._variants.get(key)
        if run is None:
            run = kernel._variants[key] = type(kernel)(
                kernel.name, derived.rebuild(variant(source, patterns, reverse)))
            run.broadcast = dict(patterns)      # a variant: eagle's plan of it does not route again
    given = dict(kwargs)
    for name, kept in patterns.items():
        given[name] = compact(name, kwargs[name], kept, heads[name], where, cell=name in reverse)
        tangent = f"{TANGENT_PREFIX}_{name}"
        if tangent in kwargs:
            got = of(name, "per_sample", kwargs[tangent], heads[name], axes, extent, where)
            if got != kept:
                raise HawkError(
                    f"{where}: argument {tangent!r} must vary along the axes of {name!r}, "
                    f"{pattern_text(kept, axes)}, got {pattern_text(got, axes) if got else 'the exact shape'}")
            given[tangent] = compact(tangent, kwargs[tangent], kept, heads[name], where)
        bar = f"{ADJOINT_PREFIX}_{name}"
        if name in reverse and bar not in kwargs:
            given[bar] = _zeros(kwargs[name], np_shape(kwargs[name]))
    slots = {n for _role, n in run.arg_spec}    # a derivative linear in an input does not take it
    return run, (), {n: v for n, v in given.items() if n in slots or n not in patterns}


def np_shape(value) -> tuple:
    return shape_of(value) or ()


def _zeros(like, shape: tuple):
    """A zeroed array of ``shape`` where ``like`` lives, in its precision."""
    import sys

    import numpy

    module = sys.modules.get(type(like).__module__.split(".")[0], numpy)
    return getattr(module, "zeros", numpy.zeros)(shape, dtype=getattr(like, "dtype", float))


def compact(name: str, value, kept: tuple, head: tuple, where: str, cell: bool = False):
    """The caller's buffer as the variant binds it: a scalar for a uniform, else
    the same memory with the size-1 axes dropped (flat when it varies along the
    runtime axis). A buffer that is not C-contiguous would need a copy: refused.
    A ``cell`` (the plane a reverse sums into) is a one-cell array, not a scalar."""
    if not any(kept) and not head:
        if cell:
            import numpy

            return numpy.asarray(value).reshape(-1)[:1]
        return value if isinstance(value, (bool, int, float)) else value.item()
    flags = getattr(value, "flags", None)
    contiguous = (flags.c_contiguous if flags is not None
                  else getattr(value, "is_contiguous", lambda: True)())
    if not contiguous:
        raise HawkError(
            f"{where}: broadcast argument {name!r} (shape {tuple(value.shape)}) must be "
            f"C-contiguous to bind without a copy; pass np.ascontiguousarray({name})")
    return value.reshape(-1) if len(head) < 2 else value.reshape(prod(head), -1)
