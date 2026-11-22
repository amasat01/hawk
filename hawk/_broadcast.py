# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A read-only per-lane input of a ``Grid`` kernel may be given broadcast.

Its shape is any that broadcasts, right-aligned, against ``(head, *lane shape)``
(``head`` is a ``Vector[w]``'s component axis, which stays exact): a scalar, axes
of size 1, fewer axes. NumPy's rule. The input binds as the CALLER's buffer,
compact and never copied: the pattern (which lane axes it varies along) picks a
twin of the kernel in which the per-lane read is a lookup at those axes
(``Table["a", "c"]``, or a flat table when the runtime axis is kept), a one-cell
table when an array varies along none, or a uniform when it is a Python number.
One twin per pattern, memoised on the kernel; the exact shape runs the kernel
untouched.

:meth:`hawk.plans.Plan.bound` classifies with :func:`routed`, which also compacts the
caller's buffers for the twin; :func:`rebound` does the same for a bound plan's
rebind. A plan with no source kernel (``hawk.plan`` over ``hawk.load``) reads every
input per lane and refuses a broadcast shape by name (:func:`loaded_only`).
Written, accumulated and mask planes are exact: only a plane the kernel reads
broadcasts."""

from __future__ import annotations

from math import prod

from . import _ir
from .errors import HawkError

#: The pattern of a Python number given for a per-lane input: the twin reads it as a uniform, by value. An array that varies
#: along no lane axis has the all-false tuple instead and is read from a one-cell table, the caller's memory.
BY_VALUE = "by value"
#: What a bare number is: the values that bind by value.
NUMBERS = (bool, int, float)
#: The roles of a read-only per-lane input; the ones that broadcast.
READ_ROLES = ("plane_in",)
#: The roles of a per-lane plane that is written or masked: exact.
EXACT_ROLES = ("mutable", "out", "terminated")


def _head(ttype) -> tuple:
    """The component axes of a per-lane slot, which stay exact: a Vector's ``(w,)``,
    a Matrix's ``(rows, cols)``, a deeper value's every extent."""
    return tuple(ttype.shape)


def shape_of(value) -> tuple | None:
    """``value``'s shape: ``()`` for a Python number, ``None`` when it has none
    (a marked or raw buffer, which the layout rules read)."""
    if isinstance(value, NUMBERS):
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


def loaded_only(kernel: str, name: str, shape: tuple, kept: tuple, head: tuple, axes) -> str:
    """The refusal of a plan with no source kernel, given a broadcast shape: a loaded artifact reads the plane per lane."""
    return (f"{kernel}: argument {name!r} has shape {shape}, which broadcasts against "
            f"{lane_text(head, axes)} along {pattern_text(kept, axes)} only; this plan was made from a loaded "
            "artifact, which reads it per lane; deploy the kernel (hawk.deploy) to bind it as given")


def _axes_text(pattern, axes) -> str:
    """A pattern as the axes it varies along, for a refusal: ``[a, c]``, ``none (a number)``, ``the exact shape``."""
    if pattern is BY_VALUE:
        return "none (a number)"
    return "the exact shape" if pattern is None else pattern_text(pattern, axes)


def _bound_as(pattern, axes) -> str:
    """How a plane was bound, for a refusal: by value, varying along some axes, or in the exact lane shape."""
    if pattern is BY_VALUE:
        return "by value"
    return "in the exact lane shape" if pattern is None else f"varying along {pattern_text(pattern, axes)}"


def check_rebind(name: str, held, got, axes) -> None:
    """Refuse binding ``name`` as a pattern other than the one a plan was bound
    with (``None`` for the exact shape): the plan reads the caller's buffer at
    the first one's strides."""
    if held != got:
        asked = "the exact lane shape" if got is None else ("by value" if got is BY_VALUE
                                                            else f"varying along {pattern_text(got, axes)}")
        raise HawkError(
            f"argument {name!r} was bound {_bound_as(held, axes)}; "
            f"it cannot be rebound {asked}. Bind afresh for the new pattern")


# -- the twin: the kernel with the per-lane read of a plane rewritten ----- #


def _primal(kernel):
    """The kernel whose inputs a call is classified over: the primal of a derivative (``vjp`` / ``jvp``), else ``kernel``."""
    source = getattr(kernel.sinks, "source", None)
    return source if source is not None and hasattr(source, "lane_space") else kernel


def twin(kernel, patterns: dict):
    """``kernel`` with each plane of ``patterns`` (``{name: pattern}``) read at the axes it varies along; the memoised twin of
    this pattern. A derivative's twin is the derivative rebuilt over its primal's twin, so the gradient of a broadcast
    input has the input's own compact shape, summed (compensated) over the axes it does not vary along."""
    key = tuple(sorted(patterns.items()))
    with kernel._lock:
        held = kernel._twins.get(key)
        if held is None:
            held = kernel._twins[key] = (_derived_twin(kernel, patterns) if _primal(kernel) is not kernel
                                         else _build(kernel, dict(patterns)))
    return held


def _derived_twin(kernel, patterns: dict):
    """The derivative ``kernel`` over the twin of its primal, stating the pattern of every plane of the primal it still binds, so
    its request and sidecar say the compact planes it reads as the primal's do."""
    derived = kernel.sinks
    primal = twin(derived.source, patterns)
    out = type(kernel)(kernel.name, derived.rebuild(primal))
    bound = {name for _role, name in _ir.arg_spec(out.walk)}
    out.broadcast = {name: kept for name, kept in primal.broadcast.items() if name in bound}
    return out


def _build(kernel, patterns: dict):
    from .trace.kernel import Kernel
    from .trace.lane_axes import trace_context
    from .trace.steps import default_step, default_steps

    step = default_step(kernel)
    if step is not kernel and step.name != kernel.name:
        raise HawkError(f"{kernel.name!r} takes several steps per launch; broadcast "
                        "inputs bind to a plain or automatic kernel")
    context = step.context
    nodes = _ir.post_order(step.sinks)
    crossed = {n.operands[0] for n in nodes if _ir.is_a(n, "at")}
    leaves: dict = {}
    for n in nodes:
        name = n.field_text("name") if _ir.is_a(n, "leaf") else None
        if name in patterns and n.field_text("role") in READ_ROLES:
            if n in crossed:
                raise HawkError(
                    f"{kernel.name!r} reads {name!r} at another lane (.at), so it "
                    "needs the exact lane shape")
            leaves.setdefault(name, []).append(n)
    planes, mapping = dict(step.planes), {}
    # the lane indices and table reads below are made from nothing, in the kernel's own context
    with trace_context(context):
        for name, found in leaves.items():
            ttype = _ir.type_of(found[0])
            if len(ttype.shape) > 1:       # a Matrix: the caller's compact plane, read at its index
                index = _index(patterns[name], step.lane_space, context)
                read = _ir.make_node("At", ttype, (found[0], index))
            else:
                planes[name], read = _read(name, ttype, patterns[name], step.lane_space, context)
            mapping.update({leaf: read for leaf in found})
    out = Kernel(step.name, _ir.swap(step.sinks, mapping), planes, kind=step.kind,
                 source_digest=step.source_digest, returns=step.returns)
    out.time, out.time_step, out.jump, out.group = step.time, step.time_step, step.jump, step.group
    # the request and the sidecar state the lane axes each input varies along; a number has none
    out.broadcast = {name: patterns[name] for name in leaves if patterns[name] is not BY_VALUE}
    if step is kernel:
        return out
    fused = default_steps(out)
    fused.time, fused.time_step, fused.jump, fused.group = (
        step.time, step.time_step, step.jump, kernel.group)
    fused.broadcast = out.broadcast
    return fused


def _index(kept: tuple, axes, context):
    """The flat index of a lane in the compact plane of pattern ``kept``: row-major
    over the kept axes, the last fastest (a runtime axis is never in the C), ``0``
    when it varies along none. Made in ``context``."""
    from .trace.lane_axes import LaneAxes

    held = [i for i, k in enumerate(kept) if k]
    if not held:
        return _ir.const(context, 0, _ir.TensorType((), "i32"))
    lane, flat, stride = LaneAxes(axes), None, 1
    for i in reversed(held):
        term = lane[i] * stride
        flat = term if flat is None else flat + term
        stride *= axes[i][1] or 1
    return flat.node


def _read(name: str, ttype, pattern, axes, context) -> tuple:
    """``(declaration, node)``: the plane ``name`` read where it varies, the
    caller's compact buffer: a uniform for a number (:data:`BY_VALUE`), a one-cell table
    for an array that varies along no lane axis, else a lookup. A flat table when
    it varies along the runtime axis (a runtime extent is never in the C), else a
    table named by its kept axes. Made in ``context``."""
    from .trace.decl import Plane, _axes
    from .trace.lane_axes import LaneAxes
    from .trace.value import TableRef

    if pattern is BY_VALUE:
        return Plane("uniform", ttype), _ir.leaf(context, "uniform", "uniform", name, ttype)
    width = ttype.shape[0] if ttype.shape else 0
    element = _ir.TensorType((), ttype.dtype)
    held = [i for i, k in enumerate(pattern) if k]
    if not held and not width:
        return Plane("table", element), TableRef(name, element).at(0).node
    lane = LaneAxes(axes)
    if pattern[0] and axes[0][1] is None:
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
    table_axes = _axes("Table", tuple(spans))
    table = TableRef(name, element, axes=table_axes)
    at = {axes[i][0]: lane[i] for i in held}
    if not width:
        return Plane("table", element, axes=table_axes), table.at(**at).node
    parts = tuple(table.at(**at, **{comp: k}).node for k in range(width))
    return Plane("table", element, axes=table_axes), _ir.make("vec", parts)


# -- a call's planes: classified, then compacted for the twin ----------------- #


def _pattern(value, kept, head: tuple):
    """The pattern of ``value`` whose kept axes are ``kept``: :data:`BY_VALUE` for a bare number beside no head, else ``kept``."""
    return BY_VALUE if isinstance(value, NUMBERS) and not any(kept) and not head else kept


def classify(kernel, given: dict, where: str, extent=None) -> tuple:
    """``(patterns, heads, extent)`` of the planes ``given`` (``{name: value}``): the pattern of each read-only per-lane input
    given broadcast (none for the exact form), the component axes of each per-lane plane, and the runtime extent (the
    label's, else that of the first lane-shaped plane, unless ``extent`` states it). A kernel that declares no ``Grid`` has no
    patterns. A derivative is classified over its primal's inputs. Refuses a shape that does not broadcast, and a broadcast
    shape on a plane that is written or masked, naming ``where``."""
    axes = kernel.lane_space
    if axes[0][0] is None:
        return {}, {}, None
    source = _primal(kernel)
    roles = {n: r for r, n in source.arg_spec}
    walk = source.walk
    heads = {n: _head(_ir.slot_type(walk, r, n)) for n, r in roles.items() if r in READ_ROLES + EXACT_ROLES}
    named = {n for n in given if getattr(source.planes.get(n), "axes", ())}
    if extent is None:
        shaped = [(shape_of(given[n]), heads[n]) for n in heads if n in given and n not in named]
        extent = runtime_extent(shaped, axes, given.get(axes[0][0]))
    patterns = {}
    for name, head in heads.items():
        if name in given and name not in named:
            kept = of(name, roles[name], given[name], head, axes, extent, where)
            if kept is not None:
                patterns[name] = _pattern(given[name], kept, head)
    return patterns, heads, extent


def _tangent_patterns(kernel, patterns: dict, given: dict, heads: dict, extent, where: str) -> dict:
    """``{tangent plane: its pattern}`` of the tangents in ``given`` of the broadcast inputs ``patterns`` of a forward derivative:
    a tangent takes its input's pattern, and any other is refused."""
    axes = kernel.lane_space
    found = {}
    for tangent, name in getattr(kernel.sinks, "tangents", {}).items():
        if name not in patterns or tangent not in given:
            continue
        kept = of(name, "plane_in", given[tangent], heads[name], axes, extent, where)
        got = None if kept is None else _pattern(given[tangent], kept, heads[name])
        if got != patterns[name]:
            raise HawkError(
                f"{where}: argument {tangent!r} must vary along the axes of {name!r}, "
                f"{_axes_text(patterns[name], axes)}, got {_axes_text(got, axes)}")
        found[tangent] = got
    return found


def _followers(kernel, patterns: dict) -> dict:
    """``{plane: the broadcast input it follows}``: each input of ``patterns`` itself, the tangent of one in a forward derivative
    and the gradient of one in a reverse derivative, which bind in the input's own pattern."""
    sinks = kernel.sinks
    owner = {name: name for name in patterns}
    owner |= {tangent: name for tangent, name in getattr(sinks, "tangents", {}).items() if name in patterns}
    owner |= {bar: name for bar, name in getattr(sinks, "gradients", {}).items() if name in patterns}
    return owner


def _compacted(planes: dict, patterns: dict, heads: dict, owner: dict, where: str) -> dict:
    """``planes`` with each broadcast input, tangent and gradient compacted to the buffer the twin binds."""
    return {name: compact(name, value, patterns[owner[name]], heads[owner[name]], where) if name in owner else value
            for name, value in planes.items()}


def routed(kernel, given: dict, where: str) -> tuple:
    """``(patterns, shown, planes)`` for a call of ``kernel`` over ``given``: ``({}, given, given)`` when every per-lane input
    has the exact shape, else the broadcast patterns, the caller's own planes (``given`` and the gradients made) and the
    planes the twin of the patterns binds, each broadcast input, tangent and gradient compacted to the caller's buffer.

    A reverse derivative's omitted gradient of a broadcast input is made, zeroed, in the input's own shape. The gradient of a
    Python number, which the twin reads by value, is refused."""
    patterns, heads, extent = classify(kernel, given, where)
    if not patterns:
        return {}, given, given
    _tangent_patterns(kernel, patterns, given, heads, extent, where)
    shown = dict(given)
    for bar, name in getattr(kernel.sinks, "gradients", {}).items():
        if name not in patterns:
            continue
        if patterns[name] is BY_VALUE:
            raise HawkError(f"{where}: argument {name!r} is a Python number, which the kernel reads by value, so it has "
                            "no gradient; pass a one-element array")
        if bar not in shown:
            shown[bar] = _zeros_like(given[name])
    return patterns, shown, _compacted(shown, patterns, heads, _followers(kernel, patterns), where)


def _zeros_like(value):
    """A zeroed array of the shape and dtype of ``value``, in the memory ``value`` lives in."""
    from . import _place  # _place loads the backend libraries, which this module's importers need not

    return _place.Maker.at(_place.device_of(value), value).like(value)


def rebound(kernel, patterns: dict, held: dict, planes: dict, lanes: int, where: str) -> tuple:
    """``(shown, planes)`` of a bound plan's rebind of ``planes`` over ``held``, the planes it has bound as the caller gave
    them: ``shown`` is ``held`` with the new values in, ``planes`` the same compacted as the twin binds them.

    A plane bound broadcast (and the tangent or gradient of one) keeps its pattern and is accepted as any shape of it or as the
    compact buffer it holds; a plane bound in the exact shape stays exact; another pattern is refused by name. ``lanes`` is
    the lane count of the plan, which settles the runtime extent."""
    shown = {**held, **planes}
    axes = kernel.lane_space
    if axes[0][0] is None:
        return shown, shown
    extent = axes[0][1] if axes[0][1] is not None else lanes // prod(e for _l, e in axes[1:])
    owner = _followers(kernel, patterns)
    heads = classify(kernel, {}, where, extent)[1]
    flat = {name: shape_of(compact(name, held[name], patterns[owner[name]], heads[owner[name]], where))
            for name in planes if name in owner and name in held}
    changed = {name: value for name, value in planes.items() if name not in flat or shape_of(value) != flat[name]}
    fresh = classify(kernel, changed, where, extent)[0]
    fresh |= _tangent_patterns(kernel, patterns | fresh, changed, heads, extent, where)
    for name in changed:
        if name in patterns or name in fresh:
            check_rebind(name, patterns.get(owner.get(name)), fresh.get(name), axes)
    return shown, _compacted(shown, patterns, heads, owner, where)


def compact(name: str, value, pattern, head: tuple, where: str):
    """The caller's buffer as the twin binds it: the number itself for a uniform (:data:`BY_VALUE`), else the same memory with
    the size-1 axes dropped (flat when it varies along the runtime axis; a one-cell array when it varies along none). A buffer
    that is not C-contiguous would need a copy: refused."""
    if pattern is BY_VALUE:
        return value
    if isinstance(value, NUMBERS):     # a Python number beside a runtime axis of one lane: the 0-d array's path
        import numpy

        value = numpy.asarray(value)
    flags = getattr(value, "flags", None)
    contiguous = (flags.c_contiguous if flags is not None
                  else getattr(value, "is_contiguous", lambda: True)())
    if not contiguous:
        raise HawkError(
            f"{where}: broadcast argument {name!r} (shape {tuple(value.shape)}) must be "
            f"C-contiguous to bind without a copy; pass np.ascontiguousarray({name})")
    return value.reshape(-1) if len(head) < 2 else value.reshape(prod(head), -1)
