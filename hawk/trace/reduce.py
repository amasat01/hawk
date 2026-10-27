# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Reductions and named axes on traced values.

A value is ONE lane's, so ``x.sum(axis=...)`` folds the value's own axes:
the components of a vector, the rows or columns of a matrix. Across lanes there
are two doors: ``hawk.sum/max/min/mean(x, over=, where=, policy=)`` reduces over
lane axes, and ``hawk.scatter(sink)[key] += v`` adds into a sink by key (another
lane's input is read with ``u.at(i)``). The labelled views ``.components``,
``.rows`` and ``.columns`` read a component, a row or a column by name.
"""

from __future__ import annotations

from typing import Any

from ..ir import HawkError
from ..ir import make as _make
from ..ir.nodes import POLICIES, SCATTER_POLICIES, Leaf, ReduceSamples
from .value import Value, _operands, index_node, n_samples, node_of, real, select, vec


def _axis(value: Value, axis: Any, op: str):
    """``axis`` as the value's own axis number (None for all), or a refusal."""
    rank = len(value.ttype.shape)
    if rank == 0:
        raise HawkError(
            f"x.{op}(): a rank-0 value is this lane's and has no axis to fold; "
            f"the {op} over the lanes is hawk.{op}(x)")
    if axis is None:
        return None
    if axis == "samples":
        raise HawkError(
            f"x.{op}(axis='samples') is gone: the {op} over every lane is "
            f"hawk.{op}(x), over some lane axes hawk.{op}(x, over=...)")
    if axis == "components" and rank == 1:
        return 0
    if axis == "rows" and rank == 2:
        return 0
    if axis == "columns" and rank == 2:
        return 1
    if isinstance(axis, int) and not isinstance(axis, bool) and 0 <= axis < rank:
        return axis
    own = "0 or 'components'" if rank == 1 else "0 / 'rows', 1 / 'columns'"
    raise HawkError(
        f"x.{op}(axis={axis!r}): a rank-{rank} value has the axes {own} "
        "(the lane axes are reduced by hawk.sum(x, over=...))")


def _entry(value: Value, index: tuple) -> Value:
    """The entry of a matrix at the static ``(row, column)``."""
    return Value(_make("component", (value.node,), index))


def _fold(op: str, items: list) -> Value:
    out = items[0]
    for item in items[1:]:
        out = out + item if op == "sum" else (
            Value(_make("max" if op == "max" else "min", (out.node, node_of(item)))))
    return out


def reduce_own(value: Value, op: str, axis: Any = None) -> Value:
    """``value.sum/max/min/mean(axis)`` over the value's own axes."""
    ax = _axis(value, axis, op)
    shape = value.ttype.shape
    base = "sum" if op == "mean" else op
    if ax is None:
        if len(shape) == 1 or base == "sum":
            out = Value(_make("sum", (value.node,))) if base == "sum" else _fold(
                base, [value[k] for k in range(shape[0])])
        else:
            out = _fold(base, [_entry(value, (r, c)) for r in range(shape[0])
                                     for c in range(shape[1])])
        count = _size(shape)
    elif len(shape) == 1:
        out = reduce_own(value, base)
        count = shape[0]
    else:
        keep = 1 - ax
        parts = [_fold(base, [_entry(value, (j, i) if ax == 0 else (i, j))
                              for j in range(shape[ax])]) for i in range(shape[keep])]
        out = vec(*parts)
        count = shape[ax]
    return out / float(count) if op == "mean" else out


def _size(shape: tuple) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


class _View:
    """``v.components`` / ``m.rows`` / ``m.columns``: a component, a row or a
    column read by name; each lowers to the existing ``component`` and ``vec``."""

    __slots__ = ("value", "axis")

    def __init__(self, value: Value, axis: str) -> None:
        self.value = value
        self.axis = axis

    def __getitem__(self, k: Any) -> Value:
        shape = self.value.ttype.shape
        if self.axis == "components":
            if isinstance(k, slice):
                return vec(*[self.value[i] for i in range(*k.indices(shape[0]))])
            return self.value[k]
        if isinstance(k, slice):
            raise HawkError(
                f"a slice of .{self.axis} is not in this release; read one "
                f"{self.axis[:-1]} at a time")
        if isinstance(k, Value):
            return _own_view(self.value, self.axis, k)
        if not isinstance(k, int) or isinstance(k, bool):
            raise HawkError(
                f".{self.axis}[i] takes an integer or a traced integer, got {k!r}")
        extent = shape[0] if self.axis == "rows" else shape[1]
        if not 0 <= k < extent:
            raise HawkError(f".{self.axis}[{k}] is outside the {extent} {self.axis}")
        if self.axis == "rows":
            return vec(*[_entry(self.value, (k, c)) for c in range(shape[1])])
        return vec(*[_entry(self.value, (r, k)) for r in range(shape[0])])


def view(value: Value, axis: str) -> _View:
    """The named view ``axis`` of ``value``, refused where the value lacks it."""
    rank = len(value.ttype.shape)
    want = 1 if axis == "components" else 2
    if rank != want:
        raise HawkError(
            f".{axis} names an axis of a rank-{want} value, got {value.ttype!r}")
    return _View(value, axis)


_MASK_STORE = ("a mask is not a key: a masked store keeps the old value where the "
               "mask is false, y = m.where(mask, v, y)")


def _identity(op: str) -> Value:
    """The identity of ``op``."""
    return real({"sum": 0.0, "max": float("-inf"), "min": float("inf")}[op])


def _own_view(value: Value, axis: str, k: Any) -> Value:
    """``view(value, axis)[k]`` where ``k`` may be a traced index: a row or a
    column picked out of a matrix by a select chain over its static extent
    (a vector's component is the ordinary ``value[k]``)."""
    if axis == "components" or isinstance(k, int):
        return view(value, axis)[k]
    rows, cols = value.ttype.shape
    if axis == "rows":
        pick = [[_entry(value, (r, c)) for r in range(rows)] for c in range(cols)]
    else:
        pick = [[_entry(value, (r, c)) for c in range(cols)] for r in range(rows)]
    out = []
    for line in pick:
        item = line[-1]
        for j in range(len(line) - 2, -1, -1):
            item = select(k == j, line[j], item)
        out.append(item)
    return vec(*out)


def _policy(policy: Any, where: str, allowed: tuple = POLICIES) -> str:
    if policy not in allowed:
        raise HawkError(f"{where}: policy={policy!r} is not one of {allowed}")
    return policy


def _term(value: Any, where: str) -> Value:
    """``value`` as the floating term a reduction or a scatter adds (a bool
    counts 1 / 0)."""
    from .value import _SinkRef

    if isinstance(value, (Reduced, _SinkRef)):
        raise HawkError(f"{where}: the term is a value of this lane, got the output "
                        f"{getattr(value, 'name', 'of a reduction')!r}")
    if not isinstance(value, Value):
        value = Value(node_of(value))
    if value.ttype.dtype == "bool":
        value = select(value, real(1.0), real(0.0))
    if value.ttype.dtype not in ("f64", "f32"):
        raise HawkError(f"{where}: a float or bool value is reduced, got {value.ttype!r}")
    return value


def _reduced_axes(over: Any, axes: tuple, where: str) -> tuple:
    """The positions of the lane axes ``over`` names (``None``: every axis)."""
    n = len(axes)
    if over is None:
        return tuple(range(n))
    labels = [label for label, _ in axes]
    out = []
    for a in over if isinstance(over, tuple) else (over,):
        if isinstance(a, str):
            if a not in labels:
                raise HawkError(
                    f"{where}: over={a!r} names no lane axis; "
                    + (f"this kernel's Grid declares {labels}" if labels[0] is not None
                       else "the kernel declares no Grid[...], so its one axis is over=0"))
            out.append(labels.index(a))
        elif isinstance(a, int) and not isinstance(a, bool) and -n <= a < n:
            out.append(a % n)
        else:
            raise HawkError(f"{where}: over= takes lane axis labels or positions "
                            f"(0 the slowest, -1 the fastest of {n}), got {a!r}")
    if len(set(out)) != len(out):
        raise HawkError(f"{where}: over={over!r} names an axis twice")
    return tuple(sorted(out))


def _extent(axes: tuple, i: int) -> Any:
    """The extent of lane axis ``i``: its declared number, or the runtime one
    read off the lane count."""
    if axes[i][1] is not None:
        return axes[i][1]
    rest = 1
    for _label, extent in axes[1:]:
        rest *= extent
    return n_samples() / rest if rest != 1 else n_samples()


def _product(items: list) -> Any:
    out: Any = 1
    for item in items:
        out = item if isinstance(out, int) and out == 1 else out * item
    return out


def _reduce(op: str, x: Any, over: Any, where_mask: Any, policy: Any) -> Any:
    from .lane_axes import LaneAxes, _trace

    call = f"hawk.{op}(x, over={over!r})" if over is not None else f"hawk.{op}(x)"
    policy = _policy(policy, call)
    value = _term(x, call)
    axes = getattr(_trace, "axes", ((None, None),))
    gone = _reduced_axes(over, axes, call)
    kept = tuple(i for i in range(len(axes)) if i not in gone)
    base = "sum" if op == "mean" else op
    if kept and base != "sum":
        raise HawkError(
            f"{call}: a {op} over some of the lane axes is in the next release; "
            f"hawk.{op}(x) (one value for the run) and hawk.sum/mean(x, over=...) "
            "work now")
    if kept and policy == "ordered":
        raise HawkError(
            f"{call}: policy='ordered' over some of the lane axes is in the next "
            "release; use policy='auto' or 'compensated' here (hawk.sum(x, "
            "policy='ordered') over every axis is ordered now)")
    if where_mask is not None:
        mask = where_mask if isinstance(where_mask, Value) else Value(node_of(where_mask))
        if mask.ttype.dtype != "bool" or mask.ttype.shape:
            raise HawkError(f"{call}: where= is this lane's bool mask, got {mask.ttype!r}")
        if op == "mean":
            raise HawkError(
                f"{call}: the mean of the masked lanes needs their count, a second "
                "reduction: hawk.sum(x, where=mask) / hawk.sum(mask)")
        value = select(mask, value, _identity(base))
    if not kept:
        term = value / real(n_samples()) if op == "mean" else value
        return Value(ReduceSamples(base, node_of(term), policy=policy))
    if op == "mean":
        value = value / real(_product([_extent(axes, i) for i in gone]))
    lane = LaneAxes(axes)
    key, stride = None, 1
    for i in reversed(kept):
        term = lane[i] if stride == 1 else lane[i] * stride
        key = term if key is None else term + key
        stride *= axes[i][1] or 1
    count = _product([_extent(axes, i) for i in kept])
    count = Value(index_node(count)) if isinstance(count, int) else count
    return Reduced(ReduceSamples("sum", node_of(value), key=node_of(key),
                                 count=node_of(count), policy=policy,
                                 kept=tuple(axes[i] for i in kept)),
                   tuple(axes[i] for i in kept))


def sum(x: Any, over: Any = None, where: Any = None, policy: str = "auto") -> Any:  # noqa: A001
    """The sum of ``x`` over the lane axes ``over`` names (a label, a position
    or a tuple of them; ``None``: every axis, one value for the run). Masked
    lanes (``where=`` false) add nothing; ``policy`` is the sum contract
    (``"auto"``, ``"compensated"``, ``"ordered"``). Over every axis the result
    may be read back later in the body (a stage boundary); over some axes it is
    an output, one value per kept coordinate, bound ``(*payload, *kept)``."""
    return _reduce("sum", x, over, where, policy)


def max(x: Any, over: Any = None, where: Any = None, policy: str = "auto") -> Any:  # noqa: A001
    """The maximum of ``x`` over every lane (NaN-ignoring, exact)."""
    return _reduce("max", x, over, where, policy)


def min(x: Any, over: Any = None, where: Any = None, policy: str = "auto") -> Any:  # noqa: A001
    """The minimum of ``x`` over every lane (NaN-ignoring, exact)."""
    return _reduce("min", x, over, where, policy)


def mean(x: Any, over: Any = None, where: Any = None, policy: str = "auto") -> Any:
    """The mean of ``x`` over the lane axes ``over`` names: the sum over the
    count of reduced lanes."""
    return _reduce("mean", x, over, where, policy)


class Reduced:
    """``hawk.sum(x, over=...)`` over some lane axes: one value per kept
    coordinate. It is returned or assigned to a ``Mutable`` declared with the
    kept axes, and it reads back in the body as the value of THIS lane's key
    (``x / hawk.sum(x, over="col")``): the kernel runs in stages, the reduction
    committed by the first."""

    __slots__ = ("node", "kept")

    def __init__(self, node: ReduceSamples, kept: tuple) -> None:
        self.node = node
        self.kept = kept

    @property
    def ttype(self) -> Any:
        return self.node.ttype

    def __bool__(self) -> bool:
        raise HawkError(
            "hawk.sum(x, over=...) read in the body is a value, not a truth: "
            "compare it, m.where(hawk.sum(x, over=...) > 0, a, b)")


def _read_back(name: str) -> Any:
    def op(self: Reduced, *args: Any) -> Any:
        return getattr(Value(self.node), name)(*args)
    op.__name__ = name
    return op


for _name in ("__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
              "__truediv__", "__rtruediv__", "__neg__", "__pow__", "__getitem__",
              "__lt__", "__le__", "__gt__", "__ge__"):
    setattr(Reduced, _name, _read_back(_name))
del _name


def axes_plane(kept: tuple) -> tuple:
    """A plane's ``dims`` for the kept lane axes ``((label, extent|None), ...)``."""
    out, stride = [], 1
    for label, extent in reversed(kept):
        out.append((label, extent, stride))
        stride = stride * extent if extent is not None and stride is not None else None
    return tuple(out[::-1])


class _Stored:
    """What a scatter leaves in the sink's name: the plane is written by key,
    so the name has no value of this lane to read."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def _hawk_refuse(self, *_: Any) -> Any:
        raise HawkError(f"{self.name!r} is written by key (hawk.scatter); it has no "
                        "value of this lane to read in this kernel")

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _hawk_refuse
    __truediv__ = __rtruediv__ = __neg__ = __pow__ = __getitem__ = _hawk_refuse


def scatter(sink: Any, policy: str = "auto") -> Any:
    """``hawk.scatter(sink)[key] += v`` adds ``v`` into ``sink`` at ``key``,
    ``= v`` stores it there. Many lanes may share a key; the adds are combined
    (atomics or a segmented sum), never a race; the order is not fixed unless
    ``policy="ordered"``. ``"auto"`` and ``"compensated"`` add with a
    correction kept in a companion plane ``<sink>_c`` that the runtime owns
    (the caller never passes it); ``"plain"`` is the atomic add alone. ``key``
    is an index, one coordinate per axis of a sink declared with named axes, an
    ``Index`` plane or a ``Sorted[Index]`` one.
    ``hawk.scatter(mask, policy="or")[word] |= bits`` sets bits of an ``Int32``
    or ``UInt32`` plane (a bit mask: the word index is the key): the commit is
    the target's atomic OR, so any number of lanes may hit one word and the
    result is the exact ``np.bitwise_or.at``; the C target ORs serially. A bit
    mask is discrete: ``vjp`` and ``jvp`` of a kernel that writes one refuse by name.
    Written only as the target of a store in a kernel body."""
    raise HawkError(
        "hawk.scatter(y) is the target of a store: write hawk.scatter(y)[key] += v "
        "or hawk.scatter(y)[key] = v as a statement of the kernel body")


def _flat_key(ref: Any, key: Any, where: str) -> Any:
    """``key`` as one flat index into ``ref``: a coordinate per declared axis of
    a named-axis sink, folded with its strides, or a flat index."""
    if isinstance(key, tuple):
        labels = [label for label, _size, _stride in ref.dims]
        if len(key) != len(labels):
            raise HawkError(
                f"{where}: a tuple key gives one coordinate per declared axis "
                f"{labels or '(none: this sink is flat, give one index)'}, got {len(key)}")
        flat = None
        for k, (_label, _size, stride) in zip(key, ref.dims):
            term = Value(index_node(k)) * stride if stride != 1 else Value(index_node(k))
            flat = term if flat is None else flat + term
        return node_of(flat)
    if isinstance(key, slice) or key is Ellipsis:
        raise HawkError(f"{where}: key is an index, an Index plane or a tuple of "
                        "coordinates")
    return index_node(key)


def scatter_store(refs: dict, name: str, key: Any, value: Any, op: str, current: Any,
                  policy: str = "auto") -> Any:
    """``hawk.scatter(name, policy=...)[key] (op)= value``, as the ast pass
    reaches it: ``+=`` accumulates (``accum_write``), ``=`` stores
    (``wide_write``)."""
    from .value import MutableRef, _SinkRef, by_key

    where = f"hawk.scatter({name})[key] {op} v"
    ref = refs.get(name)
    if not isinstance(ref, _SinkRef):
        raise HawkError(f"{where}: {name!r} is not an output plane (declare it "
                        "Mutable[...])")
    policy = _policy(policy, where, SCATTER_POLICIES)
    if isinstance(key, Value) and key.ttype.dtype == "bool":
        raise HawkError(f"{where}: {_MASK_STORE}")
    if op == "|=" and policy != "or":
        raise HawkError(f"{where}: |= sets bits, written hawk.scatter({name}, "
                        f"policy=\"or\")[key] |= bits")
    if policy == "or" and op != "|=":
        raise HawkError(f"{where}: policy='or' is for |= only")
    if op == "=" and policy != "auto":
        raise HawkError(f"{where}: a store has no sum contract; policy= is for += only")
    if policy == "ordered":
        raise HawkError(
            f"{where}: policy='ordered' on a scatter is in the next release; use "
            "policy='auto' or 'compensated'")
    if isinstance(ref, MutableRef) and isinstance(current, Value) and not (
            isinstance(current.node, Leaf) and current.node.kind == "prior_read"):
        raise HawkError(
            f"{name!r} is written per lane (locked) and by key "
            f"(hawk.scatter({name})[key]): one plane has one extent")
    by_key(ref, _flat_key(ref, key, where), value, op != "=", policy)
    return current if isinstance(current, _Stored) else _Stored(name)


def maxmin(which: str, *args: Any, **kwargs: Any) -> Any:
    """The builtin ``max``/``min`` as the ast pass reaches it: two or more
    traced operands are ``hawk.math.max/min`` (a tie keeps the first
    operand); anything untraced is the Python builtin."""
    import builtins
    traced = [a for a in args if isinstance(a, Value)]
    if len(args) == 1 and not kwargs:
        only = args[0]
        if isinstance(only, Value) or (isinstance(only, (list, tuple))
                                       and any(isinstance(a, Value) for a in only)):
            raise HawkError(
                f"{which}(x) of one traced value or an iterable of them is not a "
                f"HAWK form; compare values two at a time: {which}(a, b) is "
                f"hawk.math.{which}(a, b), and the {which} over the lanes is "
                f"hawk.{which}(x)")
    if not traced:
        return getattr(builtins, which)(*args, **kwargs)
    if kwargs:
        raise HawkError(f"{which}({', '.join(kwargs)}=...) is not a HAWK form")
    out = args[0]
    for a in args[1:]:
        out = Value(_make(which, _operands((out, a))))
    return out
