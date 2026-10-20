# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Reductions and named axes on traced values.

A value is ONE sample's, so ``x.sum(axis=...)`` folds the value's own axes:
the components of a vector, the rows or columns of a matrix. The samples
axis is reached through ``hawk.all_samples(x)``, whose ``.sum() / .max() /
.min() / .mean()`` give a reduction over every sample that is stored to a
``Mutable`` output (a cell). The labelled views ``.components``, ``.rows`` and
``.columns`` read a component, a row or a column by name.
"""

from __future__ import annotations

from typing import Any

from ..ir import HawkError
from ..ir import make as _make
from ..ir.loop_nodes import Loop, LoopCarry, LoopIndex, LoopValue
from ..ir.nodes import At, Leaf, Node, ReduceSamples, extent_of
from .value import Value, _operands, index_node, node_of, real, select, vec


def _axis(value: Value, axis: Any, op: str):
    """``axis`` as the value's own axis number (None for all), or a refusal."""
    rank = len(value.ttype.shape)
    if rank == 0:
        raise HawkError(
            f"x.{op}(): a rank-0 value is this sample's and has no axis to fold; "
            f"the reduction over samples is hawk.all_samples(x).{op}()")
    if axis is None:
        return None
    if axis == "samples":
        raise HawkError(
            f"axis='samples' on a value of one sample: the {op} over samples is "
            f"hawk.all_samples(x).{op}()")
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
        "(the samples axis is hawk.all_samples(x))")


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


from ..types import INT_PLANE_DTYPES

_INDEX_DTYPES = ("i32", "i64") + tuple(INT_PLANE_DTYPES)
_GATHER_ROLES = ("per_sample", "vec_in", "mat_in")

_BARE = ("'hawk.all_samples(x)' is every sample; compute per sample first "
         "(hawk.all_samples(x * 2)), then index, mask or reduce it")
_MASKED = ("a masked view has no fixed length; reduce it "
           "(hawk.all_samples(x)[mask].sum()) or store through it "
           "(hawk.all_samples(y)[mask] = v)")


def _identity(op: str, like: Value) -> Value:
    """The identity of ``op`` as a value broadcast against ``like``."""
    return real({"sum": 0.0, "max": float("-inf"), "min": float("inf")}[op])


def _as_float(v: Value) -> Value:
    """A bool term (a count's mask) as 1.0 / 0.0."""
    return select(v, real(1.0), real(0.0)) if v.ttype.dtype == "bool" else v


def _reread(value: Value, index: Node) -> Value:
    """``value`` as sample ``index`` computed it: every per-sample input read
    at ``index`` (pure recompute, no plane). Refused by name when the
    subtree reads something that exists only for this sample."""
    from ..ir.nodes import SampleIndex
    from ..ir.walk import substitute

    mapping: dict = {}
    stack, seen = [value.node], set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        why = None
        if isinstance(node, Leaf):
            if node.role in _GATHER_ROLES:
                mapping[id(node)] = At(node, index, node.ttype)
            elif node.kind == "prior_read":
                why = node.name
            elif node.role == "terminated":
                why = node.name
        elif isinstance(node, SampleIndex):
            mapping[id(node)] = index
        elif isinstance(node, At) and node.plane.role in _GATHER_ROLES:
            why = f"another sample's {node.plane.name!r}"
        elif getattr(node, "kind", "").startswith("random"):
            why = "a random draw"
        elif isinstance(node, (LoopCarry, LoopValue, LoopIndex, Loop)):
            why = "a loop-carried value"
        if why is not None:
            raise HawkError(
                f"'hawk.all_samples(expr)[i]' re-reads expr at sample i, but expr "
                f"reads {why}, which exists only for this sample")
        stack.extend(node.operands)
    return Value(substitute([value.node], mapping)[0])


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


class _AllView:
    """``hawk.all_samples(m).rows`` / ``.columns`` / ``.components``."""

    __slots__ = ("value", "axis")

    def __init__(self, value: Value, axis: str) -> None:
        self.value = value
        self.axis = axis

    def __getitem__(self, k: Any) -> AllSamples:
        return AllSamples(_own_view(self.value, self.axis, k))


class AllSamples:
    """``hawk.all_samples(x)``: every sample's ``x``. It is indexed (``[i]``),
    masked (``[mask]``), sliced to a component (``[:, k]``), reduced
    (``.sum()``) or stored through (``all_samples(y)[key] += t``); it is
    never arithmetic and never stored whole."""

    __slots__ = ("value",)

    def __init__(self, value: Value) -> None:
        self.value = value

    def _hawk_refuse(self, *_: Any) -> Any:
        raise HawkError(_BARE)

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _hawk_refuse
    __truediv__ = __rtruediv__ = __neg__ = __pow__ = _hawk_refuse

    @property
    def components(self) -> _AllView:
        return _AllView(self.value, view(self.value, "components").axis)

    @property
    def rows(self) -> _AllView:
        return _AllView(self.value, view(self.value, "rows").axis)

    @property
    def columns(self) -> _AllView:
        return _AllView(self.value, view(self.value, "columns").axis)

    def __getattr__(self, name: str) -> Any:
        if name in ("batch", "samples"):
            raise HawkError("the samples axis is read through hawk.all_samples(v)[i]")
        raise AttributeError(name)

    def __getitem__(self, k: Any) -> Any:
        v = self.value
        if isinstance(k, tuple):
            if not (k and isinstance(k[0], slice) and k[0] == slice(None)
                    and len(k) - 1 == len(v.ttype.shape) and len(k) > 1):
                raise HawkError(
                    "hawk.all_samples(v)[:, k] takes every sample (':') then one "
                    "static index per axis of v")
            return AllSamples(v[k[1]] if len(k) == 2 else _entry(v, tuple(k[1:])))
        if isinstance(k, Value) and k.ttype.dtype == "bool":
            if v.ttype.dtype == "bool":
                raise HawkError(
                    "a mask of a mask: hawk.all_samples(x > 0)[mask] is refused; "
                    "combine the conditions (land) and mask the value once")
            return Masked(v, k)
        if isinstance(k, (slice, bool)) or k is Ellipsis:
            raise HawkError(
                "hawk.all_samples(x)[i] takes an Index or integer value, or a "
                "bool mask; slices beyond [:, k] are not in this release")
        idx = index_node(k)
        if idx.ttype.shape != () or idx.ttype.dtype not in _INDEX_DTYPES:
            raise HawkError(
                f"hawk.all_samples(x)[i]: i must be an integer value, got {idx.ttype!r}")
        node = v.node
        if isinstance(node, Leaf) and node.kind == "prior_read":
            raise HawkError(
                f"{node.name!r} is written by this launch; another sample's value is "
                "not defined here — stage it (Staged) or pass it as an input")
        if isinstance(node, Leaf) and node.role in _GATHER_ROLES:
            return Value(At(node, idx, node.ttype))
        return _reread(v, idx)

    def __setitem__(self, k: Any, v: Any) -> None:     # pragma: no cover
        raise HawkError("a store through hawk.all_samples(y)[key] is written as a "
                        "statement of the kernel body, on a Mutable output")

    def _reduce(self, op: str, axis: Any) -> Value:
        return _cut(self.value, op, axis)

    def sum(self, axis: Any = None) -> Value:
        """The sum over every sample."""
        return self._reduce("sum", axis)

    def max(self, axis: Any = None) -> Value:
        """The maximum over every sample (NaN-ignoring, exact)."""
        return self._reduce("max", axis)

    def min(self, axis: Any = None) -> Value:
        """The minimum over every sample (NaN-ignoring, exact)."""
        return self._reduce("min", axis)

    def mean(self, axis: Any = None) -> Value:
        """The mean over every sample: the sum of each sample's term over
        ``n_samples()``."""
        return self._reduce("mean", axis)


def _cut(value: Value, op: str, axis: Any, mask: Value | None = None) -> Value:
    """The reduction of ``value`` over samples (and over the own axes ``axis``
    names beside ``"samples"``), as the transient value a store turns into
    cells."""
    own = ()
    if isinstance(axis, tuple):
        if not any(a in ("samples", 0) for a in axis):
            raise HawkError(f"all_samples(x).{op}(axis={axis!r}): name 'samples' too")
        own = tuple(a for a in axis if a not in ("samples", 0))
        axis = "samples"
    if axis not in (None, 0, "samples"):
        raise HawkError(
            f"hawk.all_samples(x).{op}(axis={axis!r}): the view has the axis "
            "'samples' (or 0), plus the value's own named axes in a tuple")
    value = _as_float(value)
    if value.ttype.dtype not in ("f64", "f32"):
        raise HawkError(f"hawk.all_samples(x).{op}(): a float or bool value, "
                        f"got {value.ttype!r}")
    for a in own:
        value = reduce_own(value, op, a)
    if axis is None and value.ttype.shape:
        value = reduce_own(value, op)
    base = "sum" if op == "mean" else op
    if mask is not None:
        if op == "mean":
            raise HawkError(
                "the mean of a masked view needs the count of the mask, which is "
                "another cell: store hawk.all_samples(x)[mask].sum() and "
                "hawk.all_samples(mask).sum() and divide on the host")
        value = select(mask, value, _identity(base, value))
    term = value / real(_n_samples()) if op == "mean" else value
    return Value(ReduceSamples(base, node_of(term)))


class Masked:
    """``hawk.all_samples(x)[mask]``: reduced, or stored through; nothing else."""

    __slots__ = ("value", "mask")

    def __init__(self, value: Value, mask: Value) -> None:
        self.value = value
        self.mask = mask

    def _hawk_refuse(self, *_: Any) -> Any:
        raise HawkError(_MASKED)

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _hawk_refuse
    __truediv__ = __rtruediv__ = __neg__ = __pow__ = __getitem__ = _hawk_refuse

    def sum(self, axis: Any = None) -> Value:
        """The sum of the masked samples."""
        return _cut(self.value, "sum", axis, self.mask)

    def max(self, axis: Any = None) -> Value:
        """The maximum of the masked samples."""
        return _cut(self.value, "max", axis, self.mask)

    def min(self, axis: Any = None) -> Value:
        """The minimum of the masked samples."""
        return _cut(self.value, "min", axis, self.mask)

    def mean(self, axis: Any = None) -> Value:
        """Refused: the count is a second cell."""
        return _cut(self.value, "mean", axis, self.mask)


def _n_samples() -> Value:
    from .value import n_samples
    return n_samples()


class _Stored:
    """What a store through ``hawk.all_samples(y)[key]`` leaves in ``y``'s name:
    the plane is written by key, so the name has no per-sample value."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def _hawk_refuse(self, *_: Any) -> Any:
        raise HawkError(f"{self.name!r} is stored by key; it has no per-sample "
                        "value to read in this kernel")

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _hawk_refuse
    __truediv__ = __rtruediv__ = __neg__ = __pow__ = __getitem__ = _hawk_refuse


def viewstore(refs: dict, name: str, key: Any, value: Any, op: str, current: Any) -> Any:
    """``hawk.all_samples(name)[key] (op)= value``, as the ast pass reaches it.

    A bool ``key`` is a mask (``y = select(mask, value, y)``, the per-sample
    store); an Index or integer ``key`` is by key: ``+=`` accumulates
    (``accum_write``), ``=`` stores (``wide_write``)."""
    from .value import AccumRef, MutableRef, _SinkRef

    ref = refs.get(name)
    if not isinstance(ref, _SinkRef):
        raise HawkError(
            f"hawk.all_samples({name})[...] stores through an output plane; "
            f"{name!r} is not one (declare it Mutable[...])")
    if isinstance(key, Value) and key.ttype.dtype == "bool":
        if not isinstance(ref, MutableRef) or not isinstance(current, Value):
            raise HawkError(f"hawk.all_samples({name})[mask] = v stores through a Mutable")
        new = current + value if op == "+=" else value
        return select(key, new, current)
    if isinstance(key, (slice, tuple)) or key is Ellipsis:
        raise HawkError(f"hawk.all_samples({name})[key]: key is an Index, an "
                        "integer value or a bool mask")
    if isinstance(ref, MutableRef):
        if isinstance(current, Value) and not (
                isinstance(current.node, Leaf) and current.node.kind == "prior_read"):
            raise HawkError(
                f"{name!r} is written per sample (locked) and by key "
                f"(hawk.all_samples({name})[key]): one plane has one extent")
    ref.by_key(key, value, accumulate=(op == "+="))
    return current if isinstance(current, _Stored) else _Stored(name)


def all_samples(x: Any) -> AllSamples:
    """Every sample's ``x``, to be indexed, masked or reduced:
    ``total = hawk.all_samples(r * r).sum()`` stored to a ``Mutable`` output
    gives one sum over every sample."""
    from .value import _SinkRef

    if isinstance(x, (AllSamples, Masked)):
        raise HawkError("hawk.all_samples(hawk.all_samples(x)): already every sample")
    if isinstance(x, _SinkRef):
        raise HawkError(
            f"{x.name!r} is written by this launch; another sample's value is not "
            "defined here — stage it (Staged) or pass it as an input")
    if not isinstance(x, Value):
        x = Value(node_of(x))
    if extent_of(x.node) == "all_samples":
        raise HawkError("hawk.all_samples(hawk.all_samples(x)): already every sample")
    return AllSamples(x)


def maxmin(which: str, *args: Any, **kwargs: Any) -> Any:
    """The builtin ``max``/``min`` as the ast pass reaches it: two or more
    traced operands are ``hawk.math.max/min`` (a tie keeps the first
    operand); anything untraced is the Python builtin."""
    import builtins
    traced = [a for a in args if isinstance(a, (Value, AllSamples))]
    if len(args) == 1 and not kwargs:
        only = args[0]
        if isinstance(only, Value) or (isinstance(only, (list, tuple))
                                       and any(isinstance(a, Value) for a in only)):
            raise HawkError(
                f"{which}(x) of one traced value or an iterable of them is not a "
                f"HAWK form; compare values two at a time: {which}(a, b) is "
                f"hawk.math.{which}(a, b), and the {which} over samples is "
                f"hawk.all_samples(x).{which}()")
    if not traced:
        return getattr(builtins, which)(*args, **kwargs)
    if kwargs:
        raise HawkError(f"{which}({', '.join(kwargs)}=...) is not a HAWK form")
    out = args[0]
    for a in args[1:]:
        out = Value(_make(which, _operands((out, a))))
    return out
