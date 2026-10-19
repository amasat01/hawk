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
from ..ir.nodes import ReduceSamples
from .value import Value, _operands, node_of, real, vec


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
        if not isinstance(k, int) or isinstance(k, bool):
            raise HawkError(
                f".{self.axis}[i] takes a STATIC integer in this release, got {k!r}")
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


class AllSamples:
    """``hawk.all_samples(x)``: every sample's ``x``. Only its reductions are
    in this release, and a reduction is stored to a ``Mutable``."""

    __slots__ = ("value",)

    def __init__(self, value: Value) -> None:
        self.value = value

    def _refuse(self, what: str):
        raise HawkError(
            f"'hawk.all_samples(x)' is every sample; {what} is not in this "
            "release — reduce it (hawk.all_samples(x).sum()) and store the "
            "result to a Mutable")

    def __getitem__(self, k: Any) -> Any:
        self._refuse("indexing, masking or gathering it")

    def __setitem__(self, k: Any, v: Any) -> None:
        self._refuse("storing through it")

    def _arith(self, *_: Any) -> Any:
        raise HawkError(
            "'hawk.all_samples(x)' is every sample; compute per sample first "
            "(hawk.all_samples(x * 2)), then reduce it")

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _arith
    __truediv__ = __rtruediv__ = __neg__ = __pow__ = _arith

    def _reduce(self, op: str, axis: Any) -> Value:
        v = self.value
        if axis not in (None, 0, "samples"):
            raise HawkError(
                f"hawk.all_samples(x).{op}(axis={axis!r}): the view has the one "
                "axis 'samples' (or 0)")
        if v.ttype.shape:
            raise HawkError(
                f"hawk.all_samples(v).{op}(): a {v.ttype!r} value reduced over "
                "samples is not in this release; reduce its components "
                f"one at a time (hawk.all_samples(v[0]).{op}())")
        term = v / real(_n_samples()) if op == "mean" else v
        return Value(ReduceSamples("sum" if op == "mean" else op, node_of(term)))

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


def _n_samples() -> Value:
    from .value import n_samples
    return n_samples()


def all_samples(x: Any) -> AllSamples:
    """Every sample's ``x``, to be reduced: ``total = hawk.all_samples(r * r).sum()``
    stored to a ``Mutable`` output gives one sum over every sample."""
    if isinstance(x, AllSamples):
        raise HawkError("hawk.all_samples(hawk.all_samples(x)): already every sample")
    if not isinstance(x, Value):
        x = Value(node_of(x))
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
