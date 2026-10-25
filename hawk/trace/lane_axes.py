# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""What a body sees of the launch shape: ``Grid[...]`` and the per-lane planes
read at another lane.

``lane.<label>``, ``lane[k]`` and ``lane.flat`` are Index values built from the
nodes :func:`hawk.math.split_index` builds (``lane / k``, ``lane - major * k``),
each divisor a ``RoleConst("lane_<label>_stride")`` that names the declared
stride. Same literals, same emitted text."""

from __future__ import annotations

import threading
from contextlib import contextmanager

from ..ir import HawkError
from ..ir import make as _make
from ..ir.nodes import At, Leaf, RoleConst
from .value import Value, index_node, sample_index


def lane_space(planes: dict) -> tuple:
    """``((label|None, extent|None), ...)``: the launch shape ``planes`` declare,
    one unlabelled runtime axis for a kernel that declares none."""
    return next((p.grid for p in planes.values() if getattr(p, "grid", None)),
                ((None, None),))


def _strides(axes: tuple) -> list:
    """Each axis' stride in lanes: the product of the extents after it."""
    out, stride = [], 1
    for _label, extent in reversed(axes):
        out.append(stride)
        stride *= extent or 1
    return out[::-1]


def check_group(name: str, group: object, planes: dict) -> int:
    """``group`` (a lane count, or the label of an axis) as a lane count."""
    axes = lane_space(planes)
    if isinstance(group, str):
        labels = [label for label, _ in axes]
        if group not in labels:
            raise HawkError(
                f"kernel {name!r}: group={group!r} names no axis; "
                + (f"this kernel's Grid declares {labels}" if axes[0][0] is not None
                   else "the kernel declares no Grid[...] parameter"))
        tail = axes[labels.index(group):]
        if any(extent is None for _label, extent in tail):
            raise HawkError(
                f"kernel {name!r}: group={group!r} includes the runtime axis "
                f"{axes[0][0]!r}, so its lane count is not known when the kernel is "
                "built; name a later axis or give a number")
        count = 1
        for _label, extent in tail:
            count *= extent
        return count
    if isinstance(group, bool) or not isinstance(group, int) or group < 1:
        raise HawkError(
            f"kernel {name!r}: group= is a lane count (an int >= 1) or an axis label; "
            f"got {group!r}")
    return group


class _NoAxis(HawkError, AttributeError):
    """``lane.<name>`` for a name the Grid does not declare (an
    ``AttributeError`` too, so ``getattr(lane, name, default)`` still works)."""


class LaneAxes:
    """The value of a ``Grid[...]`` parameter."""

    __slots__ = ("_axes", "_values")

    def __init__(self, axes: tuple) -> None:
        self._axes = axes
        self._values: dict = {}

    @property
    def labels(self) -> tuple:
        return tuple(label for label, _ in self._axes)

    def _divisor(self, i: int) -> Value:
        return Value(RoleConst(_strides(self._axes)[i], f"lane_{self._axes[i][0]}_stride"))

    def _quotient(self, i: int) -> Value:
        """``lane / stride_i`` (the lane itself for the last axis)."""
        if i == len(self._axes) - 1:
            return sample_index()
        return sample_index() / self._divisor(i)

    def _axis(self, i: int) -> Value:
        if i not in self._values:
            if i == 0:
                value = self._quotient(0)
            else:
                # the last axis' extent IS the previous stride: the node split_index multiplies by
                extent = (self._divisor(i - 1) if i == len(self._axes) - 1
                          else Value(index_node(self._axes[i][1])))
                value = self._quotient(i) - self._quotient(i - 1) * extent
            self._values[i] = value
        return self._values[i]

    def __getitem__(self, k: object) -> Value:
        if isinstance(k, bool) or not isinstance(k, int):
            raise HawkError(
                f"lane[k] takes an int position (0 is the slowest axis, -1 the "
                f"fastest), got {k!r}; an axis by name is lane.<label> {self.labels}")
        n = len(self._axes)
        if not -n <= k < n:
            raise HawkError(f"lane[{k}]: this Grid has {n} axes {self.labels}")
        return self._axis(k % n)

    def __getattr__(self, name: str) -> Value:
        if name == "flat":
            return sample_index()
        if name not in self.labels:
            raise _NoAxis(
                f"lane.{name}: this Grid has the axes {self.labels} and lane.flat")
        return self._axis(self.labels.index(name))


def _flat(name: str, axes: tuple, positional: tuple, named: dict):
    """The node of the flat lane index ``at()``'s arguments name."""
    if named and positional:
        raise HawkError(
            f"{name!r}: at() takes a flat lane index OR every Grid axis by name, never both")
    if not named:
        if len(positional) != 1:
            raise HawkError(
                f"{name!r}: at() takes ONE flat lane index, got {len(positional)}"
                + (f"; by axis: at({', '.join(f'{a}=…' for a, _ in axes)})"
                   if axes[0][0] is not None else ""))
        return index_node(positional[0])
    labels = [label for label, _ in axes]
    if axes[0][0] is None:
        raise HawkError(
            f"{name!r}: at({', '.join(named)}=…) names Grid axes, but the kernel declares "
            "no Grid[...] parameter; read at a flat index: at(i)")
    if sorted(named) != sorted(labels):
        raise HawkError(
            f"{name!r}: at() must name every Grid axis exactly once (declared {labels})"
            + (f"; missing {[a for a in labels if a not in named]}"
               if any(a not in named for a in labels) else "")
            + (f"; unknown {sorted(set(named) - set(labels))}"
               if set(named) - set(labels) else ""))
    flat = None
    for (label, _extent), stride in zip(axes, _strides(axes)):
        term = _make("mul", (index_node(named[label]), RoleConst(stride, f"lane_{label}_stride")))
        flat = term if flat is None else _make("add", (flat, term))
    return flat


_trace = threading.local()


@contextmanager
def tracing(axes: tuple):
    """While a body is traced, ``axes`` is the Grid ``u.at(...)`` names its axes by."""
    before = getattr(_trace, "axes", ((None, None),))
    _trace.axes = axes
    try:
        yield
    finally:
        _trace.axes = before


def read_at(value: Value, positional: tuple, named: dict) -> Value:
    """``u.at(...)``: the per-lane input ``value`` read at another lane, by flat
    lane index or by every Grid axis. Only a plane that existed before the launch
    can be read at another lane."""
    node = value.node
    if not (isinstance(node, Leaf) and node.role in ("per_sample", "vec_in", "mat_in")
            and node.kind != "prior_read"):
        raise HawkError(
            "at() reads an input plane at another lane; a value computed in this "
            "launch has no other lane's value here — only a plane that existed before "
            "the launch can be read at another lane (stage it, or pass it as an input)")
    return Value(At(node, _flat(node.name, _trace.axes, positional, named), node.ttype))
