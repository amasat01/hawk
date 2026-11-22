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
from typing import Any

from .. import _ir
from ..errors import HawkError
from .value import INDEX, Value, index_node, n_samples, sample_index, widened


def _stride_constant(stride: int, label: str) -> Any:
    """The ``RoleConst`` of the lane stride the axis ``label`` declares, made in the traced body's context."""
    return _ir.make_node("RoleConst", INDEX, context=current_context(), literal=stride, label=f"lane_{label}_stride")


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
    """The value of a ``Grid[...]`` parameter: this lane's coordinates, or those of the flat lane index ``flat`` when
    one is given."""

    __slots__ = ("_axes", "_flat", "_values")

    def __init__(self, axes: tuple, flat: Value | None = None) -> None:
        self._axes = axes
        self._flat = flat
        self._values: dict = {}

    def _lane(self) -> Value:
        return sample_index() if self._flat is None else self._flat

    @property
    def labels(self) -> tuple:
        return tuple(label for label, _ in self._axes)

    def _divisor(self, i: int) -> Value:
        return Value(_stride_constant(_strides(self._axes)[i], self._axes[i][0]))

    def _quotient(self, i: int) -> Value:
        """``lane / stride_i`` (the lane itself for the last axis)."""
        if i == len(self._axes) - 1:
            return self._lane()
        return self._lane() / self._divisor(i)

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
            return self._lane()
        if name not in self.labels:
            raise _NoAxis(
                f"lane.{name}: this Grid has the axes {self.labels} and lane.flat")
        return self._axis(self.labels.index(name))


_trace = threading.local()


@contextmanager
def trace_context(context):
    """While a body is traced, ``context`` is the context every node the tracer makes from nothing (a literal, a
    leaf, a lane constant) is made in."""
    before = getattr(_trace, "context", None)
    _trace.context = context
    try:
        yield
    finally:
        _trace.context = before


def current_context():
    """The context of the body being traced; refuses outside a trace, where a node (``node.ctx``) or a kernel
    (``kernel.context``) names it."""
    context = getattr(_trace, "context", None)
    if context is None:
        raise HawkError(
            "no kernel body is being traced here, so there is no ambient context: take the context "
            "of a node (node.ctx) or of a kernel (kernel.context)")
    return context


@contextmanager
def tracing(axes: tuple, name: str | None = None):
    """While a body is traced, ``axes`` is the Grid ``u.at(...)`` names its axes by and ``name`` the kernel.

    Yields ``{plane name: its leaf stating the Grid's axes}`` for the per-lane inputs read at another lane by axis, for
    :func:`placed_on_axes`."""
    before = (getattr(_trace, "axes", ((None, None),)), getattr(_trace, "name", None),
              getattr(_trace, "reads", None))
    _trace.axes, _trace.name, _trace.reads = axes, name, {}
    try:
        yield _trace.reads
    finally:
        _trace.axes, _trace.name, _trace.reads = before


EDGES = ("clamp", "wrap", "zero")


def _plane_leaf(value: Value) -> tuple:
    """``(leaf, narrow)``: the plane leaf behind ``value``, which must be an input plane that existed before the launch;
    ``narrow`` says it is an ``Index`` plane, stored 32-bit and widened at the read."""
    node = value.node
    narrow = (_ir.is_a(node, "op") and node.field_text("kind") == "cast"           # an Index plane, widened
              and _ir.is_a(node.operands[0], "leaf")
              and _ir.type_of(node.operands[0]).dtype == "int32" and _ir.type_of(node).dtype == "i32")
    if narrow:
        node = node.operands[0]
    if not (_ir.is_a(node, "leaf") and node.field_text("role") == "plane_in"
            and node.field_text("kind") != "prior_read"):
        raise HawkError(
            "at() reads an input plane at another lane; a value computed in this "
            "launch has no other lane's value here — only a plane that existed before "
            "the launch can be read at another lane (stage it, or pass it as an input)")
    return node, narrow


def _on_axes(leaf: Any) -> Any:
    """The leaf of the input plane ``leaf``, stating the Grid's axes (one value per lane, lanes in Grid order). One
    leaf per plane and trace, so that every read of the plane is over the same node."""
    name, reads = leaf.field_text("name"), _trace.reads
    if name not in reads:
        reads[name] = _ir.leaf(current_context(), leaf.field_text("kind"), "plane_in", name, _ir.type_of(leaf),
                               sorted=leaf.field_flag("sorted"), axes=_trace.axes)
    return reads[name]


def read_at(value: Value, positional: tuple, named: dict, edge: str = "clamp") -> Value:
    """``u.at(...)``: the per-lane input ``value`` read at another lane, by flat lane index or by every Grid axis, under
    the ``edge`` rule for a coordinate outside its axis (or a flat index outside the launch). Only a plane that existed
    before the launch can be read at another lane."""
    leaf, narrow = _plane_leaf(value)
    if edge not in EDGES:
        raise HawkError(f"at(edge={edge!r}): the edge rule is one of {EDGES}")
    name, axes = leaf.field_text("name"), _trace.axes
    if named and positional:
        raise HawkError(
            f"{name!r}: at() takes a flat lane index OR every Grid axis by name, never both")
    if not named:
        if len(positional) != 1:
            raise HawkError(
                f"{name!r}: at() takes ONE flat lane index, got {len(positional)}"
                + (f"; by axis: at({', '.join(f'{a}=…' for a, _ in axes)})"
                   if axes[0][0] is not None else ""))
        plane, indices = leaf, [index_node(positional[0])]
    else:
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
        plane, indices = _on_axes(leaf), [index_node(named[label]) for label in labels]
    read = _ir.make_node("At", _ir.type_of(leaf), [plane, *indices], edge=edge)
    return Value(widened(read) if narrow else read)


def near(value: Value, offsets: dict, edge: str = "clamp") -> Value:
    """``u.near(<axis>=<offset>, ...)``: the per-lane input ``value`` read at this lane's coordinates moved by the integer
    ``offsets`` on the named axes (an axis not named keeps this lane's coordinate), under the ``edge`` rule."""
    axes = _trace.axes
    labels = [label for label, _ in axes]
    if axes[0][0] is None:
        raise HawkError(
            "near() moves this lane's coordinates along Grid axes, but the kernel declares no Grid[...] "
            "parameter; read at a flat index: at(i)")
    unknown = sorted(set(offsets) - set(labels))
    if unknown:
        raise HawkError(f"near(): {unknown} are not axes of this Grid {labels}")
    bad = {label: k for label, k in offsets.items() if isinstance(k, bool) or not isinstance(k, int)}
    if bad:
        raise HawkError(f"near(): an offset is a Python integer, got {bad}")
    lane = LaneAxes(axes)
    moved = {label: lane[k] + offsets[label] if offsets.get(label) else lane[k] for k, label in enumerate(labels)}
    return read_at(value, (), moved, edge)


def _unfolded(flat: Value, edge: str, axes: tuple) -> list:
    """The coordinates, one node per axis, of the flat lane index ``flat`` read under ``edge``: a ``clamp`` or ``wrap``
    index is first brought into the launch; under ``zero`` an index outside it has a coordinate outside its axis."""
    from ..math import where  # a local import: hawk.math imports this module's package

    if edge == "clamp":
        last = n_samples() - 1
        flat = where(flat < 0, 0, where(flat > last, last, flat))
    elif edge == "wrap":
        flat = (flat % n_samples() + n_samples()) % n_samples()
    lane = LaneAxes(axes, flat)
    return [lane[k].node for k in range(len(axes))]


def placed_on_axes(sinks: tuple, by_axes: dict, axes: tuple) -> tuple:
    """``sinks`` with every other read of the planes ``by_axes`` (read at another lane by Grid axis; see :func:`tracing`)
    made over the plane that states the Grid's ``axes``: one slot is one plane, so all its reads state the same axes.
    Its flat reads become reads at the coordinates of the flat index, and this lane's own cell the read at this lane's
    coordinates."""
    if not by_axes:
        return sinks
    flat_reads = {
        node: _ir.make_node("At", _ir.type_of(node), [by_axes[node.operands[0].field_text("name")],
                            *_unfolded(Value(node.operands[1]), node.field_text("edge"), axes)],
                            edge=node.field_text("edge"))
        for node in _ir.post_order(sinks)
        if _ir.is_a(node, "at") and _is_plain_input(node.operands[0], by_axes)}
    sinks = _ir.swap(sinks, flat_reads)
    lane = LaneAxes(axes)
    coordinates = [lane[k].node for k in range(len(axes))]
    own = {leaf: _ir.make_node("At", _ir.type_of(leaf), [by_axes[leaf.field_text("name")], *coordinates],
                               edge="unchecked")
           for leaf in _ir.post_order(sinks) if _is_plain_input(leaf, by_axes)}
    return _ir.swap(sinks, own)


def _is_plain_input(node: Any, names: dict) -> bool:
    """Whether ``node`` is a leaf of an input plane in ``names`` that states no axes."""
    return (_ir.is_a(node, "leaf") and node.field_text("role") == "plane_in"
            and node.field_text("name") in names and not _ir.field_pairs(node, "axes"))
