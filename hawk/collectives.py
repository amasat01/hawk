# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Whole-plane operations called like a kernel: ``hawk.reduce``, ``hawk.scan``, ``hawk.compact``, ``hawk.sort``,
``hawk.reorder``, ``hawk.copy``, ``hawk.matmul`` and ``hawk.fft``.

One call path serves them all. Called inside a graph builder or block, the call records one node of the graph; called
anywhere else it runs once. Either way the result arrays are made at the call, in the memory of the first operand, and
``out=`` passes the caller's own instead. hawk picks no realisation: the runtime door makes the plan, checks the typing and
runs the node, and its reason for a refusal reaches the author unparsed.
"""

from __future__ import annotations

import operator
import weakref
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable

import _osprey

from . import _place, graph_builder, streams
from .errors import HawkError
from .runtime import through_door

__all__ = ["compact", "copy", "fft", "matmul", "reduce", "reorder", "scan", "sort"]

_OPS = MappingProxyType(
    {
        "sum": _osprey.EG_REDUCE_SUM,
        "min": _osprey.EG_REDUCE_MIN,
        "max": _osprey.EG_REDUCE_MAX,
        "all": _osprey.EG_REDUCE_ALL,
        "any": _osprey.EG_REDUCE_ANY,
    }
)
_SCAN_OPS = MappingProxyType(
    {
        "sum": _osprey.EG_SCAN_SUM,
        "min": _osprey.EG_SCAN_MIN,
        "max": _osprey.EG_SCAN_MAX,
    }
)
_CONTRACTS = MappingProxyType(
    {
        "compensated": _osprey.EG_REDUCE_COMPENSATED,
        "fixed": _osprey.EG_REDUCE_FIXED,
    }
)
_NORMS = MappingProxyType(
    {
        "backward": _osprey.EG_FFT_NORM_BACKWARD,
        "forward": _osprey.EG_FFT_NORM_FORWARD,
        "ortho": _osprey.EG_FFT_NORM_ORTHO,
    }
)
#: The DLPack type codes the runtime runs, as the words the sidecar spells a dtype with.
_KIND_WORDS = MappingProxyType({0: "int", 1: "uint", 2: "float"})


class Collective:
    """One whole-plane node: what it is called, the door of the memory it runs in, the arrays it touches and the door call that
    runs it over their planes.

    ``arrays`` are the operands in the order ``run`` takes their planes. ``plan`` is the door's plan handle for a transform or
    a matrix product (``None`` for the others), freed with the node. ``kind`` is the operation (``"reduce"``, ``"scan"``,
    ``"reorder"``, ``"copy"``, ``"matmul"``, ``"fft"``, ``"compact"``, ``"sort"``) and ``op`` its fold where it has one;
    ``redo(arrays)`` calls the same operation over other ``arrays`` of the same roles (the derivative of a program calls it
    over tangent planes), and is ``None`` where that has no meaning.
    """

    def __init__(
        self,
        label: str,
        door: Any,
        arrays: tuple,
        run: Callable,
        plan: Any = None,
        free: Callable | None = None,
        *,
        kind: str = "",
        op: str | None = None,
        redo: Callable | None = None,
    ) -> None:
        """Make the node ``label`` over ``arrays`` held by ``door``; ``run(door, planes)`` calls the door, ``free(door, plan)`` drops its plan."""
        self.label, self.arrays, self.run, self.plan = label, arrays, run, plan
        self.kind, self.op, self.redo = kind, op, redo
        self.door = door
        if plan is not None:
            weakref.finalize(self, free, self.door, plan)

    def record(self, door: Any, store: Any) -> None:
        """Record the node into the open recording of ``door`` over the planes the graph's ``store`` holds for its arrays."""
        through_door(self.run, door, [store.plane(array) for array in self.arrays])


def _door_of(array: Any, label: str) -> Any:
    """The door of the memory ``array`` lives in."""
    device = _place.device_of(array)
    if device is None:
        raise HawkError(
            f"hawk.{label}: {type(array).__name__} is not an array hawk can read; pass a NumPy array or one that exports DLPack"
        )
    return _place.door_of(device)


def _call(node: Collective) -> None:
    """Record ``node`` in the graph being written, or run it once when none is."""
    recorder = graph_builder.current()
    if recorder is not None:
        graph = recorder._graph()
        graph._store_over(node.door)
        graph._append(node)
        return
    door = node.door
    planes = [through_door(_place.wrap, door, array) for array in node.arrays]
    try:
        streams.apply(door)
        through_door(node.run, door, planes)
        through_door(door.sync)
    finally:
        for plane in planes:
            door.plane_free(plane)


def _word(dtype: tuple) -> str:
    """The sidecar word of the DLPack ``(code, bits, lanes)`` dtype."""
    code, bits = dtype[0], dtype[1]
    return "bool" if code == 6 else f"{_KIND_WORDS[code]}{bits}"


def _made(like: Any, shape: tuple, dtype: tuple) -> Any:
    """A zeroed array of ``shape`` and the DLPack ``dtype`` in the memory of ``like``."""
    return _place.Maker.at(_place.device_of(like), like=like).zeros(shape, _word(dtype))


def _layout(door: Any, array: Any) -> dict:
    """The door's reading of ``array``: ``shape``, ``dtype`` and ``device``."""
    return door.array_layout(array)


def _result(out: Any, like: Any, shape: tuple, dtype: tuple) -> Any:
    """``out`` as the caller gave it, else the array the door makes beside ``like``."""
    return out if out is not None else _made(like, shape, dtype)


def reduce(
    p: Any,
    op: str,
    contract: str = "compensated",
    out: Any = None,
    with_index: bool = False,
) -> Any:
    """Reduce the plane ``p`` over its samples (its last axis) with ``op``: ``"sum"``, ``"min"``, ``"max"``, ``"all"`` or ``"any"``.

    ``p`` of shape ``(columns, n)`` gives one value per column, a plane of shape ``(columns,)``; a plane of shape ``(n,)`` has
    one column and gives a plane of shape ``(1,)``. The result has the dtype of ``p``; ``"all"`` and ``"any"`` take a bool
    plane and give a bool plane, or a uint32 plane when ``out=`` is one (a cell that a branch of a graph can test).
    ``contract`` is how a sum is added: ``"compensated"`` (the default) or ``"fixed"``. Returns the result plane, which
    ``out=`` replaces by the caller's.

    ``with_index=True`` (``"min"`` and ``"max"`` only) returns ``(value, where)``: ``where`` is an int32 plane with the lane of
    each column's winner. A tie goes to the lowest lane, a NaN never wins, and a column of nothing but NaN has the value NaN and
    the lane ``n`` (no lane). Typing is the runtime's, and its reason for a refusal is shown as given."""
    if op not in _OPS:
        raise HawkError(f"hawk.reduce: op={op!r} is not one of {tuple(_OPS)}")
    if contract not in _CONTRACTS:
        raise HawkError(
            f"hawk.reduce: contract={contract!r} is not one of {tuple(_CONTRACTS)}"
        )
    door = _door_of(p, "reduce")
    layout = _layout(door, p)
    columns = (layout["shape"][0] if len(layout["shape"]) > 1 else 1,)
    result = _result(out, p, columns, layout["dtype"])
    where = _made(p, columns, (0, 32, 1)) if with_index else None

    def run(door: Any, planes: list) -> None:
        index = planes[2] if with_index else None
        door.reduce(planes[0], planes[1], _OPS[op], _CONTRACTS[contract], index=index)

    arrays = (p, result, where) if with_index else (p, result)
    _call(
        Collective(
            f"reduce {op}",
            door,
            arrays,
            run,
            kind="reduce",
            op=op,
            redo=lambda a: reduce(a[0], op, contract, a[1]),
        )
    )
    return (result, where) if with_index else result


def scan(p: Any, op: str, exclusive: bool = False, out: Any = None) -> Any:
    """The running fold of the one-axis plane ``p`` with ``op`` (``"sum"``, ``"min"`` or ``"max"``): ``out[i]`` folds
    ``p[0..i]``, or with ``exclusive=True`` ``p[0..i-1]`` with the identity of ``op`` (0, the largest value, the smallest) first.

    The result has the shape and dtype of ``p``; ``out=`` passes the caller's plane instead. An integer sum is exact (it wraps),
    a minimum and a maximum are exact, and a real sum has the same bits on every run but is not the bits of a left-to-right loop.
    Typing (one axis, a scalar value, the six numeric types) is the runtime's, and its reason for a refusal is shown as given."""
    if op not in _SCAN_OPS:
        raise HawkError(f"hawk.scan: op={op!r} is not one of {tuple(_SCAN_OPS)}")
    door = _door_of(p, "scan")
    layout = _layout(door, p)
    result = _result(out, p, tuple(layout["shape"]), layout["dtype"])
    flags = _osprey.EG_SCAN_EXCLUSIVE if exclusive else 0

    def run(door: Any, planes: list) -> None:
        door.scan(planes[0], planes[1], _SCAN_OPS[op], flags)

    _call(
        Collective(
            f"scan {op}",
            door,
            (p, result),
            run,
            kind="scan",
            op=op,
            redo=lambda a: scan(a[0], op, exclusive, a[1]),
        )
    )
    return result


def compact(keep: Any, out: tuple | None = None) -> tuple:
    """The lanes of the bool plane ``keep`` that are set: returns ``(kept, count)``, where ``kept[:count]`` holds their indices in
    lane order and ``count`` is a one-element uint32 plane with their number.

    ``kept`` has the shape of ``keep`` (int32) and its entries from ``count`` on are not written; ``count`` can be the condition
    of an ``if`` in a graph builder, or a loop limit. ``out=(kept, count)`` passes the caller's planes instead. The same result on
    every run. Typing (one axis, a bool ``keep``) is the runtime's, and its reason for a refusal is shown as given."""
    door = _door_of(keep, "compact")
    layout = _layout(door, keep)
    if out is None:
        kept = _made(keep, tuple(layout["shape"]), (0, 32, 1))
        count = _made(keep, (1,), (1, 32, 1))
    else:
        kept, count = out

    def run(door: Any, planes: list) -> None:
        door.compact(*planes)

    _call(Collective("compact", door, (keep, kept, count), run, kind="compact"))
    return kept, count


def sort(keys: Any, bound: int, out: tuple | None = None) -> tuple:
    """The stable sort of the lanes of the integer plane ``keys`` (one axis, every key in ``[0, bound)``): returns
    ``(order, starts)``, where ``order[j]`` is the lane that lands at sorted place ``j`` and ``starts[k]`` is the first sorted
    place of key ``k``, with ``starts[bound]`` the lane count.

    Lanes of equal key keep their lane order. ``order`` is an int32 plane the shape of ``keys`` and ``starts`` an int32 plane of
    ``bound + 1`` entries; ``out=(order, starts)`` passes the caller's planes instead. The same result on every run. A key outside
    ``[0, bound)`` fails the call, or the replay in a graph builder, with the number of such keys. Typing (an integer plane, one
    axis, a bound of 1 or more) is the runtime's, and its reason for a refusal is shown as given."""
    try:
        bound = operator.index(bound)
    except TypeError:
        raise HawkError(f"hawk.sort: bound={bound!r} is not an integer") from None
    door = _door_of(keys, "sort")
    layout = _layout(door, keys)
    if out is None:
        order = _made(keys, tuple(layout["shape"]), (0, 32, 1))
        starts = _made(keys, (max(bound, 0) + 1,), (0, 32, 1))
    else:
        order, starts = out

    def run(door: Any, planes: list) -> None:
        door.order_by_key(planes[0], planes[1], planes[2], bound)

    _call(Collective("sort", door, (keys, order, starts), run, kind="sort"))
    return order, starts


def reorder(planes: Any, order: Any) -> None:
    """Permute the planes in place by ``order``: afterwards ``p[j]`` holds what ``p[order[j]]`` held, for every plane ``p``.

    ``planes`` is a list of arrays of one lane extent (their last axis; a plane of a vector value moves every row by the same
    order) and ``order`` an int32 or uint32 plane of that extent, such as the one ``hawk.sort`` returns, so
    ``order, starts = hawk.sort(keys, bound=K)`` then ``hawk.reorder([pos, vel], order)`` sorts both by key. Every entry of
    ``order`` must lie in ``[0, n)``: the call, or in a graph builder the replay, fails naming how many do not and moves
    nothing. A repeated entry is not detected and leaves a lane twice. The same bytes on every run and at any thread count.
    Typing is the runtime's, and its reason for a refusal is shown as given."""
    members = tuple(planes)
    if not members:
        raise HawkError(
            "hawk.reorder: planes is empty; pass the list of arrays to move"
        )

    def run(door: Any, handles: list) -> None:
        door.reorder(handles[:-1], handles[-1])

    _call(
        Collective(
            "reorder",
            _door_of(members[0], "reorder"),
            (*members, order),
            run,
            kind="reorder",
            redo=lambda a: reorder(a[:-1], a[-1]),
        )
    )


def copy(src: Any, dst: Any) -> Any:
    """Copy the plane ``src`` into the plane ``dst`` (the same shape and dtype, in any memory); returns ``dst``."""

    def run(door: Any, planes: list) -> None:
        door.copy(planes[0], planes[1])

    _call(
        Collective(
            "copy",
            _door_of(src, "copy"),
            (src, dst),
            run,
            kind="copy",
            redo=lambda a: copy(a[0], a[1]),
        )
    )
    return dst


def matmul(
    a: Any,
    b: Any,
    out: Any = None,
    transpose_a: bool = False,
    transpose_b: bool = False,
    accumulate: bool = False,
) -> Any:
    """The matrix product ``op(a) @ op(b)`` of two row-major planes, where ``op`` is the transpose under ``transpose_a`` /
    ``transpose_b`` (the planes are stored as given). Returns the product plane, which ``out=`` replaces by the caller's;
    with ``accumulate=True`` it is added to ``out`` instead of replacing it. The planes' shapes and dtype are checked by the
    runtime, and its reason for a refusal is shown as given."""
    door = _door_of(a, "matmul")
    la, lb = _layout(door, a), _layout(_door_of(b, "matmul"), b)
    (a_rows, a_cols), (b_rows, b_cols) = la["shape"], lb["shape"]
    m, n = (a_cols if transpose_a else a_rows), (b_rows if transpose_b else b_cols)
    result = _result(out, a, (m, n), la["dtype"])
    flags = (_osprey.EG_GEMM_TRANS_A if transpose_a else 0) | (
        _osprey.EG_GEMM_TRANS_B if transpose_b else 0
    )
    flags |= _osprey.EG_GEMM_ACCUMULATE if accumulate else 0
    plan = through_door(
        door.gemm_new,
        la["shape"],
        lb["shape"],
        tuple(result.shape),
        la["dtype"][1],
        flags,
    )

    def run(door: Any, planes: list) -> None:
        door.gemm_run(plan, *planes)

    _call(
        Collective(
            "matmul",
            door,
            (a, b, result),
            run,
            plan,
            lambda d, handle: d.gemm_free(handle),
            kind="matmul",
            redo=lambda a, accumulate=accumulate: matmul(
                a[0], a[1], a[2], transpose_a, transpose_b, accumulate
            ),
        )
    )
    return result


def _transform(
    kind: int, label: str, p: Any, out: Any, axes: Any, norm: str, inverse: bool
) -> Any:
    """The Fourier transform of ``kind`` of the plane ``p`` over ``axes`` (all the data axes when ``None``), under ``norm``.

    A complex plane is a float plane whose last axis has extent 2 (real part, imaginary part)."""
    if norm not in _NORMS:
        raise HawkError(
            f"hawk.fft.{label}: norm={norm!r} is not one of {tuple(_NORMS)}"
        )
    door = _door_of(p, f"fft.{label}")
    layout = _layout(door, p)
    shape, dtype = tuple(layout["shape"]), layout["dtype"]
    complex_in = kind != _osprey.EG_FFT_R2C
    data = (
        shape[:-1] if complex_in else shape
    )  # the transform's input, without its trailing 2
    ndim = len(data)
    chosen = (
        list(range(ndim))
        if axes is None
        else [axis % ndim for axis in ((axes,) if isinstance(axes, int) else axes)]
    )
    halved = chosen[-1]
    if kind == _osprey.EG_FFT_C2C:
        real_side, made = data, shape
    elif kind == _osprey.EG_FFT_R2C:
        real_side = data
        made = (*data[:halved], data[halved] // 2 + 1, *data[halved + 1 :], 2)
    elif out is not None:
        real_side, made = tuple(out.shape), tuple(out.shape)
    else:
        # the real length of the halved axis is not in the complex plane: an even one is assumed, as a plain transform gives
        real_side = (*data[:halved], 2 * (data[halved] - 1), *data[halved + 1 :])
        made = real_side
    result = _result(out, p, made, dtype)
    plan = through_door(
        door.fft_new,
        kind,
        _NORMS[norm],
        dtype[1],
        real_side,
        chosen,
        _osprey.EG_FFT_INVERSE if inverse else 0,
    )

    def run(door: Any, planes: list) -> None:
        door.fft_run(plan, *planes)

    _call(
        Collective(
            f"fft {label}",
            door,
            (p, result),
            run,
            plan,
            lambda d, handle: d.fft_free(handle),
            kind="fft",
            redo=lambda a: _transform(kind, label, a[0], a[1], axes, norm, inverse),
        )
    )
    return result


def _c2c(
    p: Any,
    out: Any = None,
    axes: Any = None,
    norm: str = "backward",
    inverse: bool = False,
) -> Any:
    """The complex-to-complex transform of ``p`` over ``axes`` (every data axis by default); ``inverse=True`` runs the inverse
    direction. ``norm`` is ``"backward"`` (no scale forward, ``1/n`` inverse), ``"forward"`` or ``"ortho"``. Returns the
    result plane, which ``out=`` replaces by the caller's."""
    return _transform(_osprey.EG_FFT_C2C, "c2c", p, out, axes, norm, inverse)


def _r2c(
    p: Any,
    out: Any = None,
    axes: Any = None,
    norm: str = "backward",
    inverse: bool = False,
) -> Any:
    """The real-to-complex transform of the real plane ``p`` over ``axes`` (every axis by default): the last of the axes keeps
    ``n // 2 + 1`` bins. ``norm`` is as for ``c2c``; ``inverse=True`` is refused by the runtime. Returns the complex result
    plane, which ``out=`` replaces by the caller's."""
    return _transform(_osprey.EG_FFT_R2C, "r2c", p, out, axes, norm, inverse)


def _c2r(
    p: Any,
    out: Any = None,
    axes: Any = None,
    norm: str = "backward",
    inverse: bool = False,
) -> Any:
    """The complex-to-real transform of the complex plane ``p`` over ``axes`` (every data axis by default). The real length of
    the last of the axes is not in ``p``: it is ``2 * (bins - 1)`` unless ``out=`` states it by its shape. ``norm`` is as for
    ``c2c``; ``inverse=True`` is refused by the runtime. Returns the real result plane, which ``out=`` replaces."""
    return _transform(_osprey.EG_FFT_C2R, "c2r", p, out, axes, norm, inverse)


#: ``hawk.fft.c2c``, ``hawk.fft.r2c`` and ``hawk.fft.c2r``.
fft = SimpleNamespace(c2c=_c2c, r2c=_r2c, c2r=_c2r)
