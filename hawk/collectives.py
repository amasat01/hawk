# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Whole-plane operations called like a kernel: ``hawk.reduce``, ``hawk.copy``, ``hawk.matmul`` and ``hawk.fft``.

One call path serves them all. Called inside a graph builder or block, the call records one node of the graph; called
anywhere else it runs once. Either way the result arrays are made at the call, in the memory of the first operand, and
``out=`` passes the caller's own instead. hawk picks no realisation: the runtime door makes the plan, checks the typing and
runs the node, and its reason for a refusal reaches the author unparsed.
"""

from __future__ import annotations

import weakref
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable

import _osprey

from . import _place, graph_builder, streams
from .errors import HawkError
from .runtime import through_door

__all__ = ["copy", "fft", "matmul", "reduce"]

_OPS = MappingProxyType(
    {
        "sum": _osprey.EG_REDUCE_SUM,
        "min": _osprey.EG_REDUCE_MIN,
        "max": _osprey.EG_REDUCE_MAX,
        "all": _osprey.EG_REDUCE_ALL,
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
    """One whole-plane node: what it is called, the arrays it touches and the door call that runs it over their planes.

    ``arrays`` are the operands in the order ``run`` takes their planes. ``plan`` is the door's plan handle for a transform or
    a matrix product (``None`` for the others), freed with the node.
    """

    def __init__(
        self,
        label: str,
        arrays: tuple,
        run: Callable,
        plan: Any = None,
        free: Callable | None = None,
    ) -> None:
        """Make the node ``label`` over ``arrays``; ``run(door, planes)`` calls the door, ``free(door, plan)`` drops its plan."""
        self.label, self.arrays, self.run, self.plan = label, arrays, run, plan
        self.door = _door_of(arrays[0], label)
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


def _made(like: Any, shape: tuple, dtype: tuple, label: str) -> Any:
    """A zeroed array of ``shape`` and the DLPack ``dtype`` in the memory of ``like``."""
    door = _door_of(like, label)
    return _place.Maker.at(_place.device_of(like), like=like).zeros(shape, _word(dtype))


def _layout(array: Any, label: str) -> dict:
    """The door's reading of ``array``: ``shape``, ``dtype`` and ``device``."""
    return _door_of(array, label).array_layout(array)


def _result(out: Any, like: Any, shape: tuple, dtype: tuple, label: str) -> Any:
    """``out`` as the caller gave it, else the array the door makes beside ``like``."""
    return out if out is not None else _made(like, shape, dtype, label)


def reduce(p: Any, op: str, contract: str = "compensated", out: Any = None) -> Any:
    """Reduce the plane ``p`` over its samples (its last axis) with ``op``: ``"sum"``, ``"min"``, ``"max"`` or ``"all"``.

    ``p`` of shape ``(columns, n)`` gives one value per column, a plane of shape ``(columns,)``; a plane of shape ``(n,)`` has
    one column and gives a plane of shape ``(1,)``. The result has the dtype of ``p``; ``"all"`` takes a bool plane.
    ``contract`` is how a sum is added: ``"compensated"`` (the default) or ``"fixed"``. Returns the result plane, which
    ``out=`` replaces by the caller's. Typing is the runtime's, and its reason for a refusal is shown as given."""
    if op not in _OPS:
        raise HawkError(f"hawk.reduce: op={op!r} is not one of {tuple(_OPS)}")
    if contract not in _CONTRACTS:
        raise HawkError(
            f"hawk.reduce: contract={contract!r} is not one of {tuple(_CONTRACTS)}"
        )
    layout = _layout(p, "reduce")
    result = _result(
        out,
        p,
        (layout["shape"][0] if len(layout["shape"]) > 1 else 1,),
        layout["dtype"],
        "reduce",
    )

    def run(door: Any, planes: list) -> None:
        door.reduce(planes[0], planes[1], _OPS[op], _CONTRACTS[contract])

    _call(Collective(f"reduce {op}", (p, result), run))
    return result


def copy(src: Any, dst: Any) -> Any:
    """Copy the plane ``src`` into the plane ``dst`` (the same shape and dtype, in any memory); returns ``dst``."""

    def run(door: Any, planes: list) -> None:
        door.copy(planes[0], planes[1])

    _call(Collective("copy", (src, dst), run))
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
    la, lb = _layout(a, "matmul"), _layout(b, "matmul")
    (a_rows, a_cols), (b_rows, b_cols) = la["shape"], lb["shape"]
    m, n = (a_cols if transpose_a else a_rows), (b_rows if transpose_b else b_cols)
    result = _result(out, a, (m, n), la["dtype"], "matmul")
    flags = (_osprey.EG_GEMM_TRANS_A if transpose_a else 0) | (
        _osprey.EG_GEMM_TRANS_B if transpose_b else 0
    )
    flags |= _osprey.EG_GEMM_ACCUMULATE if accumulate else 0
    door = _door_of(a, "matmul")
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
            "matmul", (a, b, result), run, plan, lambda d, handle: d.gemm_free(handle)
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
    layout = _layout(p, f"fft.{label}")
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
    result = _result(out, p, made, dtype, f"fft.{label}")
    door = _door_of(p, f"fft.{label}")
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
            f"fft {label}", (p, result), run, plan, lambda d, handle: d.fft_free(handle)
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
