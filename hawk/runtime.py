# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The thin Python face over :mod:`hawk._core` — descriptor, bind, launch.

This module splits the host path by LIFETIME: everything a launch pays
for is C++ (the ``void*[]``, the mirror marshal, the entry cache, the
serial driver), and what stays in Python happens once per kernel —
turning a slot's declared role into the by-value ABI shape it rides as,
and handing the descriptor to :class:`hawk._core.ArgBlock`. It is not a
runtime; it is the last mile of the DEFINITION.

The descriptor is derived, never re-tabulated: a slot's kind comes from
the same role -> mirror map the emitter generated the entry signature
from (:data:`hawk.emit.aether.MIRROR_OF`), so a slot cannot be described
one way to the compiler and another to the marshal — the failure mode
being a 32-byte handle where the kernel reads a 40-byte mirror, which
is deterministic garbage, not a crash.

No numpy, and no allocation: planes cross through the buffer protocol,
so a numpy array, a torch CPU tensor and a plain :class:`array.array`
all bind the same way. Output planes are the caller's.

The crossing budget is visible in the method split:
:meth:`HostKernel.bind_all` is per bind, :meth:`HostKernel.rebind` is
bulk (one crossing for the whole changed set), and
:meth:`HostKernel.launch` is the single per-launch crossing.

Every bound argument is validated against the declaration in
:meth:`HostKernel.bind_all`, before any address is taken
(:func:`_check_bind`), since a wrong dtype or size used to run silently
and return garbage or segfault. Dtype is exact and fixed by role (every
integer-valued role rides a 64-bit wire regardless of how it was
declared in Python, and ``Terminated`` is always ``bool``); shape
requires a per-sample plane at least ``n_samples`` long (shorter is
always the bug) and a ``Reduce(op)`` output one slot per sample; every
plane must be C-contiguous and a sink role's plane writable, except a
sample-major plane whose transpose is C-contiguous, which binds
zero-copy (:func:`plane_layout`). :meth:`HostKernel.rebind` takes raw
pointers, which carry none of this, and is the unchecked fast path over
an already-validated bind.
"""

from __future__ import annotations

import array
import ctypes
from pathlib import Path

from . import _core
from ._bind_checks import _PER_SAMPLE_ROLES, _check_bind, _n_from
from ._plane_layout import _native_extent, plane_layout
from .emit.aether import MIRROR_OF, mirror_of
from .ir import HawkError
from .types import TensorType

__all__ = ["ArgBlock", "HostKernel", "descriptor_for_sidecar", "DEVICE_CPU",
           "kind_for", "load", "run"]

#: The DLPack CPU device code every host plane's mirror carries. Spelled
#: here rather than imported, to sever the runtime.
DEVICE_CPU = 1

#: :class:`hawk._core.ArgBlock`, re-exported so a consumer never has to
#: reach into the extension directly.
ArgBlock = _core.ArgBlock

#: The by-value ABI shape -> ``_core`` descriptor-kind vocabulary
#: (:data:`hawk.emit.aether.MIRROR_OF` on the left).
_KIND_OF_MIRROR = {"GRefMirror": "gref", "ScalarHandle": "handle"}

_INT_DTYPES = ("i32", "i64")


def kind_for(role: str, ttype: TensorType, *, scalar_type: str = "float64") -> str:
    """The ``_core`` descriptor kind ONE slot binds as.

    The mirror-shaped roles by their POD, the by-value ones by the
    element spelling the entry declares them with — the index width for
    ``nsamples``, the emitter's ``Int`` for an integer uniform, and the
    kernel's compiled ``Real`` otherwise."""
    shape = mirror_of(role, ttype)
    if shape == "value":
        if role == "nsamples":
            return "nsamples"
        if ttype.dtype in _INT_DTYPES:
            return "int"
        return "f32" if scalar_type == "float32" else "f64"
    kind = _KIND_OF_MIRROR[shape]
    if kind == "handle" and ttype.dtype in _INT_DTYPES:
        return "int_handle"
    return kind


def descriptor_for_walk(walk, *, scalar_type: str = "float64") -> list:
    """The per-slot kind list for a traced kernel, in ``arg_spec`` order."""
    return [kind_for(role, walk.slot_types[(role, name)], scalar_type=scalar_type)
            for role, name in walk.arg_spec]


def descriptor_for_sidecar(sidecar: dict) -> list:
    """The per-slot kind list for a DEPLOYED artifact, from its sidecar
    alone.

    A consumer holds a directory, not a :class:`~hawk.ir.walk.Walk`, so
    types are reconstructed from the sidecar's own blocks: ``mutables``
    (resolves a ``mutable`` role's shape — without it a
    ``Mutable[Vector[3]]`` packs as a 32-byte handle where the kernel
    expects a 40-byte mirror), ``params`` and ``scalar_type``. Widths
    never enter: shape is decided by role and rank alone."""
    scalar_type = sidecar.get("scalar_type", "float64")
    mutables = {m["name"]: m for m in sidecar.get("mutables", ())}
    params = {p["name"]: p for p in sidecar.get("params", ())}
    out = []
    for role, name in (tuple(pair) for pair in sidecar["arg_spec"]):
        if role in ("mutable", "out"):
            decl = mutables.get(name, {})
            dtype = decl.get("dtype", "scalar")
            ttype = _MUTABLE_TTYPE[dtype]
        elif role == "uniform":
            ttype = TensorType((), "i64" if params.get(name, {}).get("dtype") == "int"
                               else "f64")
        elif role in ("vec_in", "mat_in"):
            ttype = _SHAPED[role]
        else:
            ttype = TensorType((), "f64")
        out.append(kind_for(role, ttype, scalar_type=scalar_type))
    return out


#: The TensorType shape each declared ``mutables`` dtype stands for.
#: Only the RANK matters to :func:`kind_for`.
_MUTABLE_TTYPE = {
    "scalar": TensorType((), "f64"),
    "int": TensorType((), "i64"),
    "vector": TensorType((3,), "f64"),
    "matrix": TensorType((3, 3), "f64"),
}
_SHAPED = {"vec_in": TensorType((3,), "f64"), "mat_in": TensorType((3, 3), "f64")}

assert set(_KIND_OF_MIRROR) | {"value"} == set(MIRROR_OF.values()), (
    "hawk.runtime's kind vocabulary must cover every by-value shape the map "
    f"resolves to: {sorted(set(MIRROR_OF.values()))}"
)


class HostKernel:
    """One HAWK host artifact's entry, self-checked at load and bound once.

    Construction is per artifact + per kernel; everything after is per
    bind or per launch. The library is held so the entry address can
    never outlive it."""

    def __init__(self, so_path, sidecar: dict) -> None:
        self.path = Path(so_path)
        self.sidecar = sidecar
        self.arg_spec = tuple(tuple(pair) for pair in sidecar["arg_spec"])
        self.scalar_type = sidecar.get("scalar_type", "float64")
        self.descriptor = descriptor_for_sidecar(sidecar)
        entry_name = sidecar.get("host_entry")
        if not entry_name:
            raise HawkError(
                f"{self.path}: its sidecar declares no 'host_entry', so it carries "
                "no host target to run"
            )
        #: The self-check happens here, in C++: a wrong-layout artifact
        #: refuses, naming the field, rather than computing garbage.
        self.library = _core.HostLibrary(str(self.path))
        self.entry = self.library.entry(entry_name)
        self.block = _core.ArgBlock(self.descriptor)
        self.slot_of = {pair: i for i, pair in enumerate(self.arg_spec)}
        self._values = []          # keeps every by-value ctypes box alive
        self.one_step = None       # the word of 1 an unbound fused_steps reads
        self.finished = None       # the counter an unbound finished_count fills
        self._buffers = []         # keeps every exported buffer view alive

    # -- per bind ----------------------------------------------------------
    def _one_step_word(self, arrays: dict) -> dict:
        """``arrays``, plus the reserved words a runner normally provides,
        for each one this kernel uses and the caller leaves out: a
        one-cell ``fused_steps`` word of ``1`` (a launch outside a runner
        takes exactly one step) and a one-cell ``finished_count`` counter
        reset to ``0`` (a kernel that finishes its own samples counts them
        there; bind your own buffer to read the count)."""
        from .ir.nodes import FINISHED_PLANE, FUSED_STEPS_PLANE

        extra = {}
        for name in (FUSED_STEPS_PLANE, FINISHED_PLANE):
            if name in arrays or not any(n == name for _, n in self.slot_of):
                continue
            if name == FUSED_STEPS_PLANE:
                if self.one_step is None:
                    self.one_step = array.array("q", [1])
                extra[name] = self.one_step
            else:
                if self.finished is None:
                    self.finished = array.array("q", [0])
                self.finished[0] = 0
                extra[name] = self.finished
        return {**arrays, **extra} if extra else arrays

    def bind_all(self, arrays: dict, n_samples: int) -> None:
        """Bind every slot once (per bind). ``arrays`` maps a slot name to
        a buffer-protocol object; a ``uniform`` maps to a Python number
        and an ``nsamples`` role is bound from ``n_samples`` itself.

        VALIDATED FIRST, before any slot's address is taken
        (:func:`_check_bind`): dtype, shape, C-contiguity and sink
        writability, all read off the sidecar rather than hard-coded —
        a wrong dtype or size used to silently return garbage or segfault.
        Refuses naming every failing argument, not just the first. An
        automatic kernel's ``fused_steps`` word may be left out (it is
        bound to ``1``, one step per launch), and so may a finishing
        kernel's ``finished_count`` counter (bound to a fresh ``0``)."""
        arrays = self._one_step_word(arrays)
        _check_bind(self.sidecar, self.arg_spec, arrays, n_samples)
        self._values, self._buffers = [], []
        widths = self.sidecar.get("arg_shapes", {})
        for slot, (role, name) in enumerate(self.arg_spec):
            if role == "nsamples":
                self.block.bind(slot, self._value_address(slot, n_samples),
                                n_samples, 1, DEVICE_CPU, 0)
            elif role == "uniform":
                self.block.bind(slot, self._value_address(slot, arrays[name]),
                                n_samples, 1, DEVICE_CPU, 0)
            else:
                # A GRef plane's component pitch is its trailing extent L,
                # which may exceed n_samples; `ArgBlock.bind` derives the
                # pitch from `samples`, so passing n_samples here would
                # read every component after the first at the wrong offset.
                samples = n_samples
                if self.descriptor[slot] == "gref" and name in arrays:
                    samples = _native_extent(memoryview(arrays[name]),
                                             int(widths.get(name) or 1))
                self.block.bind(slot, self._buffer_address(arrays, name, role),
                                samples, 1, DEVICE_CPU, 0)

    def rebind(self, slots, ptrs) -> None:
        """The BULK pointer swap: the whole changed set in ONE crossing.

        ``slots``/``ptrs`` are int64 sequences, converted to buffers
        here once, since a Python call per slot is exactly the O(slots)
        cost this exists to avoid.

        UNCHECKED, by construction: a raw pointer carries no dtype,
        shape or writability to check — :func:`_check_bind` validated
        the original array at the :meth:`bind_all` call that produced
        this pointer, and this call only swaps an already-approved
        address. It is not where a new binding is introduced."""
        return self.block.rebind(array.array("q", (int(s) for s in slots)),
                                 array.array("q", (int(p) for p in ptrs)))

    # -- per launch --------------------------------------------------------
    def launch(self, base: int, count: int, n_samples: int) -> None:
        """THE launch crossing: one call over ``[base, base+count)``.

        SERIAL, always: the reference oracle every partitioned, tiled or
        ranked run is judged against, so it never grows a schedule."""
        self.entry.run(self.block, int(base), int(count), int(n_samples))

    # -- helpers -----------------------------------------------------------
    def _value_address(self, slot: int, value) -> int:
        box = _box(self.descriptor[slot], value)
        self._values.append(box)
        return ctypes.addressof(box)

    def _buffer_address(self, arrays: dict, name: str, role: str) -> int:
        try:
            obj = arrays[name]
        except KeyError:
            raise HawkError(
                f"{self.path.name}: this artifact declares a {role!r} slot {name!r}, "
                "but no plane was bound for it"
            ) from None
        width = int(self.sidecar.get("arg_shapes", {}).get(name) or 1)
        view = memoryview(obj)
        if role in _PER_SAMPLE_ROLES and plane_layout(view, width) == "view":
            return _transposed_address(obj, self._buffers)
        return buffer_address(obj, self._buffers)


def buffer_address(obj, keepalive: list) -> int:
    """The address of ``obj``'s buffer, through the BUFFER PROTOCOL alone.

    No numpy: any C-contiguous writable buffer exports one, and
    ``ctypes.from_buffer`` is the stdlib door to its address. Appended
    to ``keepalive``, since dropping the export would let the owner
    resize or free the memory the mirror still points at."""
    view = memoryview(obj)
    if not view.c_contiguous:
        raise HawkError(
            "hawk.runtime binds a plane at computed sample offsets, so it must be "
            "C-contiguous; got a strided buffer"
        )
    if view.readonly:
        raise HawkError(
            "hawk.runtime binds a plane by ADDRESS, never by copy, so a read-only "
            "buffer cannot be bound: a copy would silently drop the kernel's writes "
            "and silently freeze an input's later updates"
        )
    exported = (ctypes.c_char * view.nbytes).from_buffer(obj)
    keepalive.append(exported)
    return ctypes.addressof(exported)


class _PyBuffer(ctypes.Structure):
    """CPython's ``Py_buffer``, for the one export ``from_buffer`` can't
    make: a writable F-contiguous plane."""

    _fields_ = [("buf", ctypes.c_void_p), ("obj", ctypes.c_void_p),
                ("len", ctypes.c_ssize_t), ("itemsize", ctypes.c_ssize_t),
                ("readonly", ctypes.c_int), ("ndim", ctypes.c_int),
                ("format", ctypes.c_char_p), ("shape", ctypes.c_void_p),
                ("strides", ctypes.c_void_p), ("suboffsets", ctypes.c_void_p),
                ("internal", ctypes.c_void_p)]


_PYBUF_F_CONTIGUOUS_WRITABLE = 0x0058 | 0x0001
_get_buffer = ctypes.pythonapi.PyObject_GetBuffer
_get_buffer.argtypes = (ctypes.py_object, ctypes.POINTER(_PyBuffer), ctypes.c_int)
_get_buffer.restype = ctypes.c_int
_release_buffer = ctypes.pythonapi.PyBuffer_Release
_release_buffer.argtypes = (ctypes.POINTER(_PyBuffer),)
_release_buffer.restype = None


class _Export:
    """One held buffer export, released when the bind that took it is
    dropped."""

    def __init__(self, obj):
        self.view = _PyBuffer()
        _get_buffer(obj, ctypes.byref(self.view), _PYBUF_F_CONTIGUOUS_WRITABLE)

    def __del__(self):
        if self.view.obj:
            _release_buffer(ctypes.byref(self.view))
            self.view.obj = None


def _transposed_address(obj, keepalive: list) -> int:
    # Its bytes already ARE the native (w, L) plane, so no copy is made.
    export = _Export(obj)
    keepalive.append(export)
    return int(export.view.buf)


def _box(kind: str, value):
    if kind == "f64":
        return ctypes.c_double(float(value))
    if kind == "f32":
        return ctypes.c_float(float(value))
    if kind == "int":
        return ctypes.c_longlong(int(value))
    if kind == "nsamples":
        return _index_ctype()(int(value))
    raise HawkError(f"slot kind {kind!r} is not a by-value slot")


def _index_ctype():
    """The ctype of ``EAGLE_ABI_INDEX_T``, read off the binding rather
    than assumed: a hard-coded ``c_uint32`` would shift the whole
    parameter block on a 64-bit-index build."""
    width = int(_core.build_info()["index_type_bytes"])
    try:
        return {4: ctypes.c_uint32, 8: ctypes.c_uint64}[width]
    except KeyError:
        raise HawkError(
            f"hawk._core was built with a {width}-byte EAGLE_ABI_INDEX_T; the "
            "nsamples role is 4 or 8 bytes wide"
        ) from None


def load(directory, kernel: str, sidecar: dict | None = None) -> HostKernel:
    """Load ``<directory>/<kernel>.so``, reading its sidecar JSON if omitted."""
    import json

    directory = Path(directory)
    if sidecar is None:
        sidecar = json.loads((directory / f"{kernel}.json").read_text())
    return HostKernel(directory / f"{kernel}.so", sidecar)


def run(kernel_artifact: HostKernel, *, base: int = 0, count: int | None = None,
        n_samples: int | None = None, **arrays) -> None:
    """Bind ``arrays`` and run the WHOLE range in ONE serial call.

    ``count`` defaults to ``n_samples``, and ``n_samples`` to the count
    every bound per-sample plane agrees on (:func:`_n_from`). Output
    planes are the caller's, written in place; this returns nothing."""
    if n_samples is None:
        n_samples = _n_from(kernel_artifact.arg_spec, arrays,
                            kernel_artifact.sidecar.get("arg_shapes", {}))
    kernel_artifact.bind_all(arrays, n_samples)
    kernel_artifact.launch(base, n_samples if count is None else count, n_samples)
