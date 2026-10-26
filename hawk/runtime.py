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
from (:data:`hawk.emit.render.MIRROR_OF`), so a slot cannot be described
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
always the bug); every
plane must be C-contiguous and a sink role's plane writable, except a
sample-major plane whose transpose is C-contiguous, which binds
zero-copy (:func:`plane_layout`). A ``(w, w)`` plane reads both ways and is
refused unless ``layout="samples_first"``/``"samples_last"`` (per call) or
``hawk.samples_first(x)``/``hawk.samples_last(x)`` (per array, which wins) says
which axis holds the samples. :meth:`HostKernel.rebind` takes raw
pointers, which carry none of this, and is the unchecked fast path over
an already-validated bind.

Threading. Two :class:`HostKernel` objects are independent and run freely in
parallel. ONE :class:`HostKernel` shared across threads is safe: :func:`run`
(bind and launch together), :meth:`~HostKernel.bind_all` and
:meth:`~HostKernel.launch` each hold the kernel's lock, so concurrent calls
queue and each sees only its own planes. Separate ``bind_all`` and ``launch``
calls from different threads are still the caller's to order. Module-level state (the build
counters, the primitive registry, the unit publisher) is always safe to use
from any thread.
"""

from __future__ import annotations

import array
import ctypes
import threading
from pathlib import Path

from . import _cells, _core, _lane_bind
from ._cells import ROW
from ._bind_checks import _PER_SAMPLE_ROLES, _check_bind, _n_from, widen_index
from ._plane_layout import (
    LayoutChoice,
    _native_extent,
    plane_layout,
    split_marks,
)
from .emit.box import threads
from .emit.render import MIRROR_OF, mirror_of
from .emit.wrapper import fold_entry_name
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
#: (:data:`hawk.emit.render.MIRROR_OF` on the left).
_KIND_OF_MIRROR = {"GRefMirror": "gref", "ScalarHandle": "handle"}

_INT_DTYPES = ("i32", "i64", "int32", "uint32")


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
    expects a 40-byte mirror), ``vec_widths`` and ``mat_shapes`` (the
    declared extents of the vector and matrix inputs), ``params`` and
    ``scalar_type``. A slot's kind is decided by role and rank, but the
    type handed to :func:`kind_for` is the declared one, whatever the
    extents (``Matrix[R, C]``, ``Vector[W]``)."""
    scalar_type = sidecar.get("scalar_type", "float64")
    mutables = {m["name"]: m for m in sidecar.get("mutables", ())}
    wire = sidecar.get("arg_dtypes", {})
    params = {p["name"]: p for p in sidecar.get("params", ())}
    out = []
    for role, name in (tuple(pair) for pair in sidecar["arg_spec"]):
        if role in ("mutable", "out"):
            ttype = _mutable_ttype(name, mutables.get(name, {}))
        elif role == "uniform":
            ttype = TensorType((), "i64" if params.get(name, {}).get("dtype") == "int"
                               else "f64")
        elif role == "vec_in":
            ttype = TensorType((int(_extent(sidecar.get("vec_widths", {}).get(name),
                                            "vec_widths", name)),), "f64")
        elif role == "mat_in":
            shape = _extent(sidecar.get("mat_shapes", {}).get(name), "mat_shapes", name)
            ttype = TensorType(tuple(int(e) for e in shape), "f64")
        else:
            ttype = TensorType((), _WIRE_TO_DTYPE.get(wire.get(name), "f64"))
        out.append(kind_for(role, ttype, scalar_type=scalar_type))
    return out


def _extent(value, block: str, name: str):
    """``value``, the extent the artifact declares for the plane ``name`` in
    ``block``; a sidecar that omits it cannot be bound."""
    if value is None:
        raise HawkError(
            f"the sidecar declares no {block!r} entry for {name!r}, so its shape "
            "cannot be read; rebuild the artifact")
    return value


def _mutable_ttype(name: str, decl: dict) -> TensorType:
    """The type of a ``mutables`` entry: ``dtype`` names the rank, ``width`` and
    ``shape`` the extents."""
    dtype = decl.get("dtype", "scalar")
    if dtype == "vector":
        return TensorType((int(_extent(decl.get("width"), "mutables.width", name)),), "f64")
    if dtype == "matrix":
        shape = _extent(decl.get("shape"), "mutables.shape", name)
        return TensorType(tuple(int(e) for e in shape), "f64")
    return TensorType((), _MUTABLE_DTYPE[dtype])


#: The IR dtype an integer ``arg_dtypes`` wire label stands for.
_WIRE_TO_DTYPE = {"int64": "i64", "int32": "int32", "uint32": "uint32"}

#: The IR dtype of each rank-0 ``mutables`` dtype.
_MUTABLE_DTYPE = {"scalar": "f64", "int": "i64", "int32": "int32", "uint32": "uint32"}

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
        #: ``{target: entry}``: the fold of each owned companion
        #: (:func:`hawk.emit.wrapper.fold_entries`), run after the kernel.
        self.folds = {target: self.library.entry(fold_entry_name(sidecar["kernel"], target)
                                                 + "_host")
                      for target in (sidecar.get("companions") or {}).values()}
        self.block = _core.ArgBlock(self.descriptor)
        self.slot_of = {pair: i for i, pair in enumerate(self.arg_spec)}
        #: The input planes riding the 64-bit index wire: a 32-bit one is widened at bind.
        self._index_inputs = tuple(
            name for role, name in self.arg_spec
            if role in ("per_sample", "lookup", "wide_in")
            and sidecar.get("arg_dtypes", {}).get(name) == "int64")
        self._values = []          # keeps every by-value ctypes box alive
        self.one_step = None       # the word of 1 an unbound fused_steps reads
        self.finished = None       # the counter an unbound finished_count fills
        self._buffers = []         # keeps every exported buffer view alive
        #: Held across everything one call touches (the block, the keepalive
        #: lists, the shared scratch above), until its launch returns.
        self._lock = threading.Lock()

    # -- per bind ----------------------------------------------------------
    def _one_step_word(self, arrays: dict) -> dict:
        """``arrays``, plus the reserved words a runner normally provides,
        for each one this kernel uses and the caller leaves out: a
        one-cell ``fused_steps`` word of ``1`` (a launch outside a runner
        takes exactly one step) and a one-cell ``finished_count`` counter
        reset to ``0`` (a kernel that finishes its own samples counts them
        there; bind your own buffer to read the count), and a zeroed
        ``<group>_mix`` plane per ``"auto"`` dispatch (its branch counts), and
        a ``jump_seed`` of ``0`` for an alternate-path ``jump=`` derivative."""
        from .diff.jump import SEED
        from .ir.nodes import FINISHED_PLANE, FUSED_STEPS_PLANE

        extra = {}
        if SEED not in arrays and any(n == SEED for _, n in self.slot_of):
            extra[SEED] = 0
        for name in (FUSED_STEPS_PLANE, FINISHED_PLANE):
            if name in arrays or not any(n == name for _, n in self.slot_of):
                continue
            if name == FUSED_STEPS_PLANE:
                if self.one_step is None:
                    self.one_step = array.array("q", [1])
                extra[name] = self.one_step
            else:
                if self.finished is None:
                    self.finished = array.array("I", [0])
                self.finished[0] = 0
                extra[name] = self.finished
        for entry in self.sidecar.get("dispatch", ()):
            mix = entry.get("mix")
            if mix is not None and mix not in arrays:
                # an "auto" group's branch counters, left out: counted into a
                # fresh scratch plane (bind your own to read them)
                extra[mix] = array.array("I", [0] * int(entry["K"]))
        return {**arrays, **extra} if extra else arrays

    def bind_all(self, arrays: dict, n_samples: int, *, layout=None) -> None:
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
        with self._lock:
            self._bind_all(arrays, n_samples, layout)

    def _bind_all(self, arrays: dict, n_samples: int, layout) -> None:
        arrays, axes = split_marks(arrays, layout)
        choice = LayoutChoice(layout, axes)
        arrays = self._one_step_word(arrays)
        if self._index_inputs:
            arrays = widen_index(self._index_inputs, arrays)
        _check_bind(self.sidecar, self.arg_spec, arrays, n_samples, choice)
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
                    mv = memoryview(arrays[name])
                    width = int(widths.get(name) or 1)
                    samples = (mv.shape[-1] if name in (self.sidecar.get("gather_mats") or {})
                               else _native_extent(mv, width, choice.of(name, mv, width)))
                self.block.bind(slot,
                                self._buffer_address(arrays, name, role, choice),
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

    def _cell_scratch_for(self, cells: dict, n_samples: int, arrays: dict):
        """``(scratch, cell outputs, arrays)``: the caller binds each CELL
        (``(columns,)``, the plane ``name`` or ``name_k`` of a wide one is its
        column k); the rows are this kernel's own, made once per sample count."""
        missing = [c for c in cells if c not in arrays]
        if missing:
            raise HawkError(
                f"{self.path.name}: this kernel reduces over samples into the cells "
                f"{sorted(cells)}; bind {missing} (one value per column), not its rows")
        outputs = {c: arrays[c] for c in cells}
        held = self.__dict__.setdefault("_cell_scratches", {})
        if n_samples not in held:
            held[n_samples] = _cells.Scratch(cells, n_samples, self.scalar_type)
        scratch = held[n_samples]
        planes = {k: v for k, v in arrays.items() if k not in cells}
        return scratch, outputs, {**planes, **scratch.rows}

    # -- per launch --------------------------------------------------------
    def launch(self, base: int, count: int, n_samples: int) -> None:
        """THE launch crossing: one call over ``[base, base+count)``.

        SERIAL, always: the reference oracle every partitioned, tiled or
        ranked run is judged against, so it never grows a schedule."""
        with self._lock:
            self._launch(base, count, n_samples)

    def _launch(self, base: int, count: int, n_samples: int) -> None:
        self.entry.run(self.block, int(base), int(count), int(n_samples))

    # -- helpers -----------------------------------------------------------
    def _value_address(self, slot: int, value) -> int:
        box = _box(self.descriptor[slot], value)
        self._values.append(box)
        return ctypes.addressof(box)

    def _buffer_address(self, arrays: dict, name: str, role: str,
                        choice=None) -> int:
        try:
            obj = arrays[name]
        except KeyError:
            raise HawkError(
                f"{self.path.name}: this artifact declares a {role!r} slot {name!r}, "
                "but no plane was bound for it"
            ) from None
        width = int(self.sidecar.get("arg_shapes", {}).get(name) or 1)
        view = memoryview(obj)
        kind = (choice.of(name, view, width) if choice is not None
                else plane_layout(view, width))
        if role in _PER_SAMPLE_ROLES and kind == "view":
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
            "C-contiguous; got a strided buffer — pass np.ascontiguousarray(x)"
        )
    if view.readonly:
        raise HawkError(
            "hawk.runtime binds a plane by ADDRESS, never by copy, so a read-only "
            "buffer cannot be bound: a copy would silently drop the kernel's writes "
            "and silently freeze an input's later updates — pass a writable array, "
            "e.g. np.array(x)"
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
    """Load ``<directory>/<kernel>.so``, reading its sidecar JSON if omitted.

    Refuses a directory holding only ``kernel``'s standalone C unit (a
    ``<kernel>.embedded.json`` and no ``<kernel>.json``): that is source for
    an embedded build, not a host artifact."""
    import json

    directory = Path(directory)
    if sidecar is None:
        if (not (directory / f"{kernel}.json").exists()
                and (directory / f"{kernel}.embedded.json").exists()):
            raise HawkError(
                f"{kernel!r} in {directory} was built for the 'c' target only: "
                "it is a standalone C unit (<name>.c, <name>.h, "
                "<name>.embedded.json, see hawk.embedded) for an embedded build, "
                "not a host artifact hawk.load can run; build it with "
                "targets=('host',) to load it here")
        sidecar = json.loads((directory / f"{kernel}.json").read_text())
    return HostKernel(directory / f"{kernel}.so", sidecar)


def _own_companions(sidecar: dict, arrays: dict) -> tuple:
    """``(arrays, owned)``: ``arrays`` plus a zeroed companion, the target's
    shape, for each compensated keyed sum whose companion the caller did not
    bind (the run owns it and the caller never passes it), and the
    ``{companion: target}`` it made."""
    owned = {comp: target for comp, target in (sidecar.get("companions") or {}).items()
             if comp not in arrays and target in arrays}
    if not owned:
        return arrays, owned
    import numpy as np

    return {**arrays, **{comp: np.zeros_like(np.asarray(arrays[target]))
                         for comp, target in owned.items()}}, owned


def _fold_companions(kernel_artifact: HostKernel, arrays: dict, owned: dict) -> None:
    """Each owned companion into its target, after the launch, through the
    kernel's fold entry over the target's slots (the block is still bound):
    the target alone then holds the compensated sum."""
    import numpy as np

    for target in owned.values():
        size = int(np.asarray(arrays[target]).size)
        kernel_artifact.folds[target].run(kernel_artifact.block, 0, size, size)


def run(kernel_artifact: HostKernel, *, base: int = 0, count: int | None = None,
        n_samples: int | None = None, layout: str | None = None,
        **arrays) -> None:
    """Bind ``arrays`` and run the WHOLE range in ONE serial call.

    ``count`` defaults to ``n_samples``, and ``n_samples`` to the count
    every bound per-sample plane agrees on (:func:`_n_from`). Output
    planes are the caller's, written in place; this returns nothing. A
    thread-box kernel (sidecar ``launch.box``) counts ``base``/``count`` in
    threads, ``count`` defaulting to the threads that cover ``n_samples``.

    A per-sample plane whose shape reads both as component-major ``(w, N)`` and
    as sample-major ``(N, w)`` (``(w, w)``) is refused unless the call says
    which axis holds the samples: ``layout="samples_first"`` or
    ``layout="samples_last"`` resolves every such plane of the call, and
    ``hawk.samples_first(x)`` / ``hawk.samples_last(x)`` say it for one array
    (any shape; a contradiction is refused; the array's marker beats ``layout``).
    Neither copies.

    A kernel declaring ``hawk.Grid[...]`` takes its runtime extent by label
    (``run(k, x=x, y=y, channel=64)``) or from lane-shaped planes, and refuses
    ``n_samples=``."""
    sidecar = kernel_artifact.sidecar
    arrays, lane_n = _lane_bind.bind(sidecar, kernel_artifact.arg_spec, arrays,
                                     n_samples, _PER_SAMPLE_ROLES)
    arrays, owned = _own_companions(sidecar, arrays)
    plain, axes = split_marks(arrays, layout)
    if lane_n is not None:
        n_samples = lane_n
    if n_samples is None:
        compact = sidecar.get("gather_mats") or {}
        n_samples = _n_from([(r, n) for r, n in kernel_artifact.arg_spec
                             if not (r == "mat_in" and n in compact)], plain,
                            sidecar.get("arg_shapes", {}),
                            LayoutChoice(layout, axes))
        _lane_bind.check_total(sidecar, n_samples)
    if kernel_artifact.sidecar.get("stages"):
        if base != 0 or count not in (None, n_samples):
            raise HawkError(
                f"{kernel_artifact.path.name}: this kernel reads a value reduced over "
                "every sample, so it runs over every sample (no base= or count=)")
        with kernel_artifact._lock:
            _run_stages(kernel_artifact, n_samples, layout, arrays, owned)
        return
    cells = kernel_artifact.sidecar.get("cells") or {}
    with kernel_artifact._lock:
        scratch = None
        if cells:
            scratch, outputs, arrays = kernel_artifact._cell_scratch_for(
                cells, n_samples, arrays)
        kernel_artifact._bind_all(arrays, n_samples, layout)
        # a thread-box kernel's base and count count threads (sidecar launch.box)
        count = (threads(sidecar.get("launch", {}).get("box"), n_samples)
                 if count is None else count)
        kernel_artifact._launch(base, count, n_samples)
        _fold_companions(kernel_artifact, arrays, owned)
        if scratch is not None:
            scratch.finish(outputs, base // ROW, -(-(base + count) // ROW))


def _keyed_planes(first: HostKernel, held: dict, n_samples: int) -> dict:
    """The keyed planes a split kernel owns between its stages, and the
    companions of their compensated sums: made once per sample count and
    ZEROED for every run (a stage accumulates into them)."""
    import numpy as np

    out = {}
    for cell, spec in (first.sidecar.get("keyed") or {}).items():
        size = _lane_bind.keyed_size(first.sidecar, spec, n_samples)
        out[cell] = held.setdefault("keyed", {}).setdefault(
            (cell, size), np.zeros(size, dtype=first.scalar_type))
        for stage in held["stages"]:
            for comp, target in (stage.sidecar.get("companions") or {}).items():
                if target == cell:
                    out[comp] = held["keyed"].setdefault(
                        (comp, size), np.zeros(size, dtype=first.scalar_type))
    for plane in out.values():
        plane.fill(0)
    return out


def _run_stages(first: HostKernel, n_samples: int, layout, arrays: dict,
                owned: dict) -> None:
    """A split kernel: stage ``k``, then the host finish of its cells, then
    stage ``k + 1`` reading those cells as one-cell planes. A cell the caller
    does not bind (one only a later stage reads) is this kernel's own."""
    import numpy as np

    cells = first.sidecar["cells"]
    held = first.__dict__.setdefault("_split", {})
    if "stages" not in held:
        held["stages"] = [first] + [load(first.path.parent, st["kernel"])
                                    for st in first.sidecar["stages"][1:]]
        held["own"] = {c: np.zeros(spec["columns"], dtype=first.scalar_type)
                       for c, spec in cells.items()}
    values = {c: arrays[c] if c in arrays else held["own"][c] for c in cells}
    keyed = _keyed_planes(first, held, n_samples)
    for at, (stage, record) in enumerate(zip(held["stages"], first.sidecar["stages"])):
        names = {name for _, name in stage.arg_spec}
        bound = {k: v for k, v in arrays.items() if k in names and k not in cells}
        bound.update({k: v for k, v in keyed.items() if k in names})
        for cell in record["cells_in"]:
            column = np.asarray(values[cell]).reshape(-1)
            for k, plane in enumerate(_cells.planes(cell, cells[cell]["columns"])):
                bound[plane] = column[k:k + 1]
        out = {c: cells[c] for c in record["cells_out"]}
        scratch = None
        if out:
            scratch, outputs, bound = stage._cell_scratch_for(
                out, n_samples, {**bound, **{c: values[c] for c in out}})
        stage._bind_all(bound, n_samples, layout)
        stage._launch(0, n_samples, n_samples)
        later = {name for st in held["stages"][at + 1:] for _, name in st.arg_spec}
        mine = {c: t for c, t in owned.items() if c in names and c not in later}
        mine.update({c: t for c, t in (stage.sidecar.get("companions") or {}).items()
                     if t in keyed and c in keyed})
        _fold_companions(stage, bound, mine)
        if scratch is not None:
            scratch.finish(outputs, 0, -(-n_samples // ROW))
