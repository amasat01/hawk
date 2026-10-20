# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ctypes harness that CALLS an exported C unit the way a caller would.

:func:`hawk.embedded.export` writes ``<name>.c``/``<name>.h`` and the report
whose ``interface`` rows (one per ``arg_spec`` slot, in order) describe the
args struct field for field. This module builds that struct from the report
(:class:`ArgsStruct`), compiles the unit into a shared library
(:func:`cc_shared`) and runs it over a whole range or a partition
(:func:`run_c`), handing back the output planes in ``arg_spec`` order.

THE PLANE LAYOUT IS THE ORACLE'S, AND IT IS THE C ABI'S. ``_oracle.planes()``
(``tests/_oracle.py``'s ``_shape``) allocates a rank>=1 plane as a ``(W, n)``
C-contiguous array with ``W = prod(shape)`` in row-major component order, so a
``Matrix[R, C]`` plane is ``(R*C, n)`` and component ``(r, c)`` is row
``r*C + c``; a rank-0 plane is ``(n,)``. Component ``c`` of sample ``i`` is
then at flat offset ``c * n + i``: exactly the header's SoA contract,
``base[(c * n_samples) + i]``, pitch ``n_samples``. The same array can be
handed to both sides with no reshuffle, which is what lets a row compare
``tobytes()``.

It does not import ``_oracle`` (that module exists to build and run host
artifacts, which the interpreter row must not depend on): the allocation
rules are restated from the report's rows instead.

* a supplied plane is used as given (``np.ascontiguousarray``; a real plane
  cast to the mode's width as ``_oracle._host_plane`` casts it, an integer or
  mask plane at its own dtype; a ``bool`` mask is VIEWED as ``uint8``, so an
  ``inout`` mask the finish sets is written into the caller's array);
* an absent output plane (``out``/``mutable``/``wide_out``/``accum_out``) is
  zero-filled at ``(W, n)`` / ``(n,)``, integers as ``int64``;
* an absent counter (``finished_count``, a branch-mix plane) is zero-filled
  at ``prod(shape)`` cells (the caller zeroes it, the header says);
* any other absent plane is an error naming it.
"""

from __future__ import annotations

import ctypes
import math
import sys
from pathlib import Path

import numpy as np

#: The real element per scalar mode: ``(ctypes type, numpy dtype)``.
_REAL = {"float64": (ctypes.c_double, np.float64),
         "float32": (ctypes.c_float, np.float32)}
#: A non-real element per IR dtype. An index (``i32``/``i64``) is the 64-bit
#: ``hawk_int_t``; a ``bool`` plane is one byte.
_ELEM = {"i32": (ctypes.c_int64, np.int64), "i64": (ctypes.c_int64, np.int64),
         "int32": (ctypes.c_int32, np.int32), "uint32": (ctypes.c_uint32, np.uint32),
         "bool": (ctypes.c_uint8, np.uint8)}
_OUTPUT_ROLES = ("out", "mutable", "wide_out", "accum_out")
_SHARED_SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"


def corpus():
    """``(name, sinks, walk)`` of every C-target corpus case:
    ``tests/_emitted.cases()`` without ``energy`` (a mapreduce, refused)."""
    import _emitted

    return [c for c in _emitted.cases() if c[0] != "energy"]


def cc_shared(c_path, tmp) -> Path:
    """Compile the exported ``c_path`` into a shared library in ``tmp``
    (:func:`hawk.embedded.toolchain.compile_shared`: c99, -O2, no FP
    contraction)."""
    from hawk.embedded.toolchain import compile_shared

    c_path = Path(c_path)
    return compile_shared(c_path, Path(tmp) / f"{c_path.stem}{_SHARED_SUFFIX}")


def _mode(report, mode) -> str:
    return str(mode if mode is not None else report["scalar_mode"])


def element(row: dict, mode: str):
    """``(ctypes type, numpy dtype)`` of one interface row's element."""
    if row["role"] == "terminated":
        return _ELEM["bool"]
    if row["dtype"] in ("f64", "f32"):
        return _REAL[mode]
    return _ELEM[row["dtype"]]


def _rows(report) -> list:
    return [r for r in report["interface"] if r["role"] != "nsamples"]


class ArgsStruct(ctypes.Structure):
    """Base of every ``<name>_args_t`` mirror; :meth:`from_report` makes one."""

    @classmethod
    def from_report(cls, report: dict, mode=None) -> type[ctypes.Structure]:
        """The ``<name>_args_t`` struct of ``report``, fields in interface
        order: a pointer per plane (``POINTER(element)``) followed by
        ``int64_t <field>_len`` when the plane has a runtime length; a
        by-value field per ``uniform`` (the mode's real, or the 64-bit int of
        an integer Param). The ``nsamples`` row is the entry's own
        ``n_samples`` argument, not a field."""
        mode = _mode(report, mode)
        fields = []
        for row in _rows(report):
            if row["layout"] == "value":
                real = row["dtype"] in ("f64", "f32")
                fields.append((row["field"], _REAL[mode][0] if real else ctypes.c_int64))
                continue
            fields.append((row["field"], ctypes.POINTER(element(row, mode)[0])))
            if row["runtime_length"]:
                fields.append((f"{row['field']}_len", ctypes.c_int64))
        return type(f"{report['kernel']}_args_t", (cls,), {"_fields_": fields})


_LIBS: dict = {}


def _entry(so_path, name: str):
    so_path = str(Path(so_path).resolve())
    lib = _LIBS.get(so_path)
    if lib is None:
        lib = _LIBS[so_path] = ctypes.CDLL(so_path)
    return getattr(lib, f"{name}_c")


def _cells(row: dict, n: int) -> tuple:
    if row["layout"] == "partials" or not row["shape"]:
        return (n,)
    return (math.prod(row["shape"]), n)


def _bind(row: dict, n: int, mode: str, planes: dict):
    """``(array the caller sees, array the C side reads)`` for one plane row."""
    ctype, dtype = element(row, mode)
    name = row["name"]
    if name in planes:
        arr = np.ascontiguousarray(planes[name])
        if row["role"] == "terminated" or arr.dtype == np.bool_:
            assert arr.dtype.itemsize == 1, f"{name!r}: a mask is one byte per sample"
            return arr, arr.view(np.uint8)
        if arr.dtype != dtype:
            assert row["direction"] == "in", (
                f"{name!r} is {row['direction']}: pass it as {np.dtype(dtype)}, not "
                f"{arr.dtype} (a cast copy would hide the C side's writes)")
            arr = arr.astype(dtype)
        return arr, arr
    if row["role"] == "lookup" and row["direction"] == "inout":
        arr = np.zeros(math.prod(row["shape"]) if row["shape"] else 1, dtype=dtype)
        return arr, arr
    if row["role"] in _OUTPUT_ROLES:
        arr = np.zeros(_cells(row, n), dtype=dtype)
        return arr, arr
    raise KeyError(f"no plane supplied for {name!r} ({row['role']}, {row['field']})")


def run_c(so_path, name: str, report: dict, n: int, /, *, base: int = 0,
          count: int | None = None, mode=None, **planes):
    """Call ``<name>_c`` over ``[base, base + count)`` of ``n`` samples and
    return the planes it may write (direction ``out``/``inout``) in
    ``arg_spec`` order: the plane itself for a single one, never a 1-tuple
    (``_oracle.returned``'s rule). Asserts ``HAWK_OK``.

    ``planes`` maps each plane's NAME (the interface row's ``name``, as
    ``_oracle.run_kernel`` takes it) to an array, and each ``uniform`` to its
    value; see the module docstring for what is allocated when absent. A
    supplied writable plane is written in place. The four leading parameters
    are positional-only, so a plane may be named ``n`` (``tape(spin)``'s
    integer counter is)."""
    mode = _mode(report, mode)
    struct = ArgsStruct.from_report(report, mode)
    args = struct()
    keep: list = []           # ctypes holds no reference: keep every buffer alive
    returned = []
    for row in _rows(report):
        if row["layout"] == "value":
            setattr(args, row["field"], planes[row["name"]])
            continue
        seen, raw = _bind(row, n, mode, planes)
        keep.append(raw)
        setattr(args, row["field"],
                raw.ctypes.data_as(ctypes.POINTER(element(row, mode)[0])))
        if row["runtime_length"]:
            setattr(args, f"{row['field']}_len", int(raw.shape[-1]))
        if row["direction"] != "in":
            returned.append(seen)
    entry = _entry(so_path, name)
    entry.restype = ctypes.c_int32
    entry.argtypes = [ctypes.POINTER(struct), ctypes.c_int64, ctypes.c_int64,
                      ctypes.c_int64]
    status = entry(ctypes.byref(args), base, n if count is None else count, n)
    assert status == 0, (
        f"{name}_c returned status {status} (1 = HAWK_E_NULL, 2 = HAWK_E_RANGE) "
        f"for base={base}, count={count}, n_samples={n}")
    del keep
    if not returned:
        raise AssertionError(f"{name}: its interface declares no written plane")
    return returned[0] if len(returned) == 1 else tuple(returned)
