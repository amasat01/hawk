# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The SERIAL oracle a partitioned run is judged against.

``hawk.runtime`` allocates nothing and returns nothing — a plane's memory
belongs to whoever owns the run. This module is the CONSUMER half: it allocates
the planes, calls ``hawk.runtime.run`` once over the whole range, and hands back
the outputs in ``arg_spec`` order.

WHY THE ALLOCATION RULE IS COPIED FROM EAGLE, LINE FOR LINE. Rows 
compare this arm against ``eagle.plan``'s, and a paired comparison is only
worth anything if the arms differ in exactly ONE variable — the execution
structure. So the plane shapes, the dtypes, the "an integer or bool array is
passed at its own dtype" rule and the "a single output comes back as the plane,
never a 1-tuple" rule are eagle's own (``eagle/plan.py``'s ``_run_host_plan``,
``_host_plane``, ``_plane_shape``, ``_returned``), restated here rather than
imported so the oracle does not reach into another package's privates — and
pinned by a row that runs both arms over the same fixtures.

It is deliberately in ``hawk/tests/`` and not in ``hawk/hawk/``: HAWK depends on
no numpy (:mod:`hawk.runtime` binds through the buffer protocol alone), and this
allocation policy is a consumer's, not the toolchain's.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from hawk import runtime

#: The roles eagle ALLOCATES an output plane for and returns, in ``arg_spec``
#: order (``eagle.plan._OUTPUT_ROLES``).
OUTPUT_ROLES = ("out", "mutable", "wide_out", "accum_out")

#: The roles the caller supplies an input plane for (``eagle.plan._INPUT_ROLES``).
INPUT_ROLES = ("per_sample", "vec_in", "mat_in", "terminated", "lookup", "wide_in")

_NP_DTYPE = {"float64": np.float64, "float32": np.float32}


def sidecar_at(directory, kernel: str) -> dict:
    return json.loads((pathlib.Path(directory) / f"{kernel}.json").read_text())


def load(directory, kernel: str, sidecar: dict | None = None):
    """The dlopen'ed, self-checked host kernel (``hawk.runtime.load``)."""
    return runtime.load(directory, kernel, sidecar)


def planes(kernel, n: int, kw: dict) -> dict:
    """Every plane one run binds, allocated eagle's way (see the module note)."""
    dtype = np.dtype(_NP_DTYPE[kernel.scalar_type])
    # `arg_widths` covers input roles too (vec_in/mat_in/
    # per_sample/terminated, alongside the output roles this already had), but
    # it stays INT-ONLY: a runtime-length role (lookup/wide_in/wide_out/
    # accum_out) is simply ABSENT here, never `None` (hawk/artifact/
    # sidecar.py documents why an `int(v)` cast over this
    # field may never see one) -- `widths.get(name, 1)` below already handles
    # an undeclared role by falling back to width 1.
    widths = {str(k): int(v) for k, v in kernel.sidecar.get("arg_widths", {}).items()}
    # An INTEGER-typed `Mutable` (the dispatch fixture: a dispatch over
    # integer branches) declares its dtype in the sidecar's own `mutables`
    # block ("int", `hawk/artifact/sidecar.py`'s `_mutable_dtype`) -- an
    # auto-allocated output plane must honour it, or the C++ side reads an
    # int64 wire out of a float64-sized buffer and computes a pointer's bytes
    # back as a double (the "6.94e-310 signature" this codebase already knows
    # by name, `hawk/_bind_checks.py`'s `_n_from` docstring).
    int_mutables = {m["name"] for m in kernel.sidecar.get("mutables", ())
                    if m.get("dtype") == "int"}
    bound = {}
    for role, name in kernel.arg_spec:
        if role in OUTPUT_ROLES:
            out_dtype = np.dtype(np.int64) if name in int_mutables else dtype
            bound[name] = (np.ascontiguousarray(kw[name]) if name in kw
                           else np.zeros(_shape(widths.get(name, 1), n),
                                        dtype=out_dtype))
        elif role in INPUT_ROLES:
            bound[name] = _host_plane(kw[name], dtype)
        elif role == "uniform":
            bound[name] = kw[name]
    return bound


def run(directory, kernel_name: str, n: int, sidecar: dict | None = None, **kw):
    """Run ONE kernel serially through ``hawk._core`` and return its outputs.

    ``n`` is the TRUE sample count (the triple's ``nSamples``), and the whole
    range is run in one call — which is what makes this the oracle rather than
    another structure."""
    kernel = load(directory, kernel_name, sidecar)
    return run_kernel(kernel, n, **kw)


def run_kernel(kernel, n: int, *, base: int = 0, count: int | None = None, **kw):
    """The same, on an already-loaded :class:`hawk.runtime.HostKernel`."""
    bound = planes(kernel, n, kw)
    runtime.run(kernel, base=base, count=n if count is None else count,
                n_samples=n, **bound)
    return returned(kernel, bound)


def returned(kernel, bound: dict):
    """The output planes in ``arg_spec`` order — the plane itself for a
    single-output kernel, never a 1-tuple (eagle's ``_returned``)."""
    out = [bound[name] for role, name in kernel.arg_spec if role in OUTPUT_ROLES]
    if not out:
        raise AssertionError(
            f"{kernel.path.name}: its arg_spec declares no output role, so a run "
            "has nothing to hand back"
        )
    return out[0] if len(out) == 1 else tuple(out)


def _shape(width: int, n: int):
    return (n,) if int(width) <= 1 else (int(width), n)


def _host_plane(value, dtype):
    """A caller-supplied input as a contiguous host array of the artifact's Real
    dtype; an integer/bool array passes at its OWN dtype (an int lookup table or
    a bool ``terminated`` mask is not a Real plane)."""
    arr = np.asarray(value)
    if arr.dtype.kind not in "iub":
        arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr)
