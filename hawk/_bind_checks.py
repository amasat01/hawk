# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Bind-time inference and validation: the sample count read off the bound
planes, and every argument checked against the sidecar's declaration before
any address is taken."""

from __future__ import annotations

from ._plane_layout import (
    _native_extent,
    mixed_refusal,
    number_refusal,
    plane_layout,
    sample_major_refusal,
)
from .ir import HawkError
from .ir import nodes as _ir_nodes
from .types import PER_SAMPLE_ROLES

#: The roles whose plane's trailing extent is n_samples by construction
#: (:data:`hawk.types.PER_SAMPLE_ROLES`).
_PER_SAMPLE_ROLES = PER_SAMPLE_ROLES

#: Roles that NEVER vote on n_samples, whatever their trailing extent
#: happens to be. A ``lookup``/``wide_in`` plane's length is cross-sample
#: by construction. ``wide_out``/``accum_out`` are subtler: when written
#: at their own column their plane genuinely IS n_samples long, but that
#: fact lives in the walk's per-sink ``.index``, invisible to a deployed
#: artifact's flat sidecar — so both roles are excluded categorically.
#: Refusing to vote fails loudly; voting on an unrelated buffer's length
#: fails silently (see :func:`_n_from`'s docstring).
_NEVER_VOTES = ("lookup", "wide_in", "wide_out", "accum_out")

assert set(_PER_SAMPLE_ROLES) | set(_NEVER_VOTES) | {"uniform", "nsamples"}\
    == set(_ir_nodes.ROLE_ORDER), (
    "hawk.runtime's n_samples-inference vocabulary must classify EVERY "
    "role exactly once (vote, never-vote, or by-value/no-plane): "
    f"{sorted(_ir_nodes.ROLE_ORDER)}"
)


def _n_from(arg_spec, arrays: dict, arg_shapes: dict | None = None) -> int:
    """The sample count every bound PER-SAMPLE plane agrees on (the ``n``).

    An earlier version read the trailing extent of the FIRST plane in
    ``arg_spec`` order, full stop — silently wrong whenever that plane
    wasn't per-sample. A KAN fold's reverse pass leads its ``arg_spec``
    with a scattered ``accum_out`` plane whose own length has nothing to
    do with the sample count; reading its length as ``n_samples`` over-ran
    the correctly-sized cotangent plane and returned garbage with no error
    — the "computes, does not crash" failure mode this module warns about.

    The fix votes ONLY the roles :data:`_PER_SAMPLE_ROLES` names —
    :data:`_NEVER_VOTES` roles are IGNORED regardless of position, never
    merely deprioritised. Every voting plane that IS bound must AGREE (a
    truncated buffer for one input and a full one for another is a real
    mismatch, not something to average), and a kernel with no per-sample
    plane bound at all has nothing to read a count off of — both cases
    refuse, naming what disagreed or was missing, never guessing, since
    ``n_samples`` sizes the entire launch."""
    votes: dict[str, int] = {}
    single, batch = None, None
    for role, name in arg_spec:
        if role not in _PER_SAMPLE_ROLES or name not in arrays:
            continue
        value = arrays[name]
        if isinstance(value, (bool, int, float, complex)):
            raise HawkError(f"hawk.runtime.run: {number_refusal(name, value)}")
        # the NATIVE trailing extent: a sample-major (L, w) plane's count is
        # its leading axis, never its width
        width = int((arg_shapes or {}).get(name) or 1)
        mv = memoryview(value)
        votes[name] = _native_extent(mv, width)
        if plane_layout(mv, width) == "single":
            single = single or (name, tuple(mv.shape))
        else:
            batch = batch or (name, tuple(mv.shape), votes[name])
    if single is not None and batch is not None:
        raise HawkError(f"hawk.runtime.run: {mixed_refusal(single, batch)}")
    if not votes:
        raise HawkError(
            "hawk.runtime.run: no bound PER-SAMPLE plane to read the sample "
            f"count from (arg_spec carries none of {_PER_SAMPLE_ROLES} bound "
            "in arrays) — a lookup/wide_in/wide_out/accum_out plane's own "
            "length is a runtime quantity unrelated to n_samples and never "
            "votes; pass n_samples= explicitly"
        )
    lengths = set(votes.values())
    if len(lengths) != 1:
        raise HawkError(
            "hawk.runtime.run: the bound per-sample planes disagree on the "
            f"sample count: {sorted(votes.items())} — pass n_samples= "
            "explicitly to say which is right"
        )
    return lengths.pop()


# -- Bind-time validation, run in `HostKernel.bind_all` before any
# argument's address is taken. The single source of truth is the SIDECAR
# itself (`arg_dtypes`, `arg_shapes`, `params`, `exec_access`) — never a
# per-feature rule hard-coded here, so a kernel declaring a new role or
# dtype is covered for free. --------------------------------------------- #

#: Wire dtype string -> the (kind, itemsize) its buffer-protocol export
#: must report. `itemsize` is read off the buffer itself; only kind needs
#: a table.
_WIRE_SPEC = {"float64": ("f", 8), "float32": ("f", 4), "int64": ("i", 8),
             "bool": ("b", 1)}

#: `memoryview.format` -> coarse kind. Signed and unsigned integer codes
#: fold to the same two buckets so a platform's `long` vs `long long`
#: spelling of "8-byte integer" is never a false mismatch; `?` is its own
#: kind so a 1-byte integer is never mistaken for a mask.
_FORMAT_KIND = {"f": "f", "d": "f", "e": "f",
               "b": "i", "h": "i", "i": "i", "l": "i", "q": "i", "n": "i",
               "B": "u", "H": "u", "I": "u", "L": "u", "Q": "u", "N": "u", "c": "u",
               "?": "b"}

#: The `np.asarray(..., dtype=...)` spelling a FIX suggests for each wire
#: dtype string.
_NP_DTYPE_LITERAL = {"float64": "np.float64", "float32": "np.float32",
                     "int64": "np.int64", "bool": "bool"}


def _buffer_kind(mv: memoryview) -> tuple:
    """``(kind, itemsize)`` off a memoryview — the two facts every dtype
    check below compares, read from the buffer itself rather than assumed."""
    return (_FORMAT_KIND.get(mv.format, "?"), mv.itemsize)


def _dtype_name(kind: str, size: int) -> str:
    """The human-readable dtype name one ``(kind, itemsize)`` pair names, for
    an error's "got" side."""
    if kind == "b":
        return "bool"
    if kind == "f":
        return {4: "float32", 8: "float64"}.get(size, f"{size * 8}-bit float")
    if kind in ("i", "u"):
        return f"{'int' if kind == 'i' else 'uint'}{size * 8}"
    return f"<unrecognised buffer format, itemsize {size}>"


def _dtype_matches(mv: memoryview, expect: str) -> bool:
    """Whether ``mv`` carries the wire dtype ``expect`` names. Signed and
    unsigned integers of the right width both satisfy an ``intNN``
    declaration (signedness is not policed, only width and kind); an
    ``expect`` this table doesn't know is let through."""
    want_kind, want_size = _WIRE_SPEC.get(expect, (None, None))
    if want_kind is None:
        return True
    kind, size = _buffer_kind(mv)
    if want_kind == "i":
        return kind in ("i", "u") and size == want_size
    return kind == want_kind and size == want_size


def _check_plane_arg(role: str, name: str, value, arg_dtypes: dict,
                     arg_shapes: dict, n_samples: int, reduce_sized: bool,
                     errors: list) -> None:
    """Every check one PLANE-bound argument (every role but ``uniform``/
    ``nsamples``) must pass, appended to ``errors`` rather than raised —
    :func:`_check_bind` reports every failing argument together."""
    if isinstance(value, (bool, int, float, complex)):
        errors.append(number_refusal(name, value))
        return
    try:
        mv = memoryview(value)
    except TypeError:
        errors.append(
            f"argument {name!r}: expected a buffer-protocol array (a numpy "
            f"array, a torch CPU tensor, or an array.array), got "
            f"{type(value).__name__} — pass np.asarray({name}, ...)"
        )
        return

    expect_dtype = arg_dtypes.get(name)
    if expect_dtype is not None and not _dtype_matches(mv, expect_dtype):
        got_kind, got_size = _buffer_kind(mv)
        errors.append(
            f"argument {name!r}: expected dtype {expect_dtype}, got "
            f"{_dtype_name(got_kind, got_size)} — pass np.asarray({name}, "
            f"dtype={_NP_DTYPE_LITERAL.get(expect_dtype, expect_dtype)!s})"
        )

    layout = "native"
    if role in _PER_SAMPLE_ROLES:
        layout = plane_layout(mv, int(arg_shapes.get(name) or 1))
        if layout == "copy":
            errors.append(sample_major_refusal(
                name, tuple(mv.shape), int(arg_shapes.get(name) or 1)))
            return

    if not mv.c_contiguous and layout != "view":
        errors.append(
            f"argument {name!r}: hawk.runtime binds a plane at computed "
            f"sample offsets, so it must be C-contiguous, got a strided "
            f"buffer — pass np.ascontiguousarray({name})"
        )

    if role in _ir_nodes.SINK_ROLES and mv.readonly:
        errors.append(
            f"argument {name!r}: role {role!r} is written by the kernel, so "
            f"it must be writable, got a read-only array — pass a writable "
            f"array, e.g. np.array({name}) rather than a read-only view"
        )

    if role in _PER_SAMPLE_ROLES:
        # The trailing extent is a MINIMUM, not an exact match: an explicit
        # `n_samples=` may ask for fewer samples than the bound planes hold
        # (a caller allocates once, launches a sub-range without
        # rebinding), so longer is correct and shorter is always the bug.
        # The leading (per-sample width) extent is a static fact of the
        # compiled body, so it's still checked exactly.
        width = int(arg_shapes.get(name) or 1)
        shape = tuple(mv.shape)
        if layout == "view":
            shape = shape[::-1]      # the native (w, L) plane its bytes are
        elif layout == "single":
            # one sample: its contiguous bytes ARE the (w, 1) / (1,) plane
            shape = (1,) if width <= 1 else (width, 1)
        if width <= 1:
            bad = len(shape) != 1 or shape[0] < n_samples
            want = f"(L,) with L >= n_samples={n_samples}"
        else:
            bad = len(shape) != 2 or shape[0] != width or shape[1] < n_samples
            want = f"({width}, L) with L >= n_samples={n_samples}"
        if bad:
            errors.append(
                f"argument {name!r}: expected shape {want} ({width} per "
                f"sample), got {shape} — pass an array of at least that "
                f"length (never shorter than n_samples)"
            )
    elif role == "accum_out" and reduce_sized:
        # A `MapreducePartial` sink (`Reduce(op)`) always writes its own
        # column — the one case a deployed artifact's flat sidecar can
        # still tell "own column" from "scattered", since
        # `exec_access == "mapreduce"` means this is the kernel's only
        # sink. A plain scattered `Accum`/`WideOut` is NOT shape-checked
        # here: its length is a genuine runtime quantity the sidecar never
        # declares (see `_NEVER_VOTES`). Same minimum-length rule as above.
        shape = tuple(mv.shape)
        if len(shape) != 1 or shape[0] < n_samples:
            errors.append(
                f"argument {name!r}: a Reduce(...) output holds ONE slot per "
                f"sample, expected shape (L,) with L >= n_samples={n_samples}, "
                f"got {shape} — pass np.zeros(n_samples), not a single "
                f"accumulator slot"
            )


def _check_uniform_arg(name: str, value, params: dict, errors: list) -> None:
    """A ``uniform``'s value: a Python number or a 0-d buffer, matching
    the ``params`` block's declared ``int``/``float``. int->float is
    never flagged (exact and safe); float->int IS, since ``_box`` would
    silently truncate it."""
    if isinstance(value, bool):
        kind = "b"
    elif isinstance(value, int):
        kind = "i"
    elif isinstance(value, float):
        kind = "f"
    else:
        try:
            mv = memoryview(value)
        except TypeError:
            errors.append(
                f"argument {name!r}: a uniform binds a Python number (or a "
                f"0-d array), got {type(value).__name__}"
            )
            return
        if tuple(mv.shape) != ():
            errors.append(
                f"argument {name!r}: a uniform is a single broadcast value, "
                f"expected a 0-d array, got shape {tuple(mv.shape)} — pass "
                f"a Python number or {name}.item()"
            )
            return
        raw_kind = _buffer_kind(mv)[0]
        kind = "i" if raw_kind in ("i", "u") else ("f" if raw_kind == "f" else "b")

    want = params.get(name, {}).get("dtype", "float")
    if want == "int" and kind == "f":
        errors.append(
            f"argument {name!r}: expected an int uniform, got a float value "
            f"— pass int({name}) if that is exact, or declare the parameter "
            f"as a float"
        )


def _check_bind(sidecar: dict, arg_spec, arrays: dict, n_samples: int) -> None:
    """Every bound argument, checked against the kernel's own declaration
    — before any address is taken. Raises ONE :class:`~hawk.ir.HawkError`
    naming every failing argument, or returns silently when all match.

    The declaration is read from the sidecar alone — ``arg_dtypes``,
    ``arg_shapes``, ``params`` and ``exec_access`` — so nothing here
    special-cases a kernel or feature by name."""
    kernel_name = sidecar.get("kernel", "?")
    arg_dtypes = sidecar.get("arg_dtypes", {})
    arg_shapes = sidecar.get("arg_shapes", {})
    params = {p["name"]: p for p in sidecar.get("params", ())}
    reduce_sized = sidecar.get("exec_access") == "mapreduce"

    errors: list = []
    for role, name in arg_spec:
        if role == "nsamples":
            continue
        if name not in arrays:
            errors.append(
                f"argument {name!r}: this kernel declares a {role!r} slot "
                f"for it, but no value was bound"
            )
            continue
        if role == "uniform":
            _check_uniform_arg(name, arrays[name], params, errors)
        else:
            _check_plane_arg(role, name, arrays[name], arg_dtypes, arg_shapes,
                             n_samples, reduce_sized, errors)

    if not errors:
        return
    if len(errors) == 1:
        raise HawkError(f"kernel {kernel_name!r}: {errors[0]}")
    raise HawkError(
        f"kernel {kernel_name!r}: {len(errors)} arguments failed validation:\n"
        + "\n".join(f"  - {e}" for e in errors)
    )
