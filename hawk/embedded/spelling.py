# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""How the standalone C target spells things: element and aggregate types, libm
names, literals, index and trip clamps, and the integer helper names. Pure
tables and small string functions; nothing here renders a kernel."""

from __future__ import annotations

from ..emit.spelling import _real_literal
from ..ir import HawkError

#: ScalarMode id to the aether C width word used in vector and matrix type names.
WIDTH = {"float64": "f64", "float32": "f32"}

#: Element dtype to its scalar C type.
_SCALAR_CTYPE = {
    "f64": "hawk_real_t",
    "f32": "hawk_real_t",
    "i32": "hawk_int_t",
    "i64": "hawk_int_t",
    "int32": "int32_t",
    "uint32": "uint32_t",
    "bool": "bool",
}

#: C99 libm base names the C target may call (float32 appends ``f``).
#: ``isnan``/``isinf``/``isfinite`` are macros, not libm names.
LIBM_NAMES = (
    "sqrt rsqrt exp log sin cos tan tanh asin acos atan floor ceil trunc round "
    "rint exp2 expm1 log2 log10 log1p cbrt sinh cosh asinh acosh atanh erf erfc "
    "pow fmin fmax atan2 hypot copysign fmod fdim fma fabs nextafter"
).split()

#: Integer helper kinds, in the order the helpers are declared.
_INT_KINDS = (
    "add sub mul neg abs min max div floordiv mod band bor bxor shl shr invert "
    "from_real"
).split()

#: Kind to the operation word in the helper name (``band`` is ``and``, ...).
_INT_OP = {"band": "and", "bor": "or", "bxor": "xor"}

#: Dtype to the helper family prefix; i32 and i64 share the 64-bit helpers.
_INT_PREFIX = {
    "i32": "hawk_i64",
    "i64": "hawk_i64",
    "int32": "hawk_i32",
    "uint32": "hawk_u32",
}

#: ``(kind, dtype)`` to the integer helper function name.
INT_HELPER = {
    (kind, dtype): f"{prefix}_{_INT_OP.get(kind, kind)}"
    for dtype, prefix in _INT_PREFIX.items()
    for kind in _INT_KINDS
}

#: Dtype to the fixed-width literal macro.
_INT_MACRO = {
    "i32": "INT64_C",
    "i64": "INT64_C",
    "int32": "INT32_C",
    "uint32": "UINT32_C",
}


def shape_word(shape: tuple[int, ...]) -> str:
    """Name a shape for type spellings: ``scalar``, ``vec3``, ``mat2x3``."""
    if len(shape) == 0:
        return "scalar"
    if len(shape) == 1:
        return f"vec{shape[0]}"
    if len(shape) == 2:
        return f"mat{shape[0]}x{shape[1]}"
    raise HawkError(f"no C spelling for rank-{len(shape)} shape {shape}")


def ctype(ttype, mode) -> str:
    """The C type of a ``TensorType`` under a scalar mode."""
    if ttype.shape == ():
        try:
            return _SCALAR_CTYPE[ttype.dtype]
        except KeyError:
            raise HawkError(f"no C type for dtype {ttype.dtype!r}") from None
    return f"aether_c_{WIDTH[mode.id]}_{shape_word(ttype.shape)}_t"


def libm_name(fn: str, mode) -> str:
    """The C99 libm function for ``fn`` under a mode (``f`` suffix in float32)."""
    if fn not in LIBM_NAMES:
        raise HawkError(
            f"unknown libm function {fn!r}; known: {', '.join(LIBM_NAMES)}"
        )
    return fn + "f" if mode.id == "float32" else fn


def real_literal(value: float, mode) -> str:
    """A real constant as a cast double literal; the mode does not change it."""
    return f"((hawk_real_t){_real_literal(value)})"


def int_literal(value: int, dtype: str) -> str:
    """An integer (or bool) constant as a fixed-width literal macro."""
    if dtype == "bool":
        return "true" if value else "false"
    return f"{_INT_MACRO[dtype]}({int(value)})"


def clamp_index(expr: str, length: str) -> str:
    """Clamp an index expression into ``[0, length)``."""
    return f"hawk_clamp_index({expr}, {length})"


def clamp_trips(expr: str, trip: int) -> str:
    """Clamp a loop trip count into ``[0, trip]``."""
    return f"hawk_clamp_trips({expr}, {trip})"
