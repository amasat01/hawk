# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The C helpers hawk itself owns in the standalone C target: index and trip
clamps, plane load/store, aggregate construction and selection, the integer
helpers with numpy edge semantics, ``spacing`` and ``sign_nan``. Each is a text
template returned as an ``aether_dsc.c_core.CFunction``, so the translation-unit
assembler treats these and the core's arithmetic folds alike. Arithmetic folds
and the scalar ``sign`` come from the core, not from here."""

from __future__ import annotations

from aether_dsc import c_core
from aether_dsc.c_core import CFunction

from ..ir import HawkError
from .spelling import (
    INT_HELPER,
    WIDTH,
    int_literal,
    libm_name,
    real_literal,
    shape_word,
)

#: Helpers that take a shape (aggregate type), by base name.
_SHAPED = frozenset(
    "load store splat make select at set at_raw set_raw".split()
)

#: ``(prefix, op)`` pairs of the integer helpers, in declaration order.
_INT_OPS = tuple(dict.fromkeys(
    tuple(name.removeprefix("hawk_").split("_", 1)) for name in INT_HELPER.values()
))

#: Every helper name; shape-bearing ones once, by base name.
HELPER_NAMES: tuple[str, ...] = (
    "clamp_index", "clamp_trips", "clamp_count",
    "load", "store", "splat", "make", "select", "at", "set", "at_raw", "set_raw",
    *(f"{prefix}_{op}" for prefix, op in _INT_OPS),
    "spacing", "sign_nan",
)


def _text(name: str, shape, mode, signature: str, body: list[str],
          note: str | None = None) -> str:
    """One helper: the header line, an optional note, then the function."""
    lines = [f"/* hawk: {name} {shape if shape else ()} {mode.id} */"]
    if note:
        lines.append(f"/* {note} */")
    lines.append(f"static inline {signature}")
    lines.append("{")
    lines += [f"    {line}" for line in body]
    lines.append("}")
    return "\n".join(lines) + "\n"


# ---- clamps ----

def _clamp(name: str, mode) -> str:
    zero32, zero64 = int_literal(0, "int32"), int_literal(0, "i64")
    if name == "clamp_index":
        one = int_literal(1, "i64")
        return _text(
            name, None, mode, "int64_t hawk_clamp_index(int64_t k, int64_t n)",
            [f"return (k < {zero64}) ? {zero64} : ((k >= n) ? (n - {one}) : k);"],
            "precondition: n >= 1",
        )
    if name == "clamp_trips":
        return _text(
            name, None, mode, "int32_t hawk_clamp_trips(hawk_int_t n, int32_t trip)",
            [f"return (n < {zero64}) ? {zero32} : ((n > trip) ? trip : (int32_t)n);"],
        )
    top = int_literal(2147483647, "int32")
    return _text(
        name, None, mode, "int32_t hawk_clamp_count(hawk_int_t n)",
        [f"return (n <= {zero64}) ? {zero32} : "
         f"((n < {int_literal(2147483647, 'i64')}) ? (int32_t)n : {top});"],
        "returns 0 for n <= 0; the C++ _CLAMP_COUNT narrowed instead "
        "(both run no iterations for INT32_MIN <= n <= 0)",
    )


# ---- aggregates ----

def _cells(shape: tuple[int, ...]) -> list[tuple[int, str]]:
    """Row-major ``(flat index, accessor)`` per component."""
    if len(shape) == 1:
        return [(k, f"v[{k}]") for k in range(shape[0])]
    rows, cols = shape
    return [(r * cols + c, f"m[{r}][{c}]") for r in range(rows) for c in range(cols)]


def _plane(flat: int) -> str:
    if flat == 0:
        return "plane[i]"
    if flat == 1:
        return "plane[pitch + i]"
    return f"plane[({flat} * pitch) + i]"


def _unused_pitch(count: int) -> list[str]:
    """A one-component plane access never reads ``pitch``."""
    return ["(void)pitch;"] if count == 1 else []


def _clamped(shape: tuple[int, ...], count: int) -> tuple[list[str], str]:
    """Prelude lines and the index expression for a clamped slot access."""
    clamp = f"hawk_clamp_index(k, {count})"
    if len(shape) == 1:
        return [], clamp
    return [f"const int64_t j = {clamp};"], "j"


def _aggregate(name: str, shape: tuple[int, ...], mode) -> str:
    w = WIDTH[mode.id]
    t = c_core.struct(shape, w).name
    word = shape_word(shape)
    fn = f"hawk_{word}_{name}"
    cells = _cells(shape)
    count = len(cells)
    if name == "load":
        sig = f"{t} {fn}(const hawk_real_t* plane, int64_t i, int64_t pitch)"
        body = _unused_pitch(count) + [f"{t} out;"]
        body += [f"out.{acc} = {_plane(flat)};" for flat, acc in cells]
        body.append("return out;")
    elif name == "store":
        sig = f"void {fn}(hawk_real_t* plane, int64_t i, int64_t pitch, {t} x)"
        body = _unused_pitch(count)
        body += [f"{_plane(flat)} = x.{acc};" for flat, acc in cells]
    elif name == "splat":
        sig = f"{t} {fn}(hawk_real_t s)"
        body = [f"{t} out;"]
        body += [f"out.{acc} = s;" for _, acc in cells]
        body.append("return out;")
    elif name == "make":
        args = ", ".join(f"hawk_real_t a{flat}" for flat, _ in cells)
        sig = f"{t} {fn}({args})"
        body = [f"{t} out;"]
        body += [f"out.{acc} = a{flat};" for flat, acc in cells]
        body.append("return out;")
    elif name == "select":
        sig = f"{t} {fn}(bool c, {t} a, {t} b)"
        body = [f"{t} out;"]
        body += [f"out.{acc} = c ? a.{acc} : b.{acc};" for _, acc in cells]
        body.append("return out;")
    elif name in ("at", "at_raw"):
        sig = f"hawk_real_t {fn}({t} x, hawk_int_t k)"
        if name == "at":
            body, use = _clamped(shape, count)
        else:
            body, use = [], "k"
        if len(shape) == 1:
            body.append(f"return x.v[{use}];")
        else:
            body.append(f"return x.m[{use} / {shape[1]}][{use} % {shape[1]}];")
    else:
        sig = f"{t} {fn}({t} x, hawk_int_t k, hawk_real_t e)"
        if name == "set":
            body, use = _clamped(shape, count)
        else:
            body, use = [], "k"
        body.append(f"{t} out = x;")
        if len(shape) == 1:
            body.append(f"out.v[{use}] = e;")
        else:
            body.append(f"out.m[{use} / {shape[1]}][{use} % {shape[1]}] = e;")
        body.append("return out;")
    return _text(name, shape, mode, sig, body)


# ---- integer helpers: one formatter for the three widths ----

#: prefix -> (dtype key of the literal macro, value type, unsigned twin, twin
#: literal macro, bits, signed). ``i64`` takes ``hawk_int_t`` for both sides.
_WIDTHS = {
    "i64": ("i64", "hawk_int_t", "uint64_t", "UINT64_C", 64, True),
    "i32": ("int32", "int32_t", "uint32_t", "UINT32_C", 32, True),
    "u32": ("uint32", "uint32_t", "uint32_t", "UINT32_C", 32, False),
}

_RANGE = {
    "i64": (f"(-{int_literal(9223372036854775807, 'i64')} - {int_literal(1, 'i64')})",
            int_literal(9223372036854775807, "i64")),
    "i32": (f"(-{int_literal(2147483647, 'int32')} - {int_literal(1, 'int32')})",
            int_literal(2147483647, "int32")),
    "u32": (int_literal(0, "uint32"), int_literal(4294967295, "uint32")),
}


#: The saturation bounds a real is compared against, spelled as real
#: literals (exact in both widths; a cast of the composite ``-MAX - 1`` would
#: break MISRA C:2012 10.8). The values returned stay the integer :data:`_RANGE`.
_REAL_RANGE = {
    "i64": ("((hawk_real_t)-9223372036854775808.0)",
            "((hawk_real_t)9223372036854775807.0)"),
    "i32": ("((hawk_real_t)-2147483648.0)", "((hawk_real_t)2147483647.0)"),
    "u32": ("((hawk_real_t)0.0)", "((hawk_real_t)4294967295.0)"),
}


def _int_helper(prefix: str, op: str, mode) -> str:
    dtype, t, u, umacro, bits, signed = _WIDTHS[prefix]
    lit = lambda v: int_literal(v, dtype)  # noqa: E731
    ulit = lambda v: f"{umacro}({v})"  # noqa: E731
    fn = f"hawk_{prefix}_{op}"
    lo, hi = _RANGE[prefix]
    z, m1, one = lit(0), lit(-1), lit(1)
    two = f"{t} a, {t} b"
    # wrapping arithmetic runs in the unsigned twin and is named before the
    # cast back: a composite expression is never cast (MISRA C:2012 10.8)
    negate = f"const {u} n = {ulit(0)} - ({u})a;"
    if op in ("add", "sub", "mul"):
        sym = {"add": "+", "sub": "-", "mul": "*"}[op]
        sig = f"{t} {fn}({two})"
        body = [f"const {u} r = ({u})a {sym} ({u})b;", f"return ({t})r;"]
    elif op == "neg":
        sig, body = f"{t} {fn}({t} a)", [negate, f"return ({t})n;"]
    elif op == "abs":
        sig = f"{t} {fn}({t} a)"
        body = ([negate, f"return (a < {z}) ? ({t})n : a;"] if signed
                else ["return a;"])
    elif op == "min":
        sig, body = f"{t} {fn}({two})", ["return (b < a) ? b : a;"]
    elif op == "max":
        sig, body = f"{t} {fn}({two})", ["return (a < b) ? b : a;"]
    elif op == "div":
        sig = f"{t} {fn}({two})"
        body = [f"{t} result;", f"if (b == {z}) {{", f"    result = {z};", "}"]
        if signed:
            body += [f"else if ((a == {lo}) && (b == {m1})) {{",
                     f"    result = {lo};", "}"]
        body += ["else {", "    result = a / b;", "}", "return result;"]
    elif op == "floordiv":
        sig = f"{t} {fn}({two})"
        body = [f"{t} result;", f"if (b == {z}) {{", f"    result = {z};", "}"]
        if signed:
            body += [f"else if (b == {m1}) {{", f"    {negate}",
                     f"    result = ({t})n;", "}",
                     "else {",
                     f"    const {t} q = ({t})(a / b);",
                     f"    result = (((a % b) != {z}) && ((a < {z}) != (b < {z})))",
                     f"        ? ({t})(q - {one}) : q;",
                     "}"]
        else:
            body += ["else {", f"    result = ({t})(a / b);", "}"]
        body.append("return result;")
    elif op == "mod":
        sig = f"{t} {fn}({two})"
        if signed:
            body = [f"{t} result;",
                    f"if ((b == {z}) || (b == {m1})) {{", f"    result = {z};", "}",
                    "else {",
                    f"    const {t} r = ({t})(a % b);",
                    f"    result = ((r != {z}) && ((r < {z}) != (b < {z})))",
                    f"        ? ({t})(r + b) : r;",
                    "}", "return result;"]
        else:
            body = [f"{t} result;", f"if (b == {z}) {{", f"    result = {z};", "}",
                    "else {", f"    result = ({t})(a % b);", "}", "return result;"]
    elif op in ("and", "or", "xor"):
        sym = {"and": "&", "or": "|", "xor": "^"}[op]
        sig, body = f"{t} {fn}({two})", [f"return ({t})(a {sym} b);"]
    elif op == "invert":
        sig, body = f"{t} {fn}({t} a)", [f"return ({t})(~a);"]
    elif op == "shl":
        sig = f"{t} {fn}({two})"
        body = [f"{t} result;", f"if (({u})b < {ulit(bits)}) {{",
                f"    const {u} r = ({u})a << ({u})b;",
                f"    result = ({t})r;", "}",
                "else {", f"    result = {z};", "}", "return result;"]
    elif op == "shr":
        sig = f"{t} {fn}({two})"
        far = f"({t})((a < {z}) ? {m1} : {z})" if signed else lit(0)
        # a signed operand shifts in the unsigned twin (MISRA C:2012 10.1):
        # ~(~a >> b) is the arithmetic shift of a negative a, bit for bit
        shift = ([f"    const {u} ua = ({u})a;",
                  f"    const {u} r = (a < {z}) ? ~((~ua) >> ({u})b) : (ua >> ({u})b);",
                  f"    result = ({t})r;"] if signed
                 else [f"    result = ({t})(a >> ({u})b);"])
        body = [f"{t} result;", f"if (({u})b < {ulit(bits)}) {{", *shift, "}",
                "else {", f"    result = {far};", "}", "return result;"]
    elif op == "from_real":
        sig = f"{t} {fn}(hawk_real_t x)"
        rlo, rhi = _REAL_RANGE[prefix]
        body = [f"return isnan(x) ? {z} : ((x <= {rlo}) ? {lo} : "
                f"((x >= {rhi}) ? {hi} : ({t})x));"]
    else:
        raise HawkError(f"no integer helper {prefix}_{op}")
    return _text(f"{prefix}_{op}", None, mode, sig, body)


# ---- real helpers ----

def _spacing(mode) -> str:
    zero, one, two = (real_literal(v, mode) for v in (0.0, 1.0, 2.0))
    nxt = libm_name("nextafter", mode)
    return _text(
        "spacing", None, mode, "hawk_real_t hawk_spacing(hawk_real_t x)",
        [f"const hawk_real_t a = (x < {zero}) ? -x : x;",
         f"const hawk_real_t up = {nxt}(a, (a * {two}) + {one}) - a;",
         f"return (x < {zero}) ? -up : up;"],
    )


def _sign_nan(mode) -> str:
    sign = c_core.function("sign", width=WIDTH[mode.id]).name
    return _text(
        "sign_nan", None, mode, "hawk_real_t hawk_sign_nan(hawk_real_t x)",
        [f"return isnan(x) ? x : {sign}(x);"],
    )


def helper(name: str, *, mode, shape=None) -> CFunction:
    """The C helper ``name`` under a scalar mode; shape-bearing helpers need
    ``shape`` (rank 1 or 2)."""
    w = WIDTH[mode.id]
    if name in _SHAPED:
        if shape is None or len(shape) not in (1, 2):
            raise HawkError(f"helper {name!r} needs a rank-1 or rank-2 shape")
        shape = tuple(shape)
        deps = (c_core.struct(shape, w).name,)
        if name in ("at", "set"):
            deps += ("hawk_clamp_index",)
        return CFunction(f"hawk_{shape_word(shape)}_{name}",
                         _aggregate(name, shape, mode), deps, ())
    if name in ("clamp_index", "clamp_trips", "clamp_count"):
        return CFunction(f"hawk_{name}", _clamp(name, mode), (), ())
    if name == "spacing":
        libm = (libm_name("nextafter", mode),)
        return CFunction("hawk_spacing", _spacing(mode), (), libm)
    if name == "sign_nan":
        deps = (c_core.function("sign", width=w).name,)
        return CFunction("hawk_sign_nan", _sign_nan(mode), deps, ())
    prefix, _, op = name.partition("_")
    if (prefix, op) in _INT_OPS:
        return CFunction(f"hawk_{name}", _int_helper(prefix, op, mode), (), ())
    raise HawkError(f"unknown helper {name!r}; known: {', '.join(HELPER_NAMES)}")
