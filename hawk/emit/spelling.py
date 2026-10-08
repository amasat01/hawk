# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""How the aether renderer spells things: binding and parameter names, element,
carry, view and mirror types, the launch-geometry audit, literal and loop-header
text, and the op-to-aether spelling tables."""

from __future__ import annotations

import re

from ..ir import HawkError, Node
from ..ir.nodes import (
    Const,
    Op,
    Primitive,
)
from ..types import TensorType

#: The launch-geometry tokens forbidden anywhere in a body; the one place
#: they may appear is the backend's index prologue. Warp-level spelling
#: (``%laneid``, ``__activemask``, the shuffle reductions) has its own sanctioned
#: place, the CUDA backend's persistent and fast entries, and is not listed.
GEOMETRY_TOKENS = ("blockIdx", "threadIdx", "gridDim", "blockDim", "omp", "MPI_",
                   "nccl")


#: The audit matches each token at a WORD boundary, not as a bare
#: substring: ``omp`` is a substring of the ABI mirror's own
#: ``compStride_`` field, so a naive ``in`` would misreport it.
_GEOMETRY_RE = re.compile("|".join(rf"\b{re.escape(t)}" for t in GEOMETRY_TOKENS))


def geometry_hits(text: str) -> list:
    """Every launch-geometry token in ``text``, as ``(line, token, text)``."""
    out = []
    for number, line in enumerate(text.splitlines(), 1):
        out += [(number, m.group(0), line.strip()) for m in _GEOMETRY_RE.finditer(line)]
    return out


#: The identifier both backends' index prologues bind the lane's own
#: integer index to, and so the one name a
#: :class:`~hawk.ir.nodes.SampleIndex` node renders as. Spelled here
#: because the body and both wrappers must agree on it, and a row checks
#: that agreement rather than assuming it. Deliberately NOT a
#: launch-geometry token — the geometry stays inside the prologue.
SAMPLE_INDEX_IDENT = "hawk_i"


#: Role -> the identifier prefix its binding carries. A prefix, not the
#: bare name, since ``(role, name)`` is the slot identity.
IDENT_PREFIX = {
    "mutable": "mut", "out": "out", "wide_out": "wout", "accum_out": "aout",
    "vec_in": "vin", "mat_in": "mtx", "wide_in": "win", "per_sample": "psc",
    "lookup": "lut", "terminated": "trm", "uniform": "uni", "nsamples": "nsm",
}


#: The pinned role -> by-value ABI shape map, matching eagle's
#: :func:`eagle.roles.classify_arg` tags. ``mutable`` resolves by context
#: (rank); ``value`` is a by-value scalar or the sample count.
MIRROR_OF = {
    "out": "GRefMirror", "vec_in": "GRefMirror", "mat_in": "GRefMirror",
    "per_sample": "ScalarHandle", "lookup": "ScalarHandle",
    "terminated": "ScalarHandle", "wide_in": "ScalarHandle",
    "wide_out": "ScalarHandle", "accum_out": "ScalarHandle",
    "uniform": "value", "nsamples": "value",
}


#: Roles pinned to a 32-byte ``ScalarHandle``; a rank>=1 plane there would
#: need the 40-byte ``GRefMirror``, which the map forbids.
_HANDLE_ONLY = ("per_sample", "lookup", "terminated", "wide_in", "wide_out",
                "accum_out")


#: The declared aether reduction functors -> their aether spelling. Emitted
#: as an alias so an unbuilt functor fails to COMPILE rather than reaching
#: a manifest.
REDUCE_FUNCTOR = {"sum": "aether::detail::SumOp",
                  "times": "aether::detail::TimesOp",
                  "max": "aether::detail::MaxOp",
                  "land": "aether::detail::LogicalAndOp"}


#: The one-argument ``aether::math`` names a rank-0 unary lowers to.
#: Kinds with no ``cwise*`` companion are rank-0 ONLY here: ``hawk.math``
#: maps a vector or matrix operand entry by entry before the IR sees it.
_MATH1 = {"abs": "abs", "sqrt": "sqrt", "rsqrt": "rsqrt", "exp": "exp",
          "log": "log", "sin": "sin", "cos": "cos", "tan": "tan", "tanh": "tanh",
          "asin": "asin", "acos": "acos", "atan": "atan", "floor": "floor",
          "exp2": "exp2", "expm1": "expm1", "log2": "log2", "log10": "log10",
          "log1p": "log1p", "cbrt": "cbrt", "sinh": "sinh", "cosh": "cosh",
          "asinh": "asinh", "acosh": "acosh", "atanh": "atanh", "ceil": "ceil",
          "trunc": "trunc", "round": "round", "rint": "rint", "erf": "erf",
          "erfc": "erfc", "isnan": "isnan", "isinf": "isinf",
          "isfinite": "isfinite"}


#: The two-argument ``aether::math`` names a rank-0 binary lowers to.
_MATH2 = {"pow": "pow", "min": "fmin", "max": "fmax", "atan2": "atan2",
          "hypot": "hypot", "copysign": "copysign", "fmod": "fmod",
          "remainder": "remainder", "fdim": "fdim"}


#: The three-argument ``aether::math`` names a rank-0 ternary lowers to.
_MATH3 = {"fma": "fma", "clip": "clip"}


#: The two draws -> aether's stateless ``aether::random::detail`` free
#: function each lowers to — neither has a public twin shaped for this
#: op's rank-0 ``(seed, counter)`` scalar draw.
_RANDOM_FN = {"random_uniform": "uniform01", "random_normal": "standardNormal"}


#: The rank>=1 component-wise UNARY free functions, one per :data:`_MATH1`
#: entry, so a rank-1 chain never has to scalarise to reach a
#: transcendental.
_CWISE1 = {"sqrt": "cwiseSqrt", "rsqrt": "cwiseRsqrt", "exp": "cwiseExp",
           "log": "cwiseLog", "sin": "cwiseSin", "cos": "cwiseCos",
           "asin": "cwiseAsin", "acos": "cwiseAcos", "atan": "cwiseAtan",
           "tan": "cwiseTan", "tanh": "cwiseTanh"}


#: Rank-0 infix spellings.
_INFIX = {"add": "+", "sub": "-", "mul": "*", "div": "/", "lt": "<", "le": "<=",
          "gt": ">", "ge": ">=", "eq": "==", "ne": "!=", "land": "&&", "lor": "||"}


#: Rank>=1 component-wise BINARY free functions. Each carries both
#: overloads — ``f(expr, scalar_bound)`` and ``f(expr, expr)`` — so the
#: renderer never guards on the right operand's rank.
_CWISE = {"min": "aether::cwiseMin", "max": "aether::cwiseMax",
          "div": "aether::cwiseDiv", "pow": "aether::cwisePow"}


#: Rank>=1 zero-argument Expression METHODS (``aether/expr/Expression.h``).
_METHOD0 = {"norm": "norm", "sum": "sum", "transpose": "transpose",
            "quat_conj": "quatConj", "quat_recip": "quatReciprocal",
            "as_pure": "asPureQuaternion", "as_vec3": "asBack3DVector"}


#: Rank>=1 one-argument Expression METHODS.
_METHOD1 = {"dot": "dot", "cross": "cross", "mul": "cwiseMul",
            "quat_mul": "quatMul", "quat_rotate": "quatRotate"}


def _is_literal_zero(node: Node) -> bool:
    """HAWK's elision predicate: ``node`` (after :func:`_through`'s
    unwrap) is the literal zero — a bare :class:`~hawk.ir.nodes.Const` of
    value 0, or a rank>=1 ``splat`` of one."""
    node = _through(node)
    if isinstance(node, Const):
        return node.literal == 0
    if isinstance(node, Op) and node.kind == "splat" and len(node.operands) == 1:
        return _is_literal_zero(node.operands[0])
    return False


def binding_name(role: str, name: str) -> str:
    """The identifier the body reads a slot through (a view, or a by-value scalar)."""
    return f"{IDENT_PREFIX[role]}_{name}"


def finish_binding(mask: str) -> str:
    """The WRITABLE twin binding a finishing kernel marks its mask through."""
    return f"{binding_name('terminated', mask)}_w"


def param_name(role: str, name: str) -> str:
    """The identifier the entry takes the slot's mirror under."""
    return f"p_{binding_name(role, name)}"


def element_spelling(dtype: str) -> str:
    """The C++ element type an IR dtype stores as. ``f64``/``f32`` are
    both ``Real``, since the compiled scalar mode decides the alias."""
    if dtype in ("f64", "f32"):
        return "Real"
    if dtype in ("i32", "i64"):
        return "Int"
    if dtype == "bool":
        return "bool"
    raise HawkError(f"no C++ element spelling for dtype {dtype!r}")


def carry_type(ttype: TensorType) -> str:
    """The C++ type ONE loop-carried local is declared as.

    A rank-0 carry is a plain ``Real``/``Int``/``bool``. A rank>=1 carry
    is an ``aether::Item<Elem, Es...>`` and may not be anything else: an
    aether expression node stores its operands BY REFERENCE, so an
    ``auto`` carry would hold a tree over temporaries that die at the end
    of the iteration that built them."""
    elem = element_spelling(ttype.dtype)
    if not ttype.shape:
        return elem
    return f"aether::Item<{elem}, {', '.join(str(e) for e in ttype.shape)}>"


def view_type(role: str, ttype: TensorType) -> str:
    """The aether view / by-value spelling one wire binds as."""
    if mirror_of(role, ttype) == "value":
        return "EAGLE_ABI_INDEX_T" if role == "nsamples"\
            else element_spelling(ttype.dtype)
    elem, shape = element_spelling(ttype.dtype), ttype.shape
    ext = ", ".join([*(str(e) for e in shape), "aether::dyn"])
    return f"aether::View<{elem}, aether::extents<{ext}>, aether::layout_right>"


def mirror_of(role: str, ttype: TensorType) -> str:
    """The pinned by-value ABI shape: ``GRefMirror``, ``ScalarHandle`` or
    ``value``; ``mutable`` resolves by rank."""
    if role == "mutable":
        return "GRefMirror" if ttype.shape else "ScalarHandle"
    if role in _HANDLE_ONLY and ttype.shape:
        raise HawkError(
            f"role {role!r} carries a rank-{len(ttype.shape)} plane typed {ttype!r}, "
            "but the map pins it to the 32-byte ScalarHandle (eagle's classify_arg, "
            "eagle/python/eagle/roles.py:292-341) — a wider plane needs the "
            "40-byte GRefMirror, which that role has no packing for. Until eagle's "
            "packer grows a GRef accum/wide role, such a kernel cannot "
            "be emitted; a rank-0 plane in this role can."
        )
    return MIRROR_OF[role]


def reconstruct_call(role: str, ttype: TensorType, source: str) -> str:
    """The prologue expression rebuilding one slot's aether view from its
    by-value mirror — the same text on both targets."""
    if mirror_of(role, ttype) == "value":
        return source
    elem, shape = element_spelling(ttype.dtype), ttype.shape
    if len(shape) == 0:
        return f"hawk_abi::scalar_view<{elem}>({source})"
    if len(shape) == 1:
        return f"hawk_abi::vec_view<{elem}, {shape[0]}>({source})"
    return f"hawk_abi::mat_view<{elem}, {shape[0]}, {shape[1]}>({source})"


def _through(node: Node) -> Node:
    """The node a reference to ``node`` is a reference TO: a Primitive
    stands for its forward and is never rendered itself."""
    while isinstance(node, Primitive):
        node = node.forward
    return node


_IDENT_OK = re.compile(r"[^A-Za-z0-9_]")


def _ident(name: str) -> str:
    """An author's Python identifier, made safe to paste into C++. Every
    emitted loop identifier is additionally prefixed, so it cannot
    collide with a plane's binding."""
    return _IDENT_OK.sub("_", name) or "x"


def _for_header(index: str, loop, count: str | None = None) -> str:
    """The plain C++/CUDA ``for`` header asks for, and nothing else.

    The comparison direction follows the sign of the (compile-time) step,
    and a unit step collapses to ``++k``/``--k``. Both bounds are LITERALS
    in the emitted text, which is what lets the compiler see the trip
    count and decide for itself whether to unroll."""
    cmp_ = "<" if loop.step > 0 else ">"
    if loop.step == 1:
        incr = f"++{index}"
    elif loop.step == -1:
        incr = f"--{index}"
    else:
        incr = f"{index} += {loop.step}"
    if count is None:
        return (f"for (Int {index} = {loop.start}; {index} {cmp_} {loop.stop}; "
                f"{incr}) {{")
    # A RUNTIME count (a derived loop of a loop with a `break`): the first
    # `count` iterations of the static range, or the last `count` for a
    # reverse loop.
    if loop.count_tail:
        sign = "+" if loop.step > 0 else "-"
        begin = f"{loop.start} {sign} ({loop.trip} - {count}) * {abs(loop.step)}"
        end = f"{loop.stop}"
    else:
        begin = f"{loop.start}"
        end = (f"hawk_clamp_count({count})" if (loop.start, loop.step) == (0, 1)
               else f"{loop.start} + {count} * {loop.step}")
    # A 32-bit index: the count is clamped to the int range once, and every
    # other runtime bound lies inside the static range of literals.
    begin = begin if begin.lstrip("-").isdigit() else f"static_cast<int>({begin})"
    end = end if end.startswith("hawk_clamp_count(") else f"static_cast<int>({end})"
    return f"for (int {index} = {begin}; {index} {cmp_} {end}; {incr}) {{"


def _ordinal(index: str, loop) -> str:
    """The ITERATION ORDINAL of ``index`` in ``loop`` — the tape's
    subscript. The tape is written by the forward loop and read by the
    reverse one, whose header runs the same values backwards, so the two
    must agree on which slot an iteration owns: ``(k - start) / step``,
    folded away in the ordinary ``range(N)`` case where it IS the index."""
    if (loop.start, loop.step) == (0, 1):
        return index
    shifted = index if loop.start == 0 else f"({index} - {loop.start})"
    return shifted if loop.step == 1 else f"({shifted} / {loop.step})"


def _no_spelling(kind: str, rank: int) -> str:
    return f"op {kind!r} has no aether spelling at rank {rank}"


def _real_literal(value: float) -> str:
    """Shortest round-tripping C++ literal for a baked constant."""
    import math

    if math.isnan(value):
        return "NAN"
    if math.isinf(value):
        return "-INFINITY" if value < 0 else "INFINITY"
    text = repr(float(value))
    return text if ("." in text or "e" in text or "E" in text) else text + ".0"
