# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The op-KIND vocabulary and its result-type rules.

One tensor-typed node family needs one place that says which op kinds
exist and what type each returns: the tracer mints nodes through
:func:`make`, the derivative rule table keys on the same :data:`OP_KINDS`,
and neither can drift from the other. Typing is rank-generic: an
elementwise op broadcasts a rank-0 operand against a rank-1/rank-2 one,
dtypes promote along ``bool < i32 < i64 < f32 < f64``, and a quaternion
``tag`` survives only ``add``/``sub``/``neg`` and a scalar scaling — so a
tangent or adjoint carries the SAME type as its primal and the walk's
total merge never sees a spurious collision.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..types import TensorType
from .nodes import QUAT_OPS, HawkError, Node, Op

#: Rank-generic elementwise binaries (a rank-0 operand broadcasts).
ELEMENTWISE_BINARY = ("add", "sub", "mul", "div", "pow", "min", "max", "atan2",
                      "hypot", "copysign", "fmod", "remainder", "fdim")
#: Rank-generic elementwise unaries.
UNARY_MATH = ("neg", "abs", "sqrt", "rsqrt", "exp", "log", "sin", "cos", "tan",
              "tanh", "asin", "acos", "atan", "floor",
              "exp2", "expm1", "log2", "log10", "log1p", "cbrt",
              "sinh", "cosh", "asinh", "acosh", "atanh",
              "ceil", "trunc", "round", "rint", "sign", "erf", "erfc")
#: Rank-generic elementwise ternaries (every non-rank-0 operand shares one shape).
ELEMENTWISE_TERNARY = ("fma", "clip")
#: Elementwise floating-point class tests: bool-typed, zero derivative.
PREDICATE = ("isnan", "isinf", "isfinite")
#: Elementwise comparisons: bool-typed, zero derivative.
COMPARE = ("lt", "le", "gt", "ge", "eq", "ne")
#: Boolean connectives (``lnot`` is unary): bool-typed, zero derivative.
LOGICAL = ("land", "lor", "lnot")
#: Rank-changing vector algebra. ``component`` carries a literal index,
#: ``splat`` a literal shape; ``component_at`` is ``component``'s
#: RUNTIME-index twin (a loop's symbolic ``k``), a separate KIND since its
#: arity, type rule and emission differ. ``set_component_at`` is its WRITE
#: twin: a VALUE, not a statement, so ``v[k] = e`` rebinds the name to it.
VECTOR_OPS = ("dot", "cross", "norm", "sum", "component", "component_at",
              "set_component_at", "vec", "splat")
#: Matrix algebra: matrix-vector, matrix-matrix, transpose, outer product.
MATRIX_OPS = ("mv", "mm", "transpose", "outer")
#: aether's stateless, counter-based draws: arity 2, ``(seed, counter)``,
#: both rank-0 integer-typed, lowered to aether's
#: ``random::detail::{uniform01,standardNormal}`` with the kernel's
#: flattened sample index as the stream's third coordinate. A re-drawn
#: value re-evaluates the SAME two integers, never RNG state, so a
#: captured graph replays by writing the wire, never recapturing.
RANDOM_OPS = ("random_uniform", "random_normal")

#: Every op kind a traced body can mint (``at``/``select`` mint their own
#: node subclasses but are KINDS here, since the rule table keys on kind).
OP_KINDS = (ELEMENTWISE_BINARY + UNARY_MATH + COMPARE + LOGICAL + VECTOR_OPS
            + MATRIX_OPS + QUAT_OPS + RANDOM_OPS + ("select", "dispatch", "at")
            + ELEMENTWISE_TERNARY + PREDICATE)

_DTYPE_RANK = {"bool": 0, "i32": 1, "i64": 2, "f32": 3, "f64": 4}
#: Kinds under which a semantic tag survives (a quaternion sum is a quaternion).
_TAG_PRESERVING = ("add", "sub", "neg", "mul", "div", "select", "dispatch")


def promote(a: str, b: str) -> str:
    """The dtype of an elementwise combination of ``a`` and ``b``."""
    for d in (a, b):
        if d not in _DTYPE_RANK:
            raise HawkError(
                f"unknown dtype {d!r}; declared are {tuple(_DTYPE_RANK)}")
    return a if _DTYPE_RANK[a] >= _DTYPE_RANK[b] else b


def make(kind: str, operands: Sequence[Node], literal: object = None) -> Op:
    """Mint an :class:`~hawk.ir.nodes.Op` of ``kind`` with its inferred type."""
    operands = tuple(operands)
    node = Op(kind, operands, result_type(kind, operands, literal), literal)
    return node


def result_type(kind: str, operands: Sequence[Node],
                literal: object = None) -> TensorType:
    """The :class:`~hawk.types.TensorType` ``kind`` returns over ``operands``."""
    ts = [o.ttype for o in operands]
    if kind not in OP_KINDS:
        raise HawkError(
            f"unknown op kind {kind!r}; the vocabulary is {OP_KINDS}"
        )
    _arity(kind, len(ts))
    binary_logical = kind in LOGICAL and kind != "lnot"
    if kind in ELEMENTWISE_BINARY or kind in COMPARE or binary_logical:
        shape = _broadcast(kind, ts[0], ts[1])
        boolean = kind in COMPARE or kind in LOGICAL
        dtype = "bool" if boolean else promote(ts[0].dtype, ts[1].dtype)
        return TensorType(shape, dtype, _tag(kind, ts))
    if kind in UNARY_MATH or kind == "lnot" or kind in PREDICATE:
        dtype = "bool" if kind == "lnot" or kind in PREDICATE else ts[0].dtype
        return TensorType(ts[0].shape, dtype, _tag(kind, ts))
    if kind in ELEMENTWISE_TERNARY:
        shape = _broadcast(kind, _broadcast_type(kind, ts[0], ts[1]), ts[2])
        dtype = promote(promote(ts[0].dtype, ts[1].dtype), ts[2].dtype)
        return TensorType(shape, dtype)
    if kind == "select":
        if ts[0].dtype != "bool":
            raise HawkError(f"select: the condition must be bool-typed, got {ts[0]!r}")
        if ts[1] != ts[2]:
            raise HawkError(f"select: the branches disagree — {ts[1]!r} vs {ts[2]!r}")
        return ts[1]
    if kind == "dispatch":
        if ts[0].shape != () or ts[0].dtype != "i32":
            raise HawkError(
                f"dispatch: kind must be a rank-0 i32 value, got {ts[0]!r}")
        for t in ts[2:]:
            if t != ts[1]:
                raise HawkError(
                    f"dispatch: the branches disagree — {ts[1]!r} vs {t!r}")
        return ts[1]
    if kind == "at":
        return ts[0]
    if kind in RANDOM_OPS:
        for pname, t in zip(("seed", "counter"), ts):
            if t.shape != () or t.dtype not in ("i32", "i64"):
                raise HawkError(
                    f"{kind}: {pname} must be a rank-0 i32/i64 value (a literal, a "
                    f"by-value uniform wire, or a plane read), got {t!r}")
        # typed like a bare float literal (f64), so the draw composes with
        # the promotion ladder like `1.0` would; the compiled scalar MODE
        # widens it at emission time, never here.
        return TensorType((), "f64")
    return _shaped(kind, ts, literal)


def _shaped(kind: str, ts: list[TensorType], literal: object) -> TensorType:
    """The rank-CHANGING kinds: vector algebra, matrix algebra, quaternions."""
    if kind == "dot":
        _same(kind, ts[0], ts[1], rank=1)
        return TensorType((), promote(ts[0].dtype, ts[1].dtype))
    if kind == "cross":
        _same(kind, ts[0], ts[1], rank=1, extent=3)
        return TensorType((3,), promote(ts[0].dtype, ts[1].dtype))
    if kind == "norm":
        _rank(kind, ts[0], 1)
        return TensorType((), ts[0].dtype)
    if kind == "sum":
        if len(ts[0].shape) not in (1, 2):
            raise HawkError(f"sum: expected a rank-1 or rank-2 operand, got {ts[0]!r}")
        return TensorType((), ts[0].dtype)
    if kind == "splat":
        _rank(kind, ts[0], 0)
        if not isinstance(literal, tuple) or not literal:
            raise HawkError(
                f"splat: the literal must be a target shape, got {literal!r}")
        return TensorType(literal, ts[0].dtype)
    if kind == "component":
        if isinstance(literal, tuple):
            # a matrix entry: ``(row, col)``, each inside its static extent
            _rank(kind, ts[0], 2)
            index, extent = literal, ts[0].shape
        else:
            _rank(kind, ts[0], 1)
            index, extent = (literal,), ts[0].shape
        if len(index) != len(extent) or not all(
                isinstance(k, int) and not isinstance(k, bool) and 0 <= k < e
                for k, e in zip(index, extent)):
            raise HawkError(
                f"component: index {literal!r} is out of the static extent "
                f"{extent if len(extent) > 1 else extent[0]} (an extent is a type, "
                "not a runtime value)"
            )
        return TensorType((), ts[0].dtype)
    if kind == "component_at":
        _rank(kind, ts[0], 1)
        if ts[1].shape != () or ts[1].dtype not in ("i32", "i64"):
            raise HawkError(
                f"component_at: the index must be a rank-0 integer value, got "
                f"{ts[1]!r}. A runtime component index is an ADDRESS — typed "
                "f64 it would reach the emitter as a Real cast into a "
                "std::size_t (the same rule hawk.trace.value.index_node applies "
                "to a table subscript)")
        return TensorType((), ts[0].dtype)
    if kind == "set_component_at":
        _rank(kind, ts[0], 1)
        if ts[1].shape != () or ts[1].dtype not in ("i32", "i64"):
            raise HawkError(
                f"set_component_at: the index must be a rank-0 integer value, "
                f"got {ts[1]!r} (see component_at)")
        _rank(kind, ts[2], 0)
        return TensorType(ts[0].shape, promote(ts[0].dtype, ts[2].dtype),
                          ts[0].tag)
    if kind == "vec":
        for t in ts:
            _rank(kind, t, 0)
        dtype = ts[0].dtype
        for t in ts[1:]:
            dtype = promote(dtype, t.dtype)
        if literal is None:
            return TensorType((len(ts),), dtype)
        # a matrix assembled from its entries, row-major
        if (not isinstance(literal, tuple) or len(literal) != 2
                or literal[0] * literal[1] != len(ts)):
            raise HawkError(
                f"vec: a matrix shape literal must be (rows, cols) with rows * cols "
                f"== {len(ts)} entries, got {literal!r}")
        return TensorType(literal, dtype)
    if kind == "mv":
        _rank(kind, ts[0], 2), _rank(kind, ts[1], 1)
        if ts[0].shape[1] != ts[1].shape[0]:
            raise HawkError(f"mv: {ts[0].shape} does not contract with {ts[1].shape}")
        return TensorType((ts[0].shape[0],), promote(ts[0].dtype, ts[1].dtype))
    if kind == "mm":
        _rank(kind, ts[0], 2), _rank(kind, ts[1], 2)
        if ts[0].shape[1] != ts[1].shape[0]:
            raise HawkError(f"mm: {ts[0].shape} does not contract with {ts[1].shape}")
        return TensorType((ts[0].shape[0], ts[1].shape[1]),
                          promote(ts[0].dtype, ts[1].dtype))
    if kind == "outer":
        _rank(kind, ts[0], 1), _rank(kind, ts[1], 1)
        return TensorType((ts[0].shape[0], ts[1].shape[0]),
                          promote(ts[0].dtype, ts[1].dtype))
    if kind == "transpose":
        _rank(kind, ts[0], 2)
        return TensorType((ts[0].shape[1], ts[0].shape[0]), ts[0].dtype)
    return _quat(kind, ts)


def _quat(kind: str, ts: list[TensorType]) -> TensorType:
    """The quaternion ops; the tag is mandatory on every 4-wide result."""
    quat = TensorType((4,), ts[0].dtype, "quaternion")
    if kind in ("quat_mul", "quat_conj", "quat_recip", "as_pure"):
        want = 3 if kind == "as_pure" else 4
        for t in ts:
            _rank(kind, t, 1, extent=want)
        return quat
    if kind == "quat_rotate":
        _rank(kind, ts[0], 1, extent=4), _rank(kind, ts[1], 1, extent=3)
        return TensorType((3,), promote(ts[0].dtype, ts[1].dtype))
    _rank(kind, ts[0], 1, extent=4)          # as_vec3
    return TensorType((3,), ts[0].dtype)


_ARITY = {"lnot": 1, "select": 3, "fma": 3, "clip": 3,
          "isnan": 1, "isinf": 1, "isfinite": 1,
          "at": 2, "component": 1, "transpose": 1,
          "component_at": 2, "set_component_at": 3,
          "norm": 1, "sum": 1, "splat": 1, "quat_conj": 1, "quat_recip": 1,
          "as_pure": 1, "as_vec3": 1}


def _arity(kind: str, n: int) -> None:
    want = _ARITY.get(kind, 1 if kind in UNARY_MATH else 2)
    if kind == "vec":
        if n == 0:
            raise HawkError("vec: at least one component is required")
        return
    if kind == "dispatch":
        if n < 3:
            raise HawkError(
                f"dispatch: a kind plus at least 2 branches is required, "
                f"got {n} operand(s)")
        return
    if n != want:
        raise HawkError(f"op {kind!r} takes {want} operand(s), got {n}")


def _broadcast(kind: str, a: TensorType, b: TensorType) -> tuple[int, ...]:
    if a.shape == b.shape or b.shape == ():
        return a.shape
    if a.shape == ():
        return b.shape
    raise HawkError(
        f"op {kind!r}: shapes {a.shape} and {b.shape} do not broadcast — an "
        "elementwise op takes equal shapes or one rank-0 operand"
    )


def _broadcast_type(kind: str, a: TensorType, b: TensorType) -> TensorType:
    """The shape ``a`` and ``b`` broadcast to, as a type the next fold can take."""
    return TensorType(_broadcast(kind, a, b), "f64")


def _tag(kind: str, ts: list[TensorType]) -> str | None:
    """A tag survives only where a tagged value stays that thing."""
    tags = {t.tag for t in ts if t.shape != ()}
    if kind not in _TAG_PRESERVING or len(tags) != 1:
        return None
    if kind in ("mul", "div") and all(t.shape != () for t in ts):
        return None                     # elementwise product of two vectors
    return tags.pop()


def _rank(kind: str, t: TensorType, rank: int, extent: int | None = None) -> None:
    if len(t.shape) != rank or (extent is not None and t.shape[0] != extent):
        raise HawkError(
            f"op {kind!r}: operand typed {t!r} is not the expected rank-{rank}"
            + (f" extent-{extent}" if extent is not None else "") + " value"
        )


def _same(kind: str, a: TensorType, b: TensorType, rank: int,
          extent: int | None = None) -> None:
    _rank(kind, a, rank, extent), _rank(kind, b, rank, extent)
    if a.shape != b.shape:
        raise HawkError(f"op {kind!r}: operand shapes {a.shape} and {b.shape} disagree")
