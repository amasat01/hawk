# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The derivative rule table: ONE :class:`DiffRule` per node KIND.

Keyed on ``(kind, rank)`` with a rank-generic entry under ``(kind, None)``,
so a tensor-typed IR needs one rule where a per-class table needed several.
A VJP rule returns the reverse contributions ``((child, adjoint), ...)``
for one node; a JVP rule returns that node's tangent given ``t(child)``;
both build derivative IR with the same node constructors as a primal, so
the result goes through the same canonical walk. The table is
closed-world: a kind with no derivative carries an explicit zero rule
(``mode="zero"`` — comparisons, boolean connectives) or delegates to the
transform's cross-sample special case (``mode="walker"`` — ``at``, whose
reverse is a scatter-add), never absent, so :func:`rule_for` raising
names a kind that genuinely has no rule instead of silently contributing
zero.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..ir import Const, HawkError, Node, Select
from ..ir import make as _mk
from ..ir.ops import COMPARE, LOGICAL, PREDICATE, RANDOM_OPS
from ..types import TensorType

_F64 = TensorType((), "f64")


@dataclass(frozen=True)
class DiffRule:
    """One node kind's two derivative directions."""

    vjp: Callable[..., Any] | None = None
    jvp: Callable[..., Any] | None = None
    mode: str = "rule"


def _o(kind: str, *args: Node, literal: Any = None) -> Node:
    return _mk(kind, args, literal)


def _add(a: Node, b: Node) -> Node:
    return _o("add", a, b)


def _mul(a: Node, b: Node) -> Node:
    return _o("mul", a, b)


def _div(a: Node, b: Node) -> Node:
    return _o("div", a, b)


def _neg(a: Node) -> Node:
    return _o("neg", a)


def _c(x: float) -> Node:
    return Const(float(x), _F64)


def zero_like(t: TensorType) -> Node:
    """A structural zero of type ``t`` (rank-0 literal, else a splat)."""
    z = Const(0.0, TensorType((), t.dtype))
    return z if t.shape == () else _o("splat", z, literal=t.shape)


def _sum(terms: tuple[Node | None, ...]) -> Node | None:
    live = [x for x in terms if x is not None]
    if not live:
        return None
    out = live[0]
    for x in live[1:]:
        out = _add(out, x)
    return out


def _zero_vjp(n: Node, g: Node) -> tuple:
    return ()


def _zero_jvp(n: Node, t: Callable[[Node], Node | None]) -> None:
    return None


# --------------------------------------------------------------------------
# elementwise binaries (rank-generic; the transform un-broadcasts)

_BIN_VJP: dict[str, Callable[..., tuple]] = {
    "add": lambda n, g, a, b: ((a, g), (b, g)),
    "sub": lambda n, g, a, b: ((a, g), (b, _neg(g))),
    "mul": lambda n, g, a, b: ((a, _mul(g, b)), (b, _mul(g, a))),
    "div": lambda n, g, a, b: ((a, _div(g, b)), (b, _neg(_div(_mul(g, n), b)))),
    "pow": lambda n, g, a, b: (
        (a, _mul(g, _mul(b, _o("pow", a, _o("sub", b, _c(1.0)))))),
        (b, _mul(g, _mul(n, _o("log", a)))),
    ),
    "min": lambda n, g, a, b: ((a, _pick(a, b, g, True)), (b, _pick(a, b, g, False))),
    "max": lambda n, g, a, b: ((a, _pick(b, a, g, True)), (b, _pick(b, a, g, False))),
    # y = atan2(a, b) with a the numerator: da = b/(a^2+b^2), db = -a/(a^2+b^2)
    "atan2": lambda n, g, a, b: (
        (a, _div(_mul(g, b), _hypot2(a, b))),
        (b, _neg(_div(_mul(g, a), _hypot2(a, b)))),
    ),
}

_BIN_JVP: dict[str, Callable[..., tuple]] = {
    "add": lambda n, ta, tb, a, b: (ta, tb),
    "sub": lambda n, ta, tb, a, b: (ta, _neg(tb) if tb is not None else None),
    "mul": lambda n, ta, tb, a, b: (_mul(ta, b) if ta is not None else None,
                                    _mul(a, tb) if tb is not None else None),
    "div": lambda n, ta, tb, a, b: (
        _div(ta, b) if ta is not None else None,
        _neg(_div(_mul(n, tb), b)) if tb is not None else None,
    ),
    "pow": lambda n, ta, tb, a, b: (
        _mul(ta, _mul(b, _o("pow", a, _o("sub", b, _c(1.0)))))
        if ta is not None else None,
        _mul(tb, _mul(n, _o("log", a))) if tb is not None else None,
    ),
    "min": lambda n, ta, tb, a, b: (
        _pick(a, b, ta, True) if ta is not None else None,
        _pick(a, b, tb, False) if tb is not None else None),
    "max": lambda n, ta, tb, a, b: (
        _pick(b, a, ta, True) if ta is not None else None,
        _pick(b, a, tb, False) if tb is not None else None),
    "atan2": lambda n, ta, tb, a, b: (
        _div(_mul(ta, b), _hypot2(a, b)) if ta is not None else None,
        _neg(_div(_mul(tb, a), _hypot2(a, b))) if tb is not None else None,
    ),
}


def _sel(cond: Node, a: Node, b: Node) -> Node:
    """``cond ? a : b`` typed like ``a`` (``b`` is a structural zero or a constant)."""
    return Select(cond, a, b, a.ttype)


def _ramp(cond: Node, like: Node) -> Node:
    """``1`` where ``cond`` holds, ``0`` elsewhere, typed like ``like``'s scalar."""
    t = TensorType((), like.ttype.dtype)
    return Select(cond, Const(1.0, t), Const(0.0, t), t)


def _safe_div(num: Node, den: Node) -> Node:
    """``num / den``, zero where ``den`` is zero (``hypot``'s origin)."""
    q = _div(num, den)
    return _sel(_o("eq", den, zero_like(den.ttype)), zero_like(q.ttype), q)


#: ``(d/da, d/db)`` of the binaries added for numpy parity; ``None`` = no
#: dependence. Subgradients follow JAX (``fmod``/``remainder`` are piecewise
#: linear in both operands; ``copysign`` is ``|a|`` times a sign-of-``b``
#: constant, with ``abs``'s ``+1`` at zero), except where noted in
#: :mod:`hawk.math`.
_PARTIALS: dict[str, Callable[..., tuple]] = {
    "hypot": lambda n, a, b: (_safe_div(a, n), _safe_div(b, n)),
    "copysign": lambda n, a, b: (
        _mul(_o("copysign", _c(1.0), b), _sel(_o("ge", a, zero_like(a.ttype)),
                                              _c(1.0), _c(-1.0))), None),
    "fmod": lambda n, a, b: (_c(1.0), _neg(_o("trunc", _div(a, b)))),
    "remainder": lambda n, a, b: (_c(1.0), _neg(_o("floor", _div(a, b)))),
    "fdim": lambda n, a, b: (_ramp(_o("gt", a, b), a), _neg(_ramp(_o("gt", a, b), a))),
}


def _partials_vjp(kind: str):
    def fn(n, g, a, b):
        da, db = _PARTIALS[kind](n, a, b)
        return tuple((x, _mul(g, d)) for x, d in ((a, da), (b, db)) if d is not None)
    return fn


def _partials_jvp(kind: str):
    def fn(n, ta, tb, a, b):
        da, db = _PARTIALS[kind](n, a, b)
        return (_mul(ta, da) if ta is not None and da is not None else None,
                _mul(tb, db) if tb is not None and db is not None else None)
    return fn


_BIN_VJP.update({k: _partials_vjp(k) for k in _PARTIALS})
_BIN_JVP.update({k: _partials_jvp(k) for k in _PARTIALS})


def _fma_vjp(n: Node, g: Node) -> tuple:
    a, b, c = n.operands
    return ((a, _mul(g, b)), (b, _mul(g, a)), (c, g))


def _fma_jvp(n: Node, t) -> Node | None:
    a, b, c = n.operands
    ta, tb, tc = t(a), t(b), t(c)
    return _sum((_mul(ta, b) if ta is not None else None,
                 _mul(a, tb) if tb is not None else None, tc))


def _clip_masks(n: Node) -> tuple:
    """Which operand ``clip(x, lo, hi) = min(max(x, lo), hi)`` passes through:
    ``x`` on the CLOSED interval (torch's ``clamp`` rule; JAX splits a tie
    0.5/0.5), ``hi`` above it or whenever ``lo > hi``, ``lo`` below it;
    a NaN ``x`` passes nothing. The three masks partition every ordered case."""
    x, lo, hi = n.operands
    inside = _o("land", _o("ge", x, lo), _o("le", x, hi))
    above = _o("lor", _o("gt", x, hi), _o("gt", lo, hi))
    below = _o("land", _o("lt", x, lo), _o("le", lo, hi))
    return ((x, inside), (lo, below), (hi, above))


def _clip_vjp(n: Node, g: Node) -> tuple:
    return tuple((child, _sel(mask, g, zero_like(g.ttype)))
                 for child, mask in _clip_masks(n))


def _clip_jvp(n: Node, t) -> Node | None:
    terms = []
    for child, mask in _clip_masks(n):
        tc = t(child)
        terms.append(None if tc is None else _sel(mask, tc, zero_like(tc.ttype)))
    return _sum(tuple(terms))


def _bilinear(kind: str):
    """The forward rule of every BILINEAR op — ``d(a op b) = ta op b + a op tb``
    — shared by dot, cross, mv, mm, outer and the quaternion product."""
    def fn(n: Node, t) -> Node | None:
        a, b = n.operands
        ta, tb = t(a), t(b)
        return _sum((_o(kind, ta, b) if ta is not None else None,
                     _o(kind, a, tb) if tb is not None else None))
    return fn


def _hypot2(a: Node, b: Node) -> Node:
    return _add(_mul(a, a), _mul(b, b))


def _entrywise(kind: str, *nodes: Node) -> Node:
    """``kind`` (a comparison) over its operands: one op when every operand is
    rank-0, else mapped entry by entry and re-assembled — aether spells a
    comparison at rank 0 only, so a vector or matrix mask is a ``vec`` of them
    (as :func:`hawk.trace.value._elementwise` builds one)."""
    shape = next((x.ttype.shape for x in nodes if x.ttype.shape), ())
    if not shape:
        return _o(kind, *nodes)
    return _vec_like([_o(kind, *[x if not x.ttype.shape
                                 else _o("component", x, literal=k) for x in nodes])
                      for k in _entries(shape)], shape)


def _pick(lo: Node, hi: Node, g: Node, first: bool) -> Node:
    """``g`` on the branch ``lo <= hi`` selects, a structural zero on the other.
    A rank-0 ``g`` (the tangent of a broadcast operand) widens to the mask's
    shape first."""
    cond = _entrywise("le", lo, hi)
    if cond.ttype.shape and not g.ttype.shape:
        g = _o("splat", g, literal=cond.ttype.shape)
    z = zero_like(g.ttype)
    return Select(cond, g, z, g.ttype) if first else Select(cond, z, g, g.ttype)


def _abs_d(n: Node, x: Node) -> Node:
    """``+1`` where ``x >= 0`` (zero included, as JAX), ``-1`` elsewhere."""
    if not x.ttype.shape:
        return Select(_o("ge", x, zero_like(x.ttype)), _c(1.0), _c(-1.0), _F64)
    t = TensorType(x.ttype.shape, "f64")
    return Select(_entrywise("ge", x, _c(0.0)), _o("splat", _c(1.0), literal=t.shape),
                  _o("splat", _c(-1.0), literal=t.shape), t)


def _bin_vjp(n: Node, g: Node) -> tuple:
    a, b = n.operands
    return _BIN_VJP[n.kind](n, g, a, b)


def _bin_jvp(n: Node, t: Callable[[Node], Node | None]) -> Node | None:
    a, b = n.operands
    return _sum(_BIN_JVP[n.kind](n, t(a), t(b), a, b))


# --------------------------------------------------------------------------
# elementwise unaries (rank-generic)

_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)

#: ``d/dx op(x)`` as a factor multiplying the incoming adjoint/tangent;
#: ``n`` is the primal node, reused so the emitter's CSE keeps one copy.
_UNARY_D: dict[str, Callable[[Node, Node], Node]] = {
    "neg": lambda n, x: _c(-1.0),
    "abs": _abs_d,
    "sqrt": lambda n, x: _div(_c(0.5), n),
    "rsqrt": lambda n, x: _neg(_div(_mul(_c(0.5), n), x)),
    "exp": lambda n, x: n,
    "log": lambda n, x: _div(_c(1.0), x),
    "sin": lambda n, x: _o("cos", x),
    "cos": lambda n, x: _neg(_o("sin", x)),
    "tan": lambda n, x: _add(_c(1.0), _mul(n, n)),
    "tanh": lambda n, x: _o("sub", _c(1.0), _mul(n, n)),
    "asin": lambda n, x: _div(_c(1.0), _o("sqrt", _o("sub", _c(1.0), _mul(x, x)))),
    "acos": lambda n, x: _neg(_div(_c(1.0),
                                   _o("sqrt", _o("sub", _c(1.0), _mul(x, x))))),
    "atan": lambda n, x: _div(_c(1.0), _add(_c(1.0), _mul(x, x))),
    "exp2": lambda n, x: _mul(n, _c(math.log(2.0))),
    "expm1": lambda n, x: _add(n, _c(1.0)),
    "log2": lambda n, x: _div(_c(1.0 / math.log(2.0)), x),
    "log10": lambda n, x: _div(_c(1.0 / math.log(10.0)), x),
    "log1p": lambda n, x: _div(_c(1.0), _add(_c(1.0), x)),
    "cbrt": lambda n, x: _div(_c(1.0 / 3.0), _mul(n, n)),
    "sinh": lambda n, x: _o("cosh", x),
    "cosh": lambda n, x: _o("sinh", x),
    "asinh": lambda n, x: _div(_c(1.0), _o("hypot", x, _c(1.0))),
    "acosh": lambda n, x: _div(_c(1.0), _mul(_o("sqrt", _o("sub", x, _c(1.0))),
                                             _o("sqrt", _add(x, _c(1.0))))),
    "atanh": lambda n, x: _div(_c(1.0), _o("sub", _c(1.0), _mul(x, x))),
    "erf": lambda n, x: _mul(_c(_TWO_OVER_SQRT_PI), _o("exp", _neg(_mul(x, x)))),
    "erfc": lambda n, x: _mul(_c(-_TWO_OVER_SQRT_PI), _o("exp", _neg(_mul(x, x)))),
}


def _unary_vjp(n: Node, g: Node) -> tuple:
    x = n.operands[0]
    return ((x, _mul(g, _UNARY_D[n.kind](n, x))),)


def _unary_jvp(n: Node, t: Callable[[Node], Node | None]) -> Node | None:
    x = n.operands[0]
    return _mul(t(x), _UNARY_D[n.kind](n, x))


# --------------------------------------------------------------------------
# vector / matrix algebra and the quaternion ops


def _qc(x: Node) -> Node:
    return _o("quat_conj", x)


def _qm(a: Node, b: Node) -> Node:
    return _o("quat_mul", a, b)


def _entries(shape: tuple) -> list:
    """Every static index of ``shape``, row-major: the ``component`` literal
    of each entry (an ``int`` for a vector, ``(row, col)`` for a matrix)."""
    if len(shape) == 1:
        return list(range(shape[0]))
    return [(r, c) for r in range(shape[0]) for c in range(shape[1])]


def _vec_like(parts: list, shape: tuple) -> Node:
    """``parts`` (rank-0, row-major) assembled into a value of ``shape``."""
    return _o("vec", *parts, literal=shape if len(shape) == 2 else None)


def _component_vjp(n: Node, g: Node) -> tuple:
    v = n.operands[0]
    k = n.literal
    z = zero_like(TensorType((), v.ttype.dtype))
    shape = v.ttype.shape
    return ((v, _vec_like([g if i == k else z for i in _entries(shape)], shape)),)


def _component_at_vjp(n: Node, g: Node) -> tuple:
    """The reverse of a RUNTIME-indexed component read.

    ``v[k]`` with a symbolic ``k`` selects one component at a row only the
    running kernel knows, so its adjoint is ``g`` placed in that row and
    zero elsewhere — spelled with ops the table already carries
    (``vec(select(k == 0, 1, 0), ...) * g``) rather than a new node: a
    rank-1 value's fixed, small extent makes ``W`` selects a cost the
    compiler folds, and a purpose-built node would need an aether spelling
    that is a STATEMENT, putting the emitter's scope rules into this
    table."""
    v, index = n.operands
    scalar = TensorType((), v.ttype.dtype)
    one, zero = Const(1.0, scalar), Const(0.0, scalar)
    hot = _o("vec", *[
        Select(_o("eq", index, Const(i, index.ttype)), one, zero, scalar)
        for i in range(v.ttype.shape[0])])
    return ((v, _mul(hot, g)),)


def _component_at_jvp(n: Node, t: Callable[[Node], Node | None]) -> Node | None:
    """The tangent of ``v[k]`` is the same component of ``v``'s tangent.
    The index carries none, so a missing vector tangent is a constant."""
    tangent = t(n.operands[0])
    if tangent is None:
        return None
    return _o("component_at", tangent, n.operands[1])


def _set_component_at_vjp(n: Node, g: Node) -> tuple:
    """The reverse of ``v`` with component ``k`` replaced by ``e``:
    replacement, not accumulation. ``v``'s adjoint is ``g`` with that
    component zeroed, and ``e``'s adjoint is ``g[k]`` — a write at ``k``
    reverses into a read at ``k``."""
    v, index, value = n.operands
    zero = Const(0.0, TensorType((), v.ttype.dtype))
    return ((v, _o("set_component_at", g, index, zero)),
            (value, _o("component_at", g, index)))


def _set_component_at_jvp(n: Node, t: Callable[[Node], Node | None]) -> Node | None:
    """The tangent of a replacement is the replacement of tangents. A
    missing tangent on either side is a structural zero, not an absent
    one — the other side's tangent still has to flow."""
    v, index, value = n.operands
    tv, tvalue = t(v), t(value)
    if tv is None and tvalue is None:
        return None
    return _o("set_component_at",
              tv if tv is not None else zero_like(v.ttype), index,
              tvalue if tvalue is not None else zero_like(value.ttype))


def _vec_vjp(n: Node, g: Node) -> tuple:
    return tuple((c, _o("component", g, literal=i))
                 for i, c in zip(_entries(n.ttype.shape), n.operands))


def _vec_jvp(n: Node, t: Callable[[Node], Node | None]) -> Node | None:
    parts = [t(c) if t(c) is not None else zero_like(c.ttype) for c in n.operands]
    return _vec_like(parts, n.ttype.shape)


def _select_vjp(n: Node, g: Node) -> tuple:
    cond, a, b = n.operands
    z = zero_like(g.ttype)
    return ((a, Select(cond, g, z, g.ttype)), (b, Select(cond, z, g, g.ttype)))


def _select_jvp(n: Node, t: Callable[[Node], Node | None]) -> Node | None:
    cond, a, b = n.operands
    ta = t(a) if t(a) is not None else zero_like(a.ttype)
    tb = t(b) if t(b) is not None else zero_like(b.ttype)
    return Select(cond, ta, tb, a.ttype)


def _rotate_vjp(n: Node, g: Node) -> tuple:
    """``y = as_vec3(q (0,v) conj(q))``; both adjoints by the sandwich transpose."""
    q, v = n.operands
    pure_v, pure_g = _o("as_pure", v), _o("as_pure", g)
    dq = _add(_qm(_qm(pure_g, q), _qc(pure_v)), _qm(_qm(_qc(pure_g), q), pure_v))
    dv = _o("as_vec3", _qm(_qm(_qc(q), pure_g), q))
    return ((q, dq), (v, dv))


def _rotate_jvp(n: Node, t: Callable[[Node], Node | None]) -> Node | None:
    q, v = n.operands
    tq, tv = t(q), t(v)
    pure_v = _o("as_pure", v)
    terms = []
    if tq is not None:
        terms.append(_qm(_qm(tq, pure_v), _qc(q)))
        terms.append(_qm(_qm(q, pure_v), _qc(tq)))
    if tv is not None:
        terms.append(_qm(_qm(q, _o("as_pure", tv)), _qc(q)))
    return _o("as_vec3", _sum(tuple(terms)))


#: ``y = a^-1`` is a division-algebra inverse: ``dy = -y da y``.
def _recip_vjp(n: Node, g: Node) -> tuple:
    a = n.operands[0]
    return ((a, _neg(_qm(_qm(_qc(n), g), _qc(n)))),)


def _recip_jvp(n: Node, t: Callable[[Node], Node | None]) -> Node | None:
    return _neg(_qm(_qm(n, t(n.operands[0])), n))


_SHAPED: dict[str, DiffRule] = {
    "dot": DiffRule(
        vjp=lambda n, g: ((n.operands[0], _mul(g, n.operands[1])),
                          (n.operands[1], _mul(g, n.operands[0]))),
        jvp=_bilinear("dot"),
    ),
    "cross": DiffRule(
        vjp=lambda n, g: ((n.operands[0], _o("cross", n.operands[1], g)),
                          (n.operands[1], _o("cross", g, n.operands[0]))),
        jvp=_bilinear("cross"),
    ),
    "norm": DiffRule(
        vjp=lambda n, g: ((n.operands[0], _mul(_div(g, n), n.operands[0])),),
        jvp=lambda n, t: _div(_o("dot", n.operands[0], t(n.operands[0])), n),
    ),
    "sum": DiffRule(
        vjp=lambda n, g: ((n.operands[0],
                           _o("splat", g, literal=n.operands[0].ttype.shape)),),
        jvp=lambda n, t: _o("sum", t(n.operands[0])),
    ),
    "splat": DiffRule(
        vjp=lambda n, g: ((n.operands[0], _o("sum", g)),),
        jvp=lambda n, t: _o("splat", t(n.operands[0]), literal=n.ttype.shape),
    ),
    "component": DiffRule(
        vjp=_component_vjp,
        jvp=lambda n, t: _o("component", t(n.operands[0]), literal=n.literal),
    ),
    "component_at": DiffRule(vjp=_component_at_vjp, jvp=_component_at_jvp),
    "set_component_at": DiffRule(vjp=_set_component_at_vjp,
                                 jvp=_set_component_at_jvp),
    "vec": DiffRule(vjp=_vec_vjp, jvp=_vec_jvp),
    "mv": DiffRule(
        vjp=lambda n, g: ((n.operands[0], _o("outer", g, n.operands[1])),
                          (n.operands[1], _o("mv", _o("transpose", n.operands[0]), g))),
        jvp=_bilinear("mv"),
    ),
    "mm": DiffRule(
        vjp=lambda n, g: (
            (n.operands[0], _o("mm", g, _o("transpose", n.operands[1]))),
            (n.operands[1], _o("mm", _o("transpose", n.operands[0]), g)),
        ),
        jvp=_bilinear("mm"),
    ),
    "outer": DiffRule(
        vjp=lambda n, g: ((n.operands[0], _o("mv", g, n.operands[1])),
                          (n.operands[1], _o("mv", _o("transpose", g), n.operands[0]))),
        jvp=_bilinear("outer"),
    ),
    "transpose": DiffRule(
        vjp=lambda n, g: ((n.operands[0], _o("transpose", g)),),
        jvp=lambda n, t: _o("transpose", t(n.operands[0])),
    ),
    "select": DiffRule(vjp=_select_vjp, jvp=_select_jvp),
    "quat_mul": DiffRule(
        vjp=lambda n, g: ((n.operands[0], _qm(g, _qc(n.operands[1]))),
                          (n.operands[1], _qm(_qc(n.operands[0]), g))),
        jvp=_bilinear("quat_mul"),
    ),
    "quat_conj": DiffRule(
        vjp=lambda n, g: ((n.operands[0], _qc(g)),),
        jvp=lambda n, t: _qc(t(n.operands[0])),
    ),
    "quat_recip": DiffRule(vjp=_recip_vjp, jvp=_recip_jvp),
    "quat_rotate": DiffRule(vjp=_rotate_vjp, jvp=_rotate_jvp),
    "as_pure": DiffRule(
        vjp=lambda n, g: ((n.operands[0], _o("as_vec3", g)),),
        jvp=lambda n, t: _o("as_pure", t(n.operands[0])),
    ),
    "as_vec3": DiffRule(
        vjp=lambda n, g: ((n.operands[0], _o("as_pure", g)),),
        jvp=lambda n, t: _o("as_vec3", t(n.operands[0])),
    ),
}

#: The table proper: ``(kind, rank)`` -> rule, ``rank=None`` generic.
RULES: dict[tuple[str, int | None], DiffRule] = {
    **{(k, None): DiffRule(vjp=_bin_vjp, jvp=_bin_jvp) for k in _BIN_VJP},
    **{(k, None): DiffRule(vjp=_unary_vjp, jvp=_unary_jvp) for k in _UNARY_D},
    **{(k, None): DiffRule(vjp=_zero_vjp, jvp=_zero_jvp, mode="zero")
       # the rounding family and `sign` join COMPARE/LOGICAL in the
       # explicit-zero bucket: piecewise constant, so neither direction
       # propagates through it; the class tests are bool-typed.
       # RANDOM_OPS joins too (a draw is a function of integer operands
       # only), keeping the table closed-world even for a `wrt=` caller.
       for k in COMPARE + LOGICAL + RANDOM_OPS + PREDICATE
       + ("floor", "ceil", "trunc", "round", "rint", "sign")},
    ("fma", None): DiffRule(vjp=_fma_vjp, jvp=_fma_jvp),
    ("clip", None): DiffRule(vjp=_clip_vjp, jvp=_clip_jvp),
    **{(k, None): rule for k, rule in _SHAPED.items()},
    ("at", None): DiffRule(mode="walker"),
    #: A dispatch's two directions are the transform's own — K per-branch
    #: adjoints combined (VJP), a branch-wise mirror (JVP).
    ("dispatch", None): DiffRule(mode="walker"),
}


def rule_for(kind: str, rank: int) -> DiffRule:
    """The rule for one node kind: rank-specific first, then generic."""
    rule = RULES.get((kind, rank)) or RULES.get((kind, None))
    if rule is None:
        raise HawkError(
            f"no derivative rule for node kind {kind!r} (rank {rank}): HAWK's rule "
            f"table is closed-world — a kind with no derivative is registered "
            "with an explicit zero rule, never omitted, so this is a genuinely "
            "undifferentiable node, not a silent zero"
        )
    return rule

