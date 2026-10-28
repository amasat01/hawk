# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Derivative rules of the per-lane linear-algebra kinds (:mod:`hawk.ir.linalg`).

A factorisation node (``lu``, ``chol``, ``eigh``) is an identity carrier: its
tangent IS its input's tangent and the adjoint it collects IS its input's
adjoint. The consumers and projections hang the rules off it, read the same
factor node for the primal and for the derivative, and never factor again, so
a derivative holds one factorisation per call site.

``lu_solve``/``lu_det`` differentiate ``A`` as given. ``cho_solve``,
``chol_factor`` and the ``eigh`` projections differentiate the symmetric part
``S = (A + A^T) / 2`` they read, so a non-symmetric perturbation agrees with
a finite difference of the primal. The Cholesky factor follows
``dL = L Phi(L^-1 dS L^-T)`` with ``Phi`` the lower triangle at half the
diagonal, reversed by the matching rule; both are two triangular
multi-right-hand-side solves and one matrix product on the existing factor.

A ``chol_from`` factor is the one carrier of a different input: its consumers reduce to the
adjoint (tangent) of ``A = L L^T``, and it maps that to the lower triangle of the ``L`` the
program built (``dA = dL L^T + L dL^T``).

A failed lane (``info > 0``) has defined outputs of zero, so its derivatives
are zero: the Cholesky rules mask on ``info`` (their triangular solves would
divide by the partial factor), and an eigenvector derivative on a lane whose
neighbouring eigenvalues are too close (``info`` bit 2) is zero.
"""

from __future__ import annotations

from collections.abc import Callable

from ..ir import Const, Node, Select
from ..ir import make as _mk
from ..types import TensorType

_I32 = TensorType((), "int32")

Tangent = Callable[[Node], "Node | None"]


def _o(kind: str, *args: Node, literal=None) -> Node:
    return _mk(kind, args, literal)


def _c(x: float) -> Node:
    return Const(float(x), TensorType((), "f64"))


def _n(f: Node) -> int:
    return f.ttype.shape[0]


def _grid(f: Node, entry: Callable[[int, int], Node]) -> Node:
    """The ``n x n`` matrix whose ``(i, j)`` entry is ``entry(i, j)``."""
    n = _n(f)
    return _o("vec", *[entry(i, j) for i in range(n) for j in range(n)], literal=(n, n))


def _zeros(shape: tuple) -> Node:
    z = _c(0.0)
    return z if not shape else _o("splat", z, literal=shape)


def _eye(f: Node) -> Node:
    return _grid(f, lambda i, j: _c(1.0 if i == j else 0.0))


def _at(m: Node, i: int, j: int) -> Node:
    return _o("component", m, literal=(i, j))


def _t(m: Node) -> Node:
    return _o("transpose", m)


def _mm(a: Node, b: Node) -> Node:
    return _o("mm", a, b)


def _sym(m: Node) -> Node:
    """``(m + m^T) / 2``."""
    return _o("mul", _o("add", m, _t(m)), _c(0.5))


def _neg(m: Node) -> Node:
    return _o("neg", m)


def _times(a: Node, b: Node) -> Node:
    """``a b`` for a matrix ``a`` and a vector or matrix ``b``."""
    return _o("mv" if len(b.ttype.shape) == 1 else "mm", a, b)


def _outer(a: Node, b: Node) -> Node:
    """``a b^T`` for vectors, or for ``n x k`` matrices ``a`` and ``b``."""
    return _o("outer", a, b) if len(a.ttype.shape) == 1 else _mm(a, _t(b))


def _live(f: Node, kind: str, value: Node, *, below: int = 1) -> Node:
    """``value`` where the factor's ``info`` is under ``below`` (a failed
    lane has the defined output zero, so its derivative is zero), zero
    elsewhere."""
    ok = _o("lt", _o(kind, f), Const(below, _I32))
    return Select(ok, value, _zeros(value.ttype.shape), value.ttype)


# --------------------------------------------------------------------------
# the carriers

def _carry_vjp(n: Node, g: Node) -> tuple:
    return ((n.operands[0], g),)


def _carry_jvp(n: Node, t: Tangent) -> Node | None:
    return t(n.operands[0])


# --------------------------------------------------------------------------
# LU

def _lu_solve_vjp(n: Node, g: Node) -> tuple:
    f, b = n.operands
    lam = _o("lu_solve_t", f, g)
    return ((b, lam), (f, _neg(_outer(lam, n))))


def _lu_solve_jvp(n: Node, t: Tangent) -> Node | None:
    f, b = n.operands
    da, db = t(f), t(b)
    rhs = db
    if da is not None:
        drift = _neg(_times(da, n))
        rhs = drift if rhs is None else _o("add", rhs, drift)
    return _o("lu_solve", f, rhs)


def _lu_solve_t_vjp(n: Node, g: Node) -> tuple:
    f, b = n.operands
    lam = _o("lu_solve", f, g)
    return ((b, lam), (f, _neg(_outer(n, lam))))


def _lu_solve_t_jvp(n: Node, t: Tangent) -> Node | None:
    f, b = n.operands
    da, db = t(f), t(b)
    rhs = db
    if da is not None:
        drift = _neg(_times(_t(da), n))
        rhs = drift if rhs is None else _o("add", rhs, drift)
    return _o("lu_solve_t", f, rhs)


def _lu_det_vjp(n: Node, g: Node) -> tuple:
    f = n.operands[0]
    inverse_t = _o("lu_solve_t", f, _eye(f))
    return ((f, _o("mul", _o("mul", g, n), inverse_t)),)


def _lu_det_jvp(n: Node, t: Tangent) -> Node | None:
    f = n.operands[0]
    solved = _o("lu_solve", f, t(f))
    trace = _at(solved, 0, 0)
    for i in range(1, _n(f)):
        trace = _o("add", trace, _at(solved, i, i))
    return _o("mul", n, trace)


# --------------------------------------------------------------------------
# Cholesky

def _lower(f: Node) -> Node:
    return _o("chol_factor", f)


def _phi(f: Node) -> Node:
    """The mask taking a matrix to its lower triangle at half the diagonal."""
    return _grid(f, lambda i, j: _c(1.0 if i > j else 0.5 if i == j else 0.0))


def _chol_tangent(f: Node, da: Node) -> Node:
    """``dL = L Phi(L^-1 S L^-T)`` for ``S = sym(da)`` (zero on a failed lane)."""
    w = _o("tri_lower", f, _sym(da))
    x = _o("tri_lower", f, _t(w))
    return _live(f, "chol_info", _mm(_lower(f), _o("mul", x, _phi(f))))


def _chol_adjoint(f: Node, lbar: Node) -> Node:
    """The adjoint of ``A`` for the adjoint ``lbar`` of ``L``:
    ``sym(L^-T Phi(L^T lbar) L^-1)`` (zero on a failed lane). ``Phi(L^T lbar)``
    reads only the lower triangle of ``lbar``, the one ``L`` has."""
    p = _o("mul", _mm(_t(_lower(f)), lbar), _phi(f))
    y = _o("tri_upper_t", f, p)
    g = _t(_o("tri_upper_t", f, _t(y)))
    return _live(f, "chol_info", _sym(g))


def _chol_factor_vjp(n: Node, g: Node) -> tuple:
    f = n.operands[0]
    return ((f, _chol_adjoint(f, g)),)


def _chol_factor_jvp(n: Node, t: Tangent) -> Node | None:
    f = n.operands[0]
    return _chol_tangent(f, t(f))


def _tril(f: Node, m: Node) -> Node:
    """The lower triangle of ``m`` (entries above the diagonal are not read)."""
    return _grid(f, lambda i, j: _at(m, i, j) if j <= i else _c(0.0))


def _chol_from_vjp(n: Node, g: Node) -> tuple:
    """The consumers reduce to the adjoint ``g`` of ``A = L L^T``; ``dL`` enters as
    ``dL L^T + L dL^T``, so the adjoint of ``L`` is ``(g + g^T) L``, lower part."""
    f, lower = n, n.operands[0]
    return ((lower, _tril(f, _mm(_o("add", g, _t(g)), _lower(f)))),)


def _chol_from_jvp(n: Node, t: Tangent) -> Node | None:
    f, lower = n, n.operands[0]
    dl = t(lower)
    if dl is None:
        return None
    half = _mm(_tril(f, dl), _t(_lower(f)))
    return _o("add", half, _t(half))


def _cho_solve_vjp(n: Node, g: Node) -> tuple:
    f, b = n.operands
    lam = _o("cho_solve", f, g)
    return ((b, lam), (f, _neg(_sym(_outer(lam, n)))))


def _cho_solve_jvp(n: Node, t: Tangent) -> Node | None:
    f, b = n.operands
    da, db = t(f), t(b)
    rhs = db
    if da is not None:
        drift = _neg(_times(_sym(da), n))
        rhs = drift if rhs is None else _o("add", rhs, drift)
    return _o("cho_solve", f, rhs)


def _tri_lower_vjp(n: Node, g: Node) -> tuple:
    f, b = n.operands
    bbar = _live(f, "chol_info", _o("tri_upper_t", f, g))
    return ((b, bbar), (f, _chol_adjoint(f, _neg(_outer(bbar, n)))))


def _tri_lower_jvp(n: Node, t: Tangent) -> Node | None:
    f, b = n.operands
    da, db = t(f), t(b)
    rhs = db
    if da is not None:
        drift = _neg(_times(_chol_tangent(f, da), n))
        rhs = drift if rhs is None else _o("add", rhs, drift)
    return _live(f, "chol_info", _o("tri_lower", f, rhs))


def _tri_upper_t_vjp(n: Node, g: Node) -> tuple:
    f, b = n.operands
    bbar = _live(f, "chol_info", _o("tri_lower", f, g))
    return ((b, bbar), (f, _chol_adjoint(f, _neg(_outer(n, bbar)))))


def _tri_upper_t_jvp(n: Node, t: Tangent) -> Node | None:
    f, b = n.operands
    da, db = t(f), t(b)
    rhs = db
    if da is not None:
        drift = _neg(_times(_t(_chol_tangent(f, da)), n))
        rhs = drift if rhs is None else _o("add", rhs, drift)
    return _live(f, "chol_info", _o("tri_upper_t", f, rhs))


# --------------------------------------------------------------------------
# eigh

def _rotated(f: Node, da: Node) -> Node:
    """``V^T sym(da) V``."""
    v = _o("eigh_v", f)
    return _mm(_mm(_t(v), _sym(da)), v)


def _gaps(f: Node) -> Node:
    """``F_ij = 1 / (w_j - w_i)`` off the diagonal, zero on it."""
    w = _o("eigh_w", f)
    return _grid(f, lambda i, j: _c(0.0) if i == j else _o(
        "div", _c(1.0), _o("sub", _o("component", w, literal=j),
                           _o("component", w, literal=i))))


def _diag(f: Node, vector: Node) -> Node:
    return _grid(f, lambda i, j: _o("component", vector, literal=i) if i == j
                 else _c(0.0))


def _eigh_w_vjp(n: Node, g: Node) -> tuple:
    f = n.operands[0]
    v = _o("eigh_v", f)
    return ((f, _mm(_mm(v, _diag(f, g)), _t(v))),)


def _eigh_w_jvp(n: Node, t: Tangent) -> Node | None:
    f = n.operands[0]
    m = _rotated(f, t(f))
    return _o("vec", *[_at(m, i, i) for i in range(_n(f))])


def _eigh_v_vjp(n: Node, g: Node) -> tuple:
    f = n.operands[0]
    v = _o("eigh_v", f)
    inner = _o("mul", _gaps(f), _mm(_t(v), g))
    return ((f, _live(f, "eigh_info", _sym(_mm(_mm(v, inner), _t(v))), below=2)),)


def _eigh_v_jvp(n: Node, t: Tangent) -> Node | None:
    f = n.operands[0]
    v = _o("eigh_v", f)
    dv = _mm(v, _o("mul", _gaps(f), _rotated(f, t(f))))
    return _live(f, "eigh_info", dv, below=2)


# --------------------------------------------------------------------------
# QR least squares

def _qr_solve_vjp(n: Node, g: Node) -> tuple:
    f, r = n.operands
    j = f.operands[0]
    w = _o("qr_gram_solve", f, g)
    jw = _times(j, w)
    res = _o("sub", r, _times(j, n))
    rbar = _live(f, "qr_info", jw)
    return ((r, rbar), (f, _live(f, "qr_info", _o("sub", _outer(res, w), _outer(jw, n)))))


def _qr_solve_jvp(n: Node, t: Tangent) -> Node | None:
    """``dx = R^-1 R^-T (dJ^T res - J^T dJ x) + lstsq(J, dr)``: the first term by two triangular
    solves on the forward factor's ``R``, the right-hand side's by the least-squares solve itself."""
    f, r = n.operands
    j = f.operands[0]
    dj, dr = t(f), t(r)
    out = None
    if dj is not None:
        res = _o("sub", r, _times(j, n))
        rhs = _o("sub", _times(_t(dj), res), _times(_t(j), _times(dj, n)))
        out = _o("qr_gram_solve", f, rhs)
    if dr is not None:
        step = _o("qr_solve", f, dr)
        out = step if out is None else _o("add", out, step)
    return None if out is None else _live(f, "qr_info", out)


def _qr_gram_vjp(n: Node, g: Node) -> tuple:
    """``X = G^-1 B`` with ``G = J^T J``: ``B_bar = G^-1 X_bar = W`` and ``J_bar = -J (W X^T + X W^T)``."""
    f, b = n.operands
    j = f.operands[0]
    w = _o("qr_gram_solve", f, g)
    cross = _outer(w, n)
    return ((b, _live(f, "qr_info", w)),
            (f, _live(f, "qr_info", _neg(_times(j, _o("add", cross, _t(cross)))))))


def _qr_gram_jvp(n: Node, t: Tangent) -> Node | None:
    """``dX = G^-1 (dB - dJ^T J X - J^T dJ X)``."""
    f, b = n.operands
    j = f.operands[0]
    dj, db = t(f), t(b)
    terms = []
    if db is not None:
        terms.append(db)
    if dj is not None:
        terms += [_neg(_times(_t(dj), _times(j, n))), _neg(_times(_t(j), _times(dj, n)))]
    if not terms:
        return None
    rhs = terms[0]
    for term in terms[1:]:
        rhs = _o("add", rhs, term)
    return _live(f, "qr_info", _o("qr_gram_solve", f, rhs))


#: ``kind -> (vjp, jvp)``; the three ``*_info`` kinds are integer-typed and
#: sit with the explicit zero rules in :mod:`hawk.diff.rules`.
LINALG_RULES: dict[str, tuple[Callable, Callable]] = {
    "lu": (_carry_vjp, _carry_jvp),
    "chol": (_carry_vjp, _carry_jvp),
    "chol_from": (_chol_from_vjp, _chol_from_jvp),
    "eigh": (_carry_vjp, _carry_jvp),
    "qr": (_carry_vjp, _carry_jvp),
    "qr_solve": (_qr_solve_vjp, _qr_solve_jvp),
    "qr_gram_solve": (_qr_gram_vjp, _qr_gram_jvp),
    "lu_solve": (_lu_solve_vjp, _lu_solve_jvp),
    "lu_solve_t": (_lu_solve_t_vjp, _lu_solve_t_jvp),
    "lu_det": (_lu_det_vjp, _lu_det_jvp),
    "cho_solve": (_cho_solve_vjp, _cho_solve_jvp),
    "tri_lower": (_tri_lower_vjp, _tri_lower_jvp),
    "tri_upper_t": (_tri_upper_t_vjp, _tri_upper_t_jvp),
    "chol_factor": (_chol_factor_vjp, _chol_factor_jvp),
    "eigh_w": (_eigh_w_vjp, _eigh_w_jvp),
    "eigh_v": (_eigh_v_vjp, _eigh_v_jvp),
}
