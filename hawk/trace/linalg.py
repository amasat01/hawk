# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The authoring surface of per-lane linear algebra, as ``hawk.math`` names it.

Each name is a composition of the IR kinds of :mod:`hawk.ir.linalg` (one
factorisation node, its consumers and its projections) and of the matrix ops, so
nothing here is a new op. The numpy-style tuples carry the status beside the
result: ``info`` is a rank-0 ``Int32`` that is 0 when the factorisation held,
``k > 0`` for the failed pivot or minor (LAPACK's numbering), and for ``eigh`` a
bit mask (1: not converged, 2: neighbouring eigenvalues too close to tell their
vectors apart). A failed lane's results are zeros, never a partial answer.

A factor is a value too: ``lu`` and ``eigh`` factors are read by the solves and
fields below and are refused as matrices; a Cholesky factor is also the lower
triangular matrix, wherever a matrix is wanted.
"""

from __future__ import annotations

from typing import Any

from ..ir import Const, HawkError, Node
from ..ir import make as _make
from ..types import TensorType
from .value import Value, node_of, random_normal, vec

_F64 = TensorType((), "f64")


def _square(name: str, a: Any) -> Node:
    node = node_of(a)
    t = node.ttype
    if len(t.shape) != 2 or t.shape[0] != t.shape[1]:
        raise HawkError(f"{name}: a square matrix is wanted, got {t!r}")
    return node


def _at(m: Any, i: int, j: int) -> Value:
    """Entry ``(i, j)`` of a matrix value."""
    return Value(_make("component", (node_of(m),), (i, j)))


def _matrix(rows: list[list[Any]]) -> Value:
    n, m = len(rows), len(rows[0])
    parts = [node_of(x) for row in rows for x in row]
    return Value(_make("vec", parts, (n, m)))


def mat(rows: Any, *, columns: bool = False) -> Value:
    """A matrix built from traced entries: ``mat([[a, b], [c, d]])`` takes rows of
    scalars, and ``mat([v, w])`` rows of vectors (``columns=True``: the vectors are
    the columns). A row may also be a list in a list of vectors. Every row has the same
    length; a ragged or empty argument is refused."""
    if not isinstance(rows, (list, tuple)) or not rows:
        raise HawkError(f"mat: a non-empty list of rows is wanted, got {rows!r}")
    lines = []
    for k, row in enumerate(rows):
        if isinstance(row, (list, tuple)):
            lines.append([node_of(x) for x in row])
            if any(len(x.ttype.shape) for x in lines[-1]):
                raise HawkError(f"mat: row {k} has an entry that is not a scalar; "
                                "a row of scalars is a list, a row of vectors is the vector")
            continue
        node = node_of(row)
        if len(node.ttype.shape) != 1:
            raise HawkError(f"mat: row {k} is a list of scalars or a vector, got {node.ttype!r}")
        lines.append([node_of(Value(node)[i]) for i in range(node.ttype.shape[0])])
    width = len(lines[0])
    if width == 0 or any(len(r) != width for r in lines):
        raise HawkError("mat: every row needs the same non-zero length, got lengths "
                        f"{[len(r) for r in lines]}")
    if columns:
        lines = [list(c) for c in zip(*lines)]
    return Value(_make("vec", [x for row in lines for x in row], (len(lines), len(lines[0]))))


def _zero() -> Const:
    return Const(0.0, _F64)


def eye(n: int) -> Value:
    """The ``n x n`` identity."""
    _size("eye", n)
    return _matrix([[Const(1.0 if i == j else 0.0, _F64) for j in range(n)]
                    for i in range(n)])


def _size(name: str, n: Any) -> None:
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise HawkError(f"{name}: the size is a static positive int, got {n!r}")


def diag(x: Any) -> Value:
    """A vector as a diagonal matrix, or (as numpy) a matrix's diagonal."""
    node = node_of(x)
    if len(node.ttype.shape) == 2:
        return diagonal(x)
    if len(node.ttype.shape) != 1:
        raise HawkError(f"diag: a vector or a square matrix is wanted, got {node.ttype!r}")
    v, n = Value(node), node.ttype.shape[0]
    return _matrix([[v[i] if i == j else _zero() for j in range(n)] for i in range(n)])


def diagonal(a: Any) -> Value:
    """The diagonal of a square matrix, as a vector."""
    node = _square("diagonal", a)
    m = Value(node)
    return vec(*[_at(m, i, i) for i in range(node.ttype.shape[0])])


def trace(a: Any) -> Value:
    """The sum of a square matrix's diagonal."""
    node = _square("trace", a)
    m = Value(node)
    total = _at(m, 0, 0)
    for i in range(1, node.ttype.shape[0]):
        total = total + _at(m, i, i)
    return total


def tril(a: Any) -> Value:
    """A square matrix with everything above the diagonal set to zero."""
    node = _square("tril", a)
    m, n = Value(node), node.ttype.shape[0]
    return _matrix([[_at(m, i, j) if j <= i else _zero() for j in range(n)] for i in range(n)])


# --- LU --------------------------------------------------------------------------

def lu(a: Any) -> tuple[Value, Value]:
    """``P A = L U``: the factor and its ``info``. Read the factor with
    :func:`lu_solve`; it is not a matrix."""
    f = _make("lu", (_square("lu", a),))
    return Value(f), Value(_make("lu_info", (f,)))


def lu_solve(factor: Any, b: Any) -> Value:
    """``A x = b`` from :func:`lu`'s factor; ``b`` is a vector or an ``n x k``
    matrix. All zeros when the factorisation failed."""
    return Value(_make("lu_solve", (node_of(factor), node_of(b))))


def solve(a: Any, b: Any) -> tuple[Value, Value]:
    """``x, info = solve(A, b)``: ``A x = b`` by LU with partial pivoting."""
    f, info = lu(a)
    return lu_solve(f, b), info


def inv(a: Any) -> tuple[Value, Value]:
    """``Ainv, info = inv(A)``: zeros for a singular matrix."""
    node = _square("inv", a)
    f, info = lu(node)
    return lu_solve(f, eye(node.ttype.shape[0])), info


def det(a: Any) -> Value:
    """``det(A)``, 0 for a singular matrix."""
    return Value(_make("lu_det", (lu(a)[0].node,)))


# --- Cholesky ----------------------------------------------------------------------

def cholesky(a: Any) -> tuple[Value, Value]:
    """``L, info = cholesky(A)``: ``L L^T`` is the symmetric part of ``A``. ``L``
    is the lower triangular matrix (zeros when ``info > 0``) and is what
    :func:`cho_solve` reads."""
    f = _make("chol", (_square("cholesky", a),))
    return Value(f), Value(_make("chol_info", (f,)))


def chol_from(lower: Any) -> tuple[Value, Value]:
    """``factor, info = chol_from(L)``: a lower triangular ``L`` you built, taken as
    the Cholesky factor of ``L L^T``, for :func:`cho_solve` (the part above the diagonal
    is ignored). ``info`` is the first column (1-based) whose diagonal entry is not
    positive and finite, ``0`` when every one is; a failed lane's factor is zeros, so a
    solve against it is zeros. Only the diagonal is checked: a NaN below it reaches the
    solve."""
    f = _make("chol_from", (_square("chol_from", lower),))
    return Value(f), Value(_make("chol_info", (f,)))


def cho_solve(factor: Any, b: Any) -> Value:
    """``A x = b`` from :func:`cholesky`'s ``L``, or from :func:`chol_from`'s factor.
    A matrix you built or changed since is refused: it carries no status to fail on,
    so pass it through :func:`chol_from`, factor with :func:`cholesky` (or solve with
    :func:`solve`)."""
    node = node_of(factor)
    if node.ttype.tag != "chol_factor":
        raise HawkError(
            "cho_solve reads the factor that cholesky() or chol_from() returned, not a "
            f"matrix built or changed since (got {node.ttype!r}); take a lower "
            "triangular L through chol_from(L), factor with cholesky(A), or solve "
            "with solve(A, b)")
    return Value(_make("cho_solve", (node, node_of(b))))


# --- symmetric eigenproblem --------------------------------------------------------

def eigh(a: Any) -> tuple[Value, Value, Value]:
    """``w, V, info = eigh(A)`` for the symmetric part of ``A``: eigenvalues
    ascending, eigenvectors as the columns of ``V`` (each signed so its largest
    component is positive)."""
    f = _make("eigh", (_square("eigh", a),))
    return (Value(_make("eigh_w", (f,))), Value(_make("eigh_v", (f,))),
            Value(_make("eigh_info", (f,))))


# --- least squares -----------------------------------------------------------------

def _normal_equations(name: str, j: Any, r: Any, damping: Any = None):
    jn = node_of(j)
    if len(jn.ttype.shape) != 2:
        raise HawkError(f"{name}: J is a matrix, got {jn.ttype!r}")
    jv = Value(jn)
    gram = Value(_make("transpose", (jn,))) @ jv
    if damping is not None:
        gram = gram + node_of(damping) * eye(jn.ttype.shape[1])
    return cholesky(gram), Value(_make("transpose", (jn,))) @ Value(node_of(r))


def lstsq(j: Any, r: Any) -> tuple[Value, Value]:
    """``x, info = lstsq(J, r)``: minimise ``|J x - r|`` through the normal
    equations, ``J^T J x = J^T r``, by Cholesky. ``info > 0`` when ``J`` is
    rank deficient; the conditioning is squared, as in any normal-equations solve."""
    (factor, info), rhs = _normal_equations("lstsq", j, r)
    return cho_solve(factor, rhs), info


def lm_step(j: Any, r: Any, mu: Any) -> Value:
    """The Levenberg-Marquardt step ``-(J^T J + mu I)^-1 J^T r``. Zeros (no
    step) when the damped system does not factor."""
    (factor, _), rhs = _normal_equations("lm_step", j, r, mu)
    return -cho_solve(factor, rhs)


# --- sampling ----------------------------------------------------------------------

def mvn_sample(seed: Any, counter: Any, mean: Any, cov: Any) -> Value:
    """A draw from ``N(mean, cov)``: ``mean + L z`` with ``L`` the Cholesky factor
    of ``cov`` and ``z`` independent standard normals on the sub-counters
    ``counter * n + k``. Differentiable in ``mean`` and ``cov`` (the draw is
    ``z`` fixed); a ``cov`` that does not factor gives ``mean``."""
    cov_node = _square("mvn_sample", cov)
    n = cov_node.ttype.shape[0]
    factor, _ = cholesky(cov_node)
    base = counter * n
    z = vec(*[random_normal(seed, base + k) for k in range(n)])
    return Value(node_of(mean)) + factor @ z
