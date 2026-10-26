# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Per-lane linear algebra: the op kinds, their type rules and their cost.

A factorisation is ONE compute node (``lu``, ``chol``, ``eigh``, and ``chol_from``, the
Cholesky factor of a lower triangular matrix the program built) whose value
is the whole result: a square ``n x n`` :class:`~hawk.types.TensorType` tagged
with the factor it carries (``lu_factor``, ``chol_factor``, ``eigh_factor``).
Everything a program reads from it is a consumer (``lu_solve``, ``cho_solve``)
or a projection (``lu_info``, ``lu_det``, ``chol_info``, ``chol_factor``,
``eigh_w``, ``eigh_v``, ``eigh_info``) over that one node, so a factorisation
runs once however many results are read, and a derivative rule hangs off the
consumer and never refactors.

The Cholesky factor is the one factor that is also a matrix: wherever a
tagged ``chol`` node is used as a matrix it becomes its ``chol_factor``
projection (lower triangular, zero when the factorisation failed). An LU or
eigh factor used as a matrix is refused by name.

Sizes are static, ``1 <= n <= MAX_DIM``. The cost formulas are
``aether::linalg``'s (``aether/linalg/``): operations in fused multiply-adds,
bytes of storage the op holds.
"""

from __future__ import annotations

from math import prod

from ..types import TensorType
from .nodes import FACTOR_TAGS, HawkError, Node, Op

#: The largest static dimension (aether's ``linalg::max_dim``).
MAX_DIM = 32

#: The tag a factorisation node carries, by the kind that makes it.
FACTOR_TAG = {**dict(zip(("lu", "chol", "eigh"), FACTOR_TAGS)), "chol_from": "chol_factor"}

#: The tag each consumer and projection requires of its first operand.
_NEEDS = {"lu_solve": "lu_factor", "lu_solve_t": "lu_factor", "lu_info": "lu_factor",
          "lu_det": "lu_factor", "cho_solve": "chol_factor", "tri_lower": "chol_factor",
          "tri_upper_t": "chol_factor", "chol_info": "chol_factor",
          "chol_factor": "chol_factor", "eigh_w": "eigh_factor",
          "eigh_v": "eigh_factor", "eigh_info": "eigh_factor"}

#: The compute kinds: one node per factorisation.
FACTORISATIONS = tuple(FACTOR_TAG)
#: The solves against a factor. ``lu_solve_t`` solves with ``A^T``; ``tri_lower``
#: and ``tri_upper_t`` solve with the Cholesky factor's ``L`` and ``L^T`` alone,
#: the two halves of ``cho_solve`` that a derivative needs apart.
CONSUMERS = ("lu_solve", "lu_solve_t", "cho_solve", "tri_lower", "tri_upper_t")
#: The free reads of one field of a factor.
PROJECTIONS = ("lu_info", "lu_det", "chol_info", "chol_factor", "eigh_w", "eigh_v",
               "eigh_info")
#: Every linear-algebra kind.
LINALG_OPS = FACTORISATIONS + CONSUMERS + PROJECTIONS

#: Operand counts.
ARITY = {**dict.fromkeys(FACTORISATIONS, 1), **dict.fromkeys(PROJECTIONS, 1),
         **dict.fromkeys(CONSUMERS, 2)}

_FLOATS = ("f64", "f32")
_WIDTH = {"f64": 8, "f32": 4}
_INFO_BYTES = 4


def sweeps(n: int) -> int:
    """Jacobi sweeps ``eigh`` runs for dimension ``n`` (aether's ``eighSweeps``)."""
    return 2 if n <= 2 else 6 if n <= 4 else 8 if n <= 8 else 12 if n <= 16 else 15


def is_linalg(node: Node) -> bool:
    """Whether ``node`` is one of the :data:`LINALG_OPS`."""
    return isinstance(node, Op) and node.kind in LINALG_OPS


def as_matrix(node: Node) -> Node:
    """``node`` where a matrix is wanted: a Cholesky factor becomes its
    lower-triangular ``chol_factor`` projection, an LU or eigh factor is
    refused by name, anything else is itself."""
    tag = node.ttype.tag
    if tag == "chol_factor":
        return Op("chol_factor", (node,), TensorType(node.ttype.shape, node.ttype.dtype))
    if tag in ("lu_factor", "eigh_factor"):
        what = "an LU" if tag == "lu_factor" else "an eigh"
        reads = ("lu_solve, lu_det or lu_info" if tag == "lu_factor"
                 else "eigh_w, eigh_v or eigh_info")
        raise HawkError(
            f"{what} factor is not a matrix: it carries pivots and a status, so it "
            f"is read with {reads}, never as an operand of arithmetic")
    return node


def _square(kind: str, t: TensorType) -> int:
    if len(t.shape) != 2 or t.shape[0] != t.shape[1]:
        raise HawkError(f"op {kind!r}: a square matrix is factored, got {t!r}")
    n = t.shape[0]
    if t.dtype not in _FLOATS:
        raise HawkError(f"op {kind!r}: an f64/f32 matrix is factored, got {t!r}")
    if not 1 <= n <= MAX_DIM:
        raise HawkError(
            f"op {kind!r}: the dimension is static and 1 <= n <= {MAX_DIM}, got n={n}")
    return n


def _factor(kind: str, t: TensorType) -> int:
    want = _NEEDS[kind]
    if t.tag != want:
        producer = next(k for k, tag in FACTOR_TAG.items() if tag == want)
        raise HawkError(f"op {kind!r} reads the result of {producer!r} (a {want}), got {t!r}")
    return t.shape[0]


def result_type(kind: str, ts: list[TensorType]) -> TensorType:
    """The type ``kind`` returns over operands typed ``ts``."""
    if kind in FACTORISATIONS:
        n = _square(kind, ts[0])
        return TensorType((n, n), ts[0].dtype, FACTOR_TAG[kind])
    n = _factor(kind, ts[0])
    dtype = ts[0].dtype
    if kind in CONSUMERS:
        b = ts[1]
        if (b.dtype != dtype or len(b.shape) not in (1, 2) or b.shape[0] != n):
            raise HawkError(
                f"op {kind!r}: the right-hand side is a {dtype} vector ({n},) or "
                f"matrix ({n}, k) against the {n}x{n} factor, got {b!r}")
        return TensorType(b.shape, dtype)
    if kind.endswith("_info"):
        return TensorType((), "int32")
    if kind == "lu_det":
        return TensorType((), dtype)
    if kind == "eigh_w":
        return TensorType((n,), dtype)
    return TensorType((n, n), dtype)        # chol_factor, eigh_v


def _geometry(node: Op) -> tuple[int, int, int]:
    """``(n, k, bytes per scalar)`` of a linear-algebra node."""
    first = node.operands[0].ttype
    n = first.shape[0]
    rhs = node.operands[1].ttype.shape if node.kind in CONSUMERS else ()
    return n, (rhs[1] if len(rhs) == 2 else 1), _WIDTH[first.dtype]


def op_count(node: Op) -> float:
    """Fused multiply-adds ``node`` runs: ``lu`` n^3/3, ``chol`` n^3/6, ``eigh``
    S(n) n(n-1)/2 (6n+10) for S(n) sweeps, a solve n^2 k (a triangular one half
    of it), ``lu_det`` n. A projection reads a field and costs nothing."""
    n, k, _ = _geometry(node)
    kind = node.kind
    if kind == "lu":
        return n ** 3 / 3
    if kind == "chol":
        return n ** 3 / 6
    if kind == "eigh":
        return sweeps(n) * n * (n - 1) / 2 * (6 * n + 10)
    if kind in CONSUMERS:
        return n * n * k / (2 if kind in ("tri_lower", "tri_upper_t") else 1)
    return float(n) if kind == "lu_det" else 0.0


def factor_bytes(t: TensorType, width: int) -> int:
    """Bytes a factorisation of type ``t`` (tagged) holds at ``width`` bytes a
    scalar: :func:`op_bytes`' figure for the node that makes it."""
    n = t.shape[0]
    kind = {tag: k for k, tag in FACTOR_TAG.items()}[t.tag]
    return _FACTOR_BYTES[kind](n, width)


_FACTOR_BYTES = {
    "lu": lambda n, w: n * n * w + _INFO_BYTES * n + _INFO_BYTES,
    "chol": lambda n, w: n * n * w + _INFO_BYTES,
    "eigh": lambda n, w: (2 * n * n + n) * w + _INFO_BYTES,
}
_FACTOR_BYTES["chol_from"] = _FACTOR_BYTES["chol"]


def op_bytes(node: Op) -> int:
    """Bytes of storage ``node`` holds at ``w`` bytes a scalar: a factor is its
    struct (``lu`` n^2 w + 4n + 4, ``chol`` n^2 w + 4, ``eigh`` (n^2 + n) w + 4,
    plus the n^2 w working copy Jacobi rotates), a solve its result, a
    projection the field it copies."""
    n, _, w = _geometry(node)
    kind = node.kind
    if kind in _FACTOR_BYTES:
        return _FACTOR_BYTES[kind](n, w)
    if kind.endswith("_info"):
        return _INFO_BYTES
    return prod(node.ttype.shape) * _WIDTH[node.ttype.dtype]
