# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Kernels over the linear-algebra IR kinds, built as IR.

The kinds are IR nodes; these builders are the only way a test reaches them
without the authoring surface. Each returns a :class:`~hawk.trace.kernel.Kernel`
whose planes are named ``A`` (the matrix), ``b`` (a vector right-hand side) and
``B`` (an ``n x 2`` one), with the results as ``Mutable`` planes.
"""

from __future__ import annotations

from hawk import _ir
from hawk.trace.kernel import Kernel

#: Columns of the matrix right-hand side.
K = 2

F64 = "f64"


def _inputs(context, n: int):
    a = _ir.leaf(context, "vocab_read", "plane_in", "A", _ir.TensorType((n, n), F64))
    b = _ir.leaf(context, "vocab_read", "plane_in", "b", _ir.TensorType((n,), F64))
    rhs = _ir.leaf(context, "vocab_read", "plane_in", "B", _ir.TensorType((n, K), F64))
    return a, b, rhs


def _assign(name: str, value, shape: tuple, dtype: str = F64):
    """The plane ``name`` of ``shape`` committing ``value``."""
    return _ir.make_node("Assign", _ir.TensorType(shape, dtype), (value,), name=name)


def lu_kernel(context, n: int) -> Kernel:
    """``x = A \\ b``, ``X = A \\ B``, ``det(A)`` and the LU ``info``."""
    a, b, rhs = _inputs(context, n)
    f = _ir.make("lu", (a,))
    return Kernel(
        f"la_lu_{n}",
        (
            _assign("x", _ir.make("lu_solve", (f, b)), (n,)),
            _assign("X", _ir.make("lu_solve", (f, rhs)), (n, K)),
            _assign("det", _ir.make("lu_det", (f,)), ()),
            _assign("info", _ir.make("lu_info", (f,)), (), "int32"),
        ),
    )


def chol_kernel(context, n: int) -> Kernel:
    """The solves against ``chol(A)``, the lower factor and the status."""
    a, b, rhs = _inputs(context, n)
    f = _ir.make("chol", (a,))
    return Kernel(
        f"la_chol_{n}",
        (
            _assign("x", _ir.make("cho_solve", (f, b)), (n,)),
            _assign("X", _ir.make("cho_solve", (f, rhs)), (n, K)),
            _assign("L", _ir.as_matrix(f), (n, n)),
            _assign("info", _ir.make("chol_info", (f,)), (), "int32"),
        ),
    )


def chol_from_kernel(context, n: int) -> Kernel:
    """The solves against ``chol_from(A)``: ``A`` is the lower triangular factor itself."""
    a, b, rhs = _inputs(context, n)
    f = _ir.make("chol_from", (a,))
    return Kernel(
        f"la_chol_from_{n}",
        (
            _assign("x", _ir.make("cho_solve", (f, b)), (n,)),
            _assign("X", _ir.make("cho_solve", (f, rhs)), (n, K)),
            _assign("L", _ir.as_matrix(f), (n, n)),
            _assign("info", _ir.make("chol_info", (f,)), (), "int32"),
        ),
    )


def eigh_kernel(context, n: int) -> Kernel:
    """The eigenvalues, the eigenvectors and the status of ``eigh(A)``."""
    a, _, _ = _inputs(context, n)
    f = _ir.make("eigh", (a,))
    return Kernel(
        f"la_eigh_{n}",
        (
            _assign("w", _ir.make("eigh_w", (f,)), (n,)),
            _assign("V", _ir.make("eigh_v", (f,)), (n, n)),
            _assign("info", _ir.make("eigh_info", (f,)), (), "int32"),
        ),
    )


def eigh_values_kernel(context, n: int) -> Kernel:
    """The eigenvalues and the status of ``eigh(A)``: no eigenvector is read."""
    a, _, _ = _inputs(context, n)
    f = _ir.make("eigh", (a,))
    return Kernel(
        f"la_eigh_values_{n}",
        (
            _assign("w", _ir.make("eigh_w", (f,)), (n,)),
            _assign("info", _ir.make("eigh_info", (f,)), (), "int32"),
        ),
    )


def lstsq_kernel(context, m: int, n: int) -> Kernel:
    """``x, info = lstsq(J, r)`` for a tall ``m x n`` ``J`` (QR, a team stage): the solution and the status."""
    j = _ir.leaf(context, "vocab_read", "plane_in", "J", _ir.TensorType((m, n), F64))
    r = _ir.leaf(context, "vocab_read", "plane_in", "r", _ir.TensorType((m,), F64))
    f = _ir.make("qr", (j,))
    return Kernel(
        f"la_lstsq_{m}_{n}",
        (
            _assign("x", _ir.make("qr_solve", (f, r)), (n,)),
            _assign("info", _ir.make("qr_info", (f,)), (), "int32"),
        ),
    )
