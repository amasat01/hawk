# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Kernels over the linear-algebra IR kinds, built as IR.

The kinds are IR nodes; these builders are the only way a test reaches them
without the authoring surface. Each returns a :class:`~hawk.trace.kernel.Kernel`
whose planes are named ``A`` (the matrix), ``b`` (a vector right-hand side) and
``B`` (an ``n x 2`` one), with the results as ``Mutable`` planes.
"""

from __future__ import annotations

from hawk.ir import Assign, Leaf, make
from hawk.ir.linalg import as_matrix
from hawk.trace.kernel import Kernel
from hawk.types import TensorType

#: Columns of the matrix right-hand side.
K = 2

F64 = "f64"


def _inputs(n: int):
    a = Leaf("vocab_read", "mat_in", "A", TensorType((n, n), F64))
    b = Leaf("vocab_read", "vec_in", "b", TensorType((n,), F64))
    rhs = Leaf("vocab_read", "mat_in", "B", TensorType((n, K), F64))
    return a, b, rhs


def lu_kernel(n: int) -> Kernel:
    """``x = A \\ b``, ``X = A \\ B``, ``det(A)`` and the LU ``info``."""
    a, b, rhs = _inputs(n)
    f = make("lu", (a,))
    return Kernel(f"la_lu_{n}", (
        Assign("x", make("lu_solve", (f, b)), TensorType((n,), F64)),
        Assign("X", make("lu_solve", (f, rhs)), TensorType((n, K), F64)),
        Assign("det", make("lu_det", (f,)), TensorType((), F64)),
        Assign("info", make("lu_info", (f,)), TensorType((), "int32"))))


def chol_kernel(n: int) -> Kernel:
    """The solves against ``chol(A)``, the lower factor and the status."""
    a, b, rhs = _inputs(n)
    f = make("chol", (a,))
    return Kernel(f"la_chol_{n}", (
        Assign("x", make("cho_solve", (f, b)), TensorType((n,), F64)),
        Assign("X", make("cho_solve", (f, rhs)), TensorType((n, K), F64)),
        Assign("L", as_matrix(f), TensorType((n, n), F64)),
        Assign("info", make("chol_info", (f,)), TensorType((), "int32"))))


def chol_from_kernel(n: int) -> Kernel:
    """The solves against ``chol_from(A)``: ``A`` is the lower triangular factor itself."""
    a, b, rhs = _inputs(n)
    f = make("chol_from", (a,))
    return Kernel(f"la_chol_from_{n}", (
        Assign("x", make("cho_solve", (f, b)), TensorType((n,), F64)),
        Assign("X", make("cho_solve", (f, rhs)), TensorType((n, K), F64)),
        Assign("L", as_matrix(f), TensorType((n, n), F64)),
        Assign("info", make("chol_info", (f,)), TensorType((), "int32"))))


def eigh_kernel(n: int) -> Kernel:
    """The eigenvalues, the eigenvectors and the status of ``eigh(A)``."""
    a, _, _ = _inputs(n)
    f = make("eigh", (a,))
    return Kernel(f"la_eigh_{n}", (
        Assign("w", make("eigh_w", (f,)), TensorType((n,), F64)),
        Assign("V", make("eigh_v", (f,)), TensorType((n, n), F64)),
        Assign("info", make("eigh_info", (f,)), TensorType((), "int32"))))


def lstsq_kernel(m: int, n: int) -> Kernel:
    """``x, info = lstsq(J, r)`` for a tall ``m x n`` ``J`` (QR, a team stage): the solution and the status."""
    j = Leaf("vocab_read", "mat_in", "J", TensorType((m, n), F64))
    r = Leaf("vocab_read", "vec_in", "r", TensorType((m,), F64))
    f = make("qr", (j,))
    return Kernel(f"la_lstsq_{m}_{n}", (
        Assign("x", make("qr_solve", (f, r)), TensorType((n,), F64)),
        Assign("info", make("qr_info", (f,)), TensorType((), "int32"))))
