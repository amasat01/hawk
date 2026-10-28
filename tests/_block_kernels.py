# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Kernels that sum per-row products over the rows of a key (a Gram matrix and a gradient).

The lanes are ``Grid["key", "row":rows]``: the rows of a key are the lanes that share it, and
``hawk.sum(..., over="row")`` keeps one block per key. A builder takes the ``linalg_form`` the
trace runs under, which decides whether the sum is one block-sum node (``"team"``, or ``"auto"``
above 16 entries) or the per-entry accumulates (``"thread"``). No ``from __future__ import
annotations`` here: the plane annotations are read when the kernel is defined, from the
builder's arguments.
"""

import hawk
import hawk.math as m
from hawk import Grid, Matrix, Mutable, Scalar, Vector, kernel
from hawk.emit import policy


def gram_kernel(p: int, rows: int, form: str = "auto", kind=None):
    """``G = sum_rows outer(j, j)`` per key."""
    make = kernel if kind is None else kind
    with policy.using(linalg_form=form):
        @make
        def gram(lane: Grid["key", "row":rows], j: Vector[p]) -> Matrix[p, p]:
            return hawk.sum(m.outer(j, j), over="row")
    return gram


def grad_kernel(p: int, rows: int, form: str = "auto", kind=None):
    """``g = sum_rows j * r`` per key."""
    make = kernel if kind is None else kind
    with policy.using(linalg_form=form):
        @make
        def grad(lane: Grid["key", "row":rows], j: Vector[p], r: Scalar) -> Vector[p]:
            return hawk.sum(j * r, over="row")
    return grad


def step_kernel(p: int, rows: int):
    """One damped Gauss-Newton step per key: ``x = G \\ g`` with both sums read back
    in the body, so the stage chain is the per-row stores, the block sums, the per-lane read, the
    Cholesky team stage and the per-lane read of its solve."""
    @kernel
    def step(lane: Grid["key", "row":rows], j: Vector[p], r: Scalar, x: Mutable[Vector[p]]):
        gram = hawk.sum(m.outer(j, j), over="row")
        grad = hawk.sum(j * r, over="row")
        x = m.cho_solve(m.cholesky(gram)[0], grad)
    return step


def readback_kernel(p: int, rows: int):
    """Entries of both sums read back in the body, one per lane."""
    @kernel
    def readback(lane: Grid["key", "row":rows], j: Vector[p], r: Scalar) -> Vector[2]:
        gram = hawk.sum(m.outer(j, j), over="row")
        grad = hawk.sum(j * r, over="row")
        return m.vec(gram[1, 2] + 0.0 * r, grad[3])
    return readback


def scaled_kernel(p: int, rows: int):
    """A payload that is not a product of rank-1 values (a scaled outer product)."""
    @kernel
    def scaled(lane: Grid["key", "row":rows], j: Vector[p], r: Scalar) -> Matrix[p, p]:
        return hawk.sum(m.outer(j, j) * r, over="row")
    return scaled


def total_kernel(spelling: str, p: int, rows: int = 1, form: str = "auto", kind=None):
    """The sum over EVERY lane (one key) of per-lane products, in one of four spellings: ``"outer"``
    (``outer(j, j)``), ``"scaled"`` (``j * r``), ``"gram"`` (``transpose(M) @ M`` of a per-lane
    ``rows x p`` matrix) and ``"moment"`` (``transpose(M) @ r`` with ``r`` a per-lane vector of ``rows``)."""
    make = kernel if kind is None else kind
    with policy.using(linalg_form=form):
        if spelling == "outer":
            @make
            def total(j: Vector[p]) -> Matrix[p, p]:
                return hawk.sum(m.outer(j, j))
        elif spelling == "scaled":
            @make
            def total(j: Vector[p], r: Scalar) -> Vector[p]:
                return hawk.sum(j * r)
        elif spelling == "gram":
            @make
            def total(mat: Matrix[rows, p]) -> Matrix[p, p]:
                return hawk.sum(m.transpose(mat) @ mat)
        else:
            @make
            def total(mat: Matrix[rows, p], r: Vector[rows]) -> Vector[p]:
                return hawk.sum(m.transpose(mat) @ r)
    return total


def stored_and_read_kernel(p: int, rows: int):
    """The Gram per key, stored AND read back in the body (one entry): the sum keeps its block scratch."""
    @kernel
    def gram_read(lane: Grid["key", "row":rows], j: Vector[p], x: Mutable[Scalar]) -> Matrix[p, p]:
        gram = hawk.sum(m.outer(j, j), over="row")
        x = gram[1, 2]
        return gram
    return gram_read


def keyed_transpose_kernel(p: int, per: int, rows: int, kind=None):
    """``G = sum_rows transpose(M) @ M`` per key, ``M`` a per-lane ``per x p`` matrix."""
    make = kernel if kind is None else kind

    @make
    def tgram(lane: Grid["key", "row":rows], mat: Matrix[per, p]) -> Matrix[p, p]:
        return hawk.sum(m.transpose(mat) @ mat, over="row")
    return tgram
