# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Sums over the components of a wide value (``bf[j]``) and several sinks folding over one loop: the shapes that state no
``contractions`` member.

The library decides which sums of products run as a matrix product. A sum that reads components of a vector, a masked sum, or
several outputs sharing an operand is not one it routes as a matrix product today, so its description carries no member; the
kernels that must state none are the ones below.
"""

from __future__ import annotations

import pytest

import hawk
import hawk.math as m
from hawk.artifact import describe

NF, NC, NM = 512, 24, 384


@hawk.kernel
def matvec_input(bf: hawk.Vector[NF], w: hawk.Table["row":NC, "col":NF],
                 y: hawk.Mutable[hawk.Scalar]):
    """One row of a 24x512 matvec: a plane times the components of a Vector input."""
    acc = 0.0
    for j in range(NF):
        acc = acc + w.at(row=3, col=j) * bf[j]
    y = acc


@hawk.kernel
def matvec_computed(x: hawk.Vector[NF], w: hawk.Table["row":NC, "col":NF],
                    y: hawk.Mutable[hawk.Scalar]):
    """The same over a value computed in the kernel."""
    bf = m.exp(x) * 2.0
    acc = 0.0
    for j in range(NF):
        acc = acc + w.at(row=3, col=j) * bf[j]
    y = acc


@hawk.kernel
def masked_sum(bf: hawk.Vector[NM], w: hawk.Table["row":2, "col":NM],
               y: hawk.Mutable[hawk.Scalar]):
    """A masked 384-term sum: the mask reads the same component."""
    acc = 0.0
    for j in range(NM):
        acc = acc + m.select(bf[j] > 0.0, w.at(row=1, col=j) * bf[j], 0.0)
    y = acc


@hawk.kernel
def two_sinks(bf: hawk.Vector[NF], w: hawk.Table["row":2, "col":NF],
              u: hawk.Mutable[hawk.Scalar], v: hawk.Mutable[hawk.Scalar]):
    """Two outputs over one loop, sharing the operand ``bf``."""
    a = 0.0
    b = 0.0
    for j in range(NF):
        a = a + w.at(row=0, col=j) * bf[j]
        b = b + w.at(row=1, col=j) * bf[j]
    u = a
    v = b


# --- shapes that must NOT be recognised -------------------------------------
@hawk.kernel
def strided_by_literal(bf: hawk.Vector[NF], w: hawk.Table["row":2, "col":NF],
                       y: hawk.Mutable[hawk.Scalar]):
    """Every other component, the stride a bare literal."""
    acc = 0.0
    for j in range(NF // 2):
        acc = acc + w.at(row=0, col=j) * bf[2 * j]
    y = acc


@hawk.kernel
def two_loops(bf: hawk.Vector[NF], w: hawk.Table["row":2, "col":NF],
              u: hawk.Mutable[hawk.Scalar], v: hawk.Mutable[hawk.Scalar]):
    """Two separate loops are two contractions, not one with two outputs."""
    a = 0.0
    for j in range(NF):
        a = a + w.at(row=0, col=j) * bf[j]
    b = 0.0
    for j in range(NF):
        b = b + w.at(row=1, col=j) * bf[j]
    u = a
    v = b


@hawk.kernel
def fold_and_plain(bf: hawk.Vector[NF], w: hawk.Table["row":2, "col":NF],
                   u: hawk.Mutable[hawk.Scalar], v: hawk.Mutable[hawk.Scalar]):
    """One sink is a fold, the other is not: the whole is not a contraction."""
    a = 0.0
    for j in range(NF):
        a = a + w.at(row=0, col=j) * bf[j]
    u = a
    v = w.at(row=1, col=0)


@hawk.kernel
def three_operands(bf: hawk.Vector[NF], w: hawk.Table["row":2, "col":NF],
                   g: hawk.Table["col":NF], y: hawk.Mutable[hawk.Scalar]):
    """A component read is an operand: w, g and bf make three."""
    acc = 0.0
    for j in range(NF):
        acc = acc + w.at(row=0, col=j) * g.at(col=j) * bf[j]
    y = acc


@pytest.mark.parametrize("kernel", [strided_by_literal, two_loops, fold_and_plain, three_operands], ids=lambda k: k.name)
def test_a_shape_outside_the_family_states_no_contraction(kernel):
    # a stride that is a bare literal, two loops, a sink that is no fold, three operands: no single dense contraction
    assert "contractions" not in describe([kernel], targets=("host",))[kernel.name]
