# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The contraction recognizer on the ATMOSPHERIC shapes: component reads of a
wide value (``bf[j]``) and several sinks folding over one loop.

One record shape serves both consumers (a team fold and a dense-GEMM route):
a component-read operand is a plane of ONE declared axis, ``component``, of
the value's width, role ``wide`` (a ``Vector`` input) or ``computed`` (a value
built in the kernel, named ``computed#k`` in first-read order). Several sinks
put every output in ``outputs`` (first one is ``carry``), and ``operands``
unions what all the folds read. Recognition reads only: nothing here changes
a rendered kernel.
"""

from __future__ import annotations

import hawk
import hawk.math as m
from hawk.ir import recognize

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


def _wide(rec):
    return next(op for op in rec.operands if op.role in ("wide", "computed"))


def test_a_matvec_over_the_components_of_a_vector_input():
    rec = recognize(matvec_input)
    assert rec is not None and rec.dense and not rec.refusals
    assert (rec.reduce_start, rec.reduce_step, rec.reduce_extent) == (0, 1, NF)
    assert rec.operand_names == ("w", "bf")
    assert rec.operand_class == "gemm" and rec.term_form == "product_of_reads"
    assert rec.outputs == (("y", "scalar"),)
    w, bf = rec.operands
    assert (w.reduce_stride, w.reduce_dim) == (1, "col")
    assert (bf.role, bf.dims, bf.bound) == ("wide", (("component", NF, 1),), NF)
    assert (bf.reduce_stride, bf.reduce_dim, bf.dense) == (1, "component", True)


def test_the_same_matvec_over_a_computed_value():
    rec = recognize(matvec_computed)
    assert rec is not None and rec.dense and not rec.refusals
    assert rec.operand_names == ("w", "computed#0")
    assert rec.term_form == "product_of_reads"
    value = _wide(rec)
    assert (value.role, value.dims) == ("computed", (("component", NF, 1),))
    assert (value.reduce_stride, value.reduce_dim) == (1, "component")


def test_a_masked_384_term_sum():
    rec = recognize(masked_sum)
    assert rec is not None and rec.dense and not rec.refusals
    assert rec.reduce_extent == NM
    assert rec.operand_names == ("bf", "w")
    # the select makes each term a transformed one: not a plain product
    assert rec.term_form == "transformed"
    bf = _wide(rec)
    assert bf.dims == (("component", NM, 1),) and bf.reads == 2 * NM


def test_two_sinks_sharing_an_operand_are_one_contraction():
    rec = recognize(two_sinks)
    assert rec is not None and rec.dense and not rec.refusals
    assert rec.outputs == (("u", "scalar"), ("v", "scalar"))
    assert rec.carry == "u"
    assert rec.operand_names == ("w", "bf")
    assert rec.term_form == "product_of_reads"
    assert (rec.reduce_start, rec.reduce_step, rec.reduce_extent) == (0, 1, NF)
    # the shared operand is read by both folds, every iteration
    assert _wide(rec).reads == 2 * NF


def test_a_literal_component_stride_is_not_derivable():
    rec = recognize(strided_by_literal)
    assert rec is not None and not rec.dense
    bf = _wide(rec)
    assert bf.reduce_stride == 2 and bf.reduce_dim is None
    assert any("anonymous integer literal" in r for r in bf.refusals)


def test_shapes_outside_the_family_stay_unrecognised():
    # two loops: two reduce headers, so no single contraction
    assert recognize(two_loops) is None
    # a sink that is not a fold: the set is not all contraction
    assert recognize(fold_and_plain) is None
    # three operands (w, g, bf): outside the one-or-two-operand family
    assert recognize(three_operands) is None
