# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Recognising a sum of products as a CONTRACTION, on HAWK's own IR.

The record :func:`hawk.ir.recognize` produces is the ENTIRE seam between HAWK and
``eagle.gemm.plan.lower_dense_gemm``: HAWK imports no eagle and
eagle imports no producer, so nothing else crosses. The rows here therefore split
in two. Most of them mirror the previous code generator's contraction tests on the four shapes
that matter — the dense KAN ``contract``, the MoE dense block, softmax's
recognised-but-not-lowerable sum, and the index structures that refuse — and one
of them, :func:`test_the_record_carries_exactly_the_fields_eagles_lowering_reads`,
pins the record's FIELD SET, because a field renamed on this side is a
``AttributeError`` in eagle's lowering that nothing on this side would catch.

WHAT IT MATCHES. HAWK matches a ``ForLoopStmt`` with one additive carry, the
same way the previous code generator did: a bounded ``for`` LOWERS to a
:class:`~hawk.ir.loop_nodes.Loop` rather than being unrolled by trace-time execution,
so the reduce variable is a real symbolic index and each operand's stride is the
COEFFICIENT of that index, rather than a difference between
unrolled terms. This includes the rule that matters most: a stride built from a bare literal is
ANONYMOUS and not derivable from the declaration, even when its value happens to
be right (:func:`test_a_hand_folded_stride_is_not_derivable_from_the_declaration`
is that rule, and it is the reason ``Table["row":r, "col":c]`` exists at all).
"""

from __future__ import annotations

import dataclasses

import pytest

import hawk
import hawk.math as m
from hawk.diff import vjp
from hawk.ir import ContractionOperand, RecognizedContraction, recognize

ROWS, N_OUT, BATCH = 4, 3, 5
COLS = 6

#: The field NAMES ``eagle.gemm.plan`` reads off a record, transcribed from
#: ``eagle/python/eagle/gemm/contraction.py``'s two frozen dataclasses. A
#: verbatim twin in the sense ``hawk/_contracts.py`` uses the word: HAWK cannot
#: import eagle to compare, so the shape is pinned here and the executable
#: compatibility row lives on the consumer's side (a downstream package).
_RECORD_FIELDS = (
    "carry", "reduce_var", "reduce_extent", "reduce_start", "reduce_step",
    "operands", "operand_class", "term_form", "outputs", "dense", "refusals",
    "staged_fanin", "fanin_source", "adjoint_accumulates",
)
_OPERAND_FIELDS = (
    "name", "role", "dims", "bound", "reads", "reduce_stride", "reduce_dim",
    "stride_labels", "dense", "refusals",
)


# --------------------------------------------------------------------------- #
# The subjects.
# --------------------------------------------------------------------------- #
@hawk.kernel
def kan_contract(w_fold: hawk.Table["row":ROWS, "neuron":N_OUT],
                 feat: hawk.Staged[hawk.Vector[ROWS], BATCH],
                 y: hawk.Mutable[hawk.Scalar]):
    """``dense_slot.py``'s own ``dense_contract``, in HAWK's spelling: one lane
    per ``(sample, neuron)``, one sum of products over the slot axis."""
    sample, unit = m.split_index(N_OUT)
    total = 0.0
    for r in range(ROWS):
        total = total + w_fold.at(row=r, neuron=unit) * feat.at(plane=r,
                                                                sample=sample)
    y = total


@hawk.kernel
def moe_contract(w: hawk.Table["hidden":ROWS, "model":N_OUT],
                 act: hawk.Staged[hawk.Vector[ROWS], BATCH],
                 y: hawk.Mutable[hawk.Scalar]):
    """``moe_dense_block_exhibit.py``'s stage 2 — the SAME shape under different
    axis names, which is the point: a different family, the same route."""
    sample, unit = m.split_index(N_OUT)
    total = 0.0
    for h in range(ROWS):
        total = total + w.at(hidden=h, model=unit) * act.at(plane=h, sample=sample)
    y = total


@hawk.kernel
def row_sum_exp(scores: hawk.Table["row":ROWS, "col":COLS],
                peak: hawk.Table["row":ROWS],
                total: hawk.Mutable[hawk.Scalar]):
    """``softmax_exhibit.py``'s second pass: recognised, dense, and NOT
    lowerable — its per-term value is ``exp(score - shift)``, which no matmul
    over the declared planes reproduces."""
    s = m.sample_index()
    shift = peak.at(row=s)
    acc = 0.0
    for c in range(COLS):
        acc = acc + m.exp(scores.at(row=s, col=c) - shift)
    total = acc


@hawk.kernel
def gathered(base: hawk.Table["row":ROWS, "col":COLS], span: hawk.Table[hawk.Scalar],
             y: hawk.Mutable[hawk.Scalar]):
    """A data-dependent BASE: the walk is unit-stride but the row it walks is
    chosen at runtime, which is a banded gather and not a matrix."""
    start = span.at(m.sample_index())
    acc = 0.0
    for c in range(COLS):
        acc = acc + base.at(row=start, col=c)
    y = acc


@hawk.kernel
def hand_folded(flat: hawk.Table[hawk.Scalar], y: hawk.Mutable[hawk.Scalar]):
    """The SAME arithmetic as a named-axis read, with the stride multiplied out
    in Python — the number is right and nothing declares it."""
    s = m.sample_index()
    acc = 0.0
    for r in range(ROWS):
        acc = acc + flat.at(r * COLS + s)
    y = acc


@hawk.kernel
def diagonal(square: hawk.Table["row":ROWS, "col":ROWS], y: hawk.Mutable[hawk.Scalar]):
    """Both axes advance together: affine, but no SINGLE declared axis is the
    one being walked."""
    acc = 0.0
    for r in range(ROWS):
        acc = acc + square.at(row=r, col=r)
    y = acc


@hawk.kernel
def one_term(t: hawk.Table["row":ROWS, "col":COLS], y: hawk.Mutable[hawk.Scalar]):
    """One term is not a reduce."""
    row, col = m.split_index(COLS)
    y = t.at(row=row, col=col)


@hawk.kernel
def three_planes(a: hawk.Table["row":ROWS], b: hawk.Table["row":ROWS],
                 c: hawk.Table["row":ROWS], y: hawk.Mutable[hawk.Scalar]):
    """Three distinct planes: outside the one-or-two-operand family."""
    acc = 0.0
    for r in range(ROWS):
        acc = acc + a.at(row=r) * b.at(row=r) + c.at(row=r)
    y = acc


@hawk.kernel
def offset_sum(t: hawk.Table["row":ROWS], y: hawk.Mutable[hawk.Scalar]):
    """A non-zero constant addend: a contraction plus a bias is not a
    contraction, and a matmul over the declared planes would not produce it."""
    acc = 1.5
    for r in range(ROWS):
        acc = acc + t.at(row=r)
    y = acc


@hawk.kernel
def max_fold(t: hawk.Table["row":ROWS], y: hawk.Mutable[hawk.Scalar]):
    """softmax's FIRST pass: a max fold, which is not an additive chain."""
    best = t.at(row=0)
    for r in range(ROWS):
        best = m.max(best, t.at(row=r))
    y = best


# --------------------------------------------------------------------------- #
# The dense, lowerable case.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def kan():
    return recognize(kan_contract, vjp=vjp(kan_contract))


def test_the_dense_sum_of_products_is_recognized(kan):
    """Needs ``_const_factor``'s ``RoleConst`` arm, or a stride a declaration
    minted is not recognized as a compile-time factor and every operand's
    walk stops being affine::

        assert kan.dense and not kan.refusals
        AssertionError: assert (False)
    """
    assert kan is not None
    assert kan.dense and not kan.refusals
    assert kan.operand_class == "gemm"
    assert kan.term_form == "product_of_reads"
    assert (kan.reduce_start, kan.reduce_step, kan.reduce_extent) == (0, 1, ROWS)
    assert kan.carry == "y"
    assert kan.outputs == (("y", "scalar"),)
    assert kan.operand_names == ("w_fold", "feat")


def test_each_operand_walks_a_DECLARED_axis(kan):
    """The stride is not merely a number that fits: it must BE the axis's own
    declared constant, or the lowering has no way to say which axis to contract
    (``eagle.gemm.plan._resolve_operand`` refuses by name when ``reduce_dim`` is
    ``None``)."""
    w_fold, feat = kan.operands
    assert w_fold.role == "named" and feat.role == "staged"
    assert w_fold.dims == (("row", ROWS, N_OUT), ("neuron", N_OUT, 1))
    assert feat.dims == (("plane", ROWS, BATCH), ("sample", BATCH, 1))
    assert (w_fold.reduce_stride, w_fold.reduce_dim) == (N_OUT, "row")
    assert (feat.reduce_stride, feat.reduce_dim) == (BATCH, "plane")
    assert w_fold.stride_labels == ("w_fold_row_stride",)
    assert feat.stride_labels == ("feat_plane_stride",)
    assert w_fold.dense and feat.dense
    # every gate `_check_lowerable` applies, checked here so the record's
    # lowerability is a property of the record and not of eagle being present.
    assert kan.reduce_extent == w_fold.dims[0][1] == feat.dims[0][1]


def test_the_staged_fan_in_is_derived_from_the_major_half_of_the_split(kan):
    """The staged plane is addressed by the MAJOR half of ``split_index(N_OUT)``,
    so ``N_OUT`` lanes share each slot and the reverse pass accumulates exactly
    that many contributions into it. ``eagle.gemm.plan.verify_staged_fanin``
    checks this against the adjoint GEMM's contracted extent — the OTHER
    operand's free extent, which is ``w_fold``'s ``neuron`` axis, ``N_OUT``."""
    assert kan.staged_fanin == (("feat", N_OUT),)
    assert kan.fanin_source == "primal_walk"
    free_extent_of_the_other_operand = kan.operands[0].dims[1][1]
    assert free_extent_of_the_other_operand == N_OUT


def test_the_adjoint_record_comes_from_the_vjp_walk(kan):
    """``adjoint_accumulates`` is what tells a consumer the reverse direction
    lands in SHARED slots. Empty is not the same as absent, which is why the
    record carries it only when a derived IR was supplied."""
    names = {name for name, _atomic in kan.adjoint_accumulates}
    assert names == {"bar_w_fold", "bar_feat"}, kan.adjoint_accumulates
    assert all(atomic is False for _n, atomic in kan.adjoint_accumulates), (
        "hawk.ext names an `atomic` sink policy but does not build it, so no "
        "derived kernel of HAWK's accumulates atomically")
    without = recognize(kan_contract)
    assert without.adjoint_accumulates == ()
    assert without.dense and without.staged_fanin == (("feat", N_OUT),)


def test_a_second_basis_family_recognizes_identically():
    """The point, and the MoE exhibit's: a different family under different
    axis names is the SAME record shape, so one route serves both."""
    moe = recognize(moe_contract)
    assert moe.dense and moe.term_form == "product_of_reads"
    assert moe.operands[0].reduce_dim == "hidden"
    assert moe.operands[1].reduce_dim == "plane"
    assert moe.reduce_extent == ROWS


# --------------------------------------------------------------------------- #
# Recognised, NOT lowerable.
# --------------------------------------------------------------------------- #
def test_a_transformed_term_is_recognized_but_not_lowerable():
    """``sum(x[s,c])`` and ``sum(exp(x[s,c] - shift))`` walk the same indices at
    the same stride and are equally DENSE; only the first is a row sum. Without
    ``term_form`` their records would compare equal and a lowering would compute
    the first one's answer for both."""
    record = recognize(row_sum_exp)
    assert record is not None
    assert record.dense and not record.refusals
    assert record.term_form == "transformed"
    shift = next(op for op in record.operands if op.name == "peak")
    assert shift.reduce_stride == 0, (
        "the shift is read once per term at the SAME address: it is a "
        "coefficient of the contraction, not a second thing being summed")
    walked = next(op for op in record.operands if op.name == "scores")
    assert (walked.reduce_stride, walked.reduce_dim) == (1, "col")


# --------------------------------------------------------------------------- #
# The index structures that refuse.
# --------------------------------------------------------------------------- #
def test_a_data_dependent_base_is_recognized_and_refused_by_name():
    record = recognize(gathered)
    assert record is not None and not record.dense
    assert any("chosen at runtime" in r for r in record.refusals), record.refusals


def test_a_hand_folded_stride_is_not_derivable_from_the_declaration():
    """Needs the base case of ``_step`` to distinguish the two: if it
    returns ``anonymous=False`` unconditionally, a stride folded into a
    literal by Python reads exactly like a declared one::

        assert record is not None and not record.dense
        AssertionError: assert (RecognizedContraction(carry='y', ...,
        term_form='bare_read', outputs=(('y', 'scalar')), dense=True,
        refusals=, ...) is not None and not True)

    The value is right in both spellings; only one of them is a declaration, and
    reading the other would be reading a coincidence."""
    record = recognize(hand_folded)
    assert record is not None and not record.dense
    assert any("anonymous integer literal" in r for r in record.refusals), (
        record.refusals)
    assert record.operands[0].reduce_stride == COLS
    assert record.operands[0].reduce_dim is None


def test_two_axes_advancing_together_name_no_single_reduce_axis():
    record = recognize(diagonal)
    assert record is not None
    assert record.operands[0].reduce_dim is None, (
        "the walk is affine but no SINGLE declared axis is the contracted one; "
        "attaching a label the arithmetic does not support is what the "
        "stride_labels check is there to prevent")
    assert len(record.operands[0].stride_labels) == 2


@pytest.mark.parametrize("kernel,why", [
    (one_term, "one term is not a reduce"),
    (three_planes, "three distinct planes is outside the family"),
    (offset_sum, "a non-zero constant addend is a bias, not a term"),
    (max_fold, "a max fold is not an additive chain"),
], ids=["one_term", "three_planes", "offset_sum", "max_fold"])
def test_a_body_outside_the_family_is_not_recognized(kernel, why):
    assert recognize(kernel) is None, why


def test_a_derivative_ir_and_a_non_kernel_are_not_recognized():
    """``recognize`` reads a PRIMAL; handed a reverse IR (many sinks, gradients
    and scatters) or something that is not a kernel at all, it says ``None``
    rather than describing whatever it found."""
    assert recognize(vjp(kan_contract)) is None
    assert recognize(object()) is None


# --------------------------------------------------------------------------- #
# The record's shape — the whole seam.
# --------------------------------------------------------------------------- #
def test_the_record_carries_exactly_the_fields_eagles_lowering_reads(kan):
    """A field renamed here is an ``AttributeError`` inside eagle's lowering,
    which nothing on this side of the line would otherwise catch (HAWK
    cannot import eagle to compare)."""
    assert tuple(f.name for f in dataclasses.fields(RecognizedContraction))\
        == _RECORD_FIELDS
    assert tuple(f.name for f in dataclasses.fields(ContractionOperand))\
        == _OPERAND_FIELDS
    for field in _RECORD_FIELDS:
        assert hasattr(kan, field), field
    for operand in kan.operands:
        for field in _OPERAND_FIELDS:
            assert hasattr(operand, field), field
    assert kan.operand_count == 2
    assert isinstance(kan.reduce_extent, int)
    assert all(isinstance(part, (str, int)) for dim in kan.operands[0].dims
               for part in dim)


def test_recognition_mutates_nothing():
    """Reads only: it compiles nothing, launches nothing and rewrites no IR, so
    calling it leaves the kernel's emitted body byte-identical."""
    from hawk.emit import render_body

    before = render_body(kan_contract.sinks, kan_contract.walk).text
    digest = kan_contract.walk.digest
    recognize(kan_contract, vjp=vjp(kan_contract))
    assert render_body(kan_contract.sinks, kan_contract.walk).text == before
    assert kan_contract.walk.digest == digest


# --------------------------------------------------------------------------- #
# A fold committed as ONE live component of a vector output.
# --------------------------------------------------------------------------- #
@hawk.kernel
def vec_committed(t: hawk.Table["row":ROWS], y: hawk.Mutable[hawk.Vector[2]]):
    """A banded KAN stage commits ``y = vec(total, 0.0)``: the same
    fold, padded to the declared width with literal zeros."""
    acc = 0.0
    for r in range(ROWS):
        acc = acc + t.at(row=r)
    y = m.vec(acc, 0.0)


@hawk.kernel
def vec_committed_twice(t: hawk.Table["row":ROWS], y: hawk.Mutable[hawk.Vector[2]]):
    """Two live components are not ONE fold, even when both read the same carry."""
    acc = 0.0
    for r in range(ROWS):
        acc = acc + t.at(row=r)
    y = m.vec(acc, acc)


def test_a_fold_committed_through_a_vector_constructor_is_recognized():
    """A prior gap: ``_fold`` required the
    sink's value to BE the carried slot, so ``y = vec(total, 0.0)`` went
    unrecognised (``None``) and the consumer's routing record had nothing to
    read for a declaration that IS one additive fold."""
    rec = recognize(vec_committed)
    assert rec is not None
    assert rec.carry == "y" and rec.outputs == (("y", "vector"),)
    assert (rec.reduce_start, rec.reduce_step, rec.reduce_extent) == (0, 1, ROWS)
    assert rec.operand_class == "gemv"
    assert recognize(vec_committed_twice) is None
