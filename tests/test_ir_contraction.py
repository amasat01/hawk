# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The dense contraction a kernel is, as the library states it in ``describe(...)[kernel]["contractions"]``.

The library decides which sums of products run as a matrix product and which run as the kernel's own loop; ``describe`` (and the
sidecar a build writes) carries its answer: the ``unit``, the ``route`` and the ``products`` with their shapes. A kernel that is no
contraction the library routes states no such member. The kernels here are the shapes that matter: the dense spline contraction,
the same shape under other axis names, a launch-counted staged buffer, the operand halves of a split lane, and the index
structures and bodies that state none.
"""

from __future__ import annotations

import numpy as np
import pytest

import hawk
import hawk.math as m
from hawk.artifact import describe

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
    "staged_fanin", "fanin_source", "adjoint_accumulates", "lane_major",
)
_OPERAND_FIELDS = (
    "name", "role", "dims", "bound", "reads", "reduce_stride", "reduce_dim",
    "stride_labels", "dense", "refusals",
)


# --------------------------------------------------------------------------- #
# The subjects.
# --------------------------------------------------------------------------- #
@hawk.kernel
def spline_contract(w_fold: hawk.Table["row":ROWS, "neuron":N_OUT],
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
def _dot_staged(count):
    """One lane per sample (the lane IS the sample), a dot over the plane axis of a staged buffer of ``count`` samples."""
    @hawk.kernel
    def dot_staged(w: hawk.Table["row":ROWS], feat: hawk.Staged[hawk.Vector[ROWS], count], y: hawk.Mutable[hawk.Scalar]):
        s = m.sample_index()
        total = 0.0
        for r in range(ROWS):
            total = total + w.at(r) * feat.at(plane=r, sample=s)
        y = total
    return dot_staged


@hawk.kernel
def spline_contract_n(w_fold: hawk.Table["row":ROWS, "neuron":N_OUT],
                   feat: hawk.Staged[hawk.Vector[ROWS], m.n_samples],
                   y: hawk.Mutable[hawk.Scalar]):
    sample, unit = m.split_index(N_OUT)
    total = 0.0
    for r in range(ROWS):
        total = total + w_fold.at(row=r, neuron=unit) * feat.at(plane=r, sample=sample)
    y = total


@pytest.mark.parametrize("n", [7, 64])
def test_the_launch_counted_form_matches_the_int_form_bitwise(n, tmp_path, cache_dir):
    from conftest import sidecar_of
    from hawk import runtime as rt
    from hawk.artifact import build_bundle
    rng = np.random.default_rng(3)
    w, feat = rng.uniform(-1, 1, ROWS), rng.uniform(-1, 1, ROWS * n)
    got = {}
    for tag, count in (("int", n), ("sentinel", m.n_samples)):
        k = _dot_staged(count)
        bundle = build_bundle([k], tmp_path / tag, targets=("host",), cache_dir=cache_dir, host_profile="native")
        y = np.zeros(n)
        rt.run(rt.load(bundle.directory, k.name, sidecar_of(bundle, k.name)), w=w, feat=feat, y=y, n_samples=n)
        got[tag] = y
    assert np.array_equal(got["int"], got["sentinel"])
    np.testing.assert_allclose(got["int"], np.einsum("r,rs->s", w, feat.reshape(ROWS, n)), rtol=1e-13)


# --------------------------------------------------------------------------- #
# Recognised, NOT lowerable.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The index structures that refuse.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The record's shape — the whole seam.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# A fold committed as ONE live component of a vector output.
# --------------------------------------------------------------------------- #
@hawk.kernel
def vec_committed(t: hawk.Table["row":ROWS], y: hawk.Mutable[hawk.Vector[2]]):
    """A banded spline stage commits ``y = vec(total, 0.0)``: the same
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


def test_the_launch_counted_kernel_is_one_build_for_every_size():
    assert repr(_dot_staged(7).planes["feat"]) != repr(_dot_staged(64).planes["feat"])
    assert repr(_dot_staged(m.n_samples).planes["feat"]) == repr(_dot_staged(m.n_samples).planes["feat"])
    assert "7" not in repr(_dot_staged(m.n_samples).planes["feat"])


# --------------------------------------------------------------------------- #
# ``lane_major``: which operand the lanes are major in.
# --------------------------------------------------------------------------- #
@hawk.kernel
def major_is_second(
    a: hawk.Table["row":ROWS, "col":N_OUT],
    b: hawk.Table["row":ROWS, "col":BATCH],
    y: hawk.Mutable[hawk.Scalar],
):
    """``a`` is read through the minor half of the lane, ``b`` through the major half."""
    major, minor = m.split_index(N_OUT)
    total = 0.0
    for r in range(ROWS):
        total = total + a.at(row=r, col=minor) * b.at(row=r, col=major)
    y = total


@hawk.kernel
def major_is_first(
    a: hawk.Table["row":ROWS, "col":BATCH],
    b: hawk.Table["row":ROWS, "col":N_OUT],
    y: hawk.Mutable[hawk.Scalar],
):
    """The mirror of :func:`major_is_second`: the operands swap their halves."""
    major, minor = m.split_index(N_OUT)
    total = 0.0
    for r in range(ROWS):
        total = total + a.at(row=r, col=major) * b.at(row=r, col=minor)
    y = total


@hawk.kernel
def major_for_both(
    a: hawk.Table["row":ROWS, "col":BATCH],
    b: hawk.Table["row":ROWS, "col":BATCH],
    y: hawk.Mutable[hawk.Scalar],
):
    """Both operands are read through the major half: no operand is the minor one."""
    major, _minor = m.split_index(N_OUT)
    total = 0.0
    for r in range(ROWS):
        total = total + a.at(row=r, col=major) * b.at(row=r, col=major)
    y = total


# --------------------------------------------------------------------------- #
# What the library states.
# --------------------------------------------------------------------------- #
def _stated(kernel):
    """The ``contractions`` member of the description of ``kernel`` for the host, or None when it states none."""
    return describe([kernel], targets=("host",))[kernel.name].get("contractions")


def _shape(call):
    """``(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols)`` of one product call."""
    return tuple(call[k] for k in ("a_rows", "a_cols", "b_rows", "b_cols", "c_rows", "c_cols"))


def test_the_dense_sum_of_products_states_a_matrix_product_route_and_its_shapes():
    [found] = _stated(spline_contract)
    assert (found["unit"], found["route"]) == ("spline_contract", "gemm")
    [call] = found["products"]["calls"]
    assert _shape(call) == (ROWS, BATCH, ROWS, N_OUT, BATCH, N_OUT)     # the reduce extent is the shared row count


def test_a_staged_buffer_counted_by_the_launch_states_run_time_extents_and_the_lane_fan_in():
    [found] = _stated(spline_contract_n)
    [call] = found["products"]["calls"]
    assert found["route"] == "gemm"
    assert _shape(call) == (ROWS, -1, ROWS, N_OUT, -1, N_OUT)           # -1: counted by the launch
    assert found["products"]["lane_fanin"] == N_OUT


def test_a_second_basis_family_states_the_same_route_and_products():
    [moe] = _stated(moe_contract)
    [spline] = _stated(spline_contract)
    assert moe["route"] == spline["route"] and moe["products"] == spline["products"]


@pytest.mark.parametrize("kernel", [
    row_sum_exp, gathered, hand_folded, diagonal, one_term, three_planes, offset_sum, max_fold, vec_committed_twice,
], ids=lambda k: k.name)
def test_a_body_that_is_no_matrix_product_states_no_contraction(kernel):
    assert _stated(kernel) is None


def test_the_operand_slots_follow_the_half_of_the_lane_each_operand_is_read_through():
    [second] = _stated(major_is_second)
    [first] = _stated(major_is_first)
    second_call, = second["products"]["calls"]
    first_call, = first["products"]["calls"]
    assert (second_call["a_slot"], second_call["b_slot"]) == (2, 1)
    assert (first_call["a_slot"], first_call["b_slot"]) == (1, 2)
    assert _stated(major_for_both) is None      # no operand is the minor one: no product
