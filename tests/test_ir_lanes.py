# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The lane-space facts record: :mod:`hawk.ir.lanes`.

Kernels are written with ``split_index`` and flat forms; the axes come from
the divisors the body uses and ``labels=`` names them. Every row states a fact
the record must hold and the conservative answer where it cannot prove one
(unknown = per-lane). The ``Sorted`` rows mark the plane's ``Leaf`` by hand: the
declaration itself is the tracer's.
"""

from __future__ import annotations

import pytest

import hawk
import hawk.math as m
from hawk import Index, Mutable, Scalar, Table, Vector
from hawk.ir import lanes
from hawk.ir.lanes import (AFFINE, DATA_DEPENDENT, INVARIANT, NON_AFFINE, ORIGIN_AXIS,
                           ORIGIN_DATA, ORIGIN_SORTED, ORIGIN_UNSORTED, PROVEN, QUOTIENT,
                           SORTED, TESTED)
from hawk.ir.nodes import Leaf

NS = 8192          # the fastest axis of the keyed-accumulate kernels
TILE = {"kind": "BF", "SB": 32, "T": 16, "R": 1, "EB": 1, "smem": 48 * 1024}
CARD = {"T": 16, "itemsize": 4, "tile": TILE}


def f(x, s):
    return x * x + 0.25 * x + s * 1.0e-4


def keyed(kernel, **kw):
    return [s for s in lanes.facts(kernel, **kw).sinks
            if s.keyed and s.name not in kernel.walk.companions]


def one(kernel, **kw):
    (s,) = keyed(kernel, **kw)
    return s


def run_of(kernel):
    (s,) = [n for n in kernel.sinks if getattr(n, "index", None) is not None
            and n.name not in kernel.walk.companions]
    return lanes.key_run(kernel.sinks, s)


def declare_sorted(kernel, plane):
    marked = [n for n in lanes._nodes_of(kernel) if isinstance(n, Leaf) and n.name == plane]
    assert marked
    for n in marked:
        n.sorted = True
    return kernel


def undeclare_sorted(kernel):
    for n in lanes._nodes_of(kernel):
        if isinstance(n, Leaf):
            n.sorted = False


# ---- keyed accumulates over a (channel, sample) lane space ----------------
@hawk.kernel
def k_uni(x: Scalar, acc: Mutable[Scalar]):
    j, s = m.split_index(NS)
    hawk.scatter(acc)[j] += f(x, s)


@hawk.kernel
def k_runtime_stride(x: Scalar, acc: Mutable[Scalar]):
    j, s = m.split_index(NS)
    hawk.scatter(acc)[j * m.n_samples() + 3] += f(x, s)


@hawk.kernel
def k_straddle(x: Scalar, acc: Mutable[Scalar]):
    j, s = m.split_index(48)
    hawk.scatter(acc)[j] += f(x, s)


@hawk.kernel
def k_tested(x: Scalar, acc: Mutable[Scalar]):
    lane = m.sample_index()
    hawk.scatter(acc)[lane / 64] += f(x, lane - (lane / 6) * 6)


@hawk.kernel
def k_plane(x: Scalar, key: Index, acc: Mutable[Scalar]):
    j, s = m.split_index(NS)
    hawk.scatter(acc)[key] += f(x, s)


@hawk.kernel
def k_plane_const(x: Scalar, key: Index, acc: Mutable[Scalar]):
    j, s = m.split_index(NS)
    hawk.scatter(acc)[key + 3] += f(x, s)


@hawk.kernel
def k_plane_minus(x: Scalar, key: Index, acc: Mutable[Scalar]):
    j, s = m.split_index(NS)
    hawk.scatter(acc)[key - 3] += f(x, s)


@hawk.kernel
def k_plane_twice(x: Scalar, key: Index, acc: Mutable[Scalar]):
    j, s = m.split_index(NS)
    hawk.scatter(acc)[key * 2] += f(x, s)


@hawk.kernel
def k_plane_negated(x: Scalar, key: Index, acc: Mutable[Scalar]):
    j, s = m.split_index(NS)
    hawk.scatter(acc)[3 - key] += f(x, s)


@hawk.kernel
def k_plane_plus_lane(x: Scalar, key: Index, acc: Mutable[Scalar]):
    j, s = m.split_index(NS)
    hawk.scatter(acc)[key + s] += f(x, s)


@hawk.kernel
def k_clamp(x: Scalar, acc: Mutable[Scalar]):
    hawk.scatter(acc)[m.index(x, 9)] += x


@hawk.kernel
def k_clamp_shifted(x: Scalar, acc: Mutable[Scalar]):
    hawk.scatter(acc)[5 + m.index(x, 9)] += x


@hawk.kernel
def k_const(x: Scalar, acc: Mutable[Scalar]):
    hawk.scatter(acc)[7] += x


@hawk.kernel
def k_static_axes(x: Scalar, acc: Mutable[Scalar]):
    lane = m.sample_index()
    a = lane / 32
    b = lane / 8 - a * 4
    c = lane - (lane / 8) * 8
    hawk.scatter(acc)[b * 8 + c] += x


@hawk.kernel
def k_key_by_fastest(x: Scalar, acc: Mutable[Scalar]):
    j, s = m.split_index(64)
    hawk.scatter(acc)[s] += x


# ---- FEM scatter ---------------------------------------------------------
@hawk.kernel
def k_fem(val: Scalar, conn: Table[Index], out: Mutable[Scalar]):
    e, v = m.split_index(4)
    hawk.scatter(out)[conn.at(e * 4 + v)] += val


@hawk.kernel
def k_fem_shifted(val: Scalar, conn: Table[Index], out: Mutable[Scalar]):
    e, v = m.split_index(4)
    hawk.scatter(out)[conn.at(e * 4 + v) + 1] += val


# ---- the contraction forward and partial-sum forms --------------------------------
N_IN, N_OUT, BATCH, G, K = 8, 8, 256, 5, 2
NB = G + K
S_ROW = NB + 2
E_ALL = N_IN * N_OUT
CHUNK = BATCH // 2
L_ALL = N_IN * BATCH
O_N, O_S = 1, 2 * K + 3


def _feat(feat, slot, l):
    return feat.at(slot * L_ALL + l)


def _edge(feat, th, l, b):
    i = m.index(_feat(feat, 0, l), G)
    acc = 0.0
    for r in range(K + 1):
        acc = acc + _feat(feat, O_N + r, l) * th.at(b + 2 + i + r)
    return acc


@hawk.kernel
def kg_fwd(lane: Index, feat: Table[Scalar], th: Table[Scalar], y: Mutable[Scalar]):
    j, s = m.split_index(BATCH)
    acc = 0.0 * m.real(lane)
    for e in range(N_IN):
        l = e * BATCH + s
        b = (j * N_IN + e) * S_ROW
        acc = acc + th.at(b) * _feat(feat, O_S, l) + th.at(b + 1) * _edge(feat, th, l, b)
    y = acc


def _zeros():
    return m.vec(*([0.0] * S_ROW))


def _terms(feat, l, ybs):
    return m.vec(ybs * _feat(feat, O_S, l), 0.0, *[ybs * _feat(feat, O_N + r, l) for r in range(NB)])


def _finish(acc, th, b):
    return m.vec(acc[0], th.at(b + 1) * acc[2], *[th.at(b + 2 + g) * acc[2 + g] for g in range(NB)])


@hawk.kernel
def kg_tpart(lane: Index, feat: Table[Scalar], yb: Table[Scalar], th: Table[Scalar],
             part: Mutable[Vector[S_ROW]]):
    c, q = m.split_index(E_ALL)
    e = m.index((m.real(q) + 0.5) / N_OUT, N_IN)
    j = q - e * N_OUT
    b = (j * N_IN + e) * S_ROW + 0 * lane
    acc = _zeros()
    for t in range(CHUNK):
        l = e * BATCH + c * CHUNK + t
        acc = acc + _terms(feat, l, yb.at(j * BATCH + c * CHUNK + t))
    part = _finish(acc, th, b)


@hawk.kernel
def kg_wide(feat: Table[Scalar], th: Table[Scalar], y: Mutable[Scalar]):
    j, s = m.split_index(BATCH)
    y = th.at(j * 48 + m.index(feat.at(s), 40))


@hawk.kernel
def kg_flat(feat: Table[Scalar], th: Table[Scalar], y: Mutable[Scalar]):
    y = th.at(m.index(feat.at(m.sample_index()), G) + 2)


# ---- stencils, gathers, lane algebra --------------------------------------
@hawk.kernel
def k_stencil(i: Index, c: Table[Scalar], y: Mutable[Scalar]):
    y = c.at(i) + c.at(m.sample_index() + 1) + c.at(m.sample_index() - 1)


@hawk.kernel
def k_quarter(c: Table[Scalar], y: Mutable[Scalar]):
    y = c.at(m.sample_index() / 4)


@hawk.kernel
def k_two_quotients(c: Table[Scalar], y: Mutable[Scalar]):
    y = c.at(m.sample_index() / 4) + c.at(m.sample_index() / 6)


@hawk.kernel
def k_square(c: Table[Scalar], y: Mutable[Scalar]):
    lane = m.sample_index()
    y = c.at(lane * lane)


@hawk.kernel
def k_break(x: Table[Scalar], y: Mutable[Scalar]):
    j, s = m.split_index(8)
    acc = 0.0 * m.real(m.sample_index())
    for e in range(16):
        acc = acc + x.at(e * 8 + s)
        if acc > 3.0:
            break
    y = acc


# ---------------------------------------------------------------------------
# axes, reads, sink keys
# ---------------------------------------------------------------------------
def test_an_undeclared_kernel_has_one_unlabeled_runtime_axis():
    fa = lanes.facts(k_stencil)
    assert [(a.label, a.extent, a.stride) for a in fa.axes] == [(None, None, 1)]
    assert (fa.group, fa.required_group) == (1, 1)


def test_labels_name_the_axes_in_numpy_order_and_none_stays_unlabeled():
    fa = lanes.facts(kg_fwd, labels=("channel", "sample"))
    assert [(a.label, a.extent, a.stride) for a in fa.axes] == [("channel", None, BATCH),
                                                              ("sample", BATCH, 1)]
    assert [a.label for a in lanes.facts(kg_fwd, labels=(None, "sample")).axes] == [None, "sample"]
    with pytest.raises(ValueError, match="do not match"):
        lanes.facts(kg_fwd, labels=("a", "b", "c"))
    with pytest.raises(ValueError, match="group"):
        lanes.facts(kg_fwd, group=0)


def test_contraction_forward_read_forms():
    fa = lanes.facts(kg_fwd, labels=("channel", "sample"))
    assert fa.read_classes("feat", 0) == (INVARIANT,)
    assert fa.read_classes("feat", "sample") == (AFFINE,)
    assert {(r.forms[1].cls, r.gather) for r in fa.reads if r.plane == "th"} >= {(INVARIANT, False)}
    assert all(r.base[1].cls == INVARIANT for r in fa.reads if r.plane == "th")
    gathered = [r for r in fa.reads if r.plane == "th" and r.gather]
    assert gathered and all(r.forms[1].cls == DATA_DEPENDENT and r.forms[0].cls == AFFINE
                          for r in gathered)
    (y,) = fa.sinks
    assert not y.keyed and [x.cls for x in y.forms] == [AFFINE, AFFINE]


def test_contraction_partial_read_forms():
    ft = lanes.facts(kg_tpart, labels=("chunk", "edge"))
    assert all(r.forms[0].cls == INVARIANT for r in ft.reads if r.plane == "th")
    assert all(r.forms[0].cls == AFFINE for r in ft.reads if r.plane in ("feat", "yb"))
    assert all(r.forms[1].cls == NON_AFFINE for r in ft.reads)
    (part,) = ft.sinks
    assert not part.keyed and part.key_origin is None
    assert [x.cls for x in part.forms] == [AFFINE, AFFINE]


def test_stencil_offsets_and_gather():
    fa = lanes.facts(k_stencil)
    seen = {(r.forms[0].cls, r.gather, r.lane_offset) for r in fa.reads}
    assert (DATA_DEPENDENT, True, None) in seen
    assert (AFFINE, False, 1) in seen and (AFFINE, False, -1) in seen
    assert {r.offset for r in fa.reads if r.lane_offset is not None} == {1, -1}


def test_a_lone_divisor_is_a_two_axis_lane_space():
    fa = lanes.facts(k_quarter)
    assert [a.extent for a in fa.axes] == [None, 4]
    assert [x.cls for x in fa.reads[0].forms] == [AFFINE, INVARIANT]


def test_divisors_that_are_not_a_chain_fall_back_to_one_flat_axis():
    fa = lanes.facts(k_two_quotients)
    assert len(fa.axes) == 1 and fa.notes
    assert sorted(str(r.forms[0].coefficient) for r in fa.reads) == ["1/4", "1/6"]
    assert all(r.forms[0].cls == QUOTIENT for r in fa.reads)


def test_a_product_of_lane_terms_is_non_affine():
    assert lanes.facts(k_square).reads[0].forms[0].cls == NON_AFFINE


def test_keyed_sink_invariant_axes():
    fa = lanes.facts(k_uni, labels=("channel", "sample"))
    (s,) = [x for x in fa.sinks if x.keyed and x.name not in k_uni.walk.companions]
    assert s.invariant_axes(fa) == ("sample",)
    fb = lanes.facts(k_key_by_fastest, labels=("c", "q"))
    (s,) = [x for x in fb.sinks if x.keyed and x.name not in k_key_by_fastest.walk.companions]
    assert s.invariant_axes(fb) == ("c",)
    fp = lanes.facts(k_plane)
    (s,) = [x for x in fp.sinks if x.keyed and x.name not in k_plane.walk.companions]
    assert s.gather and s.invariant_axes(fp) == ()


# ---- key origin, extents ---------------------------------------------------
@pytest.mark.parametrize("kernel, origin", [(k_uni, ORIGIN_AXIS), (k_straddle, ORIGIN_AXIS),
                                            (k_tested, ORIGIN_AXIS), (k_plane, ORIGIN_UNSORTED),
                                            (k_fem, ORIGIN_UNSORTED),
                                            (k_fem_shifted, ORIGIN_UNSORTED)])
def test_key_origin_without_a_declaration(kernel, origin):
    s = one(kernel)
    assert s.key_origin == origin
    assert s.key_extent is None and s.slot_extent is None and s.slot_base_forms == ()


@pytest.mark.parametrize("kernel, origin", [
    (k_plane, ORIGIN_SORTED), (k_plane_const, ORIGIN_SORTED), (k_plane_minus, ORIGIN_SORTED),
    (k_plane_twice, ORIGIN_DATA),             # a product is opaque plane data: conservative
    (k_plane_negated, ORIGIN_DATA),           # order reversed
    (k_plane_plus_lane, ORIGIN_UNSORTED)])    # the added part moves with a lane axis
def test_key_origin_of_a_sorted_plane(kernel, origin):
    declare_sorted(kernel, "key")
    try:
        s = one(kernel)
        assert s.key_origin == origin and s.key_extent is None
    finally:
        undeclare_sorted(kernel)


def test_the_sorted_flag_defaults_false_and_is_outside_the_digest():
    leaf = Leaf("table_read", "lookup", "seg", m.index(1.0, 3).ttype)
    assert leaf.sorted is False
    assert Leaf("table_read", "lookup", "seg", leaf.ttype, sorted=True).sorted is True
    before = k_plane.walk.digest
    declare_sorted(k_plane, "key")
    try:
        assert k_plane.walk.digest == before
    finally:
        undeclare_sorted(k_plane)


def test_static_key_extents():
    s = one(k_const)
    assert s.key_origin == ORIGIN_AXIS and s.key_extent == 8
    s = one(k_clamp)
    assert (s.key_origin, s.key_extent, s.slot_extent) == (ORIGIN_DATA, 9, 9)
    s = one(k_clamp_shifted)
    assert (s.key_origin, s.key_extent, s.slot_extent) == (ORIGIN_DATA, 14, 9)
    assert all(x.cls == INVARIANT for x in s.slot_base_forms)
    s = one(k_static_axes)
    assert s.key_origin == ORIGIN_AXIS and s.key_extent == 32


def test_policy_is_compensated_when_a_companion_plane_is_carried():
    assert {s.name: s.policy for s in lanes.facts(k_uni).sinks} == {"acc": "compensated",
                                                                    "acc_c": "auto"}
    comp = hawk.ext.Kind("lanes_comp", sink=hawk.ext.sink_policy("compensated", into="acc_c"))

    @hawk.kernel(kind=comp)
    def k_comp(x: Scalar, acc: Mutable[Scalar], acc_c: Mutable[Scalar]):
        j, s = m.split_index(NS)
        hawk.scatter(acc, policy="plain")[j] += f(x, s)
        hawk.scatter(acc_c, policy="plain")[j] += 0.0

    assert {s.name: s.policy for s in lanes.facts(k_comp).sinks} == {"acc": "compensated",
                                                                   "acc_c": "auto"}


def test_policy_reads_the_scatter_policy():
    @hawk.kernel
    def k_policy(x: Scalar, acc: Mutable[Scalar]):
        j, s = m.split_index(NS)
        hawk.scatter(acc, policy="compensated")[j] += f(x, s)

    assert {s.name: s.policy for s in lanes.facts(k_policy).sinks} == {"acc": "compensated",
                                                                     "acc_c": "auto"}


# ---- key runs and the required group --------------------------------------
def test_key_run_forms():
    proven = run_of(k_uni)
    assert (proven.form, proven.width) == (PROVEN, 32)
    straddle = run_of(k_straddle)               # 48-wide rows: a warp may straddle two keys
    assert (straddle.form, straddle.width) == (PROVEN, 16)
    tested = run_of(k_tested)
    assert (tested.form, tested.width) == (TESTED, 32)
    assert run_of(k_plane) is None              # a gather key
    assert run_of(k_key_by_fastest) is None     # moves along the fastest axis
    assert run_of(k_fem) is None


def test_a_runtime_stride_does_not_move_with_the_lane():
    """``n_samples`` is the same for every lane, so a key that scales a lane
    axis by it still holds one value along the other axis."""
    run = run_of(k_runtime_stride)
    assert (run.form, run.width) == (PROVEN, 32)


def test_key_run_of_an_own_column_sink_is_none():
    (sink,) = kg_fwd.sinks
    assert lanes.key_run(kg_fwd.sinks, sink) is None


def test_key_run_of_a_sorted_plane_is_runtime_found():
    declare_sorted(k_plane, "key")
    try:
        r = run_of(k_plane)
        assert (r.form, r.width) == (SORTED, 32)
    finally:
        undeclare_sorted(k_plane)


@pytest.mark.parametrize("kernel, declared, required", [
    (k_uni, 1, 32), (k_straddle, 48, 48), (k_uni, 8, 32), (k_straddle, 1, 16),
    (k_tested, 1, 1), (k_plane, 1, 1), (kg_fwd, 1, 1), (kg_fwd, 12, 12)])
def test_required_group(kernel, declared, required):
    fa = lanes.facts(kernel, group=declared)
    assert (fa.group, fa.required_group) == (declared, required)


# ---- thread-box candidates and the block box --------------------------------
def test_box_candidates_of_the_contraction_forms():
    fa = lanes.facts(kg_fwd, labels=("channel", "sample"))
    (c,) = fa.box_candidates
    assert (c.axis, c.planes, c.fastest_extent) == ("channel", ("feat",), BATCH)
    assert c.reuse is None and not c.static             # the first axis is a runtime extent
    ft = lanes.facts(kg_tpart, labels=("chunk", "edge"))
    assert [c.planes for c in ft.box_candidates] == [("th",)]


@pytest.mark.parametrize("kernel", [k_fem, k_clamp, k_const, k_uni, k_tested])
def test_no_box_candidate_for_scatter_and_histogram(kernel):
    fa = lanes.facts(kernel)
    assert fa.box_candidates == ()
    assert lanes.block_box_eligible(fa, reuse=64) == (False, "no invariant-axis read")


def test_block_box_eligible_reasons():
    fa = lanes.facts(kg_fwd, labels=("channel", "sample"))
    assert lanes.block_box_eligible(fa) == (False, "reuse extent is runtime")
    for g in (None, 1, 32, BATCH):
        ok, _ = lanes.block_box_eligible(fa, required_group=g, reuse=N_OUT)
        assert ok
    assert lanes.block_box_eligible(fa, required_group=8, reuse=N_OUT) == (False, "group 8 < box 32")
    ok, why = lanes.block_box_eligible(fa, required_group=48, reuse=N_OUT)
    assert not ok and why == "group 48 not a multiple of box 32"
    assert lanes.block_box_eligible(fa, reuse=N_OUT, uniform_control=False) \
        == (False, "control diverges: masked-alive path")
    assert lanes.block_box_eligible(fa, reuse=1) == (False, "reuse 1 < 2")
    import dataclasses
    (c,) = fa.box_candidates
    runtime = dataclasses.replace(fa, box_candidates=(dataclasses.replace(
        c, fastest_extent=None, static=False),))
    assert lanes.block_box_eligible(runtime, reuse=N_OUT) == (False, "fastest extent is runtime")


def test_block_box_reads_the_records_own_group():
    fa = lanes.facts(kg_fwd, labels=("channel", "sample"), group=8)
    assert lanes.block_box_eligible(fa, reuse=N_OUT) == (False, "group 8 < box 32")


def test_uniform_control():
    assert lanes.facts(kg_fwd).uniform_control
    assert lanes.facts(k_clamp).uniform_control
    assert not lanes.facts(k_break).uniform_control


def test_choose_mapping_records_the_decision_from_the_card():
    fa = lanes.facts(kg_fwd, labels=("channel", "sample"))
    got = lanes.choose_mapping(fa, card=CARD, reuse=N_OUT)
    assert got is fa.mapping
    assert got["P1"] == {"axis": "channel", "index": 0, "T": 16}
    assert got["block_box"]["planes"] == ("feat",)
    small = lanes.choose_mapping(fa, card=dict(CARD, tile=dict(TILE, smem=64)), reuse=N_OUT)
    assert small["block_box"] is None and "smem" in small["block_box_reason"]
    unknown = lanes.choose_mapping(fa, card=CARD)
    assert unknown["P1"] and unknown["block_box"] is None


def test_facts_are_deterministic():
    assert lanes.facts(kg_fwd) == lanes.facts(kg_fwd)


# ---- the gather tile ---------------------------------------------------------
def test_the_contraction_forward_has_one_gather_tile():
    fa = lanes.facts(kg_fwd, labels=("channel", "sample"))
    (g,) = fa.gather_tiles
    assert (g.plane, g.slot_extent, g.row_words) == ("th", G, G + K + 2)
    assert g.fits_warp_row
    assert lanes.gather_tile_eligible(fa, tile=64) == (True, f"plane th: row {S_ROW} words, slot {G}")
    assert lanes.facts(kg_wide).gather_tiles[0].row_words == 40


@pytest.mark.parametrize("kernel", [k_fem, k_clamp, k_const, k_uni, k_tested, k_stencil, kg_tpart])
def test_no_gather_tile_for_scatter_histogram_and_stencil(kernel):
    fa = lanes.facts(kernel)
    assert fa.gather_tiles == ()
    assert lanes.gather_tile_eligible(fa) == (False, "no gather-tile read")


def test_gather_tile_eligible_reasons():
    fa = lanes.facts(kg_fwd, labels=("channel", "sample"))
    assert lanes.gather_tile_eligible(fa, required_group=8) == (False, "group 8 < box 32")
    assert lanes.gather_tile_eligible(fa, required_group=48) \
        == (False, "group 48 not a multiple of box 32")
    assert lanes.gather_tile_eligible(fa, tile=48) == (False, f"fastest extent {BATCH} not a multiple of tile 48")
    assert lanes.gather_tile_eligible(fa, uniform_control=False) \
        == (False, "control diverges: masked-alive path")
    assert lanes.gather_tile_eligible(lanes.facts(kg_flat)) == (False, "fastest extent is runtime")
    assert lanes.gather_tile_eligible(lanes.facts(kg_wide)) \
        == (False, "row 40 words exceeds 32-word bank width")


# ---- the card: tile fields and the contraction choice -------------------------
BT = dict(TILE, kind="BT", SB=128, T=16, R=1, EB=4)


def test_no_card_or_no_tile_fields_choose_no_block_lowering():
    fa = lanes.facts(kg_fwd, labels=("channel", "sample"))
    for card in (None, {"T": 8}, {"T": 8, "itemsize": 4}):
        got = lanes.choose_mapping(fa, card=card, reuse=N_OUT)
        assert got["block_box"] is None and got["gather_tile"] is None
        assert got["contraction"] == "gemm"
    assert lanes.choose_mapping(fa, card=None)["P1"] is None
    assert lanes.choose_mapping(fa, card={"T": 8})["P1"]["T"] == 8


def test_a_tile_kind_selects_one_lowering():
    fa = lanes.facts(kg_fwd, labels=("channel", "sample"))
    gt = lanes.choose_mapping(fa, card={"T": 8, "itemsize": 4, "tile": BT}, reuse=N_OUT)
    assert gt["block_box"] is None and gt["block_box_reason"] == "tile kind is BT"
    assert gt["gather_tile"] == {"plane": "th", "row_words": S_ROW, "slot_extent": G,
                                 "SB": 128, "T": 16, "R": 1, "EB": 4}
    bf = lanes.choose_mapping(fa, card=CARD, reuse=N_OUT)
    assert bf["gather_tile"] is None and bf["gather_tile_reason"] == "tile kind is BF"
    assert bf["block_box"]["planes"] == ("feat",)


def test_a_gather_tile_must_fit_the_card_and_the_kernel():
    fa = lanes.facts(kg_fwd, labels=("channel", "sample"))
    small = lanes.choose_mapping(fa, card={"T": 8, "itemsize": 4, "tile": dict(BT, smem=64)})
    assert small["gather_tile"] is None and "smem" in small["gather_tile_reason"]
    wide = lanes.choose_mapping(lanes.facts(kg_wide), card={"T": 8, "tile": BT})
    assert wide["gather_tile"] is None and "bank width" in wide["gather_tile_reason"]


@pytest.mark.parametrize("tile", [{"kind": "BT", "SB": 128}, dict(TILE, kind="XX")])
def test_a_malformed_tile_is_refused(tile):
    fa = lanes.facts(kg_fwd, labels=("channel", "sample"))
    with pytest.raises(ValueError, match="card tile needs"):
        lanes.choose_mapping(fa, card={"T": 8, "tile": tile})


def test_the_contraction_choice_is_the_cards_per_dtype(tmp_path, monkeypatch):
    import json

    from hawk.ir import recognize
    from hawk.targets import cards
    from test_ir_contraction import kan_contract as shared_contract

    fa = lanes.facts(kg_fwd)
    assert [lanes.choose_mapping(fa, card=cards.thread_box(t))["contraction"]
            for t in ("float32", "float64")] == ["gemm", "gemm"]
    doc = json.loads(cards.THREAD_BOX_FILE.read_text())
    doc["rows"] = [{"cc": "sm_00", "device": "Test Card", "dtype": "float64", "contraction": "direct"}]
    path = tmp_path / "thread_box.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setattr(cards, "THREAD_BOX_FILE", path)
    monkeypatch.setattr(cards, "_device", lambda: ("sm_00", "Test Card"))
    direct = dict(lanes.choose_mapping(fa, card=cards.thread_box("float64")))
    assert direct["contraction"] == "direct"
    assert lanes.choose_mapping(fa, card=cards.thread_box("float32"))["contraction"] == "gemm"
    assert recognize(shared_contract) is not None
    assert recognize(shared_contract, mapping=direct) is None
    assert recognize(shared_contract, mapping={"contraction": "gemm"}) is not None
    doc["contraction"]["float32"] = "blas"
    path.write_text(json.dumps(doc))
    with pytest.raises(hawk.HawkError, match="contraction"):
        cards.thread_box("float32")


def test_the_committed_card_names_its_tile_examples_only_as_examples():
    from hawk.targets import cards
    doc = __import__("json").loads(cards.THREAD_BOX_FILE.read_text())
    assert doc["rows"] == [] and len(doc["examples"]) == 2
    assert "tile" not in cards.thread_box("float32")


# --------------------------------------------------------------------------
# warp-uniform values


def _keys(kernel):
    from hawk.ir import canonical_nodes
    from hawk.ir.loop_nodes import Loop

    uniform = lanes.warp_uniform(kernel.sinks)
    loop = next(n for n in canonical_nodes(kernel.sinks)[0] if isinstance(n, Loop))
    return [uniform(n.operands[1]) for n in loop.body_nodes() if n.kind == "set_component_at"]


def _quotient_keys(width):
    """A store keyed by ``floor(minor / width)`` of a split lane."""
    def build(t, o):
        major, minor = m.split_index(256)
        row = m.index((m.real(minor) + 0.5) / width, 8)
        acc = m.vec(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for k in range(3):
            acc[row + k] = acc[row + k] + 1.0
        o = acc
    build.__annotations__ = {"t": Scalar, "o": Mutable[Vector[10]]}
    return hawk.kernel(build)


def _input_keys():
    def build(t, o):
        row = m.index(t, 8)
        acc = m.vec(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for k in range(3):
            acc[row + k] = acc[row + k] + 1.0
        o = acc
    build.__annotations__ = {"t": Scalar, "o": Mutable[Vector[10]]}
    return hawk.kernel(build)


def test_a_key_that_is_a_quotient_of_the_lane_by_a_warp_multiple_is_uniform():
    assert _keys(_quotient_keys(64)) == [True]
    assert _keys(_quotient_keys(16)) == [False]            # 16 lanes: half a warp
    assert _keys(_input_keys()) == [False]                 # a per-lane input
