# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A group-wide reduce and scan run on the host, from the author's ``g.sum``, ``g.min``, ``g.max``, ``g.any`` and ``g.all``.

Every value is checked against a reference written here from the contract, never against the kernel's own output: the sum in
the ``tree`` order is the halving tree of the runtime's in-leaf reduction (slots of 256, then the width halving; an absent partner
is carried) written out in NumPy scalars, and also stays within ``(n - 1) * eps * sum(|x|)`` of ``math.fsum``; the ``lane`` order
is a serial loop; a minimum or maximum is ``numpy.nanmin`` / ``numpy.nanmax`` (a NaN term is ignored); ``any`` and ``all`` are
NumPy's; the scan is ``numpy.cumsum`` per group.
"""

from __future__ import annotations

import functools
import math
import re

import numpy as np
import pytest

import hawk
from hawk import Grid, Mutable, Scalar
from hawk import math as m
from hawk.artifact import build_bundle, emit_source
from hawk.trace.value import Value, node_of

RNG = np.random.default_rng(20261126)
SIZES = (1, 2, 3, 7, 8, 64, 100)
SLOTS = 256


def _groups(field, lanes):
    """The consecutive groups of ``lanes`` entries of the lane vector ``field``; the last may be shorter."""
    return [field[start : start + lanes] for start in range(0, len(field), lanes)]


def _tree_sum(terms, dtype):
    """The sum of ``terms`` as the runtime's in-leaf tree adds them, written from its definition: lane ``l`` above the slot count
    folds into slot ``l % 256`` in lane order, then for the width ``128, 64, ..., 1`` every slot ``t < width`` whose partner
    ``t + width`` is present takes it."""
    slots = [dtype(x) for x in terms]
    for lane in range(SLOTS, len(slots)):
        slots[lane % SLOTS] = dtype(slots[lane % SLOTS] + slots[lane])
    width = SLOTS // 2
    while width >= 1:
        for slot in range(width):
            if slot + width < min(len(slots), SLOTS):
                slots[slot] = dtype(slots[slot] + slots[slot + width])
        width //= 2
    return slots[0]


def _lane_sum(terms, dtype):
    """The terms added one by one in rank order, from the first."""
    return functools.reduce(lambda acc, x: dtype(acc + x), (dtype(x) for x in terms))


def _scan_of(value):
    """The inclusive running sum of ``value`` over the lanes of the group, recorded as the library's ``GroupScan`` node (the
    authoring has no name for it yet, so the test builds the node the way ``g.sum`` builds its reduce)."""
    node = node_of(value)
    return Value(
        node.ctx.new(
            "GroupScan", "(('op', 'sum'), ('value', #0), ('inclusive', True))", [node]
        )
    )


def _sum_kernel(lanes, order):
    @hawk.kernel
    def summed(x: Scalar, y: Mutable[Scalar]):
        with hawk.group(lanes=lanes) as g:
            y = g.sum(x, order=order)

    summed.name = f"sum_{order}_{lanes}"
    return summed


def _scan_kernel(lanes):
    @hawk.kernel
    def scanned(x: Scalar, y: Mutable[Scalar]):
        with hawk.group(lanes=lanes) as g:
            y = _scan_of(x)

    scanned.name = f"scan_{lanes}"
    return scanned


@hawk.kernel
def group_min(x: Scalar, y: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        y = g.min(x)


@hawk.kernel
def group_max(x: Scalar, y: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        y = g.max(x)


@hawk.kernel
def group_any(x: Scalar, y: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        y = m.where(g.any(x > 0.9), 1.0, 0.0)


@hawk.kernel
def group_all(x: Scalar, y: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        y = m.where(g.all(x > -0.9), 1.0, 0.0)


@hawk.kernel
def share(x: Scalar, y: Mutable[Scalar]):
    """The value of a reduce read by every lane of a later phase: each lane's share of its group's sum."""
    with hawk.group(lanes=8) as g:
        t = g.scratch("t", Scalar)
        t[g.rank] = x * x
        total = g.sum(t[g.rank])
        y = t[g.rank] / total


@hawk.kernel
def sum_of_sums(x: Scalar, y: Mutable[Scalar]):
    """A reduce whose operand holds another reduce: the inner one is evaluated a phase before the outer one."""
    with hawk.group(lanes=8) as g:
        y = g.sum(x - g.sum(x) / 8.0)


@hawk.kernel
def over_cols(lane: Grid["row", "col":4], x: Scalar, y: Mutable[Scalar]):  # noqa: F821 axis labels
    with hawk.group(over="col") as g:
        y = g.sum(x)


@hawk.kernel
def over_rows(lane: Grid["row":3, "col":5], x: Scalar, y: Mutable[Scalar]):  # noqa: F821 axis labels
    with hawk.group(over="row") as g:
        y = g.sum(x, order="lane")


BIG = (300, 1000, 4096)


@hawk.kernel
def rows_of_runtime_extent(lane: Grid["row", "col":4], x: Scalar, y: Mutable[Scalar]):  # noqa: F821 axis labels
    with hawk.group(over="row") as g:
        y = g.sum(x)


KERNELS = (
    [_sum_kernel(n, order) for n in SIZES for order in ("tree", "lane", "any")]
    + [_scan_kernel(n) for n in (1, 3, 8, 100)]
    + [
        group_min,
        group_max,
        group_any,
        group_all,
        share,
        sum_of_sums,
        over_cols,
        over_rows,
    ]
)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    """Every kernel above built for the host in 64-bit floats, once."""
    return build_bundle(
        KERNELS,
        tmp_path_factory.mktemp("group_reduce"),
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )


@pytest.fixture(scope="module")
def big_bundle(tmp_path_factory, cache_dir):
    """The tree sums over groups larger than the slots of the in-leaf tree, built apart."""
    return build_bundle(
        [_sum_kernel(n, "tree") for n in BIG],
        tmp_path_factory.mktemp("group_reduce_big"),
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )


@pytest.fixture(scope="module")
def bundle32(tmp_path_factory, cache_dir):
    """The sums built for the host in 32-bit floats."""
    return build_bundle(
        [_sum_kernel(n, order) for n in SIZES for order in ("tree", "lane")],
        tmp_path_factory.mktemp("group_reduce32"),
        mode="float32",
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )


def _run(bundle, name, **planes):
    hawk.run(hawk.load(bundle.directory, name), **planes)


def _expected(rows, lanes, one, dtype):
    """``one`` of each group of ``lanes`` lanes of ``rows``, spread to every lane of the group."""
    return np.concatenate(
        [np.full(len(g), one(g), dtype=dtype) for g in _groups(rows, lanes)]
    )


@pytest.mark.parametrize("lanes", SIZES)
@pytest.mark.parametrize("n", (100, 107))
def test_a_tree_sum_is_the_halving_tree_of_the_runtime_and_near_fsum_in_float64(
    bundle, lanes, n
):
    x = RNG.normal(size=n) * 10.0 ** RNG.integers(-3, 4, size=n)
    y = np.zeros(n)
    _run(bundle, f"sum_tree_{lanes}", x=x, y=y)
    np.testing.assert_array_equal(
        y, _expected(x, lanes, lambda g: _tree_sum(g, np.float64), np.float64)
    )
    for group, got in zip(_groups(x, lanes), _groups(y, lanes)):
        bound = (
            (len(group) - 1)
            * np.finfo(np.float64).eps
            * math.fsum(abs(v) for v in group)
        )
        assert abs(got[0] - math.fsum(group)) <= bound


@pytest.mark.parametrize("lanes", SIZES)
@pytest.mark.parametrize("n", (100, 107))
def test_a_tree_sum_is_the_halving_tree_of_the_runtime_and_near_fsum_in_float32(
    bundle32, lanes, n
):
    x = (RNG.normal(size=n) * 10.0 ** RNG.integers(-2, 3, size=n)).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)
    _run(bundle32, f"sum_tree_{lanes}", x=x, y=y)
    np.testing.assert_array_equal(
        y, _expected(x, lanes, lambda g: _tree_sum(g, np.float32), np.float32)
    )
    for group, got in zip(_groups(x, lanes), _groups(y, lanes)):
        bound = (
            (len(group) - 1)
            * np.finfo(np.float32).eps
            * math.fsum(abs(float(v)) for v in group)
        )
        assert abs(float(got[0]) - math.fsum(float(v) for v in group)) <= bound


@pytest.mark.parametrize("lanes", SIZES)
def test_a_sum_in_the_any_order_is_the_tree(bundle, lanes):
    x = RNG.normal(size=100)
    y = np.zeros(100)
    _run(bundle, f"sum_any_{lanes}", x=x, y=y)
    np.testing.assert_array_equal(
        y, _expected(x, lanes, lambda g: _tree_sum(g, np.float64), np.float64)
    )


@pytest.mark.parametrize("lanes", SIZES)
def test_a_sum_in_the_lane_order_is_the_serial_loop_in_float64(bundle, lanes):
    x = RNG.normal(size=100) * 10.0 ** RNG.integers(-3, 4, size=100)
    y = np.zeros(100)
    _run(bundle, f"sum_lane_{lanes}", x=x, y=y)
    np.testing.assert_array_equal(
        y, _expected(x, lanes, lambda g: _lane_sum(g, np.float64), np.float64)
    )


@pytest.mark.parametrize("lanes", SIZES)
def test_a_sum_in_the_lane_order_is_the_serial_loop_in_float32(bundle32, lanes):
    x = (RNG.normal(size=100) * 10.0 ** RNG.integers(-2, 3, size=100)).astype(
        np.float32
    )
    y = np.zeros(100, dtype=np.float32)
    _run(bundle32, f"sum_lane_{lanes}", x=x, y=y)
    np.testing.assert_array_equal(
        y, _expected(x, lanes, lambda g: _lane_sum(g, np.float32), np.float32)
    )


def test_the_tree_and_the_lane_order_differ_where_the_terms_do_not_commute_in_rounding(
    bundle,
):
    """The two orders are two contracts: this input is chosen so that the serial and the tree sums round differently."""
    x = np.array([1.0, 1e-16, 1e-16, 1e-16, 1e-16, 1e-16, 1e-16, 1e-16] * 4)
    tree, lane = np.zeros(32), np.zeros(32)
    _run(bundle, "sum_tree_8", x=x, y=tree)
    _run(bundle, "sum_lane_8", x=x, y=lane)
    assert tree[0] != lane[0]


def test_a_minimum_and_a_maximum_ignore_a_nan_term_and_are_nan_only_when_every_term_is(
    bundle,
):
    x = RNG.normal(size=50)
    x[[3, 11, 12, 40]] = np.nan
    x[16:24] = np.nan  # a whole group of NaN
    with np.errstate(all="ignore"), pytest.warns(RuntimeWarning, match="All-NaN"):
        for name, reference in (("group_min", np.nanmin), ("group_max", np.nanmax)):
            y = np.zeros(50)
            _run(bundle, name, x=x, y=y)
            np.testing.assert_array_equal(y, _expected(x, 8, reference, np.float64))


def test_any_and_all_are_numpys_per_group_with_a_short_last_group(bundle):
    x = RNG.uniform(-1.0, 1.0, size=50)
    x[8:16] = 0.0
    x[24:32] = 0.95
    x[40:48] = -0.95
    x[49] = 0.99
    for name, reference in (
        ("group_any", lambda g: np.any(g > 0.9)),
        ("group_all", lambda g: np.all(g > -0.9)),
    ):
        y = np.zeros(50)
        _run(bundle, name, x=x, y=y)
        np.testing.assert_array_equal(
            y, _expected(x, 8, lambda g: float(reference(g)), np.float64)
        )


@pytest.mark.parametrize("n", (8, 64, 100, 19))
def test_a_value_read_in_a_later_phase_is_the_group_sum_every_lane_normalises_by(
    bundle, n
):
    x = RNG.uniform(0.5, 2.0, size=n)
    y = np.zeros(n)
    _run(bundle, "share", x=x, y=y)
    squares = x * x
    want = np.concatenate([g / g.sum() for g in _groups(squares, 8)])
    np.testing.assert_allclose(y, want, rtol=1e-14, atol=0)


def test_a_reduce_whose_operand_holds_a_reduce_is_evaluated_after_the_inner_one(bundle):
    x = RNG.normal(size=40)
    y = np.zeros(40)
    _run(bundle, "sum_of_sums", x=x, y=y)
    want = np.concatenate(
        [np.full(8, (g - g.sum() / 8.0).sum()) for g in _groups(x, 8)]
    )
    np.testing.assert_allclose(y, want, rtol=0, atol=1e-13)


@pytest.mark.parametrize("n", (1, 5, 64))
def test_a_group_over_a_trailing_axis_sums_that_axis(bundle, n):
    x = RNG.normal(size=(n, 4))
    y = np.zeros_like(x)
    _run(bundle, "over_cols", x=x, y=y)
    np.testing.assert_array_equal(
        y,
        np.repeat(np.array([_tree_sum(r, np.float64) for r in x])[:, None], 4, axis=1),
    )


def test_a_group_over_a_leading_axis_sums_that_axis_in_lane_order(bundle):
    x = RNG.normal(size=(3, 5))
    y = np.zeros_like(x)
    _run(bundle, "over_rows", x=x, y=y)
    column = np.array([_lane_sum(x[:, c], np.float64) for c in range(5)])
    np.testing.assert_array_equal(y, np.broadcast_to(column, (3, 5)))


@pytest.mark.parametrize("lanes", (1, 3, 8, 100))
def test_a_scan_is_numpys_cumulative_sum_of_each_group_with_a_short_last_group(
    bundle, lanes
):
    x = RNG.normal(size=107)
    y = np.zeros(107)
    _run(bundle, f"scan_{lanes}", x=x, y=y)
    np.testing.assert_array_equal(
        y, np.concatenate([np.cumsum(g) for g in _groups(x, lanes)])
    )


# ---- the text ----


def _host_text(kernel):
    return emit_source(kernel, targets=("host",)).sources["host"].text


def test_a_tree_sum_is_written_as_loops_that_add_the_two_halves_of_the_slots():
    text = _host_text(_sum_kernel(100, "tree"))
    assert "hawk_work_" in text
    # the widths below the group's size, largest first, each a literal distance between the halves
    widths = [int(w) for w in re.findall(r"hawk_t < hawk_m(\d+); \+\+hawk_t\)", text)]
    assert widths == [64, 32, 16, 8, 4, 2, 1]
    assert re.search(
        r"hawk_work_\d+\[hawk_t\] = hawk_work_\d+\[hawk_t\] \+ hawk_work_\d+\[hawk_t \+ 64\];",
        text,
    )


def test_a_sum_over_more_lanes_than_the_slots_folds_the_extra_lanes_into_the_slots_first():
    text = _host_text(_sum_kernel(1000, "tree"))
    assert "hawk_k += 256" in text
    assert "hawk_t + 128" in text


def test_a_lane_order_sum_is_one_serial_loop_in_rank_order():
    text = _host_text(_sum_kernel(8, "lane"))
    assert re.search(r"hawk_acc = hawk_acc \+ hawk_work_\d+\[hawk_k\];", text)
    assert "hawk_m" not in text


def test_a_scan_is_one_serial_loop_over_the_work_array_read_at_the_lanes_own_place():
    text = _host_text(_scan_kernel(8))
    assert re.search(
        r"hawk_work_(\d+)\[hawk_k\] = hawk_work_\1\[hawk_k - 1\] \+ hawk_work_\1\[hawk_k\];",
        text,
    )
    assert re.search(r"= hawk_work_\d+\[hawk_rank\];", text)


@pytest.mark.parametrize("lanes", BIG)
def test_a_tree_sum_over_more_lanes_than_the_slots_folds_the_extra_lanes_into_the_slots_first(
    big_bundle, lanes
):
    n = lanes * 2 + 77
    x = RNG.normal(size=n)
    y = np.zeros(n)
    _run(big_bundle, f"sum_tree_{lanes}", x=x, y=y)
    np.testing.assert_array_equal(
        y, _expected(x, lanes, lambda g: _tree_sum(g, np.float64), np.float64)
    )


def _refusal(kernel):
    with pytest.raises(hawk.HawkError) as raised:
        emit_source(kernel, targets=("host",))
    return str(raised.value)


def test_a_tree_sum_over_more_lanes_than_the_tree_holds_is_refused_by_name_and_the_any_order_adds_in_rank_order():
    assert (
        "a group sum in the tree order over more than 4096 lanes is not written on the host yet"
        in _refusal(_sum_kernel(5000, "tree"))
    )
    text = _host_text(_sum_kernel(5000, "any"))
    assert "hawk_acc" in text


def test_a_reduce_over_a_group_with_no_static_size_is_refused_by_name():
    assert (
        "a group reduce over a group with no static size is not written on the host yet"
        in _refusal(rows_of_runtime_extent)
    )
