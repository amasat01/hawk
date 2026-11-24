# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.reduce`` with a position (``with_index=True``) and the op ``"any"``: values, lanes, ties, NaN, refusals, replay.

Oracles: ``np.min`` / ``np.max`` and ``np.argmin`` / ``np.argmax`` (the first occurrence, so the lowest lane) for the value and
the lane; ``np.nanmin`` / ``np.nanargmin`` (and the max pair) where some element is NaN; ``np.any`` for ``"any"``. A column of
nothing but NaN has the value NaN and the lane ``n``, which no NumPy call gives: it is stated here.
"""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import HawkError, Mutable, Scalar

RNG = np.random.default_rng(20261130)
#: One, two, around the 256-lane slot and the 4096-sample leaf of the fold, a few leaves, and a million.
SIZES = (1, 2, 255, 256, 257, 4095, 4096, 4097, 8193, 50_000, 1_000_000)
DTYPES = (np.float64, np.float32, np.int32, np.uint32)
PAIRS = {
    "min": (np.min, np.argmin, np.nanmin, np.nanargmin),
    "max": (np.max, np.argmax, np.nanmax, np.nanargmax),
}


@hawk.kernel
def double(x: Scalar, y: Mutable[Scalar]):
    y = x * 2.0


def ties(dtype, n):
    """Values from a small set, so that every extreme repeats."""
    return ((np.arange(n) * 37) % 101 + 10).astype(dtype)


def block_ties(dtype, n):
    """A constant plane with the extremes repeated across a slot (255, 256) and a leaf (4095, 4096) boundary."""
    x = np.full(n, 50, dtype=dtype)
    for lane in (255, 256, 4095, 8191):
        x[lane:][:1] = 150
    for lane in (257, 4096, 4097, 8192):
        x[lane:][:1] = 3
    return x


def ends(dtype, n):
    """The minimum at both ends and the maximum at the lanes next to them."""
    x = np.full(n, 50, dtype=dtype)
    x[[0, n - 1]] = 3
    x[[min(1, n - 1), max(n - 2, 0)]] = np.where(n > 2, 150, 3)
    return x


PLANES = {
    "many ties": ties,
    "constant": lambda dtype, n: np.full(n, 50, dtype=dtype),
    "ends": ends,
    "block boundaries": block_ties,
}


@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", sorted(PLANES))
def test_the_value_and_lane_equal_numpy_for_every_type_and_size(op, dtype, shape):
    value, lane = PAIRS[op][:2]
    for n in SIZES:
        x = PLANES[shape](dtype, n)
        got, where = hawk.reduce(x, op, with_index=True)
        assert got.dtype == dtype and where.dtype == np.int32
        assert got[0] == value(x) and where[0] == lane(x), (n, got, where)


@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_a_nan_is_never_the_winner_and_the_lane_equals_the_nan_ignoring_numpy_call(
    op, dtype
):
    _, _, value, lane = PAIRS[op]
    for n in SIZES:
        x = ties(dtype, n)
        x[::5] = np.nan
        if np.isnan(x).all():
            continue  # n == 1: the all-NaN case has its own test
        got, where = hawk.reduce(x, op, with_index=True)
        assert got[0] == value(x) and where[0] == lane(x), (n, got, where)


@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_a_column_of_nothing_but_nan_has_the_value_nan_and_the_lane_n(op, dtype):
    """The value is what the reduction without a position returns for it, NaN."""
    for n in (1, 300, 5000):
        x = np.full(n, np.nan, dtype=dtype)
        got, where = hawk.reduce(x, op, with_index=True)
        assert np.isnan(got[0]) and where[0] == n
        assert np.isnan(hawk.reduce(x, op)[0])


@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_each_column_has_its_own_value_and_lane(op, dtype):
    value, lane = PAIRS[op][:2]
    x = (RNG.integers(0, 29, size=(3, 5000)) + 5).astype(dtype)
    x[1, [4096, 4097]] = 1 if op == "min" else 200
    got, where = hawk.reduce(x, op, with_index=True)
    np.testing.assert_array_equal(got, value(x, axis=1))
    np.testing.assert_array_equal(where, lane(x, axis=1))


@pytest.mark.parametrize("op", ["min", "max"])
def test_the_result_is_the_same_with_one_thread_and_with_the_default(op):
    x = RNG.integers(0, 40, size=1_000_000).astype(np.float64)
    wanted = hawk.host_threads()
    try:
        hawk.set_host_threads(1)
        one = hawk.reduce(x, op, with_index=True)
        hawk.set_host_threads(None)
        many = hawk.reduce(x, op, with_index=True)
    finally:
        hawk.set_host_threads(wanted)
    assert one[0][0] == many[0][0] and one[1][0] == many[1][0]


def test_a_plane_of_no_sample_is_refused_by_name():
    with pytest.raises(HawkError, match="in holds no samples"):
        hawk.reduce(np.zeros(0), "min", with_index=True)


# ---- any ----


@pytest.mark.parametrize("n", [1, 2, 4097, 1_000_000])
def test_any_equals_numpy_any_for_all_false_one_true_at_the_last_lane_and_all_true(n):
    mask = np.zeros(n, dtype=bool)
    assert hawk.reduce(mask, "any")[0] == np.any(mask)
    mask[-1] = True
    assert hawk.reduce(mask, "any")[0] == np.any(mask)
    mask[:] = True
    assert hawk.reduce(mask, "any")[0] == np.any(mask)


def test_any_of_several_columns_gives_each_column_and_a_uint32_out_gives_a_cell():
    mask = np.zeros((3, 70), dtype=bool)
    mask[1, -1] = True
    np.testing.assert_array_equal(hawk.reduce(mask, "any"), mask.any(axis=1))
    cell = np.zeros(3, dtype=np.uint32)
    hawk.reduce(mask, "all", out=cell)
    np.testing.assert_array_equal(cell, mask.all(axis=1))


# ---- recorded ----


def test_a_recorded_position_equals_numpy_and_a_replay_reads_the_changed_plane():
    @hawk.Graph.builder
    def build(x) -> hawk.Graph:
        g = hawk.Graph()
        g.low, g.where = hawk.reduce(x, "min", with_index=True)
        return g

    x = ties(np.float64, 6000).reshape(2, 3000)
    graph = build(x)
    graph.launch()
    np.testing.assert_array_equal(graph.low, x.min(axis=1))
    np.testing.assert_array_equal(graph.where, x.argmin(axis=1))
    x[0, 2999] = -1.0  # a replay reads the plane as it is then
    graph.launch()
    assert graph.low[0] == -1.0 and graph.where[0] == 2999
    assert graph.node_labels() == ["reduce"]
    graph.close()


@hawk.Graph.builder
def guarded(mask, x, y) -> hawk.Graph:
    g = hawk.Graph()
    found = np.zeros(1, dtype=np.uint32)
    hawk.reduce(mask, "any", out=found)
    if found:
        double(x, y=y)
    g.found, g.result = found, y
    return g


def test_a_branch_on_any_is_taken_only_while_a_lane_is_set_and_a_changed_mask_is_read():
    """Oracle: ``np.any`` decides the branch; the arm writes ``2 * x`` into ``y``."""
    mask = np.zeros(2000, dtype=bool)
    x = RNG.normal(size=2000)
    y = np.full(2000, -5.0)
    graph = guarded(mask, x, y)
    graph.launch()
    assert (
        graph.found[0] == 0 and (y == -5.0).all()
    )  # nothing set: the arm is not taken
    mask[1999] = True
    graph.launch()
    assert graph.found[0] == 1
    np.testing.assert_array_equal(y, 2.0 * x)
    graph.close()


@hawk.kernel
def tenfold(r: Scalar, w: Mutable[Scalar]):
    w = r * 10.0


@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("with_index", [False, True])
def test_the_reverse_of_a_minimum_or_maximum_goes_to_the_winner_lane_with_or_without_a_position(
    op, with_index
):
    """Oracle: ``np.argmin`` / ``np.argmax`` (the first of equal values) gets ``20 * seed`` and every other lane zero."""

    def build(x) -> hawk.Graph:
        g = hawk.Graph(record=True)
        if with_index:
            value, g.where = hawk.reduce(double(x), op, with_index=True)
        else:
            value = hawk.reduce(double(x), op)
        g.result = tenfold(value)
        return g

    x = RNG.integers(0, 5, size=100).astype(np.float64)  # ties
    graph = hawk.Graph.builder(build)(x)
    graph.launch()
    seed = RNG.normal(size=1)
    gradients = hawk.reverse(graph.record, {"w": seed})
    want = np.zeros(100)
    want[np.argmin(x) if op == "min" else np.argmax(x)] = 20.0 * seed[0]
    np.testing.assert_allclose(gradients["x"], want, rtol=1e-14)
    graph.close()


# ---- refusals ----


@pytest.mark.parametrize("op", ["sum", "all", "any"])
def test_a_position_with_another_op_is_refused_by_name(op):
    with pytest.raises(HawkError, match="MIN or MAX only"):
        hawk.reduce(np.ones(10, dtype=bool), op, with_index=True)


def test_any_over_a_float_plane_and_a_uint32_sum_are_refused_by_name():
    with pytest.raises(HawkError, match="ANY takes a bool plane"):
        hawk.reduce(np.ones(10), "any")
    with pytest.raises(HawkError, match="uint32 is served by ops MIN and MAX only"):
        hawk.reduce(np.ones(10, dtype=np.uint32), "sum")
