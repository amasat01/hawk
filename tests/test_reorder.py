# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.reorder``: planes permuted in place by an order, eager and recorded.

Oracle: NumPy fancy indexing, ``p[order]``, exact; after a sort, ``p[np.argsort(keys, kind="stable")]``.
"""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import HawkError

RNG = np.random.default_rng(20261201)
#: The integer dtypes a plane carries today, and the two real ones.
DTYPES = (np.int8, np.int32, np.uint32, np.int64, np.float32, np.float64)
SIZES = (0, 1, 255, 256, 257, 100_000)


def plane_of(n, dtype, rows=None):
    """A plane of ``n`` lanes of random values (``(rows, n)`` for a vector value)."""
    shape = (n,) if rows is None else (rows, n)
    return (RNG.integers(0, 100, size=shape)).astype(dtype)


def permutation(n):
    """A random permutation of ``n`` lanes as an int32 plane."""
    return RNG.permutation(n).astype(np.int32)


@pytest.mark.parametrize("n_planes", [1, 3, 64])
@pytest.mark.parametrize("dtype", DTYPES)
def test_planes_equal_numpy_fancy_indexing(n_planes, dtype):
    order = permutation(1000)
    planes = [plane_of(1000, dtype) for _ in range(n_planes)]
    wanted = [p[order] for p in planes]
    hawk.reorder(planes, order)
    for got, want in zip(planes, wanted):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("n", SIZES)
def test_the_sizes_around_a_pass_and_a_chunk_equal_numpy(n):
    order = permutation(n)
    plane = plane_of(n, np.float64)
    wanted = plane[order]
    hawk.reorder([plane], order)
    np.testing.assert_array_equal(plane, wanted)


def test_a_plane_of_a_vector_value_moves_every_row_by_the_same_order():
    order = permutation(500)
    plane = plane_of(500, np.float64, rows=3)
    wanted = plane[:, order]
    hawk.reorder([plane], order)
    np.testing.assert_array_equal(plane, wanted)


def test_a_uint32_order_is_taken_as_it_is():
    order = permutation(300).astype(np.uint32)
    plane = plane_of(300, np.int32)
    wanted = plane[order]
    hawk.reorder([plane], order)
    np.testing.assert_array_equal(plane, wanted)


def test_the_result_is_the_same_with_one_thread_and_with_the_default():
    order = permutation(100_000)
    base = plane_of(100_000, np.float64)
    wanted = hawk.host_threads()
    try:
        hawk.set_host_threads(1)
        one = base.copy()
        hawk.reorder([one], order)
        hawk.set_host_threads(None)
        many = base.copy()
        hawk.reorder([many], order)
    finally:
        hawk.set_host_threads(wanted)
    np.testing.assert_array_equal(one, many)


@pytest.mark.parametrize("n", [255, 257, 100_000])
def test_sort_then_reorder_equals_numpy_on_the_stable_argsort(n):
    keys = RNG.integers(0, 300, size=n).astype(np.int32)
    pos, vel = plane_of(n, np.float64), plane_of(n, np.int64)
    wanted = [
        pos[np.argsort(keys, kind="stable")],
        vel[np.argsort(keys, kind="stable")],
    ]
    order, starts = hawk.sort(keys, bound=300)
    hawk.reorder([pos, vel], order)
    np.testing.assert_array_equal(pos, wanted[0])
    np.testing.assert_array_equal(vel, wanted[1])
    assert starts[300] == n


# ---- recorded: a replay reads the order and the planes as they are then ----


@hawk.Graph.builder
def reordered(a, b, order) -> hawk.Graph:
    g = hawk.Graph()
    hawk.reorder([a, b], order)
    return g


def test_a_recorded_reorder_is_one_node_and_a_replay_uses_the_order_as_it_is_then():
    a0, b0 = plane_of(2000, np.float64), plane_of(2000, np.int32)
    a, b, order = a0.copy(), b0.copy(), permutation(2000)
    graph = reordered(a, b, order)
    np.testing.assert_array_equal(a, a0)  # recording moves nothing
    assert graph.node_labels() == ["reorder"]
    graph.launch()
    np.testing.assert_array_equal(a, a0[order])
    np.testing.assert_array_equal(b, b0[order])
    a[:], b[:] = a0, b0
    order[:] = permutation(2000)
    graph.launch()
    np.testing.assert_array_equal(a, a0[order])
    np.testing.assert_array_equal(b, b0[order])
    graph.close()


def test_an_entry_outside_the_extent_at_a_replay_fails_with_the_count_and_moves_nothing():
    a0 = plane_of(100, np.float64)
    a, order = a0.copy(), permutation(100)
    graph = reordered(a, a.copy(), order)
    order[[3, 50]] = [100, -1]
    with pytest.raises(
        HawkError, match=r"reorder: 2 entries of order are outside \[0, 100\)"
    ):
        graph.launch()
    np.testing.assert_array_equal(a, a0)
    graph.close()


@hawk.kernel
def double(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    y = x * 2.0


@hawk.kernel
def tenfold(r: hawk.Scalar, w: hawk.Mutable[hawk.Scalar]):
    w = r * 10.0


def test_the_reverse_of_a_reorder_is_the_scatter_through_the_order():
    """Oracle: ``w = 10 * (2 x)[order]``, so ``dJ/dx[order[j]] = 20 * seed[j]``."""

    def build(order, x) -> hawk.Graph:
        g = hawk.Graph(record=True)
        y = double(x)
        hawk.reorder([y], order)
        g.result = tenfold(y)
        return g

    order, x, seed = permutation(300), RNG.normal(size=300), RNG.normal(size=300)
    graph = hawk.Graph.builder(build)(order, x)
    graph.launch()
    gradients = hawk.reverse(graph.record, {"w": seed})
    want = np.zeros(300)
    want[order] = 20.0 * seed
    np.testing.assert_allclose(gradients["x"], want, rtol=1e-14)
    graph.close()


def test_the_reverse_through_a_sort_and_the_reorder_by_its_order_equals_numpy():
    """Oracle: ``order = argsort(keys, kind="stable")``, then the same scatter as above; the sort carries no gradient."""

    def build(keys, x) -> hawk.Graph:
        g = hawk.Graph(record=True)
        order, starts = hawk.sort(keys, bound=8)
        y = double(x)
        hawk.reorder([y], order)
        g.result = tenfold(y)
        return g

    keys = RNG.integers(0, 8, size=200).astype(np.int32)
    x, seed = RNG.normal(size=200), RNG.normal(size=200)
    graph = hawk.Graph.builder(build)(keys, x)
    graph.launch()
    gradients = hawk.reverse(graph.record, {"w": seed})
    want = np.zeros(200)
    want[np.argsort(keys, kind="stable")] = 20.0 * seed
    np.testing.assert_allclose(gradients["x"], want, rtol=1e-14)
    graph.close()


# ---- refusals ----


def test_an_entry_outside_the_extent_is_refused_with_the_count_and_moves_nothing():
    plane = plane_of(16, np.float64)
    before = plane.copy()
    order = np.arange(16, dtype=np.int32)
    order[[2, 9, 11]] = [16, -1, 99]
    with pytest.raises(
        HawkError, match=r"reorder: 3 entries of order are outside \[0, 16\)"
    ):
        hawk.reorder([plane], order)
    np.testing.assert_array_equal(plane, before)


def test_no_planes_are_refused_by_name():
    with pytest.raises(HawkError, match="planes is empty"):
        hawk.reorder([], np.arange(4, dtype=np.int32))


def test_an_order_that_is_not_an_index_plane_is_refused_by_name():
    with pytest.raises(HawkError, match="order must be an int32 or uint32 plane"):
        hawk.reorder([np.zeros(4)], np.arange(4, dtype=np.float64))


def test_planes_of_differing_lane_extents_are_refused_by_name():
    with pytest.raises(HawkError, match="plane 1 has 5 lanes, order has 4"):
        hawk.reorder([np.zeros(4), np.zeros(5)], np.arange(4, dtype=np.int32))


def test_one_plane_twice_is_refused_by_name():
    plane = np.zeros(4)
    with pytest.raises(HawkError, match="overlap; the planes must be distinct"):
        hawk.reorder([plane, plane], np.arange(4, dtype=np.int32))
