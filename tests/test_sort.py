# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.sort``: the stable permutation by integer key and the first place of every key, eager and recorded.

Oracles: ``np.argsort(keys, kind="stable")`` for ``order`` and ``np.searchsorted(keys[order], np.arange(bound + 1))`` for
``starts``, both exact.
"""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import HawkError, Mutable, Scalar

RNG = np.random.default_rng(20261130)
#: Empty, one, around a radix pass of 256 buckets, and a size above the chunk the host splits by.
SIZES = (0, 1, 255, 256, 257, 100_000)
#: The integer dtypes a plane carries.
DTYPES = (
    np.int8,
    np.uint8,
    np.int16,
    np.uint16,
    np.int32,
    np.uint32,
    np.int64,
    np.uint64,
)


def keys_of(n, bound, dtype=np.int32, skew=False):
    """``n`` keys in ``[0, bound)``; with ``skew`` half of the lanes are in key 0."""
    keys = RNG.integers(0, bound, size=n)
    if skew:
        keys[RNG.random(n) < 0.5] = 0
    return keys.astype(dtype)


def expect_sorted(keys, bound, order, starts):
    """``order`` and ``starts`` equal the stable argsort and the first place of each key."""
    np.testing.assert_array_equal(order, np.argsort(keys, kind="stable"))
    np.testing.assert_array_equal(
        starts, np.searchsorted(keys[order], np.arange(bound + 1))
    )


@hawk.kernel
def double(x: Scalar, y: Mutable[Scalar]):
    y = x * 2.0


@pytest.mark.parametrize("skew", [False, True])
@pytest.mark.parametrize("bound", [16, 4096, 1 << 20, 100_000])
def test_order_and_starts_equal_the_stable_argsort_and_searchsorted_in_the_eight_cases(
    bound, skew
):
    keys = keys_of(100_000, bound, skew=skew)
    order, starts = hawk.sort(keys, bound=bound)
    expect_sorted(keys, bound, order, starts)


@pytest.mark.parametrize("n", SIZES)
def test_the_sizes_around_a_pass_and_a_chunk_equal_the_stable_argsort(n):
    keys = keys_of(n, 300)
    order, starts = hawk.sort(keys, bound=300)
    expect_sorted(keys, 300, order, starts)
    assert order.dtype == np.int32 and starts.dtype == np.int32
    assert order.shape == (n,) and starts.shape == (301,)


@pytest.mark.parametrize("dtype", DTYPES)
def test_every_integer_plane_dtype_equals_the_stable_argsort(dtype):
    bound = min(70_000, np.iinfo(dtype).max)
    keys = keys_of(50_000, bound, dtype)
    order, starts = hawk.sort(keys, bound=bound)
    expect_sorted(keys, bound, order, starts)


def test_the_planes_passed_as_out_are_the_ones_written():
    keys = keys_of(1000, 10)
    order = np.full(1000, -1, dtype=np.int32)
    starts = np.full(11, -1, dtype=np.int32)
    got_order, got_starts = hawk.sort(keys, bound=10, out=(order, starts))
    assert got_order is order and got_starts is starts
    expect_sorted(keys, 10, order, starts)


def test_the_result_is_the_same_with_one_thread_and_with_the_default():
    keys = keys_of(100_000, 4096, skew=True)
    wanted = hawk.host_threads()
    try:
        hawk.set_host_threads(1)
        one = hawk.sort(keys, bound=4096)
        hawk.set_host_threads(None)
        many = hawk.sort(keys, bound=4096)
    finally:
        hawk.set_host_threads(wanted)
    np.testing.assert_array_equal(one[0], many[0])
    np.testing.assert_array_equal(one[1], many[1])


# ---- recorded: a replay reads the keys as they are then ----


@hawk.Graph.builder
def sorted_then_doubled(keys, x, y) -> hawk.Graph:
    g = hawk.Graph()
    order, starts = hawk.sort(keys, bound=8)
    double(x, y=y)
    g.order, g.starts, g.result = order, starts, y
    return g


def test_a_recorded_sort_is_one_node_and_a_replay_sorts_the_keys_as_they_are_then():
    keys = keys_of(5000, 8)
    x = RNG.normal(size=5000)
    y = np.zeros(5000)
    graph = sorted_then_doubled(keys, x, y)
    graph.launch()
    expect_sorted(keys, 8, graph.order, graph.starts)
    assert graph.node_labels()[0] == "sort"
    keys[:] = keys_of(5000, 8, skew=True)
    graph.launch()
    expect_sorted(keys, 8, graph.order, graph.starts)
    np.testing.assert_array_equal(y, 2.0 * x)
    graph.close()


def test_a_key_outside_the_bound_at_a_replay_fails_with_the_count():
    keys = keys_of(100, 8)
    graph = sorted_then_doubled(keys, RNG.normal(size=100), np.zeros(100))
    keys[[3, 50]] = 8
    with pytest.raises(HawkError, match=r"sort: 2 keys are outside \[0, 8\)"):
        graph.launch()
    graph.close()


def test_a_recording_that_holds_a_sort_reverses_with_no_adjoint_through_order_and_starts():
    """Oracle: ``order`` and ``starts`` are integers; the kernel beside the sort reverses as it does alone, ``dJ/dx = 2 * seed``."""

    def build(keys, x) -> hawk.Graph:
        g = hawk.Graph(record=True)
        hawk.sort(keys, bound=8)
        g.result = double(x)
        return g

    graph = hawk.Graph.builder(build)(keys_of(100, 8), RNG.normal(size=100))
    graph.launch()
    seed = RNG.normal(size=100)
    gradients = hawk.reverse(graph.record, {"y": seed})
    np.testing.assert_allclose(gradients["x"], 2.0 * seed, rtol=1e-14)
    graph.close()


# ---- refusals ----


def test_real_keys_are_refused_by_name():
    with pytest.raises(HawkError, match="real keys are not written"):
        hawk.sort(RNG.random(10), bound=4)


def test_a_dtype_no_plane_carries_is_refused_by_the_runtime_with_its_reason():
    with pytest.raises(HawkError, match="is not one eagle runs"):
        hawk.sort(np.zeros(10, dtype=np.complex64), bound=4)


def test_a_bound_below_one_is_refused_by_name():
    with pytest.raises(HawkError, match="bound is 0"):
        hawk.sort(np.zeros(10, dtype=np.int32), bound=0)


def test_a_bound_that_is_not_an_integer_is_refused_by_name():
    with pytest.raises(HawkError, match="bound=2.5 is not an integer"):
        hawk.sort(np.zeros(10, dtype=np.int32), bound=2.5)


def test_keys_outside_the_bound_are_refused_with_their_count():
    keys = np.array([0, 4, 1, -1, 2, 5, 3], dtype=np.int32)
    with pytest.raises(HawkError, match=r"sort: 3 keys are outside \[0, 4\)"):
        hawk.sort(keys, bound=4)


def test_keys_of_two_axes_are_refused_naming_the_axes():
    with pytest.raises(HawkError, match="run-time axes"):
        hawk.sort(np.zeros((3, 50), dtype=np.int32), bound=4)


def test_planes_of_the_wrong_extent_or_type_are_refused_by_name():
    keys = keys_of(10, 4)
    with pytest.raises(HawkError, match="starts has 4 elements, bound . 1 is 5"):
        hawk.sort(keys, bound=4, out=(np.zeros(10, np.int32), np.zeros(4, np.int32)))
    with pytest.raises(HawkError, match="order extent differs"):
        hawk.sort(keys, bound=4, out=(np.zeros(9, np.int32), np.zeros(5, np.int32)))
    with pytest.raises(HawkError, match="order must be an int32 or uint32 plane"):
        hawk.sort(keys, bound=4, out=(np.zeros(10, np.float64), np.zeros(5, np.int32)))


# ---- device row (runs on a machine with a card, once the device sort is behind the door) ----


@pytest.mark.gpu
def test_a_device_sort_equals_the_stable_argsort():
    cupy = pytest.importorskip("cupy")
    keys = keys_of(1_000_000, 4096)
    order, starts = hawk.sort(cupy.asarray(keys), bound=4096)
    expect_sorted(keys, 4096, cupy.asnumpy(order), cupy.asnumpy(starts))
