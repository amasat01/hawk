# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.scan``: values, bounds and refusals, eager and recorded.

Oracles: NumPy's sequential ``add.accumulate``, ``minimum.accumulate`` and ``maximum.accumulate`` for integers and for
minimum and maximum (exact, an unsigned or signed sum wraps as they do); an exact integer sum of the float64 values for a real
sum, against the bound ``|out[i] - exact| <= i * u * sum(|in[0..i]|)``; and the float64 sum for float32. The exclusive form is
the inclusive one shifted with the identity first.
"""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import HawkError, Scalar, Mutable

RNG = np.random.default_rng(20261128)
#: One below, at and above the 1024-element block, a few blocks, and a million.
SIZES = (1, 2, 1023, 1024, 1025, 2048, 4097, 1_000_000)
INTEGERS = (np.int32, np.int64, np.uint32, np.uint64)
ACCUMULATE = {
    "sum": lambda x: np.add.accumulate(x, dtype=x.dtype),
    "min": np.minimum.accumulate,
    "max": np.maximum.accumulate,
}


@hawk.kernel
def double(x: Scalar, y: Mutable[Scalar]):
    y = x * 2.0


@hawk.kernel
def tenfold(r: Scalar, w: Mutable[Scalar]):
    w = r * 10.0


def integers(dtype, n):
    """Values over the whole range of ``dtype``, so a sum wraps."""
    info = np.iinfo(dtype)
    return RNG.integers(info.min, info.max, size=n, dtype=dtype, endpoint=True)


def identity(op, dtype):
    """What an exclusive scan starts with: 0, the largest value for min, the smallest for max (infinite for a real type)."""
    if op == "sum":
        return dtype(0)
    real = np.issubdtype(dtype, np.floating)
    low = op == "max"
    if real:
        return dtype(-np.inf if low else np.inf)
    info = np.iinfo(dtype)
    return dtype(info.min if low else info.max)


@pytest.mark.parametrize("op", ["sum", "min", "max"])
@pytest.mark.parametrize("dtype", INTEGERS)
@pytest.mark.parametrize("n", SIZES)
def test_an_integer_scan_equals_the_sequential_loop_exactly_in_both_forms(dtype, op, n):
    x = integers(dtype, n)
    want = ACCUMULATE[op](x)
    np.testing.assert_array_equal(hawk.scan(x, op), want)
    shifted = np.concatenate([[identity(op, dtype)], want[:-1]]).astype(dtype)
    np.testing.assert_array_equal(hawk.scan(x, op, exclusive=True), shifted)


@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", SIZES)
def test_a_real_minimum_or_maximum_scan_is_exact_in_both_forms(dtype, op, n):
    x = RNG.normal(size=n).astype(dtype)
    want = ACCUMULATE[op](x)
    np.testing.assert_array_equal(hawk.scan(x, op), want)
    shifted = np.concatenate([[identity(op, dtype)], want[:-1]]).astype(dtype)
    np.testing.assert_array_equal(hawk.scan(x, op, exclusive=True), shifted)


def test_a_float64_sum_scan_is_within_the_bound_of_the_exact_sum_in_both_forms():
    """Oracle: the exact integer sum of the values; ``u`` is 2 ** -53."""
    x = RNG.normal(size=100_000) * RNG.choice([1e-3, 1.0, 1e3], size=100_000)
    scaled = [
        int(v * 2.0**60) for v in x
    ]  # a multiple of 2 ** -60 holds every value here to well under u
    exact, running = [], 0
    for v in scaled:
        running += v
        exact.append(running / 2.0**60)
    exact = np.array(exact)
    magnitude = np.cumsum(np.abs(x))
    unit = 2.0**-53
    inclusive = hawk.scan(x, "sum")
    # the 2 ** -60 grid is finer than u times the smallest term, so it adds at most the grid's own error to the bound
    slack = 1e5 * 2.0**-60
    assert np.all(
        np.abs(inclusive - exact) <= (np.arange(len(x)) + 1) * unit * magnitude + slack
    )
    exclusive = hawk.scan(x, "sum", exclusive=True)
    assert exclusive[0] == 0.0
    assert np.all(
        np.abs(exclusive[1:] - exact[:-1])
        <= (np.arange(len(x) - 1) + 1) * unit * magnitude[:-1] + slack
    )


@pytest.mark.parametrize("n", [1000, 1_000_000])
def test_a_float32_sum_scan_is_within_the_bound_of_the_float64_sum(n):
    """Oracle: ``np.cumsum`` of the same values in float64 (its own error is a factor 2 ** 29 below float32's); ``u`` is 2 ** -24."""
    x = RNG.normal(size=n).astype(np.float32)
    want = np.cumsum(x.astype(np.float64))
    magnitude = np.cumsum(np.abs(x.astype(np.float64)))
    got = hawk.scan(x, "sum").astype(np.float64)
    assert np.all(np.abs(got - want) <= (np.arange(n) + 1) * 2.0**-24 * magnitude)


def test_a_real_sum_scan_is_the_same_bits_with_one_thread_and_with_the_default():
    x = RNG.normal(size=1_000_000) * 1e3
    wanted = hawk.host_threads()
    try:
        hawk.set_host_threads(1)
        one = hawk.scan(x, "sum")
        hawk.set_host_threads(None)
        many = hawk.scan(x, "sum")
    finally:
        hawk.set_host_threads(wanted)
    np.testing.assert_array_equal(one, many)


def test_a_size_zero_scan_runs_and_writes_nothing():
    for exclusive in (False, True):
        assert hawk.scan(np.zeros(0), "sum", exclusive=exclusive).shape == (0,)


def test_out_passes_the_callers_plane_and_the_scan_fills_it():
    x = RNG.integers(-50, 50, size=3000).astype(np.int64)
    out = np.zeros_like(x)
    assert hawk.scan(x, "sum", out=out) is out
    np.testing.assert_array_equal(out, np.cumsum(x))


# ---- recorded ----


def test_a_recorded_scan_between_two_kernels_equals_the_eager_one_and_numpy_and_replays_changed_inputs():
    """Oracle: ``np.cumsum`` of twice the input, scaled by ten, over whole numbers (exact in any grouping)."""

    def build(x) -> hawk.Graph:
        g = hawk.Graph()
        g.running = hawk.scan(double(x), "sum")
        g.result = tenfold(g.running)
        return g

    x = RNG.integers(-100, 100, size=5000).astype(np.float64)
    graph = hawk.Graph.builder(build)(x)
    graph.launch()
    np.testing.assert_array_equal(graph.running, np.cumsum(2.0 * x))
    np.testing.assert_array_equal(graph.result, 10.0 * np.cumsum(2.0 * x))
    np.testing.assert_array_equal(graph.running, hawk.scan(2.0 * x, "sum"))
    assert graph.node_labels() == ["double", "scan", "tenfold"]
    x += 1.0  # a replay reads the planes as they are then
    graph.launch()
    np.testing.assert_array_equal(graph.running, np.cumsum(2.0 * x))
    graph.close()


def reverse_of(build, *arrays, seed):
    """The gradients of ``seed`` on the result ``w`` of the graph ``build`` writes with ``record=True``, replayed once."""
    graph = hawk.Graph.builder(build)(*arrays)
    graph.launch()
    return hawk.reverse(graph.record, {"w": seed})


@pytest.mark.parametrize("exclusive", [False, True])
def test_the_reverse_of_a_sum_scan_is_the_suffix_scan_of_the_cotangent(exclusive):
    """Oracle: ``r = scan(2 x)``, ``w = 10 r``: ``dJ/dx = 20 * cumsum(seed[::-1])[::-1]``, shifted by one lane when exclusive."""

    def build(x) -> hawk.Graph:
        g = hawk.Graph(record=True)
        g.result = tenfold(hawk.scan(double(x), "sum", exclusive=exclusive))
        return g

    x, seed = RNG.normal(size=100), RNG.normal(size=100)
    suffix = np.cumsum(seed[::-1])[::-1]
    if exclusive:
        suffix = np.append(suffix[1:], 0.0)
    gradients = reverse_of(build, x, seed=seed)
    np.testing.assert_allclose(gradients["x"], 20.0 * suffix, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("exclusive", [False, True])
def test_the_reverse_of_a_minimum_or_maximum_scan_goes_to_the_running_winner(
    op, exclusive
):
    """Oracle: a loop over the lanes giving the cotangent of ``out[k]`` to the first lane of the running extreme."""

    def build(x) -> hawk.Graph:
        g = hawk.Graph(record=True)
        g.result = tenfold(hawk.scan(double(x), op, exclusive=exclusive))
        return g

    x = RNG.integers(0, 6, size=100).astype(np.float64)  # ties
    seed = RNG.normal(size=100)
    want, winner = np.zeros(100), -1
    for k in range(100):
        if exclusive and winner >= 0:
            want[winner] += 20.0 * seed[k]
        if winner < 0 or (x[k] < x[winner] if op == "min" else x[k] > x[winner]):
            winner = k
        if not exclusive:
            want[winner] += 20.0 * seed[k]
    gradients = reverse_of(build, x, seed=seed)
    np.testing.assert_allclose(gradients["x"], want, rtol=1e-12, atol=1e-12)


# ---- refusals ----


def test_a_plane_of_two_axes_is_refused_naming_the_group_as_the_way():
    with pytest.raises(HawkError, match=r"run-time axes.*g\.scan"):
        hawk.scan(RNG.normal(size=(3, 50)), "sum")


def test_a_type_outside_the_six_is_refused_naming_them():
    with pytest.raises(
        HawkError, match="int32, int64, uint32, uint64, float32, float64"
    ):
        hawk.scan(np.zeros(10, dtype=np.int8), "sum")


def test_out_that_is_the_input_or_of_another_type_or_extent_is_refused_by_name():
    x = RNG.normal(size=100)
    with pytest.raises(HawkError, match="overlaps"):
        hawk.scan(x, "sum", out=x)
    with pytest.raises(HawkError, match="dtype differs"):
        hawk.scan(x, "sum", out=np.zeros(100, dtype=np.float32))
    with pytest.raises(HawkError, match="extent differs"):
        hawk.scan(x, "sum", out=np.zeros(99))


def test_an_op_the_scan_does_not_have_is_refused_by_name():
    with pytest.raises(
        HawkError, match=r"op='mean' is not one of \('sum', 'min', 'max'\)"
    ):
        hawk.scan(np.zeros(4), "mean")


# ---- device row (runs on a machine with a card, once the device scan is written) ----


@pytest.mark.gpu
def test_a_device_scan_equals_numpy():
    cupy = pytest.importorskip("cupy")
    x = RNG.integers(-1000, 1000, size=1_000_000).astype(np.int32)
    np.testing.assert_array_equal(
        cupy.asnumpy(hawk.scan(cupy.asarray(x), "sum")), np.cumsum(x, dtype=np.int32)
    )
