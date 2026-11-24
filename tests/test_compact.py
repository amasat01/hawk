# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.compact``: the kept lanes in lane order, the count, the untouched tail, and a graph that branches on the count.

Oracles: ``np.flatnonzero(keep)`` for ``kept[:count]`` and ``np.count_nonzero(keep)`` for ``count``, exact.
"""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import HawkError, Mutable, Scalar

RNG = np.random.default_rng(20261129)
#: Empty, one, around the scan block of 1024, and a million.
SIZES = (0, 1, 2, 1023, 1024, 1025, 4097, 1_000_000)
MARKER = -7


@hawk.kernel
def double(x: Scalar, y: Mutable[Scalar]):
    y = x * 2.0


def masks(n):
    """All kept, none kept, alternating and a random mask of density one in four."""
    return {
        "all": np.ones(n, dtype=bool),
        "none": np.zeros(n, dtype=bool),
        "alternating": np.arange(n) % 2 == 1,
        "random": RNG.random(n) < 0.25,
    }


@pytest.mark.parametrize("pattern", ["all", "none", "alternating", "random"])
@pytest.mark.parametrize("n", SIZES)
def test_kept_and_count_equal_flatnonzero_and_count_nonzero_and_the_tail_is_not_written(
    n, pattern
):
    keep = masks(n)[pattern]
    kept = np.full(n, MARKER, dtype=np.int32)
    count = np.full(1, 99, dtype=np.uint32)
    got_kept, got_count = hawk.compact(keep, out=(kept, count))
    assert got_kept is kept and got_count is count
    wanted = np.flatnonzero(keep)
    assert count[0] == np.count_nonzero(keep)
    np.testing.assert_array_equal(kept[: count[0]], wanted)
    assert (kept[count[0] :] == MARKER).all()


def test_the_planes_the_library_makes_have_the_shape_and_types_of_the_result():
    keep = RNG.random(5000) < 0.5
    kept, count = hawk.compact(keep)
    assert kept.shape == (5000,) and kept.dtype == np.int32
    assert count.shape == (1,) and count.dtype == np.uint32
    np.testing.assert_array_equal(kept[: count[0]], np.flatnonzero(keep))


def test_the_result_is_the_same_with_one_thread_and_with_the_default():
    keep = RNG.random(1_000_000) < 0.3
    wanted = hawk.host_threads()
    try:
        hawk.set_host_threads(1)
        one, one_count = hawk.compact(keep)
        hawk.set_host_threads(None)
        many, many_count = hawk.compact(keep)
    finally:
        hawk.set_host_threads(wanted)
    assert one_count[0] == many_count[0]
    np.testing.assert_array_equal(one[: one_count[0]], many[: many_count[0]])


# ---- recorded: a branch on the count ----


@hawk.Graph.builder
def guarded(keep, x, y) -> hawk.Graph:
    g = hawk.Graph()
    kept, count = hawk.compact(keep)
    if count:
        double(x, y=y)
    g.kept, g.count, g.result = kept, count, y
    return g


def test_a_branch_on_the_count_is_taken_only_when_something_is_kept_and_a_changed_mask_is_read():
    """Oracle: ``np.count_nonzero`` decides the branch; the arm writes ``2 * x`` into ``y``."""
    keep = np.zeros(2000, dtype=bool)
    x = RNG.normal(size=2000)
    y = np.full(2000, -5.0)
    graph = guarded(keep, x, y)
    graph.launch()
    assert (
        graph.count[0] == 0 and (y == -5.0).all()
    )  # nothing kept: the arm is not taken
    keep[[5, 1999]] = True  # a replay reads the mask as it is then
    graph.launch()
    assert graph.count[0] == 2
    np.testing.assert_array_equal(graph.kept[:2], [5, 1999])
    np.testing.assert_array_equal(y, 2.0 * x)  # kept: taken
    assert graph.node_labels()[0] == "compact"
    graph.close()


def test_a_recording_that_holds_a_compaction_reverses_with_no_adjoint_through_kept_and_count():
    """Oracle: ``kept`` and ``count`` are integers; the kernel beside the compaction reverses as it does alone, ``dJ/dx = 2 * seed``."""

    def build(keep, x) -> hawk.Graph:
        g = hawk.Graph(record=True)
        kept, count = hawk.compact(keep)
        g.result = double(x)
        return g

    graph = hawk.Graph.builder(build)(RNG.random(100) < 0.5, RNG.normal(size=100))
    graph.launch()
    seed = RNG.normal(size=100)
    gradients = hawk.reverse(graph.record, {"y": seed})
    np.testing.assert_allclose(gradients["x"], 2.0 * seed, rtol=1e-14)
    graph.close()


# ---- refusals ----


def test_a_keep_of_two_axes_is_refused_naming_the_axes():
    with pytest.raises(HawkError, match="run-time axes"):
        hawk.compact(np.ones((3, 50), dtype=bool))


def test_a_keep_that_is_not_a_bool_plane_is_refused_by_name():
    with pytest.raises(HawkError, match="keep must be a bool plane"):
        hawk.compact(np.ones(10, dtype=np.float64))


def test_a_kept_of_another_extent_or_a_count_of_two_is_refused_by_name():
    keep = np.ones(10, dtype=bool)
    with pytest.raises(HawkError, match="extent differs"):
        hawk.compact(
            keep, out=(np.zeros(9, dtype=np.int32), np.zeros(1, dtype=np.uint32))
        )
    with pytest.raises(HawkError, match="one-element uint32"):
        hawk.compact(
            keep, out=(np.zeros(10, dtype=np.int32), np.zeros(2, dtype=np.uint32))
        )


# ---- device row (runs on a machine with a card, once the device compaction is behind the door) ----


@pytest.mark.gpu
def test_a_device_compaction_equals_flatnonzero():
    cupy = pytest.importorskip("cupy")
    keep = RNG.random(1_000_000) < 0.25
    kept, count = hawk.compact(cupy.asarray(keep))
    np.testing.assert_array_equal(
        cupy.asnumpy(kept)[: int(cupy.asnumpy(count)[0])], np.flatnonzero(keep)
    )
