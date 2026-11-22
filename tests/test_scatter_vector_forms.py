# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A vector value stored, minimised or maximised by key acts per component at the indexed element.

One update of the non-scalar value is written component by component into a sink declared with named axes. Storage order is the
value's components first, then the axes; the oracle of each test is NumPy applied to the component planes (``np.minimum.at``,
``np.maximum.at``, indexed assignment with unique keys) over a buffer in that order.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import _launch

import hawk
from hawk import HawkError, Index, Mutable, Vector, kernel
from hawk.artifact import build_bundle, emit_source
from hawk.diff import jvp, vjp

N = 1500


@kernel
def vector_extremes(
    i: Index, k: Index, v: Vector[2],
    lo: Mutable[Vector[2], "row":3, "col":4], hi: Mutable[Vector[2], "row":3, "col":4],  # noqa: F821 axis labels
):
    hawk.scatter(lo)[i, k].min(v)
    hawk.scatter(hi)[i, k].max(v)


@kernel
def vector_store(i: Index, k: Index, v: Vector[2], y: Mutable[Vector[2], "row":3, "col":4]):  # noqa: F821 axis labels
    hawk.scatter(y)[i, k] = v


@kernel
def matrix_extremes(i: Index, m: Vector[4], lo: Mutable[Vector[4], "row":3]):  # noqa: F821 axis labels
    hawk.scatter(lo)[i].min(m)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    return build_bundle([vector_extremes, vector_store, matrix_extremes], tmp_path_factory.mktemp("vector_forms"),
                        mode="float64", targets=("cuda", "host"), cache_dir=cache_dir)


def _crowded(rng, n):
    """Keys with many lanes on few cells: most lanes hit cell 0."""
    return np.where(rng.random(N) < 0.5, 0, rng.integers(0, n, N)).astype(np.int32)


def test_a_vector_min_and_max_over_two_key_axes_equal_minimum_at_and_maximum_at_per_component(bundle):
    rng = np.random.default_rng(1)
    i, k, v = _crowded(rng, 3), _crowded(rng, 4), rng.normal(size=(N, 2))
    lo, hi = np.full((2, 3, 4), np.inf), np.full((2, 3, 4), -np.inf)
    want_lo, want_hi = np.full((2, 3, 4), np.inf), np.full((2, 3, 4), -np.inf)
    for c in range(2):
        np.minimum.at(want_lo[c], (i, k), v[:, c])
        np.maximum.at(want_hi[c], (i, k), v[:, c])
    planes = {"i": i, "k": k, "v": np.ascontiguousarray(v.T), "lo": lo, "hi": hi}
    _launch(bundle, "vector_extremes", "host", planes).launch()
    assert lo.tobytes() == want_lo.tobytes()
    assert hi.tobytes() == want_hi.tobytes()


def test_a_four_component_min_by_one_key_axis_equals_minimum_at_per_component(bundle):
    rng = np.random.default_rng(3)
    i, m = _crowded(rng, 3), rng.normal(size=(N, 4))
    lo = np.full((4, 3), np.inf)
    want = lo.copy()
    for c in range(4):
        np.minimum.at(want[c], i, m[:, c])
    _launch(bundle, "matrix_extremes", "host", {"i": i, "m": np.ascontiguousarray(m.T), "lo": lo}).launch()
    assert lo.tobytes() == want.tobytes()


def test_a_vector_store_with_unique_keys_equals_indexed_assignment_per_component(bundle):
    lanes = np.arange(12)
    i, k = (lanes // 4).astype(np.int32), (lanes % 4).astype(np.int32)
    v = np.random.default_rng(4).normal(size=(12, 2))
    y = np.zeros((2, 3, 4))
    want = np.zeros((2, 3, 4))
    for c in range(2):
        want[c][i, k] = v[:, c]
    _launch(bundle, "vector_store", "host", {"i": i, "k": k, "v": np.ascontiguousarray(v.T), "y": y}).launch()
    assert y.tobytes() == want.tobytes()


@pytest.mark.gpu
def test_the_device_vector_min_and_max_equal_the_numpy_folds_per_component(bundle):
    import cupy as cp

    rng = np.random.default_rng(1)
    i, k, v = _crowded(rng, 3), _crowded(rng, 4), rng.normal(size=(N, 2))
    lo, hi = cp.full((2, 3, 4), np.inf), cp.full((2, 3, 4), -np.inf)
    planes = {"i": cp.asarray(i), "k": cp.asarray(k), "v": cp.asarray(np.ascontiguousarray(v.T)), "lo": lo, "hi": hi}
    _launch(bundle, "vector_extremes", "cuda", planes).launch()
    cp.cuda.Device().synchronize()
    want = np.full((2, 3, 4), np.inf)
    for c in range(2):
        np.minimum.at(want[c], (i, k), v[:, c])
    assert lo.get().tobytes() == want.tobytes()


@pytest.mark.parametrize("target", ["host", "cuda"])
def test_the_text_of_a_vector_min_calls_the_helper_once_per_component(target):
    text = emit_source(vector_extremes, targets=(target,)).sources[target].text
    assert text.count("aether::accum::atomic::minAt(") == 2 and text.count("aether::accum::atomic::maxAt(") == 2
    assert "aether::accum::atomic::minAt(wout_lo(0, " in text and "aether::accum::atomic::minAt(wout_lo(1, " in text


def test_a_vector_value_into_a_sink_without_axes_is_refused_by_name():
    with pytest.raises(HawkError, match=r"a vector payload is written by key into a sink declared with named axes"):
        @kernel
        def k(key: Index, v: Vector[3], lo: Mutable[Vector[3]]):
            hawk.scatter(lo)[key].min(v)


@pytest.mark.parametrize("derive", [vjp, jvp], ids=["vjp", "jvp"])
def test_a_derivative_of_a_vector_min_is_refused_by_name_as_the_scalar_one_is(derive):
    with pytest.raises(HawkError, match=r"is a keyed minimum.*write it as `hawk.scatter\(y, contract=\"ordered\"\)"):
        derive(matrix_extremes)
