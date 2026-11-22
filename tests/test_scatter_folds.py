# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.scatter(y)[key].min(v)``, ``.max(v)``, ``.and_(b)``, ``.xor(b)``: folds by key.

Every lane's value is combined with the plane's initial value, in any order, so the plane ends as the NumPy fold
``np.minimum.at``, ``np.maximum.at``, ``np.bitwise_and.at`` or ``np.bitwise_xor.at`` gives, for any number of lanes on
one cell, bit for bit. The oracle of each test is that NumPy function.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import _launch

import hawk
from hawk import HawkError, Index, Int32, Mutable, Scalar, UInt32, kernel
from hawk.artifact import build_bundle, emit_source

N = 2000
CELLS = 5


@kernel
def reals(key: Index, x: Scalar, lo: Mutable[Scalar], hi: Mutable[Scalar]):
    hawk.scatter(lo)[key].min(x)
    hawk.scatter(hi)[key].max(x)


@kernel
def reals_two_axes(
    i: Index, k: Index, x: Scalar, lo: Mutable[Scalar, "row":3, "col":4], hi: Mutable[Scalar, "row":3, "col":4]  # noqa: F821
):
    hawk.scatter(lo)[i, k].min(x)
    hawk.scatter(hi)[i, k].max(x)


@kernel
def ints(key: Index, v: Int32, lo: Mutable[Int32], hi: Mutable[Int32]):
    hawk.scatter(lo)[key].min(v)
    hawk.scatter(hi)[key].max(v)


@kernel
def bits(key: Index, b: UInt32, both: Mutable[UInt32], flip: Mutable[UInt32]):
    hawk.scatter(both)[key].and_(b)
    hawk.scatter(flip)[key].xor(b)


@pytest.fixture(scope="module", params=["float64", "float32"])
def mode(request):
    return request.param


@pytest.fixture(scope="module")
def bundle(mode, tmp_path_factory, cache_dir):
    return build_bundle([reals, reals_two_axes, ints, bits], tmp_path_factory.mktemp("folds"), mode=mode,
                        targets=("cuda", "host"), cache_dir=cache_dir)


def _real(mode):
    return np.float32 if mode == "float32" else np.float64


def _crowded_keys(rng, cells):
    """Keys with many lanes on few cells: most lanes hit cell 0."""
    return np.where(rng.random(N) < 0.5, 0, rng.integers(0, cells, N)).astype(np.int32)


def test_min_and_max_by_key_equal_minimum_at_and_maximum_at_on_the_host(bundle, mode):
    real = _real(mode)
    rng = np.random.default_rng(1)
    key, x = _crowded_keys(rng, CELLS), rng.normal(size=N).astype(real)
    lo, hi = np.full(CELLS, np.inf, real), np.full(CELLS, -np.inf, real)
    want_lo, want_hi = lo.copy(), hi.copy()
    np.minimum.at(want_lo, key, x)
    np.maximum.at(want_hi, key, x)
    _launch(bundle, "reals", "host", {"key": key, "x": x, "lo": lo, "hi": hi}).launch()
    assert lo.tobytes() == want_lo.tobytes() and hi.tobytes() == want_hi.tobytes()


def test_a_fold_keeps_the_value_the_caller_gave_the_plane(bundle, mode):
    real = _real(mode)
    key, x = np.zeros(4, np.int32), np.array([3, 4, 5, 6], real)
    lo, hi = np.array([4.5], real), np.array([4.5], real)
    _launch(bundle, "reals", "host", {"key": key, "x": x, "lo": lo, "hi": hi}).launch()
    assert (lo[0], hi[0]) == (3, 6)
    lo, hi = np.array([-1.0], real), np.array([9.0], real)
    _launch(bundle, "reals", "host", {"key": key, "x": x, "lo": lo, "hi": hi}).launch()
    assert (lo[0], hi[0]) == (-1, 9)


def test_min_and_max_over_two_key_axes_equal_minimum_at_and_maximum_at(bundle, mode):
    real = _real(mode)
    rng = np.random.default_rng(2)
    i, k = (np.where(rng.random(N) < 0.5, 1, rng.integers(0, n, N)).astype(np.int32) for n in (3, 4))
    x = rng.normal(size=N).astype(real)
    lo, hi = np.full((3, 4), np.inf, real), np.full((3, 4), -np.inf, real)
    want_lo, want_hi = lo.copy(), hi.copy()
    np.minimum.at(want_lo, (i, k), x)
    np.maximum.at(want_hi, (i, k), x)
    _launch(bundle, "reals_two_axes", "host", {"i": i, "k": k, "x": x, "lo": lo, "hi": hi}).launch()
    assert lo.tobytes() == want_lo.tobytes() and hi.tobytes() == want_hi.tobytes()


def test_a_nan_absorbs_in_min_and_max_whatever_its_lane(bundle, mode):
    real = _real(mode)
    for at in (0, N // 2, N - 1):
        x = np.arange(N, dtype=real)
        x[at] = np.nan
        lo, hi = np.full(1, np.inf, real), np.full(1, -np.inf, real)
        _launch(bundle, "reals", "host", {"key": np.zeros(N, np.int32), "x": x, "lo": lo, "hi": hi}).launch()
        assert np.isnan(lo[0]) and np.isnan(hi[0])


def test_integer_min_and_max_by_key_equal_minimum_at_and_maximum_at(bundle):
    rng = np.random.default_rng(3)
    key, v = _crowded_keys(rng, CELLS), rng.integers(-5000, 5000, N).astype(np.int32)
    lo, hi = np.full(CELLS, np.iinfo(np.int32).max, np.int32), np.full(CELLS, np.iinfo(np.int32).min, np.int32)
    want_lo, want_hi = lo.copy(), hi.copy()
    np.minimum.at(want_lo, key, v)
    np.maximum.at(want_hi, key, v)
    _launch(bundle, "ints", "host", {"key": key, "v": v, "lo": lo, "hi": hi}).launch()
    assert (lo == want_lo).all() and (hi == want_hi).all()


def test_and_and_xor_by_key_equal_bitwise_and_at_and_bitwise_xor_at(bundle):
    rng = np.random.default_rng(4)
    key = _crowded_keys(rng, CELLS)
    b = rng.integers(0, 2**32, N, dtype=np.uint64).astype(np.uint32) | np.uint32(0x0F0F0F0F)
    both, flip = np.full(CELLS, 0xFFFFFFFF, np.uint32), np.zeros(CELLS, np.uint32)
    want_both, want_flip = both.copy(), flip.copy()
    np.bitwise_and.at(want_both, key, b)
    np.bitwise_xor.at(want_flip, key, b)
    _launch(bundle, "bits", "host", {"key": key, "b": b, "both": both, "flip": flip}).launch()
    assert (both == want_both).all() and (flip == want_flip).all()


@pytest.mark.gpu
def test_the_device_folds_equal_the_numpy_folds(bundle, mode):
    import cupy as cp

    real = _real(mode)
    rng = np.random.default_rng(1)
    key, x = _crowded_keys(rng, CELLS), rng.normal(size=N).astype(real)
    lo, hi = cp.full(CELLS, np.inf, real), cp.full(CELLS, -np.inf, real)
    _launch(bundle, "reals", "cuda", {"key": cp.asarray(key), "x": cp.asarray(x), "lo": lo, "hi": hi}).launch()
    cp.cuda.Device().synchronize()
    want_lo, want_hi = np.full(CELLS, np.inf, real), np.full(CELLS, -np.inf, real)
    np.minimum.at(want_lo, key, x)
    np.maximum.at(want_hi, key, x)
    assert lo.get().tobytes() == want_lo.tobytes() and hi.get().tobytes() == want_hi.tobytes()


def test_a_method_that_is_not_an_operation_is_refused_by_name():
    with pytest.raises(HawkError, match=r"\.minimum\(\.\.\.\) is not an operation; the forms are: .*\.min\(v\)"):
        @kernel
        def k(key: Index, x: Scalar, y: Mutable[Scalar]):
            hawk.scatter(y)[key].minimum(x)


def test_a_fold_on_a_type_it_does_not_act_on_is_refused_by_name():
    with pytest.raises(HawkError, match=r"'and' acts on the bits of an integer plane, not on f64"):
        @kernel
        def k(key: Index, x: Scalar, y: Mutable[Scalar]):
            hawk.scatter(y)[key].and_(x)


@pytest.mark.parametrize(
    "kernel_, calls",
    [
        (reals, ["aether::accum::atomic::minAt(wout_lo(", "aether::accum::atomic::maxAt(wout_hi("]),
        (ints, ["aether::accum::atomic::minAt(wout_lo(", "aether::accum::atomic::maxAt(wout_hi("]),
        (bits, ["aether::accum::atomic::andAt(wout_both(", "aether::accum::atomic::xorAt(wout_flip("]),
    ],
    ids=["reals", "ints", "bits"],
)
def test_the_device_text_calls_one_library_helper_per_fold(kernel_, calls):
    text = emit_source(kernel_, targets=("cuda",)).sources["cuda"].text
    for call in calls:
        assert call in text
    assert "#include <aether/accum/ElementAtomic.h>" in text
