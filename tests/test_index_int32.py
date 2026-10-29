# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""An ``Index`` plane binds ``int32``: the sidecar records ``int32``, a gather by it
(``Table.at``, a ``Table[Index]``, a read at another lane) reads the right rows with and
without an index-set launch, a 64-bit ``int64`` column is refused by name, and
``Index[int64]`` is the plane bound at 64 bits."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import sidecar_of

import hawk
from hawk import HawkError, Index, Mutable, Scalar, Table
from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.ext import IndexSet, Kind

SET = Kind("lane_set", lanes=IndexSet())
N = 24
LIVE = np.array([19, 3, 8, 0, 14], dtype=np.int32)


def _gather(x: Table[Scalar], idx: Index, y: Mutable[Scalar]):
    y = x.at(idx) * 2.0


def _gather64(x: Table[Scalar], idx: Index[np.int64], y: Mutable[Scalar]):
    y = x.at(idx) * 2.0


def _chain(x: Table[Scalar], perm: Table[Index], idx: Index, y: Mutable[Scalar]):
    y = x.at(perm.at(idx)) + idx.at(0) * 0.5


def _wide_math(idx: Index, y: Mutable[Scalar]):
    # arithmetic on an index stays 64-bit: this product overflows 32 bits
    y = hawk.math.real(idx * 3_000_000_000 // 3_000_000_000)


KERNELS = {"gather": hawk.kernel(_gather), "gather64": hawk.kernel(_gather64),
           "chain": hawk.kernel(_chain), "wide": hawk.kernel(_wide_math),
           "gather_set": SET(_gather)}


@pytest.fixture(scope="module")
def built(tmp_path_factory, cache_dir):
    out = {}
    for name, k in KERNELS.items():
        out[name] = build_bundle([k], tmp_path_factory.mktemp(name), targets=("cuda", "host"),
                                 cache_dir=cache_dir, host_profile="native")
    return out


def _load(built, name):
    bundle, k = built[name], KERNELS[name]
    return rt.load(bundle.directory, k.name, sidecar_of(bundle, k.name))


def _data(dtype=np.int32):
    rng = np.random.default_rng(3)
    return (rng.uniform(1.0, 5.0, N), rng.permutation(N).astype(dtype))


def test_the_sidecar_records_int32_for_an_index_plane_and_int64_for_index_int64(built):
    assert sidecar_of(built["gather"], "_gather")["index_planes"] == ["idx"]
    assert sidecar_of(built["chain"], "_chain")["index_planes"] == ["idx", "perm"]
    assert "index_planes" not in sidecar_of(built["gather64"], "_gather64")
    assert sidecar_of(built["gather"], "_gather")["arg_dtypes"]["idx"] == "int32"
    assert sidecar_of(built["chain"], "_chain")["arg_dtypes"]["perm"] == "int32"
    assert sidecar_of(built["gather64"], "_gather64")["arg_dtypes"]["idx"] == "int64"


def test_a_gather_by_an_int32_index_reads_the_right_rows_on_the_host(built):
    x, idx = _data()
    y = np.zeros(N)
    rt.run(_load(built, "gather"), x=x, idx=idx, y=y, n_samples=N)
    np.testing.assert_array_equal(y, x[idx] * 2.0)


def test_a_gather_by_an_int32_index_under_an_index_set_reads_the_right_rows_on_the_host(built):
    x, idx = _data()
    y = np.zeros(N)
    rt.run(_load(built, "gather_set"), x=x, idx=idx, y=y, n_samples=N, active_map=LIVE,
           active_count=np.array([len(LIVE)], np.uint32))
    want = np.zeros(N)
    want[LIVE] = x[idx[LIVE]] * 2.0
    np.testing.assert_array_equal(y, want)


def test_a_table_of_index_and_a_read_at_another_lane_take_int32_on_the_host(built):
    x, idx = _data()
    perm = np.random.default_rng(5).permutation(N).astype(np.int32)
    y = np.zeros(N)
    rt.run(_load(built, "chain"), x=x, perm=perm, idx=idx, y=y, n_samples=N)
    np.testing.assert_array_equal(y, x[perm[idx]] + idx[0] * 0.5)


def test_index_arithmetic_is_not_narrowed_to_32_bits(built):
    y = np.zeros(4)
    rt.run(_load(built, "wide"), idx=np.array([1, 7, 2**31 - 1, 5], np.int32), y=y, n_samples=4)
    np.testing.assert_array_equal(y, [1.0, 7.0, 2.0**31 - 1, 5.0])


def test_index_int64_binds_int64_and_widens_int32_on_the_host(built):
    x, idx = _data(np.int64)
    for column in (idx, idx.astype(np.int32)):
        y = np.zeros(N)
        rt.run(_load(built, "gather64"), x=x, idx=column, y=y, n_samples=N)
        np.testing.assert_array_equal(y, x[idx] * 2.0)


def test_an_int64_column_that_fits_is_converted_to_int32_on_the_host(built):
    x, idx = _data(np.int64)
    y = np.zeros(N)
    rt.run(_load(built, "gather"), x=x, idx=idx, y=y, n_samples=N)
    np.testing.assert_array_equal(y, x[idx] * 2.0)


def test_an_int64_column_with_a_value_past_int32_is_refused_by_name(built):
    x, idx = _data(np.int64)
    idx[3] = 2**31
    with pytest.raises(HawkError, match=r"'idx': values exceed int32; declare Index\[int64\]"):
        rt.run(_load(built, "gather"), x=x, idx=idx, y=np.zeros(N), n_samples=N)


def test_an_int32_column_binds_without_a_copy(built):
    from hawk._bind_checks import narrow_index

    _, idx = _data()
    assert narrow_index(["idx"], {"idx": idx})["idx"] is idx


def test_a_float_column_is_still_refused_by_name(built):
    x, idx = _data()
    with pytest.raises(HawkError, match=r"'idx'.*expected dtype int32, got float64"):
        rt.run(_load(built, "gather"), x=x, idx=idx.astype(np.float64), y=np.zeros(N), n_samples=N)


def _device(built, name, tag_set=False, **planes):
    import _deploy as L
    import cupy as cp
    import eagle.exec as eexec
    from eagle import plan as eplan

    dev = {k: cp.asarray(v) for k, v in planes.items()}
    plugin = L.device_plugin(built[name].directory, KERNELS[name].name, sidecar_of(built[name], KERNELS[name].name))
    eplan.plan(plugin, structure=eexec.DeviceKernel).bind(**dev).launch()
    cp.cuda.Device().synchronize()
    return dev["y"].get()


@pytest.mark.gpu
def test_a_gather_by_an_int32_index_reads_the_right_rows_on_the_device(built):
    x, idx = _data()
    np.testing.assert_array_equal(_device(built, "gather", x=x, idx=idx, y=np.zeros(N)), x[idx] * 2.0)


@pytest.mark.gpu
def test_a_gather_by_an_int32_index_under_an_index_set_reads_the_right_rows_on_the_device(built):
    x, idx = _data()
    got = _device(built, "gather_set", x=x, idx=idx, y=np.zeros(N), active_map=LIVE,
                  active_count=np.array([len(LIVE)], np.uint32))
    want = np.zeros(N)
    want[LIVE] = x[idx[LIVE]] * 2.0
    np.testing.assert_array_equal(got, want)


@pytest.mark.gpu
def test_an_index_int64_plane_reads_the_right_rows_on_the_device(built):
    x, idx = _data(np.int64)
    np.testing.assert_array_equal(_device(built, "gather64", x=x, idx=idx, y=np.zeros(N)), x[idx] * 2.0)
