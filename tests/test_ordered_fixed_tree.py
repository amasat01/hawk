# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""An ordered sum has a FIXED grouping that depends on the lane count only: 65536 virtual lanes (lane ``l`` adds
rows ``l, l + 65536, ...`` in order), folded in groups of 1024 by the halving tree ``s[i] += s[i + w]``, then the
64 group results by the same tree. The host and the device give the same bits, equal to the NumPy emulation here.
An ordered scatter's device walk (shared bins, a warp per 32 rows) has the bits of the host's serial chunk walk."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import sidecar_of

import hawk
from hawk import Grid, Index, Mutable, Scalar, kernel
from hawk.artifact import build_bundle
from hawk.ext import IndexSet, Kind
from hawk.ir.team import THREADS_ENV

TARGETS = ["host", pytest.param("cuda", marks=pytest.mark.gpu)]
V, G, W = 65536, 64, 1024


def _tree(s):
    s = s.copy()
    w = len(s) // 2
    while w:
        s[:w] += s[w:2 * w]
        w //= 2
    return s[0]


def tree_sum(x):
    """The fixed grouping in NumPy, in the dtype of ``x``."""
    part = np.zeros(V, x.dtype)
    for r in range(0, len(x), V):
        row = x[r:r + V]
        part[:len(row)] += row
    return _tree(np.array([_tree(part[g * W:(g + 1) * W]) for g in range(G)], x.dtype))


def _total(samples):
    @kernel
    def total(lane: Grid["key", "sample":samples], x: Scalar) -> Scalar:
        return hawk.sum(x, over="sample", policy="ordered")
    return total


@kernel
def add(key: Index, x: Scalar, acc: Mutable[Scalar]):
    hawk.scatter(acc, policy="ordered")[key] += x


def _exec(bundle, name, target, planes, out):
    planes = {k: v.copy() if k == out else v for k, v in planes.items()}
    if target == "host":
        from hawk import runtime as rt

        rt.run(rt.load(bundle.directory, name, sidecar_of(bundle, name)), **planes)
        return planes[out]
    import cupy as cp
    from _team_stages import run_device

    dev = {k: cp.asarray(v) for k, v in planes.items()}
    run_device(bundle, name, dev, len(next(iter(planes.values()))) if "key" not in planes else len(planes["key"]))
    return cp.asnumpy(dev[out])


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("n", [1, 1000, 65536, 3 * 65536 + 777, 200003])
def test_ordered_sum_is_the_numpy_tree_bitwise_and_replays(tmp_path, cache_dir, target, n):
    k = _total(n)
    bundle = build_bundle([k], tmp_path, targets=("cuda", "host"), cache_dir=cache_dir, host_profile="native")
    x = np.random.default_rng(n).normal(size=n) * np.exp(np.random.default_rng(n + 1).normal(size=n))
    runs = [_exec(bundle, "total", target, {"x": x, "out": np.zeros(1)}, "out") for _ in range(3)]
    assert runs[0].tobytes() == runs[1].tobytes() == runs[2].tobytes()
    assert runs[0][0] == tree_sum(x)


def _omp_threads(count):
    """Set the host OpenMP team size of this process (skips where libgomp is not the runtime)."""
    import ctypes

    try:
        ctypes.CDLL("libgomp.so.1").omp_set_num_threads(count)
    except OSError:
        pytest.skip("libgomp is not loadable")


@pytest.mark.parametrize("threads", [1, 2, 6])
@pytest.mark.parametrize("n", [70001, 1 << 24])         # 70001: a partial second stride block; 2^24: 256 rows per virtual lane
def test_host_ordered_sum_is_the_numpy_tree_bitwise_across_thread_counts(tmp_path, cache_dir, threads, n):
    _omp_threads(threads)
    bundle = build_bundle([_total(n)], tmp_path, targets=("host",), cache_dir=cache_dir, host_profile="native")
    x = np.random.default_rng(n).normal(size=n) * np.exp(np.random.default_rng(n + 1).normal(size=n))
    got = _exec(bundle, "total", "host", {"x": x, "out": np.zeros(1)}, "out")
    assert got[0] == tree_sum(x)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("cells", [7, 1000, 1500])      # 1500: more cells than the shared bins hold
def test_ordered_scatter_walk_has_the_serial_bits(tmp_path, cache_dir, target, cells):
    bundle = build_bundle([add], tmp_path, targets=("cuda", "host"), cache_dir=cache_dir, host_profile="native")
    rng = np.random.default_rng(cells)
    n = 30000
    key = rng.integers(0, cells, n).astype(np.int32)
    key[::7] = -1
    x = rng.normal(size=n) * np.exp(rng.normal(size=n))
    got = _exec(bundle, "add", target, {"key": key, "x": x, "acc": np.zeros(cells)}, "acc")
    host = _exec(bundle, "add", "host", {"key": key, "x": x, "acc": np.zeros(cells)}, "acc")
    assert got.tobytes() == host.tobytes()


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("threads", [32, 40, 64, 100, 128, 256])     # 40 and 100: blocks that are not whole warps
def test_ordered_sum_bits_do_not_depend_on_the_team_size(tmp_path, cache_dir, monkeypatch, target, threads):
    monkeypatch.setenv(THREADS_ENV, str(threads))
    n = 2 * 65536 + 4321
    bundle = build_bundle([_total(n)], tmp_path, targets=("cuda", "host"), cache_dir=cache_dir, host_profile="native")
    x = np.random.default_rng(threads).normal(size=n)
    got = _exec(bundle, "total", target, {"x": x, "out": np.zeros(1)}, "out")
    assert got[0] == tree_sum(x)


SET = Kind("lane_set", lanes=IndexSet())


def _set_total(rows):
    def total_set(lane: Grid["key", "row":rows], x: Scalar) -> Scalar:
        return hawk.sum(x, over="row", policy="ordered")
    return hawk.kernel(total_set, kind=SET)


def test_an_index_set_build_keeps_the_general_block_sum(tmp_path, cache_dir):
    """An index-set build has no virtual-lane stage: the listed rows of each key add in order (host run)."""
    rows, keys = 300, 5
    bundle = build_bundle([_set_total(rows)], tmp_path, targets=("host",), cache_dir=cache_dir, host_profile="native")
    x = np.random.default_rng(3).normal(size=keys * rows)
    live = np.sort(np.random.default_rng(4).choice(keys * rows, 700, replace=False)).astype(np.int32)
    out = np.zeros(keys)
    from hawk import runtime as rt

    rt.run(rt.load(bundle.directory, "total_set", sidecar_of(bundle, "total_set")), x=x, out=out,
           active_map=live, active_count=np.array([len(live)], np.uint32))
    want = np.zeros(keys)
    for k in range(keys):
        for i in live[(live >= k * rows) & (live < (k + 1) * rows)]:
            want[k] += x[i]
    np.testing.assert_allclose(out, want, rtol=1e-12)
