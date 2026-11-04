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
