# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A compensated sum over a sample axis commits once per block when the key is uniform over the block.

``hawk.sum(x, over="sample")`` on ``Grid["key", "sample":E]`` with ``E`` a multiple of 1024 folds the warps of a full
block through shared memory and issues one compensated commit per block (``addCompensatedBlock``); any other extent keeps
the commit per warp. The delivered value (the library folds the companion into the plane) stays within the compensated bound of ``math.fsum``, the
partial last block and a launch that is not block-aligned included (they take the per-warp commit).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import _launch, _xp

import hawk
from hawk import Grid, Kernel, Scalar, kernel
from hawk.artifact import build_bundle
from hawk.targets import TARGETS


def _sum(extent, contract="compensated"):
    @kernel
    def ksum(lane: Grid["key", "sample":extent], x: Scalar) -> Scalar:
        return hawk.sum(x, over="sample", contract=contract)
    return Kernel(f"ksum{extent}", ksum.sinks, ksum.planes)


def _scatter():
    @kernel
    def kscat(lane: Grid["key", "sample":4096], k: hawk.Index, v: Scalar, acc: hawk.Mutable[Scalar]):
        hawk.scatter(acc)[k] += v
    kern = Kernel("kscat", kscat.sinks, kscat.planes)
    return kern.sinks, kern.walk, TARGETS["cuda"]


@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("extent, keys", [(2048, 1), (2048, 3), (1000, 2), (1 << 16, 2)])
def test_the_delivered_sum_is_within_the_compensated_bound(tmp_path, cache_dir, target, extent, keys):
    kern = _sum(extent)
    bundle = build_bundle([kern], tmp_path, mode="float64", targets=("cuda", "host"), cache_dir=cache_dir,
                          host_profile="native")
    xp = _xp(target)
    x = np.random.default_rng(extent + keys).uniform(-1.0, 1.0, (keys, extent)) * 10.0 ** np.random.default_rng(3).integers(-3, 3, (keys, extent))
    # the library makes the companion plane and folds it into ``out``
    planes = {"x": xp.asarray(x.reshape(-1)), "out": xp.zeros(keys)}
    bound = _launch(bundle, f"ksum{extent}", target, planes)
    bound.launch()
    bound.wait()                    # a launch over device memory is queued: the plane is read after the wait
    got = np.array(planes["out"].get() if hasattr(planes["out"], "get") else planes["out"])
    for k in range(keys):
        assert abs(got[k] - math.fsum(x[k])) <= 2.0 * np.finfo(np.float64).eps * np.abs(x[k]).sum()
