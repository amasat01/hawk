# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A compensated sum over a sample axis commits once per block when the key is uniform over the block.

``hawk.sum(x, over="sample")`` on ``Grid["key", "sample":E]`` with ``E`` a multiple of 1024 folds the warps of a full
block through shared memory and issues one compensated commit per block (``addCompensatedBlock``); any other extent keeps
the commit per warp. The delivered value (plane plus companion) stays within the compensated bound of ``math.fsum``, the
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
from hawk.emit.wrapper import render_source
from hawk.targets import TARGETS


def _sum(extent):
    @kernel
    def ksum(lane: Grid["key", "sample":extent], x: Scalar) -> Scalar:
        return hawk.sum(x, over="sample", policy="compensated")
    return Kernel(f"ksum{extent}", ksum.sinks, ksum.planes)


def _text(extent):
    k = _sum(extent)
    return render_source(f"ksum{extent}", k.sinks, k.walk, TARGETS["cuda"]).text


def test_a_block_uniform_key_commits_once_per_block_and_any_other_extent_per_warp():
    assert "addCompensatedBlock" in _text(2048) and "blockFull(base, count)" in _text(2048)
    assert "addCompensatedBlock" not in _text(1000) and "addCompensated<8>" in _text(1000)


@pytest.mark.gpu
@pytest.mark.parametrize("extent, keys", [(2048, 1), (2048, 3), (1000, 2), (1 << 16, 2)])
def test_the_delivered_sum_is_within_the_compensated_bound(tmp_path, cache_dir, extent, keys):
    kern = _sum(extent)
    bundle = build_bundle([kern], tmp_path, mode="float64", targets=("cuda", "host"), cache_dir=cache_dir,
                          host_profile="native")
    cp = _xp("cuda")
    x = np.random.default_rng(extent + keys).uniform(-1.0, 1.0, (keys, extent)) * 10.0 ** np.random.default_rng(3).integers(-3, 3, (keys, extent))
    planes = {"x": cp.asarray(x.reshape(-1)), "out": cp.zeros(keys), "out_c": cp.zeros(keys)}
    _launch(bundle, f"ksum{extent}", "cuda", planes).launch()
    got = planes["out"].get() + planes["out_c"].get()
    for k in range(keys):
        assert abs(got[k] - math.fsum(x[k])) <= 2.0 * np.finfo(np.float64).eps * np.abs(x[k]).sum()
