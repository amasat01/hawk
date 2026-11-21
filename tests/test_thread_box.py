# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The contraction forward ``kg_fwd`` on the host: its values are the interpreter's.

The old rows compared the boxed host build with a box-free build bit for bit; the box-free arm is gone,
so the value is read back against the ``tests/_eval.py`` interpreter instead, sample by sample."""

from __future__ import annotations

import numpy as np
import pytest
from _eval import evaluate
from conftest import _launch
from _lane_kernels import BATCH, L_ALL, N_IN, S_ROW, kg_fwd

from hawk.artifact import build_bundle

ROWS = (32, 21, 13)


def _data(rows, mode="float64"):
    rng = np.random.default_rng(rows)
    feat = rng.uniform(-1, 1, 8 * L_ALL).astype(mode)
    feat[:L_ALL] = rng.integers(0, 5, L_ALL)
    th = rng.standard_normal(rows * N_IN * S_ROW).astype(mode)
    return feat, th


def _run(bundle, target, rows, mode):
    xp = np if target == "host" else pytest.importorskip("cupy")
    n = rows * BATCH
    feat, th = _data(rows, mode)
    y = xp.full(n, np.nan, dtype=mode)
    bound = _launch(bundle, "kg_fwd", target, {"lane": xp.arange(n, dtype=np.int32),
                                               "feat": xp.asarray(feat), "th": xp.asarray(th), "y": y})
    bound.launch()
    bound.wait()                    # a launch over device memory is queued: ``y`` is read after the wait
    return np.asarray(y.get() if hasattr(y, "get") else y)


@pytest.mark.parametrize("rows", ROWS)
def test_host_values_equal_the_interpreter(tmp_path, cache_dir, rows):
    bundle = build_bundle([kg_fwd], tmp_path / "h", mode="float64", targets=("host",),
                          host_profile="native", cache_dir=cache_dir)
    n = rows * BATCH
    feat, th = _data(rows)
    y = np.full(n, np.nan)
    _launch(bundle, "kg_fwd", "host", {"lane": np.arange(n, dtype=np.int32), "feat": feat,
                                       "th": th, "y": y}).launch()
    assert not np.isnan(y).any()
    for lane in range(0, n, 97):
        want = float(evaluate(kg_fwd.sinks, {"lane": lane, "feat": feat, "th": th}, lanes=n, lane=lane)["y"])
        assert y[lane] == pytest.approx(want, rel=1e-12, abs=1e-12), lane


@pytest.mark.gpu
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("mode", ["float64", "float32"])
def test_device_values_are_the_box_free_values_and_near_the_host(tmp_path, cache_dir, mode, rows):
    """The device build of ``kg_fwd`` (boxed by the card default) against the host build of it.

    The old row also compared the boxed device build with a box-free one made under ``HAWK_BOX=0``; that switch is
    deleted (DECISIONS.md U-133 (3): "the HAWK_BOX environment switch is deleted"), so that arm has no way to be built."""
    host = build_bundle([kg_fwd], tmp_path / "h", mode=mode, targets=("host",), host_profile="native", cache_dir=cache_dir)
    boxed = build_bundle([kg_fwd], tmp_path / "d", mode=mode, targets=("cuda", "host"), host_profile="native", cache_dir=cache_dir)
    got = _run(boxed, "cuda", rows, mode)
    want = _run(host, "host", rows, mode)
    # the device may fuse a multiply into an add the host keeps apart
    np.testing.assert_allclose(got, want, rtol=64 * np.finfo(mode).eps,
                               atol=64 * np.finfo(mode).eps * np.abs(want).max())
