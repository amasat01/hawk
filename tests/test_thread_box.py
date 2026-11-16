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
from test_ir_lanes import BATCH, L_ALL, N_IN, S_ROW, kg_fwd

from hawk.artifact import build_bundle

ROWS = (32, 21, 13)


def _data(rows):
    rng = np.random.default_rng(rows)
    feat = rng.uniform(-1, 1, 8 * L_ALL)
    feat[:L_ALL] = rng.integers(0, 5, L_ALL)
    th = rng.standard_normal(rows * N_IN * S_ROW)
    return feat, th


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
