# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.max`` and ``hawk.min`` over the samples of a warp row, built in the ``banded`` scalar mode: the fold is exact
(the certified ``fmax``/``fmin`` carry no accumulator state), so a lane holds a bare ``Band`` and the warp tree is
eagle's ``shflXor`` plus the same exact op. Each row is held BITWISE against NumPy's max/min of the decoded terms."""

from __future__ import annotations

import numpy as np
import pytest

import hawk

N = 1000
ROWS = -(-N // 32)


@hawk.kernel
def peak(x: hawk.Scalar, hi: hawk.Mutable[hawk.Scalar]):
    hi = hawk.max(x)


@hawk.kernel
def trough(x: hawk.Scalar, lo: hawk.Mutable[hawk.Scalar]):
    lo = hawk.min(x)


def _data():
    """Signed terms over several binades, ties and both signs of zero included; float32-exact."""
    rng = np.random.default_rng(11)
    x = rng.uniform(-2.0, 2.0, N) * np.exp2(rng.integers(-20, 20, N))
    x[::97] = 0.0
    x[5::97] = -0.0
    return x.astype(np.float32).astype(np.float64)


def _run(kernel, name, x, target, tmp_path):
    import _deploy as L
    import eagle
    import eagle.exec as eexec
    from conftest import sidecar_of
    from eagle import arrays
    from eagle import plan as eplan
    from eagle._alloc import synchronize

    from hawk.artifact import build_bundle

    bundle = build_bundle([kernel], tmp_path / target, mode="banded", targets=(target,), host_profile="native")
    sidecar = sidecar_of(bundle, kernel.name)
    plugin, structure = L.device_plugin(bundle.directory, kernel.name, sidecar), eexec.DeviceKernel
    planes = {"x": x, name: np.zeros(ROWS)}
    planes = {k: eagle.banded_pack(arrays.asarray(v)) for k, v in planes.items()}
    if target == "cuda":
        planes = {k: arrays.asarray(v, device="cuda:0") for k, v in planes.items()}
    eplan.plan(plugin, structure=structure).bind(**planes).launch()
    if target == "cuda":
        synchronize()
    out = planes[name].get() if target == "cuda" else planes[name]
    return np.asarray(eagle.banded_unpack(out)).copy(), np.asarray(out).view(np.uint64).copy()


def _decoded(x):
    import eagle
    from eagle import arrays

    return np.asarray(eagle.banded_unpack(eagle.banded_pack(arrays.asarray(x)))).copy()


CASES = ((peak, "hi", np.max), (trough, "lo", np.min))
IDS = ("max", "min")


@pytest.mark.gpu
@pytest.mark.parametrize("kernel,name,ref", CASES, ids=IDS)
def test_a_banded_row_extreme_on_the_gpu_is_bitwise_numpys(kernel, name, ref, tmp_path):
    pytest.importorskip("eagle")
    x = _data()
    got, _ = _run(kernel, name, x, "cuda", tmp_path)
    d = _decoded(x)
    want = np.array([ref(d[r * 32:(r + 1) * 32]) for r in range(ROWS)])
    assert np.array_equal(got.view(np.uint64), want.view(np.uint64))
