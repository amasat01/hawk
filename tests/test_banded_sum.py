# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.sum`` over the samples of a warp row, built in the ``banded`` scalar mode: each lane folds its term into
aether's compensated ``BandAccum`` and the warp's tree merges whole accumulators (eagle's ``shflXor``). The row sum is held against an exact rational reference, on a series that cancels by
about 2^30 as well as on ordinary data."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

import hawk

N = 1000
ROWS = -(-N // 32)
#: The compensated fold's bound on a series cancelling by 2^30: within 0.01 ULP53 of the exact row sum.
CANCEL_ULP53 = 0.01


@hawk.kernel
def total(x: hawk.Scalar, s: hawk.Mutable[hawk.Scalar], s_c: hawk.Mutable[hawk.Scalar]):
    s = hawk.sum(x)
    s_c = hawk.sum(x * x)


def _ordinary():
    return np.random.default_rng(5).uniform(-2.0, 2.0, N).astype(np.float32).astype(np.float64)


def _cancelling():
    """Rows of ``2^30, 1, -2^30, 1, ...`` (and a trailing partial row): every term float32-exact, every row's
    exact sum small against its terms."""
    x = np.tile([2.0 ** 30, 1.0, -(2.0 ** 30), 1.0], N // 4 + 1)[:N]
    return x.astype(np.float32).astype(np.float64)


def _exact_rows(x):
    return [sum(Fraction(float(v)) for v in x[r * 32:(r + 1) * 32]) for r in range(ROWS)]


def _run(x, target, tmp_path):
    import _deploy as L
    import eagle
    import eagle.exec as eexec
    from conftest import sidecar_of
    from eagle import arrays
    from eagle import plan as eplan
    from eagle._alloc import synchronize

    from hawk.artifact import build_bundle

    bundle = build_bundle([total], tmp_path / target, mode="banded", targets=(target,), host_profile="native")
    sidecar = sidecar_of(bundle, "total")
    plugin, structure = L.device_plugin(bundle.directory, "total", sidecar), eexec.DeviceKernel
    planes = {"x": x, "s": np.zeros(ROWS), "s_comp": np.zeros(ROWS), "s_c": np.zeros(ROWS), "s_c_comp": np.zeros(ROWS)}
    planes = {k: eagle.banded_pack(arrays.asarray(v)) for k, v in planes.items()}
    if target == "cuda":
        planes = {k: arrays.asarray(v, device="cuda:0") for k, v in planes.items()}
    eplan.plan(plugin, structure=structure).bind(**planes).launch()
    if target == "cuda":
        synchronize()
    out = planes["s"].get() if target == "cuda" else planes["s"]
    return np.asarray(eagle.banded_unpack(out)).copy(), np.asarray(out).view(np.uint64).copy()


def _check(x, target, tmp_path, bound):
    pytest.importorskip("eagle")
    got, cells = _run(x, target, tmp_path)
    for g, want in zip(got, _exact_rows(x)):
        ulp = abs(Fraction(float(g)) - want) / Fraction(float(np.spacing(abs(float(want)))))
        assert ulp <= bound, (float(g), float(want), float(ulp))
    return cells


@pytest.mark.gpu
@pytest.mark.parametrize("data,bound", ((_ordinary, 0.6), (_cancelling, CANCEL_ULP53)), ids=("ordinary", "cancelling"))
def test_a_banded_row_sum_on_the_gpu_is_within_its_bound_of_the_exact_sum(data, bound, tmp_path):
    _check(data(), "cuda", tmp_path, bound)


@pytest.mark.parametrize("target_id", ["cuda"])
def test_a_banded_row_sum_folds_into_bandaccum_and_a_float_one_does_not(target_id):
    from hawk.emit import BANDED, FLOAT64, render_source
    from hawk.targets import TARGETS

    banded = render_source("total", total.sinks, total.walk, TARGETS[target_id], mode=BANDED).text
    assert "aether::banded::BandAccum hawk_" in banded and "hawk_ss[0].addTerm(aether::banded::Band(" in banded
    assert "hawk_abi::ssOut_(hawk_s)" in banded and "BandAccum" not in render_source(
        "total", total.sinks, total.walk, TARGETS[target_id], mode=FLOAT64).text
