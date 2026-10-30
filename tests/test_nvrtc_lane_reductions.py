# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A lane-set sum, max and min compile under a FORCED NVRTC and run bit-identically to nvcc.

The reductions' padding identity (``ssIdentity_``) once spelled infinity ``__builtin_inf()``,
which NVRTC rejects, so the shipped (nvcc-less) tier could not build any ``hawk.max``/``min`` over
lanes while every nvcc-on-PATH row stayed green. These rows force ``HAWK_DEVICE_COMPILER=nvrtc``.
The NVRTC build also records ptxas's per-entry figures (``--ptxas-options=-v``) in the sidecar.
"""

from __future__ import annotations

import numpy as np
import pytest
from _team_stages import run_device
from _toolchain import real_arch
from conftest import sidecar_of

import hawk
from hawk import Mutable, Scalar, kernel
from hawk.artifact import build_bundle

N = 1000
ROWS = -(-N // 32)      # one partial per 32-lane row; the host side folds them


@kernel
def lane_sum(x: Scalar, s: Mutable[Scalar]):
    s = hawk.sum(x, where=x > 0.0)


@kernel
def lane_max(x: Scalar, top: Mutable[Scalar]):
    top = hawk.max(x, where=x > 0.0)


@kernel
def lane_min(x: Scalar, low: Mutable[Scalar]):
    low = hawk.min(x, where=x > 0.0)


CASES = [(lane_sum, "s", lambda p: sum(v.sum() for v in p.values()), lambda x: x[x > 0].sum()),
         (lane_max, "top", lambda p: p["top"].max(), lambda x: x[x > 0].max()),
         (lane_min, "low", lambda p: p["low"].min(), lambda x: x[x > 0].min())]


def _run(kern, out, tmp, compiler, monkeypatch, x):
    import cupy as cp
    monkeypatch.setenv("HAWK_DEVICE_COMPILER", compiler)
    bundle = build_bundle([kern], tmp / compiler, targets=("cuda",), cache_dir=str(tmp / f"c_{compiler}"),
                    device_arch=real_arch())
    outs = {n: cp.full(ROWS, cp.nan) for _, n in kern.arg_spec if n != "x"}
    dev = {"x": cp.asarray(x), **outs}
    run_device(bundle, kern.name, dev, N)
    return {k: dev[k].get() for k in outs}, sidecar_of(bundle, kern.name)


@pytest.mark.gpu
@pytest.mark.parametrize("kern,out,fold,ref", CASES, ids=[c[1] for c in CASES])
def test_nvrtc_builds_the_lane_reduction_and_matches_nvcc(kern, out, fold, ref, tmp_path, monkeypatch):
    x = np.random.default_rng(3).uniform(-2, 2, N)
    got, side = _run(kern, out, tmp_path, "nvrtc", monkeypatch, x)
    want, nvcc_side = _run(kern, out, tmp_path, "nvcc", monkeypatch, x)
    assert all(np.array_equal(got[k], want[k]) for k in got)
    np.testing.assert_allclose(fold(got), ref(x), rtol=1e-12)
    row, ref_row = side["ptxas"][kern.name], nvcc_side["ptxas"][kern.name]
    # NVRTC's figures come off the CUBIN: registers and frame equal nvcc's, spills are unknown (None)
    assert (row["regs"], row["stack_B"]) == (ref_row["regs"], ref_row["stack_B"])
    assert row["spill_store_B"] is None and row["spill_load_B"] is None


def test_nvrtc_ptxas_figures_equal_nvcc_with_a_local_frame(tmp_path, monkeypatch):
    """A kernel with a real stack frame (a thread-form LU): the CUBIN-read figures equal nvcc's."""
    from _linalg_kernels import lu_kernel
    k = lu_kernel(8)
    rows = {}
    for comp in ("nvrtc", "nvcc"):
        monkeypatch.setenv("HAWK_DEVICE_COMPILER", comp)
        b = build_bundle([k], tmp_path / comp, targets=("cuda",), cache_dir=str(tmp_path / f"c_{comp}"),
                         device_arch=real_arch())
        side = sidecar_of(b, k.name)
        # an nvcc not newer than the driver compiles PTX, which carries no ptxas figures: this row needs a CUBIN from both
        assert "ptxas" in side, f"{comp} compiled no CUBIN here (a PTX build has no ptxas figures): point $HAWK_NVCC at an nvcc newer than the driver"
        rows[comp] = side["ptxas"][k.name]
    assert rows["nvrtc"]["stack_B"] == rows["nvcc"]["stack_B"] > 0
    assert rows["nvrtc"]["regs"] == rows["nvcc"]["regs"]
