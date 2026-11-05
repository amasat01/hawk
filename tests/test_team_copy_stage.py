# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A team-form factorisation's operands reach its scratch through a copy stage of their own: a block per lane whose
threads copy the lane's slice together, not one thread per lane walking it. The copy is exact; a stage that holds
anything besides operand copies keeps the per-lane launch."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import Int32, Matrix, Mutable, Vector
from hawk.artifact import build_bundle
from hawk.ir.team import COPY_THREADS

N, LANES = 40, 5


@hawk.kernel
def solve(A: Matrix[N, N], b: Vector[N], x: Mutable[Vector[N]], info: Mutable[Int32]):
    x, info = m.solve(A, b)


def _case():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(LANES, N, N))
    a = a @ a.transpose(0, 2, 1) + N * np.eye(N)
    return a, rng.normal(size=(LANES, N))


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    return build_bundle([solve], tmp_path_factory.mktemp("copy"), targets=("cuda", "host"), cache_dir=cache_dir,
                        host_profile="native")


def test_the_first_stage_is_a_block_per_lane_copy(bundle):
    first = sidecar_of(bundle, "solve")["stages"][0]
    assert first["team"]["threads"] == COPY_THREADS and "grid" not in first["team"]
    assert "team" in sidecar_of(bundle, "solve")["stages"][1]


def test_host_solves_through_the_copy_stage(bundle):
    from hawk import runtime as rt

    a, b = _case()
    x, info = np.zeros((N, LANES)), np.zeros(LANES, np.int32)
    rt.run(rt.load(bundle.directory, "solve", sidecar_of(bundle, "solve")),
           A=a.reshape(LANES, N * N).T.copy(), b=b.T.copy(), x=x, info=info)
    assert not info.any()
    np.testing.assert_allclose(x.T, np.linalg.solve(a, b[:, :, None])[:, :, 0], rtol=1e-9, atol=1e-9)


@pytest.mark.gpu
def test_device_copy_is_exact(bundle):
    import cupy as cp
    from _team_stages import run_device

    a, b = _case()
    planes = {"A": cp.asarray(a.reshape(LANES, N * N).T.copy()), "b": cp.asarray(b.T.copy()),
              "x": cp.zeros((N, LANES)), "info": cp.zeros(LANES, cp.int32)}
    scratch = run_device(bundle, "solve", planes, LANES, only=1)
    name = next(k for k in scratch if k.endswith("_a"))
    assert np.array_equal(cp.asnumpy(scratch[name]).reshape(LANES, N * N), a.reshape(LANES, N * N))
    assert np.array_equal(cp.asnumpy(scratch[next(k for k in scratch if k.endswith("_x0"))]).reshape(LANES, N), b)
