# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A per-lane ``Matrix[R, C]`` of any extent binds and runs through the runtime's
own descriptor, which is read from the sidecar's declared shapes: inputs
(``mat_shapes``, ``vec_widths``) and outputs (``mutables``)."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from conftest import sidecar_of

import hawk
from hawk import Matrix, Mutable, Vector
from hawk import runtime as rt
from hawk.artifact import build_bundle

N = 6


@hawk.kernel
def products(A: Matrix[4, 5], B: Matrix[5, 3], v: Vector[5],
             C: Mutable[Matrix[4, 3]], y: Mutable[Vector[4]]):
    C = A @ B
    y = A @ v


@hawk.kernel
def square(S: Matrix[7, 7], T: Mutable[Matrix[7, 7]]):
    T = S @ S


@pytest.fixture
def host(tmp_path, cache_dir):
    def load(kernel):
        bundle = build_bundle([kernel], pathlib.Path(tmp_path) / kernel.name, targets=("host",),
                              cache_dir=cache_dir, host_profile="native")
        return rt.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name)), bundle
    return load


def test_the_sidecar_declares_the_extents_the_descriptor_reads(host):
    kernel, bundle = host(products)
    sidecar = sidecar_of(bundle, "products")
    assert sidecar["mat_shapes"] == {"A": [4, 5], "B": [5, 3]}
    assert sidecar["vec_widths"] == {"v": 5}
    shapes = {m["name"]: m.get("shape") for m in sidecar["mutables"]}
    assert shapes == {"C": [4, 3], "y": None}
    assert rt.descriptor_for_sidecar(sidecar) == ["gref"] * 5
    assert kernel.descriptor == ["gref"] * 5


def test_rectangular_inputs_and_outputs_run(host):
    kernel, _ = host(products)
    rng = np.random.default_rng(3)
    A, B, v = (rng.standard_normal(s) for s in ((4, 5, N), (5, 3, N), (5, N)))
    C, y = np.zeros((12, N)), np.zeros((4, N))
    rt.run(kernel, A=A.reshape(20, N), B=B.reshape(15, N), v=v, C=C, y=y)
    np.testing.assert_allclose(C.reshape(4, 3, N), np.einsum("rkn,kcn->rcn", A, B), atol=1e-13)
    np.testing.assert_allclose(y, np.einsum("rkn,kn->rn", A, v), atol=1e-13)


def test_a_large_square_matrix_runs(host):
    kernel, _ = host(square)
    S = np.random.default_rng(4).standard_normal((7, 7, N))
    T = np.zeros((49, N))
    rt.run(kernel, S=S.reshape(49, N), T=T)
    np.testing.assert_allclose(T.reshape(7, 7, N), np.einsum("rkn,kcn->rcn", S, S), atol=1e-12)
