# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A product past the fold threshold is spelled as its ``aether::rt`` twin.

``A @ B`` and ``transpose(A) @ A`` build and match NumPy at sizes the static
text cannot compile (``excessive recursion`` in aether's product unroll), on the
host and as device source; a product at or under the threshold keeps the static
text; the linear-algebra calls hand the library plane proxies and fill their
factors in place, with no ``n x n`` identity literal for an inverse.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import Matrix, Mutable, Vector
from hawk import runtime as rt
from hawk.artifact import build_bundle

LANES = 8


def mm_kernel(n):
    @hawk.kernel
    def k_mm(A: Matrix[n, n], B: Matrix[n, n], C: Mutable[Matrix[n, n]]):
        C = A @ B
    return k_mm


def ata_kernel(n):
    @hawk.kernel
    def k_ata(A: Matrix[n, n], C: Mutable[Matrix[n, n]]):
        C = m.transpose(A) @ A
    return k_ata


def mv_kernel(r, k):
    @hawk.kernel
    def k_mv(A: Matrix[r, k], v: Vector[k], y: Mutable[Vector[r]]):
        y = A @ v
    return k_mv


def lstsq_kernel(n):
    k = n + 2

    @hawk.kernel
    def k_lstsq(J: Matrix[k, n], r: Vector[k], x: Mutable[Vector[n]], i_x: Mutable[hawk.Int32]):
        x, i_x = m.lstsq(J, r)
    return k_lstsq


def inv_kernel(n):
    @hawk.kernel
    def k_inv(A: Matrix[n, n], Ainv: Mutable[Matrix[n, n]], i_x: Mutable[hawk.Int32]):
        Ainv, i_x = m.inv(A)
    return k_inv


def _plane(a):
    p = np.ascontiguousarray(a.reshape(LANES, -1).T)
    return p[0] if p.shape[0] == 1 else p


@pytest.fixture(scope="module")
def build(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("rolled")

    def make_(kernel, targets=("host",)):
        bundle = build_bundle([kernel], pathlib.Path(root) / kernel.name, targets=targets,
                              cache_dir=cache_dir, host_profile="native", mode="float64")
        return bundle, (rt.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name))
                        if "host" in targets else None)
    return make_


@pytest.mark.parametrize("n", [16, 24, 32])
def test_products_past_the_static_compile_wall_match_numpy(build, n):
    rng = np.random.default_rng(n)
    a, b = rng.standard_normal((2, LANES, n, n))
    for kernel, want, planes in ((mm_kernel(n), a @ b, dict(A=_plane(a), B=_plane(b))),
                                 (ata_kernel(n), np.swapaxes(a, 1, 2) @ a, dict(A=_plane(a)))):
        _, loaded = build(kernel)
        c = np.zeros((n * n, LANES))
        rt.run(loaded, **planes, C=c)
        np.testing.assert_allclose(c.T.reshape(LANES, n, n), want, rtol=1e-12, atol=1e-12 * n)


@pytest.mark.parametrize("n", [16, 24, 32])
def test_lstsq_past_the_static_compile_wall_matches_numpy(build, n):
    rng = np.random.default_rng(100 + n)
    j, r = rng.standard_normal((LANES, n + 2, n)), rng.standard_normal((LANES, n + 2))
    _, loaded = build(lstsq_kernel(n))
    x, info = np.zeros((n, LANES)), np.zeros(LANES, np.int32)
    rt.run(loaded, J=_plane(j), r=_plane(r), x=x, i_x=info)
    want = np.stack([np.linalg.solve(a.T @ a, a.T @ v) for a, v in zip(j, r)])
    np.testing.assert_allclose(x.T, want, rtol=1e-8, atol=1e-8)
    assert not info.any()


@pytest.mark.parametrize("make_", [lambda: lstsq_kernel(32), lambda: mm_kernel(32), lambda: ata_kernel(32)])
def test_the_device_source_of_a_wide_product_compiles(build, make_):
    kernel = make_()
    bundle, _ = build(kernel, targets=("cuda",))
    entry = bundle.artifacts[0].entries["cuda"]
    assert entry.artifact.stat().st_size > 0 and "aether::rt::matMat(" in entry.source.read_text()


