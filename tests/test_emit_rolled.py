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
import re

import numpy as np
import pytest
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import _broadcast as broadcast
from hawk import Matrix, Mutable, Vector
from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.artifact.layout import exports
from hawk.emit import policy, render_source
from hawk.emit.wrapper import scalar_mode
from hawk.targets import TARGETS

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


def _text(kernel, target="cuda"):
    return render_source(kernel.name, kernel.sinks, kernel.walk, TARGETS[target],
                         mode=scalar_mode("float64"), exports=exports(target)).text


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


def test_a_product_at_the_threshold_keeps_the_static_text():
    text = _text(mm_kernel(4))
    assert "aether::rt::" not in text and "(mtx_A_i * mtx_B_i)" in text
    assert "aether::rt::matMat(" in _text(mm_kernel(5))         # 25 results: past the threshold
    assert "aether::rt::matVecItems(" in _text(mv_kernel(3, 17))    # a fold of 17
    assert "aether::rt::" not in _text(mv_kernel(16, 16))


def test_a_transposed_operand_is_read_through_not_built():
    text = _text(ata_kernel(24))
    assert "aether::rt::transposed(mtx_A_i)" in text and ".transpose()" not in text
    assert "const auto mtx_A_i = mtx_A[i];" in text             # a proxy, not a copy of the plane


def test_the_threshold_is_one_constant(monkeypatch):
    monkeypatch.setattr(policy, "ROLLED_FOLD_MIN", 2)
    assert "aether::rt::matVecItems(" in _text(mv_kernel(3, 3))


def test_a_factorisation_reads_the_plane_through_its_proxy_and_an_inverse_builds_no_identity():
    text = _text(inv_kernel(32))
    assert re.search(r"aether::linalg::lu<32>\(hawk_lf\d+, mtx_A_i\);", text)
    assert "aether::linalg::luInverse(hawk_lf" in text
    assert "static_cast<Real>(1.0), static_cast<Real>(0.0)" not in text
    assert ".get()" not in text.split("hawk_flat >= count")[1]


def test_a_broadcast_matrix_is_read_through_its_proxy_not_copied():
    @hawk.kernel
    def k_bmv(lane: hawk.Grid["a", "b":4, "c":6], M: Matrix[20, 20], v: Vector[20]) -> hawk.Scalar:
        return m.sum(M @ v)

    run, _, _ = broadcast.route(k_bmv, (), dict(M=np.zeros((20, 20, 3, 1, 6)), v=np.zeros((20, 3, 4, 6)), a=3))
    body = _text(run).split("hawk_flat >= count")[1]
    assert re.search(r"aether::rt::matVecItems\(hawk_la\d+, t\d+, vin_v_i\);", body)
    assert re.search(r"const auto t\d+ = mtx_M\[aether::SampleIndex::make\(", body)
    assert ".get()" not in body and "Item<Real, 20, 20>" not in body and "mtx_M_i" not in body
