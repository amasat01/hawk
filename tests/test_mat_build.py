# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.math.mat``: a matrix built from traced entries, rows or columns.

It lowers onto the IR's rank-2 ``vec``. Rows: the front end and its refusals;
a per-lane Newton step on a 2-bus system and on a 22 x 22 dense system, the
Jacobian assembled in the kernel and solved with ``m.solve``, against NumPy;
the C unit's bits against the host's, the strict-C and cppcheck checks, and the
device source compiling. Derivatives are rows of ``test_autodiff_matrix``.
"""

from __future__ import annotations

import pathlib

import _embedded as E
import numpy as np
import pytest
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import Matrix, Mutable, Vector
from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.embedded import export
from hawk.ir import HawkError

LANES = 6


def _jac(c, t, n):
    return m.mat([[(3.0 * t[i] * t[i] + c[i, j]) if i == j else c[i, j] for j in range(n)]
                  for i in range(n)])


def assembled_solve(n):
    @hawk.kernel
    def k_asm(C: Matrix[n, n], t: Vector[n], b: Vector[n], x: Mutable[Vector[n]], i_x: Mutable[hawk.Int32]):
        x, i_x = m.solve(_jac(C, t, n), b)
    return k_asm


def newton(n, steps):
    """``F(x) = x^3 + C x - b`` by Newton, ``J = diag(3 x^2) + C`` assembled per step; the
    steps are a Python loop (traced out), so the kernel holds ``steps`` factorisations."""
    @hawk.kernel
    def k_newton(C: Matrix[n, n], b: Vector[n], x0: Vector[n], x: Mutable[Vector[n]], i_x: Mutable[hawk.Int32]):
        xs = x0
        for _ in range(steps):
            f = xs * xs * xs + C @ xs - b
            dx, i_x = m.solve(_jac(C, xs, n), f)
            xs = xs - dx
        x = xs
    return k_newton


def two_bus():
    @hawk.kernel
    def k_two(x0: Vector[2], b: Vector[2], x: Mutable[Vector[2]]):
        xs = x0
        for _ in range(4):
            f = m.vec(xs[0] * xs[0] + xs[1] - b[0], xs[0] * xs[1] - b[1])
            dx, _info = m.solve(m.mat([[2.0 * xs[0], 1.0], [xs[1], xs[0]]]), f)
            xs = xs - dx
        x = xs
    return k_two


def _plane(a):
    p = np.ascontiguousarray(a.reshape(LANES, -1).T)
    return p[0] if p.shape[0] == 1 else p


@pytest.fixture(scope="module")
def build(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("mat")

    def make(kernel, targets=("host",), mode="float64"):
        bundle = build_bundle([kernel], pathlib.Path(root) / f"{kernel.name}_{mode}", targets=targets,
                              cache_dir=cache_dir, host_profile="native", mode=mode)
        return bundle, (rt.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name))
                        if "host" in targets else None)
    return make


# --- the front end ---------------------------------------------------------------

def _trace(fn):
    @hawk.kernel
    def k(a: Vector[3], b: Vector[3], s: hawk.Scalar, y: Mutable[Matrix[2, 3]]):
        y = fn(a, b, s)
    return k


def test_rows_of_scalars_of_vectors_and_columns_agree():
    rows = _trace(lambda a, b, s: m.mat([[a[0], a[1], a[2]], [b[0], b[1], s]]))
    vecs = _trace(lambda a, b, s: m.mat([a, m.vec(b[0], b[1], s)]))
    cols = _trace(lambda a, b, s: m.mat([m.vec(a[0], b[0]), m.vec(a[1], b[1]), m.vec(a[2], s)], columns=True))
    assert {rows.sinks[0].value.ttype.shape, vecs.sinks[0].value.ttype.shape,
            cols.sinks[0].value.ttype.shape} == {(2, 3)}


@pytest.mark.parametrize("bad,why", [
    (lambda a, b, s: m.mat([[a[0], a[1]], [b[0]]]), "same"),
    (lambda a, b, s: m.mat([]), "non-empty"),
    (lambda a, b, s: m.mat([[a, a]]), "not a scalar"),
    (lambda a, b, s: m.mat([s, s]), "list of scalars or a vector"),
    (lambda a, b, s: m.mat(a), "non-empty list"),
])
def test_a_malformed_matrix_is_refused_by_name(bad, why):
    with pytest.raises(HawkError, match=why):
        _trace(bad)


def test_vec_concatenates_vectors_and_a_matrix_entry_reads_by_index():
    @hawk.kernel
    def k(a: Vector[2], B: Matrix[2, 2], y: Mutable[Vector[4]], z: Mutable[hawk.Scalar]):
        y = m.vec(a, B[0, 1], B[1, 0])
        z = B[1, 1]
    assert k.sinks[0].value.ttype.shape == (4,)


# --- values ----------------------------------------------------------------------

def test_a_two_bus_newton_matches_numpy(build):
    rng = np.random.default_rng(2)
    x0 = rng.uniform(0.8, 1.4, (LANES, 2)); b = rng.uniform(1.5, 2.5, (LANES, 2))
    _, loaded = build(two_bus())
    x = np.zeros((2, LANES))
    rt.run(loaded, x0=_plane(x0), b=_plane(b), x=x)
    want = x0.copy()
    for _ in range(4):
        for k in range(LANES):
            a = want[k]
            f = np.array([a[0] ** 2 + a[1] - b[k][0], a[0] * a[1] - b[k][1]])
            want[k] = a - np.linalg.solve(np.array([[2 * a[0], 1.0], [a[1], a[0]]]), f)
    np.testing.assert_allclose(x.T, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n", [3, 22])
def test_an_assembled_jacobian_solves_like_numpy(build, n):
    rng = np.random.default_rng(n)
    c, t, b = rng.standard_normal((LANES, n, n)) + 0.0, rng.uniform(0.5, 1.5, (LANES, n)), rng.standard_normal((LANES, n))
    c = c + 3.0 * np.sqrt(n) * np.eye(n)
    _, loaded = build(assembled_solve(n))
    x, info = np.zeros((n, LANES)), np.zeros(LANES, np.int32)
    rt.run(loaded, C=_plane(c), t=_plane(t), b=_plane(b), x=x, i_x=info)
    want = np.stack([np.linalg.solve(c[k] + np.diag(3 * t[k] ** 2), b[k]) for k in range(LANES)])
    np.testing.assert_allclose(x.T, want, rtol=1e-9, atol=1e-9)
    assert not info.any()


def test_a_22_by_22_newton_matches_numpy(build):
    n, steps = 22, 3
    rng = np.random.default_rng(22)
    c = 0.05 * rng.standard_normal((LANES, n, n)) + np.eye(n) * 2.0
    b, x0 = rng.uniform(0.5, 1.5, (LANES, n)), rng.uniform(0.3, 0.9, (LANES, n))
    _, loaded = build(newton(n, steps))
    x, info = np.zeros((n, LANES)), np.zeros(LANES, np.int32)
    rt.run(loaded, C=_plane(c), b=_plane(b), x0=_plane(x0), x=x, i_x=info)
    want = x0.copy()
    for _ in range(steps):
        for k in range(LANES):
            f = want[k] ** 3 + c[k] @ want[k] - b[k]
            want[k] = want[k] - np.linalg.solve(np.diag(3 * want[k] ** 2) + c[k], f)
    np.testing.assert_allclose(x.T, want, rtol=1e-9, atol=1e-9)
    assert not info.any()


# --- the C unit and the device source ----------------------------------------------

@pytest.mark.parametrize("n", [3, 22])
def test_the_c_unit_computes_the_hosts_bits_and_passes_the_checks(tmp_path, build, n):
    kernel = assembled_solve(n)
    rng = np.random.default_rng(n)
    planes = dict(C=rng.standard_normal((LANES, n, n)) + 3.0 * np.sqrt(n) * np.eye(n),
                  t=rng.uniform(0.5, 1.5, (LANES, n)), b=rng.standard_normal((LANES, n)))
    _, loaded = build(kernel)
    h = {"x": np.zeros((n, LANES)), "i_x": np.zeros(LANES, np.int32)}
    rt.run(loaded, **{k: _plane(v) for k, v in planes.items()}, **h)
    art = export(kernel, tmp_path, mode="float64")
    cppcheck = art.report["checks"]["cppcheck"]
    assert "skipped" not in cppcheck and cppcheck["findings"] == [] and art.report["audit"] == {"hits": []}
    so = E.cc_shared(art.c_path, art.directory)
    written = [r["name"] for r in art.report["interface"] if r["direction"] != "in" and r["role"] != "nsamples"]
    got = E.run_c(so, kernel.name, art.report, LANES, mode="float64", **{k: _plane(v) for k, v in planes.items()})
    got = got if isinstance(got, tuple) else (got,)
    for name, plane in zip(written, got):
        assert np.asarray(plane).reshape(h[name].shape).tobytes() == h[name].tobytes(), name


def test_the_device_source_compiles_and_assembles_entry_by_entry(build):
    bundle, _ = build(assembled_solve(22), targets=("cuda",))
    text = bundle.artifacts[0].entries["cuda"].source.read_text()
    assert bundle.artifacts[0].entries["cuda"].artifact.stat().st_size > 0
    assert "hawk_la0(21, 21) =" in text and "static_cast<Real>(0.0), static_cast<Real>(0.0)" not in text
