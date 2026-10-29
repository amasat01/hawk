# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The linear-algebra kinds lowered through ``aether::linalg`` and run on the host.

Every kind is read back against NumPy (and LAPACK's ``info`` for the failing
lanes), at sizes 1, 5, 22 and 32 in both scalar modes. A lane that fails to
factor must come back as zeros with ``info`` set; the neighbouring lanes must
not feel it. The tolerances are ``K n eps cond`` for solves and determinants,
``K n eps ||A||`` for factors and eigenvalues, with ``K = 100``.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pytest
from _linalg_kernels import K as RHS_COLUMNS
from _linalg_kernels import chol_from_kernel, chol_kernel, eigh_kernel, lu_kernel
from conftest import sidecar_of
from scipy.linalg import lapack

from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.ir import Assign, Dispatch, Leaf, make
from hawk.trace.kernel import Kernel
from hawk.types import TensorType
from hawk.emit import render_source
from hawk.emit.wrapper import scalar_mode
from hawk.artifact.layout import exports
from hawk.targets import TARGETS

SIZES = (1, 5, 22, 32)
MODES = ("float64", "float32")
LANES = 8
#: The lane every kernel is given a failing input in.
BAD = 3
FACTOR = 100.0

_DTYPE = {"float64": np.float64, "float32": np.float32}
_EPS = {"float64": np.finfo(np.float64).eps, "float32": np.finfo(np.float32).eps}


@pytest.fixture(scope="module")
def build(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("la_artifacts")

    def make(kernel, mode):
        bundle = build_bundle([kernel], pathlib.Path(root) / f"{kernel.name}_{mode}",
                              targets=("host",), cache_dir=cache_dir, host_profile="native",
                              mode=mode)
        return rt.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name))
    return make


def _rng(n, mode):
    return np.random.default_rng(1000 * n + (mode == "float32"))


def _plane(a):
    """``(LANES, width)`` as the plane the kernel binds: ``(width, LANES)``, flat for width 1."""
    plane = np.ascontiguousarray(a.T)
    return plane[0] if plane.shape[0] == 1 else plane


def _planes(mats, vecs, rhs, mode):
    """Per-lane arrays ``(LANES, ...)`` as the sample-minor planes the kernels read."""
    dt = _DTYPE[mode]
    flat = lambda a: _plane(a.reshape(LANES, -1).astype(dt))      # noqa: E731
    return {"A": flat(mats), "b": flat(vecs), "B": flat(rhs)}, mats.shape[-1]


def _run(kernel, planes, outs, mode):
    dt = _DTYPE[mode]
    sink = {name: np.zeros(LANES if width == 1 else (width, LANES),
                           dtype=np.int32 if name == "info" else dt)
            for name, width in outs.items()}
    declared = {name for _, name in kernel.arg_spec}
    rt.run(kernel, **{k: v for k, v in planes.items() if k in declared}, **sink)
    return {name: plane.reshape(outs[name], LANES).T for name, plane in sink.items()}


def _inputs(n, mode, matrices):
    rng = _rng(n, mode)
    mats = np.stack([matrices(rng, n, lane) for lane in range(LANES)])
    dt = _DTYPE[mode]
    return (mats.astype(dt).astype(np.float64),
            rng.standard_normal((LANES, n)).astype(dt).astype(np.float64),
            rng.standard_normal((LANES, n, RHS_COLUMNS)).astype(dt).astype(np.float64))


def _tol(mode, n, scale):
    return FACTOR * n * _EPS[mode] * scale


def _general(rng, n, lane):
    a = rng.standard_normal((n, n))
    if lane == BAD:
        a[min(1, n - 1)] = 0.0                    # an exactly zero row: singular
    return a


def _spd(rng, n, lane):
    g = rng.standard_normal((n, n))
    a = g @ g.T + n * np.eye(n)
    if lane == BAD:
        a[n // 2, n // 2] = -1.0
    if lane == BAD + 1 and n > 1:
        a[np.tril_indices(n, -1)] += 0.3 * rng.standard_normal(n * (n - 1) // 2)   # not symmetric
    return a


def _built_factor(rng, n, lane):
    """A lower triangular factor with garbage above the diagonal; lanes ``BAD`` to ``BAD + 3``
    carry a negative, a zero, an infinite and a NaN diagonal entry."""
    a = np.tril(rng.standard_normal((n, n)), -1) + np.diag(1.0 + rng.random(n)) \
        + 0.5 * np.triu(rng.standard_normal((n, n)), 1)
    if lane == BAD:
        a[n // 2, n // 2] = -1.0
    elif lane == BAD + 1:
        a[0, 0] = 0.0
    elif lane == BAD + 2:
        a[n - 1, n - 1] = np.inf
    elif lane == BAD + 3:
        a[n - 1, n - 1] = np.nan
    return a


def _symmetric(rng, n, lane):
    g = rng.standard_normal((n, n))
    return g if lane == BAD else (g + g.T) / 2    # lane BAD is not symmetric


# --- LU ---------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n", SIZES)
def test_lu_solve_det_and_info_match_numpy_and_lapack(build, n, mode):
    mats, vecs, rhs = _inputs(n, mode, _general)
    planes, _ = _planes(mats, vecs, rhs, mode)
    out = _run(build(lu_kernel(n), mode), planes,
               {"x": n, "X": n * RHS_COLUMNS, "det": 1, "info": 1}, mode)
    for lane in range(LANES):
        a = mats[lane]
        if lane == BAD:
            _, _, info = lapack.dgetrf(a)
            assert out["info"][lane, 0] == info > 0
            for name in ("x", "X", "det"):
                assert np.all(out[name][lane] == 0) and not np.any(np.signbit(out[name][lane]))
            continue
        assert out["info"][lane, 0] == 0
        cond = np.linalg.cond(a)
        x = np.linalg.solve(a, vecs[lane])
        X = np.linalg.solve(a, rhs[lane])
        np.testing.assert_allclose(out["x"][lane], x, rtol=0,
                                   atol=_tol(mode, n, cond * max(1.0, np.abs(x).max())))
        np.testing.assert_allclose(out["X"][lane].reshape(n, RHS_COLUMNS), X, rtol=0,
                                   atol=_tol(mode, n, cond * max(1.0, np.abs(X).max())))
        det = np.linalg.det(a)
        assert abs(out["det"][lane, 0] - det) <= _tol(mode, n, cond) * abs(det)


# --- Cholesky ---------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n", SIZES)
def test_cholesky_factor_solve_and_info_match_numpy_and_lapack(build, n, mode):
    mats, vecs, rhs = _inputs(n, mode, _spd)
    planes, _ = _planes(mats, vecs, rhs, mode)
    out = _run(build(chol_kernel(n), mode), planes,
               {"x": n, "X": n * RHS_COLUMNS, "L": n * n, "info": 1}, mode)
    for lane in range(LANES):
        a = mats[lane]
        if lane == BAD:
            _, info = lapack.dpotrf(a, lower=1, clean=1)
            assert out["info"][lane, 0] == info > 0
            for name in ("x", "X", "L"):
                assert np.all(out[name][lane] == 0) and not np.any(np.signbit(out[name][lane]))
            continue
        assert out["info"][lane, 0] == 0
        s = (a + a.T) / 2
        cond = np.linalg.cond(s)
        np.testing.assert_allclose(out["L"][lane].reshape(n, n), np.linalg.cholesky(s), rtol=0,
                                   atol=_tol(mode, n, np.abs(s).max()))
        x = np.linalg.solve(s, vecs[lane])
        X = np.linalg.solve(s, rhs[lane])
        np.testing.assert_allclose(out["x"][lane], x, rtol=0,
                                   atol=_tol(mode, n, cond * max(1.0, np.abs(x).max())))
        np.testing.assert_allclose(out["X"][lane].reshape(n, RHS_COLUMNS), X, rtol=0,
                                   atol=_tol(mode, n, cond * max(1.0, np.abs(X).max())))
        assert np.all(np.triu(out["L"][lane].reshape(n, n), 1) == 0)


# --- the Cholesky factor of a lower triangular matrix ----------------------------

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n", SIZES)
def test_a_built_factor_solves_and_fails_on_its_diagonal(build, n, mode):
    mats, vecs, rhs = _inputs(n, mode, _built_factor)
    planes, _ = _planes(mats, vecs, rhs, mode)
    out = _run(build(chol_from_kernel(n), mode), planes,
               {"x": n, "X": n * RHS_COLUMNS, "L": n * n, "info": 1}, mode)
    first_bad = {BAD: n // 2 + 1, BAD + 1: 1, BAD + 2: n, BAD + 3: n}
    for lane in range(LANES):
        a = mats[lane]
        if lane in first_bad:
            assert out["info"][lane, 0] == first_bad[lane]
            for name in ("x", "X", "L"):
                assert np.all(out[name][lane] == 0) and not np.any(np.signbit(out[name][lane]))
            continue
        low = np.tril(a)
        s = low @ low.T
        cond = np.linalg.cond(s)
        assert out["info"][lane, 0] == 0
        np.testing.assert_array_equal(out["L"][lane].reshape(n, n), low)
        x, big = np.linalg.solve(s, vecs[lane]), np.linalg.solve(s, rhs[lane])
        np.testing.assert_allclose(out["x"][lane], x, rtol=0,
                                   atol=_tol(mode, n, cond * max(1.0, np.abs(x).max())))
        np.testing.assert_allclose(out["X"][lane].reshape(n, RHS_COLUMNS), big, rtol=0,
                                   atol=_tol(mode, n, cond * max(1.0, np.abs(big).max())))


def test_a_built_factor_is_read_through_the_plane_proxy():
    text = _body(chol_from_kernel(22), "cuda").text
    assert re.search(r"aether::linalg::choleskyFrom<22>\(hawk_lf\d+, mtx_A_i\);", text)
    assert "const auto mtx_A_i = mtx_A[i];" in text


# --- eigh -------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n", SIZES)
def test_eigh_values_vectors_and_info_match_numpy(build, n, mode):
    mats, vecs, rhs = _inputs(n, mode, _symmetric)
    planes, _ = _planes(mats, vecs, rhs, mode)
    out = _run(build(eigh_kernel(n), mode), planes, {"w": n, "V": n * n, "info": 1}, mode)
    for lane in range(LANES):
        s = (mats[lane] + mats[lane].T) / 2        # what eigh reads, whatever the input
        w, v = out["w"][lane].astype(np.float64), out["V"][lane].reshape(n, n).astype(np.float64)
        assert out["info"][lane, 0] & 1 == 0
        ref_w, ref_v = np.linalg.eigh(s)
        norm = np.abs(s).max() * n
        tol = _tol(mode, n, norm)
        np.testing.assert_allclose(w, ref_w, rtol=0, atol=tol)
        np.testing.assert_allclose(s @ v, v * w, rtol=0, atol=tol)
        np.testing.assert_allclose(v.T @ v, np.eye(n), rtol=0, atol=FACTOR * n * _EPS[mode])
        assert np.all(np.diff(w) >= 0)
        big = np.argmax(np.abs(v), axis=0)
        assert np.all(v[big, np.arange(n)] > 0), "the largest component of each vector is positive"
        if n > 1:
            gap = np.diff(ref_w).min()
            if gap > 1e-2 * norm:
                np.testing.assert_allclose(np.abs(v), np.abs(ref_v), rtol=0, atol=tol / gap * norm)
        else:
            assert v[0, 0] == 1.0


# --- inside a dispatch -------------------------------------------------------------

@pytest.mark.parametrize("policy", ["switch", "predicated"])
def test_factorisations_inside_the_arms_of_a_dispatch_run_per_case(build, policy):
    n = 4
    f64 = "f64"
    a = Leaf("vocab_read", "mat_in", "A", TensorType((n, n), f64))
    b = Leaf("vocab_read", "vec_in", "b", TensorType((n,), f64))
    k = Leaf("vocab_read", "per_sample", "k", TensorType((), "i32"))
    lu, chol = make("lu", (a,)), make("chol", (a,))
    arms = (make("lu_solve", (lu, b)), make("cho_solve", (chol, b)),
            make("mul", (b, make("lu_det", (lu,)))))
    kernel = Kernel(f"la_dispatch_{policy}", (
        Assign("x", Dispatch(k, arms, policy, TensorType((n,), f64)), TensorType((n,), f64)),))
    rng = np.random.default_rng(7)
    g = rng.standard_normal((LANES, n, n))
    mats = np.einsum("lij,lkj->lik", g, g) + n * np.eye(n)
    vecs = rng.standard_normal((LANES, n))
    kinds = np.arange(LANES, dtype=np.int64) % 3
    x = np.zeros((n, LANES))
    rt.run(build(kernel, "float64"), A=_plane(mats.reshape(LANES, -1)), b=_plane(vecs),
           k=kinds, x=x)
    want = np.stack([np.linalg.solve(mats[i], vecs[i]) if kinds[i] < 2
                     else vecs[i] * np.linalg.det(mats[i]) for i in range(LANES)])
    np.testing.assert_allclose(x.T, want, rtol=1e-12, atol=0)


# --- the emitted text ---------------------------------------------------------

def _body(kernel, target="host", mode="float64"):
    return render_source(kernel.name, kernel.sinks, kernel.walk, TARGETS[target],
                         mode=scalar_mode(mode), exports=exports(target))


@pytest.mark.parametrize("target", ["host", "cuda"])
@pytest.mark.parametrize("make,fn,struct", [(lu_kernel, "lu", "Lu"), (chol_kernel, "cholesky", "Cholesky"),
                                            (eigh_kernel, "eigh", "Eigh"),
                                            (chol_from_kernel, "choleskyFrom", "Cholesky")])
def test_a_factorisation_is_computed_once_into_a_named_struct(make, fn, struct, target):
    source = _body(make(6), target)
    assert len(re.findall(rf"aether::linalg::{fn}<6>\(", source.text)) == 1
    assert "#include <aether/linalg/linalg.h>" in source.text
    # the struct is declared once and filled in place: never returned by value, never copied
    assert len(re.findall(rf"aether::linalg::{struct}<Real, 6> hawk_lf\d+;", source.text)) == 1
    assert re.search(rf"aether::linalg::{fn}<6>\(hawk_lf\d+, \w+\);", source.text)


@pytest.mark.parametrize("make,n", [(lu_kernel, 5), (chol_kernel, 22), (eigh_kernel, 32),
                                    (chol_from_kernel, 22)])
def test_the_device_source_compiles_to_ptx(make, n, tmp_path, cache_dir):
    kernel = make(n)
    bundle = build_bundle([kernel], tmp_path / kernel.name, targets=("cuda",), cache_dir=cache_dir)
    entry = bundle.artifacts[0].entries["cuda"]
    assert entry.artifact.stat().st_size > 0 and "aether::linalg::" in entry.source.read_text()


def test_the_header_is_included_only_where_it_is_called():
    from _kernels import scale

    assert "linalg" not in _body(scale).text
