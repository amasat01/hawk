# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Per-lane linear algebra written through the front end.

``hawk.math``'s linear-algebra names are traced inside ``@hawk.kernel`` bodies,
built for the host and read back against NumPy. The rows after the surface are
the places a kernel puts them: inside a ``for`` that leaves early, inside a
``Kind``'s vocabulary, beside a broadcast input of a ``Grid`` launch, and under
``vjp`` and ``jvp``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from _eval import evaluate
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import Grid, Index, Matrix, Mutable, Param, Scalar, Vector
from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp
from hawk.ext import Guard, Kind
from hawk.ir import HawkError
from hawk.types import TensorType

LANES = 8
BAD = 3
I32 = TensorType((), "int32")


@pytest.fixture(scope="module")
def host(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("la_front")

    def build(kernel, primal=None):
        kw = dict(targets=("host",), cache_dir=cache_dir, host_profile="native")
        extra = {}
        if primal is not None:
            home = build_bundle([primal], pathlib.Path(root) / primal.name, **kw)
            extra = {"derivative": {kernel.name: (primal.name, home.digest)}}
        bundle = build_bundle([kernel], pathlib.Path(root) / kernel.name, **kw, **extra)
        return rt.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name))
    return build


def _plane(a, lanes=LANES):
    plane = np.ascontiguousarray(np.asarray(a).reshape(lanes, -1).T)
    return plane[0] if plane.shape[0] == 1 else plane


def _run(compiled, inputs, outs, lanes=LANES):
    """``outs``: name -> width (``info`` planes are ``int32``)."""
    sink = {name: np.zeros(lanes if width == 1 else (width, lanes),
                           dtype=np.int32 if name.startswith("i_") else np.float64)
            for name, width in outs.items()}
    rt.run(compiled, **{k: _plane(v, lanes) for k, v in inputs.items()}, **sink)
    return {name: plane.reshape(outs[name], lanes).T for name, plane in sink.items()}


def _spd(rng, n):
    g = rng.standard_normal((n, n))
    return g @ g.T + n * np.eye(n)


# --- the surface -----------------------------------------------------------------

@hawk.kernel
def surface(M: Matrix[3, 3], S: Matrix[3, 3], b: Vector[3], J: Matrix[4, 3],
            r: Vector[4], mu: Param,
            x: Mutable[Vector[3]], i_x: Mutable[hawk.Int32],
            Ainv: Mutable[Matrix[3, 3]], d: Mutable[Scalar],
            L: Mutable[Matrix[3, 3]], xc: Mutable[Vector[3]], i_c: Mutable[hawk.Int32],
            w: Mutable[Vector[3]], V: Mutable[Matrix[3, 3]], i_e: Mutable[hawk.Int32],
            xl: Mutable[Vector[3]], i_l: Mutable[hawk.Int32], step: Mutable[Vector[3]],
            shape: Mutable[Matrix[3, 3]]):
    x, i_x = m.solve(M, b)
    Ainv, _ = m.inv(M)
    d = m.det(M)
    chol, i_c = m.cholesky(S)
    L = chol
    xc = m.cho_solve(chol, b)
    w, V, i_e = m.eigh(S)
    xl, i_l = m.lstsq(J, r)
    step = m.lm_step(J, r, mu)
    shape = m.tril(M) + m.diag(m.diagonal(M)) + m.eye(3) * m.trace(M)


def test_the_surface_matches_numpy_and_zeroes_a_failed_lane(host):
    rng = np.random.default_rng(5)
    mats = rng.standard_normal((LANES, 3, 3)) + 3.0 * np.eye(3)
    spd = np.stack([_spd(rng, 3) for _ in range(LANES)])
    mats[BAD, 1] = 0.0                                   # singular
    spd[BAD, 1, 1] = -1.0                                # not positive definite
    jac = rng.standard_normal((LANES, 4, 3))
    inputs = {"M": mats, "S": spd, "b": rng.standard_normal((LANES, 3)), "J": jac,
              "r": rng.standard_normal((LANES, 4)), "mu": 0.25}
    widths = {"x": 3, "i_x": 1, "Ainv": 9, "d": 1, "L": 9, "xc": 3, "i_c": 1, "w": 3, "V": 9,
              "i_e": 1, "xl": 3, "i_l": 1, "step": 3, "shape": 9}
    compiled = host(surface)
    sink = {name: np.zeros(LANES if width == 1 else (width, LANES),
                           dtype=np.int32 if name.startswith("i_") else np.float64)
            for name, width in widths.items()}
    planes = {k: (v if k == "mu" else _plane(v)) for k, v in inputs.items()}
    rt.run(compiled, **planes, **sink)
    got = {name: plane.reshape(widths[name], LANES).T for name, plane in sink.items()}
    for lane in range(LANES):
        a, s, rhs = mats[lane], spd[lane], inputs["b"][lane]
        if lane == BAD:
            assert got["i_x"][lane, 0] > 0 and got["i_c"][lane, 0] > 0
            for name in ("x", "Ainv", "d", "L", "xc"):
                assert not np.any(got[name][lane]), name
        else:
            assert got["i_x"][lane, 0] == 0 and got["i_c"][lane, 0] == 0
            np.testing.assert_allclose(got["x"][lane], np.linalg.solve(a, rhs), rtol=1e-12)
            np.testing.assert_allclose(got["Ainv"][lane].reshape(3, 3), np.linalg.inv(a),
                                       rtol=1e-11, atol=1e-13)
            np.testing.assert_allclose(got["d"][lane, 0], np.linalg.det(a), rtol=1e-12)
            np.testing.assert_allclose(got["L"][lane].reshape(3, 3), np.linalg.cholesky(s),
                                       rtol=1e-12, atol=1e-13)
            np.testing.assert_allclose(got["xc"][lane], np.linalg.solve(s, rhs), rtol=1e-12)
        ws, vs = np.linalg.eigh((s + s.T) / 2)
        np.testing.assert_allclose(got["w"][lane], ws, rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(np.abs(got["V"][lane].reshape(3, 3)), np.abs(vs), atol=1e-10)
        j, res = jac[lane], inputs["r"][lane]
        np.testing.assert_allclose(got["xl"][lane], np.linalg.lstsq(j, res, rcond=None)[0],
                                   rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(
            got["step"][lane], -np.linalg.solve(j.T @ j + 0.25 * np.eye(3), j.T @ res),
            rtol=1e-11, atol=1e-13)
        want = np.tril(a) + np.diag(np.diag(a)) + np.eye(3) * np.trace(a)
        np.testing.assert_allclose(got["shape"][lane].reshape(3, 3), want, rtol=1e-13)


@hawk.kernel
def from_factor(L: Matrix[3, 3], b: Vector[3], x: Mutable[Vector[3]], i_x: Mutable[hawk.Int32],
                F: Mutable[Matrix[3, 3]]):
    factor, i_x = m.chol_from(m.mat([[L[0, 0], 0.0, 0.0], [L[1, 0], L[1, 1], 0.0],
                                     [L[2, 0], L[2, 1], L[2, 2]]]))
    F = factor
    x = m.cho_solve(factor, b)


def test_a_factor_built_in_the_kernel_solves_and_reports_its_diagonal(host):
    rng = np.random.default_rng(8)
    low = np.stack([np.tril(rng.standard_normal((3, 3)), -1) + np.diag(1.0 + rng.random(3))
                    for _ in range(LANES)])
    low[BAD, 1, 1] = 0.0
    rhs = rng.standard_normal((LANES, 3))
    got = _run(host(from_factor), {"L": low, "b": rhs}, {"x": 3, "i_x": 1, "F": 9})
    for lane in range(LANES):
        if lane == BAD:
            assert got["i_x"][lane, 0] == 2 and not np.any(got["x"][lane]) and not np.any(got["F"][lane])
        else:
            assert got["i_x"][lane, 0] == 0
            np.testing.assert_allclose(got["x"][lane], np.linalg.solve(low[lane] @ low[lane].T, rhs[lane]),
                                       rtol=1e-12)
            np.testing.assert_array_equal(got["F"][lane].reshape(3, 3), low[lane])


def test_chol_from_refuses_a_matrix_that_is_not_square():
    with pytest.raises(HawkError, match="chol_from: a square matrix is wanted"):
        @hawk.kernel
        def tall(J: Matrix[4, 3], x: Mutable[Vector[3]]):
            f, _ = m.chol_from(J)
            x = m.cho_solve(f, x)


def test_cho_solve_refuses_a_matrix_it_did_not_factor():
    with pytest.raises(HawkError, match="cho_solve reads the factor that cholesky"):
        @hawk.kernel
        def built(S: Matrix[3, 3], b: Vector[3], x: Mutable[Vector[3]]):
            x = m.cho_solve(m.tril(S), b)


def test_an_lu_factor_is_not_a_matrix():
    with pytest.raises(HawkError, match="an LU factor is not a matrix"):
        @hawk.kernel
        def arithmetic(A: Matrix[3, 3], y: Mutable[Matrix[3, 3]]):
            factor, _ = m.lu(A)
            y = factor * 2.0


# --- a pathwise Gaussian draw ---------------------------------------------------------

@hawk.kernel
def draw(seed: Index, counter: Index, mu: Vector[3], S: Matrix[3, 3],
         y: Mutable[Vector[3]]):
    y = m.mvn_sample(seed, counter, mu, S)


@hawk.kernel
def draw_by_hand(seed: Index, counter: Index, mu: Vector[3], S: Matrix[3, 3],
                 y: Mutable[Vector[3]]):
    chol, _ = m.cholesky(S)
    y = m.random_multivariate_normal(seed, counter, mu, chol)


def test_mvn_sample_is_the_mean_plus_the_cholesky_factor_times_the_draws(host):
    count = 20000
    cov = np.array([[2.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.5]])
    mean = np.array([1.0, -2.0, 0.5])
    inputs = {"seed": np.full(count, 7, np.int32), "counter": np.zeros(count, np.int32),
              "mu": np.tile(mean, (count, 1)), "S": np.tile(cov.ravel(), (count, 1))}
    sink = {"y": np.zeros((3, count))}
    rt.run(host(draw), **{k: _plane(v, count) for k, v in inputs.items()}, **sink)
    twin = {"y": np.zeros((3, count))}
    rt.run(host(draw_by_hand), **{k: _plane(v, count) for k, v in inputs.items()}, **twin)
    assert np.array_equal(sink["y"], twin["y"])
    y = sink["y"].T
    assert np.all(np.abs(y.mean(axis=0) - mean) < 5 * np.sqrt(np.diag(cov) / count))
    assert np.abs(np.cov(y.T) - cov).max() < 5 * cov.max() * np.sqrt(2.0 / count)


# --- inside a for that leaves early ---------------------------------------------------

CAP = 6


@hawk.kernel
def newton(A: Matrix[3, 3], b: Vector[3], tol: Scalar, y: Mutable[Vector[3]]):
    """Newton's method for ``A x + 0.1 x^3 = b``: a solve in every trip."""
    x = b
    for _k in range(CAP):
        f = A @ x + 0.1 * x * x * x - b
        dx, _ = m.solve(A + m.diag(0.3 * x * x), f)
        x = x - dx
        if m.norm(f) < tol:
            break
    y = x


def _newton_reference(a, b, tol):
    x = b.copy()
    for _ in range(CAP):
        f = a @ x + 0.1 * x ** 3 - b
        x = x - np.linalg.solve(a + np.diag(0.3 * x * x), f)
        if np.linalg.norm(f) < tol:
            break
    return x


def _newton_inputs(rng):
    mats = np.stack([np.array([[4.0, 1, 0], [1, 3, 0.5], [0, 0.5, 2]]) + 0.2 * rng.standard_normal((3, 3))
                     for _ in range(LANES)])
    return mats, rng.standard_normal((LANES, 3))


def test_a_solve_in_a_loop_that_breaks_runs_on_the_host_and_in_the_interpreter(host):
    rng = np.random.default_rng(9)
    mats, rhs = _newton_inputs(rng)
    tols = np.array([1e-12, 1e-12, 1e-12, 1e-12, 5.0, 1.0, 1e-3, 1e-12])   # some leave early
    got = _run(host(newton), {"A": mats, "b": rhs, "tol": tols}, {"y": 3})["y"]
    for lane in range(LANES):
        want = _newton_reference(mats[lane], rhs[lane], tols[lane])
        np.testing.assert_allclose(got[lane], want, rtol=1e-12, atol=1e-13)
        interp = evaluate(newton.sinks, {"A": mats[lane], "b": rhs[lane], "tol": tols[lane]})["y"]
        np.testing.assert_allclose(interp, want, rtol=1e-12, atol=1e-13)


def test_the_loop_with_a_solve_differentiates_both_ways():
    rng = np.random.default_rng(10)
    mats, rhs = _newton_inputs(rng)
    a, b, tol = mats[0], rhs[0], 1e-12
    env = {"A": a, "b": b, "tol": tol}
    seed, tangent, h = np.array([0.3, -1.0, 0.7]), np.array([0.2, 0.1, -0.4]), 1e-6
    reverse = evaluate(vjp(newton), {**env, "bar_y": seed})
    forward = evaluate(jvp(newton), {**env, "dot_b": tangent, "dot_A": np.zeros((3, 3)),
                                     "dot_tol": 0.0})["dot_y"]
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        fd = (_newton_reference(a, b + e, tol) - _newton_reference(a, b - e, tol)) @ seed / (2 * h)
        assert reverse["bar_b"][i] == pytest.approx(fd, rel=1e-6, abs=1e-8)
    fd = (_newton_reference(a, b + h * tangent, tol) - _newton_reference(a, b - h * tangent, tol)) / (2 * h)
    np.testing.assert_allclose(forward, fd, rtol=1e-6, atol=1e-8)


# --- inside a Kind's vocabulary -------------------------------------------------------

@Kind("la_vocab", vocabulary={"damping": Scalar, "A": Matrix[3, 3]}, guard=Guard(mask=None))
def vocab_solve(b: Vector[3], damping, A, y: Mutable[Vector[3]]):
    """``damping`` and ``A`` are the kind's vocabulary: declared once, bound by name."""
    x, _ = m.solve(A + m.eye(3) * damping, b)
    y = x


def test_a_solve_inside_a_kinds_vocabulary_binds_the_declared_matrix(host):
    rng = np.random.default_rng(12)
    mats = rng.standard_normal((LANES, 3, 3)) + 3.0 * np.eye(3)
    rhs = rng.standard_normal((LANES, 3))
    assert ("mat_in", "A") in vocab_solve.arg_spec
    got = _run(host(vocab_solve), {"A": mats, "b": rhs, "damping": np.linspace(0.1, 0.8, LANES)},
               {"y": 3})["y"]
    for lane, damping in enumerate(np.linspace(0.1, 0.8, LANES)):
        np.testing.assert_allclose(
            got[lane], np.linalg.solve(mats[lane] + damping * np.eye(3), rhs[lane]), rtol=1e-12)


# --- beside a broadcast input ---------------------------------------------------------

A_AXIS, B_AXIS, C_AXIS = 3, 4, 6


@hawk.kernel
def grid_solve(lane: Grid["a", "b":4, "c":6], M: Matrix[4, 4], rhs: Vector[4]) -> Vector[4]:
    x, _ = m.solve(M, rhs)
    return x


def test_a_per_lane_matrix_beside_a_broadcast_vector(cache_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("HAWK_CACHE_DIR", str(tmp_path / "cache"))
    rng = np.random.default_rng(1)
    mats = rng.standard_normal((4, 4, A_AXIS, B_AXIS, C_AXIS)) + 5.0 * np.eye(4)[:, :, None, None, None]
    rhs = rng.standard_normal((4, 1, 1, C_AXIS))            # one vector per column of the grid
    out = grid_solve(mats.reshape(16, -1), rhs, a=A_AXIS)
    assert out.shape == (4, A_AXIS, B_AXIS, C_AXIS)
    for i in range(A_AXIS):
        for j in range(B_AXIS):
            for k in range(C_AXIS):
                np.testing.assert_allclose(out[:, i, j, k],
                                           np.linalg.solve(mats[:, :, i, j, k], rhs[:, 0, 0, k]),
                                           rtol=1e-12)
    with pytest.raises(ValueError, match="16 component"):
        grid_solve(mats[:, :, :1].reshape(16, -1), rhs, a=A_AXIS)      # a matrix plane is exact


# --- the compiled derivative ----------------------------------------------------------

@pytest.mark.parametrize("derive", [vjp, jvp], ids=["vjp", "jvp"])
def test_the_compiled_derivative_of_the_loop_matches_the_interpreter(host, derive):
    rng = np.random.default_rng(11)
    mats, rhs = _newton_inputs(rng)
    tols = np.full(LANES, 1e-12)
    tols[::2] = 1.0                                          # half the lanes leave early
    kernel = hawk.Kernel(f"newton_{derive.__name__}", derive(newton, wrt=("A", "b")))
    seeds = rng.standard_normal((LANES, 3))
    tangents = {"dot_A": rng.standard_normal((LANES, 3, 3)), "dot_b": rng.standard_normal((LANES, 3))}
    extra = {"bar_y": seeds} if derive is vjp else tangents
    outs = {"bar_A": 9, "bar_b": 3} if derive is vjp else {"dot_y": 3}
    got = _run(host(kernel, primal=newton), {"A": mats, "b": rhs, "tol": tols, **extra}, outs)
    for lane in range(LANES):
        env = {"A": mats[lane], "b": rhs[lane], "tol": tols[lane],
               **{k: v[lane] for k, v in extra.items()}}
        want = evaluate(kernel.sinks, env)
        for name, width in outs.items():
            np.testing.assert_allclose(got[name][lane], np.asarray(want[name]).reshape(width),
                                       rtol=1e-11, atol=1e-12, err_msg=f"{name} lane {lane}")
