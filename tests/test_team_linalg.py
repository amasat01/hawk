# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Team-form linear algebra: one matrix per lane worked by a team of threads.

A matrix above ``MAX_DIM`` (32) is lowered as stages (:mod:`hawk.ir.team`): the per-lane stage
stores the operand into scratch, a team stage factors it in place and runs the solves, and the
per-lane stage after it reads the results. These rows run every factorisation and its readers
on the host (the team of one) and on the device, against numpy and LAPACK, at sizes below and above
the one-thread cap (``linalg_form="team"`` reaches the small ones), and check that the
one-thread and the team form agree to a few ULP, that the team size never changes a bit, and that
the refusals name what they refuse.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from _block_kernels import gram_kernel, grad_kernel, keyed_transpose_kernel, readback_kernel, scaled_kernel, step_kernel, stored_and_read_kernel, total_kernel
from _linalg_kernels import chol_from_kernel, chol_kernel, eigh_kernel, lstsq_kernel, lu_kernel
from conftest import sidecar_of
from scipy.linalg import lapack

import hawk
from hawk import HawkError
from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.ir import Assign, Leaf, make
from hawk.ir.linalg import MAX_DIM
from hawk.ir.team import BlockSumWrite
from hawk.ir.split import stages
from hawk.trace.kernel import Kernel
from hawk.types import TensorType

LANES = 5
K = 2                                    # right-hand side columns of the kernels
BAD = 2                                  # the lane given a failing input
EPS = np.finfo(np.float64).eps
FORCED = (4, 16, 32)                     # under linalg_form="team"
AUTO = (33, 64, 71, 128)                 # above the one-thread cap: a team, whatever the build asks
BIG = 512                                # lu and cholesky only (eigh runs N^2 rotations in series)
DEVICE = pytest.mark.gpu


def _plane(a):
    """``(LANES, ...)`` as the sample-minor plane a kernel binds, flat when one wide."""
    p = np.ascontiguousarray(a.reshape(LANES, -1).T)
    return p[0] if p.shape[0] == 1 else p


def _general(rng, n, lane):
    a = rng.standard_normal((n, n)) / np.sqrt(n) + 0.5 * np.eye(n)
    if lane == BAD:
        a[:, min(1, n - 1)] = 0.0                      # an exactly zero column: singular
    return a


def _spd(rng, n, lane):
    g = rng.standard_normal((n, n)) / np.sqrt(n)
    a = g @ g.T + np.eye(n)
    if lane == BAD:
        a[n // 2, n // 2] = -1.0                       # the leading minor n // 2 + 1 fails
    return a


def _lower(rng, n, lane):
    a = np.tril(rng.standard_normal((n, n)), -1) / np.sqrt(n) + np.diag(1.0 + rng.random(n))
    a += 0.5 * np.triu(rng.standard_normal((n, n)), 1)  # ignored above the diagonal
    if lane == BAD:
        a[0, 0] = 0.0
    return a


def _symmetric(rng, n, lane):
    g = rng.standard_normal((n, n))
    s = (g + g.T) / 2
    if lane == BAD:                                    # a repeated eigenvalue
        q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        s = (q * np.r_[np.ones(n // 2), np.linspace(2, 3, n - n // 2)]) @ q.T
    return s


def _data(n, make_matrix):
    rng = np.random.default_rng(1000 + n)
    mats = np.stack([make_matrix(rng, n, lane) for lane in range(LANES)])
    return mats, rng.standard_normal((LANES, n)), rng.standard_normal((LANES, n, K))


def _build(kernel, tmp, target, form):
    targets = ("host",) if target == "host" else ("cuda",)
    return build_bundle([kernel], tmp, targets=targets, host_profile="native", linalg_form=form)


def _build_all(kernels, tmp, target):
    return build_bundle(kernels, tmp, targets=("host",) if target == "host" else ("cuda",),
                        host_profile="native", linalg_form="auto")


def _run(bundle, kernel, mats, vecs, rhs, outs, target):
    """The outputs ``{name: (LANES, width)}`` of one run of ``kernel`` over the data."""
    planes = {"A": _plane(mats), "b": _plane(vecs), "B": _plane(rhs)}
    planes = {k: v for k, v in planes.items() if any(name == k for _, name in kernel.arg_spec)}
    sink = {name: np.zeros(LANES if width == 1 else (width, LANES),
                           dtype=np.int32 if name == "info" else np.float64)
            for name, width in outs.items()}
    if target == "host":
        rt.run(rt.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name)),
               **planes, **sink)
    else:
        import cupy as cp
        from _team_stages import run_device

        dev = {k: cp.asarray(v) for k, v in {**planes, **sink}.items()}
        run_device(bundle, kernel.name, dev, LANES)
        sink = {k: dev[k].get() for k in sink}
    return {name: plane.reshape(outs[name], LANES).T for name, plane in sink.items()}


def _back(a, x, b, n):
    """The backward error ``|A x - b| / (n eps |A| |x|)``: within a few tens when a solve is right."""
    r = np.abs(a @ x - b).max()
    return r / (n * EPS * (np.abs(a).max() * np.abs(x).max() + np.abs(b).max()))


CASES = [(n, "team", "host") for n in FORCED] + [(n, "auto", "host") for n in AUTO]
CASES += [(n, "auto", "cuda") for n in (33, 71, 128)] + [(16, "team", "cuda")]


def _ids(case):
    return f"n{case[0]}-{case[1]}-{case[2]}"


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("team_linalg")


def _marked(cases):
    return [pytest.param(*c, marks=DEVICE, id=_ids(c)) if c[2] == "cuda" else pytest.param(*c, id=_ids(c))
            for c in cases]


@pytest.mark.parametrize("n,form,target", _marked(CASES))
def test_lu_solve_det_and_info_match_numpy_and_lapack(cache, n, form, target):
    kernel = lu_kernel(n)
    bundle = _build(kernel, cache / f"lu_{n}_{form}_{target}", target, form)
    mats, vecs, rhs = _data(n, _general)
    out = _run(bundle, kernel, mats, vecs, rhs,
               {"x": n, "X": n * K, "det": 1, "info": 1}, target)
    for lane in range(LANES):
        a = mats[lane]
        if lane == BAD:
            _, _, info = lapack.dgetrf(a)
            assert out["info"][lane, 0] == info > 0
            assert not np.any(out["x"][lane]) and not np.any(out["X"][lane]) and out["det"][lane, 0] == 0
            continue
        assert out["info"][lane, 0] == 0
        assert _back(a, out["x"][lane], vecs[lane], n) < 50
        assert _back(a, out["X"][lane].reshape(n, K), rhs[lane], n) < 50
        sign, logdet = np.linalg.slogdet(a)
        assert out["det"][lane, 0] == pytest.approx(sign * np.exp(logdet), rel=1e-9 * n)


@pytest.mark.parametrize("n,form,target", _marked(CASES))
def test_cholesky_solve_factor_and_info_match_numpy_and_lapack(cache, n, form, target):
    kernel = chol_kernel(n)
    bundle = _build(kernel, cache / f"chol_{n}_{form}_{target}", target, form)
    mats, vecs, rhs = _data(n, _spd)
    out = _run(bundle, kernel, mats, vecs, rhs, {"x": n, "X": n * K, "L": n * n, "info": 1}, target)
    for lane in range(LANES):
        a = mats[lane]
        if lane == BAD:
            assert out["info"][lane, 0] == n // 2 + 1
            assert not np.any(out["x"][lane]) and not np.any(out["X"][lane])
            assert not np.any(np.tril(out["L"][lane].reshape(n, n), -1)[:, n // 2:])
            continue
        assert out["info"][lane, 0] == 0
        low = out["L"][lane].reshape(n, n)
        assert np.abs(low - np.linalg.cholesky(a)).max() < 50 * n * EPS * np.abs(a).max()
        assert not np.any(np.triu(low, 1))
        assert _back(a, out["x"][lane], vecs[lane], n) < 50
        assert _back(a, out["X"][lane].reshape(n, K), rhs[lane], n) < 50


@pytest.mark.parametrize("n,form,target", _marked(CASES))
def test_chol_from_solves_against_the_lower_factor_and_reports_the_bad_diagonal(cache, n, form, target):
    kernel = chol_from_kernel(n)
    bundle = _build(kernel, cache / f"cfrom_{n}_{form}_{target}", target, form)
    mats, vecs, rhs = _data(n, _lower)
    out = _run(bundle, kernel, mats, vecs, rhs, {"x": n, "X": n * K, "L": n * n, "info": 1}, target)
    for lane in range(LANES):
        low = np.tril(mats[lane])
        if lane == BAD:
            assert out["info"][lane, 0] == 1 and not np.any(out["x"][lane])
            continue
        assert out["info"][lane, 0] == 0
        np.testing.assert_array_equal(out["L"][lane].reshape(n, n), low)
        assert _back(low @ low.T, out["x"][lane], vecs[lane], n) < 50


@pytest.mark.parametrize("n,form,target", _marked([c for c in CASES if c[0] <= 71 and c[2] == "host"
                                                  or c[0] in (16, 33)]))
def test_eigh_values_vectors_and_info_match_numpy(cache, n, form, target):
    kernel = eigh_kernel(n)
    bundle = _build(kernel, cache / f"eigh_{n}_{form}_{target}", target, form)
    mats, vecs, rhs = _data(n, _symmetric)
    out = _run(bundle, kernel, mats, vecs, rhs, {"w": n, "V": n * n, "info": 1}, target)
    for lane in range(LANES):
        s = (mats[lane] + mats[lane].T) / 2
        w, v = out["w"][lane], out["V"][lane].reshape(n, n)
        assert out["info"][lane, 0] & 1 == 0
        tol = 100 * n * EPS * np.abs(s).max() * n
        np.testing.assert_allclose(w, np.linalg.eigvalsh(s), rtol=0, atol=tol)
        np.testing.assert_allclose(s @ v, v * w, rtol=0, atol=tol)
        np.testing.assert_allclose(v.T @ v, np.eye(n), rtol=0, atol=100 * n * EPS)
        if lane == BAD:
            assert out["info"][lane, 0] & 2, "a repeated eigenvalue is reported (info bit 2)"


@pytest.mark.parametrize("target", [pytest.param("host", id="host"), pytest.param("cuda", marks=DEVICE, id="cuda")])
def test_a_matrix_of_512_runs_in_scratch_a_lane_could_not_hold(cache, target):
    n = BIG
    a = Leaf("vocab_read", "mat_in", "A", TensorType((n, n), "f64"))
    b = Leaf("vocab_read", "vec_in", "b", TensorType((n,), "f64"))
    f = make("chol", (a,))
    kernel = Kernel(f"big_chol_{n}", (Assign("x", make("cho_solve", (f, b)), TensorType((n,), "f64")),
                                      Assign("info", make("chol_info", (f,)), TensorType((), "int32"))))
    bundle = _build(kernel, cache / f"chol_big_{target}", target, "auto")
    mats, vecs, rhs = _data(n, _spd)
    out = _run(bundle, kernel, mats, vecs, rhs, {"x": n, "info": 1}, target)
    for lane in (0, 1, 3, 4):
        assert out["info"][lane, 0] == 0
        assert _back(mats[lane], out["x"][lane], vecs[lane], n) < 50


def _per_lane_factor(cache, target, n=128):
    """The Cholesky kernel at ``n``: its solution and its ``n x n`` factor, a per-lane matrix output."""
    kernel = chol_kernel(n)
    bundle = _build(kernel, cache / f"chol_out_{n}_{target}", target, "auto")
    mats, vecs, rhs = _data(n, _spd)
    return mats, _run(bundle, kernel, mats, vecs, rhs, {"x": n, "X": n * K, "L": n * n, "info": 1}, target)


def test_a_per_lane_matrix_output_of_128_compiles_on_the_host_in_seconds(cache):
    """The factor read back from the team stage is copied into its output plane by a loop: held as a
    value the host compiler would unroll its 16384 elements one by one (minutes)."""
    import time

    start = time.perf_counter()
    mats, out = _per_lane_factor(cache, "host")
    assert time.perf_counter() - start < 60
    for lane in (0, 1, 3, 4):
        assert out["info"][lane, 0] == 0
        np.testing.assert_allclose(out["L"][lane].reshape(128, 128), np.linalg.cholesky(mats[lane]),
                                   rtol=1e-10, atol=1e-12)


@DEVICE
def test_a_per_lane_matrix_output_of_128_matches_the_device(cache):
    mats, host = _per_lane_factor(cache, "host")
    _, device = _per_lane_factor(cache, "cuda")
    assert (host["info"] == device["info"]).all()
    for name in ("L", "x"):
        np.testing.assert_allclose(host[name][[0, 1, 3, 4]], device[name][[0, 1, 3, 4]], rtol=1e-10, atol=1e-12)


# --- least squares by QR ------------------------------------------------------------

LSQ_CASES = [(n, "auto", "host") for n in (33, 64, 71)] + [(33, "auto", "cuda"), (71, "auto", "cuda")]


@pytest.mark.parametrize("n,form,target", _marked(LSQ_CASES))
def test_lstsq_by_qr_matches_numpy_and_reports_a_zero_column(cache, n, form, target):
    m = 2 * n
    kernel = lstsq_kernel(m, n)
    bundle = _build(kernel, cache / f"lsq_{n}_{target}", target, form)
    rng = np.random.default_rng(500 + n)
    j = rng.standard_normal((LANES, m, n))
    r = rng.standard_normal((LANES, m))
    j[BAD, :, 1] = 0.0                                  # a zero column: R has an exactly zero diagonal
    sink = _launch(bundle, kernel, {"J": _plane(j), "r": _plane(r)},
                   {"x": np.zeros((n, LANES)), "info": np.zeros(LANES, dtype=np.int32)}, target)
    for lane in range(LANES):
        if lane == BAD:
            assert sink["info"][lane] == 2 and not np.any(sink["x"][:, lane])
            continue
        want = np.linalg.lstsq(j[lane], r[lane], rcond=None)[0]
        assert sink["info"][lane] == 0
        cond = np.linalg.cond(j[lane])
        assert np.abs(sink["x"][:, lane] - want).max() <= 50 * n * EPS * cond * np.abs(want).max()


def _launch(bundle, kernel, planes, sink, target):
    """One run of ``kernel`` over ``planes`` into the zeroed ``sink`` planes, which come back as arrays."""
    if target == "host":
        rt.run(rt.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name)), **planes, **sink)
        return sink
    import cupy as cp
    from _team_stages import run_device

    dev = {k: cp.asarray(v) for k, v in {**planes, **sink}.items()}
    run_device(bundle, kernel.name, dev, LANES)
    return {k: dev[k].get() for k in sink}


def _lstsq_x(m, n):
    """The solution of ``lstsq(J, r)`` as a kernel of its own (what a derivative differentiates)."""
    return Kernel(f"lsq_x_{m}_{n}", tuple(s for s in lstsq_kernel(m, n).sinks if s.name == "x"))


DERIV_CASES = [(33, "host"), (71, "host"), (33, "cuda")]


@pytest.mark.parametrize("n,form,target", _marked([(n, "auto", t) for n, t in DERIV_CASES]))
def test_lstsq_by_qr_derivatives_match_finite_differences(cache, n, form, target):
    from hawk.diff import jvp, vjp

    m = n + 9
    primal = _lstsq_x(m, n)
    pull = Kernel(f"lsq_vjp_{n}", vjp(primal, wrt=("J", "r")))
    push = Kernel(f"lsq_jvp_{n}", jvp(primal, wrt=("J", "r")))
    bundle = _build_all([primal, pull, push], cache / f"lsq_d_{n}_{target}", target)
    rng = np.random.default_rng(700 + n)
    j, r = rng.standard_normal((LANES, m, n)), rng.standard_normal((LANES, m))
    dj, dr = rng.standard_normal((LANES, m, n)), rng.standard_normal((LANES, m))
    bar = rng.standard_normal((LANES, n))
    zeros = lambda *shape: np.zeros(shape)             # noqa: E731

    def solve(jj, rr):
        return np.stack([np.linalg.lstsq(jj[i], rr[i], rcond=None)[0] for i in range(LANES)])

    got = _launch(bundle, pull, {"J": _plane(j), "r": _plane(r), "bar_x": _plane(bar)},
                  {"bar_J": zeros(m * n, LANES), "bar_r": zeros(m, LANES)}, target)
    bar_j = got["bar_J"].T.reshape(LANES, m, n)
    bar_r = got["bar_r"].T
    out = _launch(bundle, push, {"J": _plane(j), "r": _plane(r), "dot_J": _plane(dj), "dot_r": _plane(dr)},
                  {"dot_x": zeros(n, LANES)}, target)["dot_x"].T
    h = 1e-6
    want_dot = (solve(j + h * dj, r + h * dr) - solve(j - h * dj, r - h * dr)) / (2 * h)
    assert np.abs(out - want_dot).max() <= 1e-5 * np.abs(want_dot).max()
    for lane, row, col in ((0, 0, 0), (1, m - 1, n - 1), (3, m // 2, 1), (4, 2, n // 2)):
        e = np.zeros((LANES, m, n))
        e[lane, row, col] = h
        fd = np.sum(bar * (solve(j + e, r) - solve(j - e, r)), axis=1)[lane] / (2 * h)
        assert abs(bar_j[lane, row, col] - fd) <= 1e-5 * max(1.0, abs(fd))
    for lane, row in ((0, 0), (2, m - 1), (4, m // 2)):
        e = np.zeros((LANES, m))
        e[lane, row] = h
        fd = np.sum(bar * (solve(j, r + e) - solve(j, r - e)), axis=1)[lane] / (2 * h)
        assert abs(bar_r[lane, row] - fd) <= 1e-5 * max(1.0, abs(fd))


def _exact_lstsq_tangent(j, r, dj, dr):
    """``(dx, x)`` of ``x = argmin |J x - r|`` for the float64 inputs, taken exactly: the normal
    equations solved in 80-digit decimals, so the conditioning of ``J`` costs the reference nothing."""
    from decimal import Decimal, getcontext

    getcontext().prec = 80
    dec = lambda a: [[Decimal(float(v)) for v in row] for row in np.atleast_2d(a)]   # noqa: E731
    (jj, djj), (rr, drr), (m, n) = (dec(j), dec(dj)), (dec(r)[0], dec(dr)[0]), j.shape

    def solve(rhs):
        a = [[sum(jj[i][p] * jj[i][q] for i in range(m)) for q in range(n)] + [rhs[p]] for p in range(n)]
        for c in range(n):
            top = max(range(c, n), key=lambda i: abs(a[i][c]))
            a[c], a[top] = a[top], a[c]
            for i in range(c + 1, n):
                f = a[i][c] / a[c][c]
                a[i] = [u - f * w for u, w in zip(a[i], a[c])]
        x = [Decimal(0)] * n
        for i in reversed(range(n)):
            x[i] = (a[i][n] - sum(a[i][q] * x[q] for q in range(i + 1, n))) / a[i][i]
        return x

    gather = lambda a, v: [sum(a[i][p] * v[i] for i in range(m)) for p in range(n)]   # noqa: E731
    x = solve(gather(jj, rr))
    res = [rr[i] - sum(jj[i][q] * x[q] for q in range(n)) for i in range(m)]
    dx_part = [sum(djj[i][q] * x[q] for q in range(n)) for i in range(m)]
    rhs = [u - w + z for u, w, z in zip(gather(djj, res), gather(jj, dx_part), gather(jj, drr))]
    return np.array([float(v) for v in solve(rhs)])


def test_lstsq_by_qr_derivative_keeps_the_conditioning_of_the_matrix(cache):
    """cond(J) = 1e8: the derivative agrees with the exact one to the conditioning, where solving
    the normal equations (cond 1e16) is off by percents."""
    from hawk.diff import jvp

    m, n = 40, 33
    rng = np.random.default_rng(5)
    u, _ = np.linalg.qr(rng.standard_normal((m, n)))
    v, _ = np.linalg.qr(rng.standard_normal((n, n)))
    j = (u * np.logspace(0, -8, n)) @ v.T
    r = j @ rng.standard_normal(n) + 1e-3 * rng.standard_normal(m)
    dj, dr = rng.standard_normal((m, n)), rng.standard_normal(m)
    want = _exact_lstsq_tangent(j, r, dj, dr)
    primal = _lstsq_x(m, n)
    push = Kernel("lsq_ill_jvp", jvp(primal, wrt=("J", "r")))
    bundle = _build_all([primal, push], cache / "lsq_ill", "host")
    rep = lambda a: _plane(np.repeat(a[None], LANES, axis=0))               # noqa: E731
    got = _launch(bundle, push, {"J": rep(j), "r": rep(r), "dot_J": rep(dj), "dot_r": rep(dr)},
                  {"dot_x": np.zeros((n, LANES))}, "host")["dot_x"][:, 0]
    scale = np.abs(want).max()
    x = np.linalg.lstsq(j, r, rcond=None)[0]
    squared = np.linalg.solve(j.T @ j, dj.T @ (r - j @ x) - j.T @ (dj @ x) + j.T @ dr)
    assert np.abs(got - want).max() <= 1e-6 * scale
    assert np.abs(squared - want).max() >= 1e-3 * scale          # the normal-equation form is not


def test_lstsq_is_a_qr_team_stage_above_32_unknowns_and_normal_equations_below():
    import hawk.math as hm
    from hawk.ir import Leaf as L
    from hawk.trace.value import node_of
    from hawk.types import TensorType as T

    def solve(m, n):
        j = hawk.Value(L("vocab_read", "mat_in", "J", T((m, n), "f64")))
        r = hawk.Value(L("vocab_read", "vec_in", "r", T((m,), "f64")))
        return node_of(hm.lstsq(j, r)[0])

    assert solve(80, 40).kind == "qr_solve" and solve(80, 30).kind == "cho_solve"
    assert solve(80, 40).operands[0].ttype.tag == "qr_factor"
    found = stages("ls", lstsq_kernel(80, 40).sinks)
    assert [s.team is not None for s in found] == [False, True, False]
    steps = found[1].team.steps
    assert [(s.op, s.m, s.n) for s in steps] == [("qr", 80, 40), ("qr_solve", 80, 40)]
    assert {p.name: p.per_lane for p in found[1].team.planes}["tm0_work"] == 40 * 2 + 1


def test_a_qr_is_refused_under_the_thread_form_and_when_wide(cache):
    with pytest.raises(HawkError, match="least squares by QR is worked by a team"):
        _build(lstsq_kernel(80, 40), cache / "lsq_thread", "host", "thread")
    with pytest.raises(HawkError, match="a tall matrix"):
        lstsq_kernel(30, 40)


# --- the split, the policy and the refusals ------------------------------------------

def test_a_team_matrix_makes_a_store_stage_a_team_stage_and_a_read_stage():
    found = stages("t", chol_kernel(40).sinks)
    assert [s.team is not None for s in found] == [False, True, False]
    spec = found[1].team
    assert spec.threads == 128 and [s.op for s in spec.steps] == ["chol", "cho_solve", "cho_solve"]
    assert {p.name: p.per_lane for p in spec.planes}["tm0_a"] == 40 * 40
    assert stages("t", chol_kernel(40).sinks)[1].team.threads == 128
    assert stages("t", chol_kernel(200).sinks)[1].team.threads == 256


def test_a_matrix_in_one_lane_is_not_split_unless_the_build_asks_for_a_team():
    from hawk.emit import policy

    assert stages("t", chol_kernel(MAX_DIM).sinks) is None
    with policy.using(linalg_form="team"):
        assert stages("t", chol_kernel(MAX_DIM).sinks) is not None


def test_the_sidecar_names_the_team_stage_and_its_scratch(cache):
    kernel = lu_kernel(40)
    bundle = _build(kernel, cache / "sidecar", "host", "auto")
    side = sidecar_of(bundle, kernel.name)
    team = [s for s in side["stages"] if "team" in s]
    assert len(team) == 1 and team[0]["team"] == {"threads": 128}
    planes = side["team_planes"]
    assert planes["tm0_a"] == {"dtype": "real", "per_lane": 1600}
    assert planes["tm0_piv"] == {"dtype": "int32", "per_lane": 40}
    assert planes["tm0_info"] == {"dtype": "int32", "per_lane": 1}


def test_a_thread_form_matrix_above_32_is_refused_by_name(cache):
    with pytest.raises(HawkError, match=r"linalg_form='thread' keeps the matrix in one lane"):
        _build(chol_kernel(64), cache / "thread64", "host", "thread")


def test_the_form_must_be_one_of_the_three(cache):
    with pytest.raises(HawkError, match="linalg_form must be one of"):
        _build(chol_kernel(4), cache / "bad_form", "host", "fast")


def test_a_team_matrix_in_a_loop_body_is_refused_by_name():
    import hawk
    import hawk.math as hm
    from hawk import Index, Matrix, Mutable, Vector

    @hawk.kernel
    def iterate(a: Matrix[40, 40], b: Vector[40], n: Index, y: Mutable[Vector[40]]):
        s = b
        for _ in range(n, max=3):
            s = hm.solve(a, s)[0]
        y = s

    with pytest.raises(HawkError, match="launch of its own and cannot sit inside a loop"):
        stages("iterate", iterate.sinks)


def test_a_team_op_in_an_arm_of_a_dispatch_is_refused_by_name_unless_the_body_also_reads_it_outside():
    from hawk.ir import Dispatch

    n = 40
    a = Leaf("vocab_read", "mat_in", "A", TensorType((n, n), "f64"))
    b = Leaf("vocab_read", "vec_in", "b", TensorType((n,), "f64"))
    k = Leaf("vocab_read", "per_sample", "k", TensorType((), "i32"))
    solve = make("cho_solve", (make("chol", (a,)), b))
    plain = make("mul", (b, b))
    arm = Dispatch(k, [solve, plain], "switch", solve.ttype)
    with pytest.raises(HawkError, match=r"op 'chol' at n=40.*inside an arm of a dispatch"):
        stages("arm", (Assign("x", arm, solve.ttype),))
    both = (Assign("x", arm, solve.ttype), Assign("y", solve, solve.ttype))
    assert any(s.team for s in stages("arm", both))                  # read outside too: every lane's


def test_a_block_sum_in_an_arm_of_a_dispatch_is_refused_by_name():
    import hawk.math as hm
    from hawk import Int32, Matrix, Vector

    @hawk.kernel
    def armed(j: Vector[40], k: Int32) -> Matrix[40, 40]:
        return hm.dispatch(k, [hawk.sum(hm.outer(j, j)), hm.outer(j, j)])

    with pytest.raises(HawkError, match="sum.*dense block.*inside an arm of a dispatch"):
        stages("armed", armed.sinks)


def test_the_standalone_c_target_refuses_a_team_matrix(cache):
    with pytest.raises(HawkError, match="standalone C target has no team form"):
        build_bundle([chol_kernel(40)], cache / "c_unit", targets=("c",))


def test_the_per_lane_writer_refuses_a_matrix_above_the_cap():
    from hawk.emit import render_source
    from hawk.emit.wrapper import scalar_mode
    from hawk.artifact.layout import exports
    from hawk.targets import TARGETS

    kernel = chol_kernel(40)
    with pytest.raises(HawkError, match="reached the per-lane writer"):
        render_source(kernel.name, kernel.sinks, kernel.walk, TARGETS["host"],
                      mode=scalar_mode("float64"), exports=exports("host"))


def test_the_dimension_bound_is_4096():
    with pytest.raises(HawkError, match=r"1 <= n <= 4096, got n=4097"):
        make("lu", (Leaf("vocab_read", "mat_in", "A", TensorType((4097, 4097), "f64")),))


# --- the bits -----------------------------------------------------------------------

def _bits_run(cache, tag, target, threads):
    old = os.environ.get("HAWK_TEAM_THREADS")
    os.environ["HAWK_TEAM_THREADS"] = str(threads)
    try:
        kernel = lu_kernel(71)
        bundle = _build(kernel, cache / f"bits_{tag}_{target}_{threads}", target, "auto")
        mats, vecs, rhs = _data(71, _general)
        return _run(bundle, kernel, mats, vecs, rhs, {"x": 71, "X": 71 * K, "det": 1, "info": 1}, target)
    finally:
        if old is None:
            os.environ.pop("HAWK_TEAM_THREADS")
        else:
            os.environ["HAWK_TEAM_THREADS"] = old


@pytest.mark.parametrize("target", [pytest.param("host", id="host"), pytest.param("cuda", marks=DEVICE, id="cuda")])
def test_the_team_size_never_changes_a_bit(cache, target):
    runs = [_bits_run(cache, "a", target, t) for t in ((1, 8) if target == "host" else (64, 128, 256))]
    for other in runs[1:]:
        for name, plane in runs[0].items():
            assert plane.tobytes() == other[name].tobytes(), name


@pytest.mark.parametrize("n", (16, 32))
@pytest.mark.parametrize("target", [pytest.param("host", id="host"), pytest.param("cuda", marks=DEVICE, id="cuda")])
def test_team_and_thread_forms_agree_to_a_few_ulp(cache, n, target):
    kernel = lu_kernel(n)
    mats, vecs, rhs = _data(n, _general)
    outs = {"x": n, "X": n * K, "det": 1, "info": 1}
    got = {form: _run(_build(kernel, cache / f"ab_{n}_{form}_{target}", target, form),
                      kernel, mats, vecs, rhs, outs, target) for form in ("thread", "team")}
    for lane in range(LANES):
        if lane == BAD:
            assert got["thread"]["info"][lane, 0] == got["team"]["info"][lane, 0] > 0
            continue
        for name in ("x", "X"):
            a, b = got["thread"][name][lane], got["team"][name][lane]
            assert np.abs(a - b).max() <= 50 * n * EPS * np.abs(a).max()
        assert got["thread"]["info"][lane, 0] == got["team"]["info"][lane, 0] == 0


# --- block sums: the rows of a key into a dense block -------------------------------

KEYS, ROWS = 3, 5
SUM_BOUND = 8                      # |error| <= 8 eps sum|terms| against the exact sum, plain adds


def _rows(p, seed=7):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((KEYS * ROWS, p)), rng.standard_normal(KEYS * ROWS)


def _exact(j, r):
    """Per key, the Gram, the gradient and the sums of the absolute terms (the error scale)."""
    jk, rk = j.reshape(KEYS, ROWS, -1), r.reshape(KEYS, ROWS)
    gram = np.stack([[[math.fsum(jk[k, :, a] * jk[k, :, b]) for b in range(jk.shape[2])]
                      for a in range(jk.shape[2])] for k in range(KEYS)])
    grad = np.stack([[math.fsum(jk[k, :, a] * rk[k]) for a in range(jk.shape[2])] for k in range(KEYS)])
    scale = np.abs(jk).transpose(0, 2, 1) @ np.abs(jk), np.einsum("kra,kr->ka", np.abs(jk), np.abs(rk))
    return gram, grad, scale


def _build_sum(kernel, tmp, target, form="auto"):
    targets = ("host",) if target == "host" else ("cuda",)
    return build_bundle([kernel], tmp, targets=targets, host_profile="native", linalg_form=form)


def _run_sum(bundle, kernel, planes, outs, target):
    """``{name: array}`` of the output planes after one run (``outs`` gives their zero arrays)."""
    lanes = KEYS * ROWS
    if target == "host":
        rt.run(rt.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name)), **planes, **outs)
        return outs
    import cupy as cp
    from _team_stages import run_device

    dev = {k: cp.asarray(v) for k, v in {**planes, **outs}.items()}
    run_device(bundle, kernel.name, dev, lanes)
    return {k: dev[k].get() for k in outs}


def _gram_planes(j, r):
    return {"j": np.ascontiguousarray(j.T), "r": r}


def _only(kernel, planes):
    names = {name for _, name in kernel.arg_spec}
    return {k: v for k, v in planes.items() if k in names}


def _sums_of(kernel, cache, tag, target, form, j, r, p):
    bundle = _build_sum(kernel, cache / tag, target, form)
    out = {"out": np.zeros((p, p, KEYS) if kernel.name == "gram" else (p, KEYS))}
    return _run_sum(bundle, kernel, _only(kernel, _gram_planes(j, r)), out, target)["out"]


SUM_CASES = [(8, "team", "host"), (8, "team", "cuda"), (71, "auto", "host"), (71, "auto", "cuda")]


@pytest.mark.parametrize("p,form,target", _marked(SUM_CASES))
def test_block_sums_match_the_exact_sum_and_the_per_entry_path(cache, p, form, target):
    """The block form against ``math.fsum`` and, at p = 8, against the per-entry accumulates (the
    ``thread`` form). Both are ordered or compensated sums: the contract is a ULP bound on the
    sum of the absolute terms, ``SUM_BOUND`` eps, not bit identity."""
    j, r = _rows(p)
    gram, grad, (gram_scale, grad_scale) = _exact(j, r)
    for build, ref, scale in ((gram_kernel, gram.transpose(1, 2, 0), gram_scale.transpose(1, 2, 0)),
                              (grad_kernel, grad.T, grad_scale.T)):
        kernel = build(p, ROWS, form)
        got = _sums_of(kernel, cache, f"blk_{build.__name__}_{p}_{target}", target, form, j, r, p)
        assert np.all(np.abs(got - ref) <= SUM_BOUND * EPS * scale + 1e-300), build.__name__
        if p == 8:
            per_entry = build(p, ROWS, "thread")
            assert not any(isinstance(s, BlockSumWrite) for s in per_entry.sinks)
            other = _sums_of(per_entry, cache, f"ent_{build.__name__}_{p}_{target}", target, "thread", j, r, p)
            assert np.all(np.abs(got - other) <= 2 * SUM_BOUND * EPS * scale + 1e-300)


@pytest.mark.parametrize("target", [pytest.param("host", id="host"), pytest.param("cuda", marks=DEVICE, id="cuda")])
def test_a_sum_written_into_its_output_plane_equals_the_one_copied_from_scratch_bit_for_bit(cache, target):
    """Stored only, the block lands in the output plane by strides; stored and read back, it is summed in
    scratch and copied: the same bits."""
    p = 24
    j, r = _rows(p)
    direct = _sums_of(gram_kernel(p, ROWS), cache, f"blk_direct_{target}", target, "auto", j, r, p)
    kernel = stored_and_read_kernel(p, ROWS)
    bundle = _build_sum(kernel, cache / f"blk_copied_{target}", target)
    out = {"gram": np.zeros((p, p, KEYS)), "x": np.zeros(KEYS * ROWS)}
    copied = _run_sum(bundle, kernel, {"j": np.ascontiguousarray(j.T)}, out, target)["gram"]
    assert direct.tobytes() == copied.tobytes()


@pytest.mark.parametrize("make", (gram_kernel, grad_kernel))
@pytest.mark.parametrize("target", [pytest.param("host", id="host"), pytest.param("cuda", marks=DEVICE, id="cuda")])
def test_the_vjp_through_a_block_sum_matches_finite_differences(cache, make, target):
    """The primal (a team stage that reads the factors back by key) and its vjp (a write per lane) are
    two execution axes, so two bundles: the vjp names the primal by its unit digest."""
    from hawk.diff import vjp

    p = 20
    primal = make(p, ROWS)
    pull = Kernel(f"{primal.name}_pull", vjp(primal, wrt=("j",)))
    targets = ("host",) if target == "host" else ("cuda",)
    first = build_bundle([primal], cache / f"vjp_primal_{primal.name}_{target}", targets=targets, host_profile="native")
    bundle = build_bundle([pull], cache / f"vjp_pull_{primal.name}_{target}", targets=targets, host_profile="native",
                          derivative={pull.name: (primal.name, first.digest)})
    j, r = _rows(p)
    rng = np.random.default_rng(9)
    gram = make is gram_kernel
    bar = rng.standard_normal((p, p, KEYS) if gram else (p, KEYS))
    planes = {"j": np.ascontiguousarray(j.T), "r": r, "bar_out": bar}
    got = _run_sum(bundle, pull, _only(pull, planes), {"bar_j": np.zeros((p, KEYS * ROWS))}, target)["bar_j"].T

    def objective(jj):
        jk, rk = jj.reshape(KEYS, ROWS, p), r.reshape(KEYS, ROWS)
        sums = (np.einsum("kra,krb->abk", jk, jk) if gram else np.einsum("kra,kr->ak", jk, rk))
        return float(np.sum(bar * sums))

    h = 1e-6
    for lane, col in ((0, 0), (4, p - 1), (7, 3), (14, 11)):
        e = np.zeros_like(j)
        e[lane, col] = h
        fd = (objective(j + e) - objective(j - e)) / (2 * h)
        assert abs(got[lane, col] - fd) <= 1e-7 * max(1.0, abs(fd))


@pytest.mark.parametrize("make", (gram_kernel, grad_kernel))
def test_a_keyed_product_sum_at_p71_lowers_to_one_block_sum(make):
    kernel = make(71, ROWS)
    assert [type(s) for s in kernel.sinks] == [BlockSumWrite]
    found = stages(kernel.name, kernel.sinks)
    assert [s.team is not None for s in found] == [False, True]
    (step,) = found[1].team.steps
    assert (step.op, found[1].team.grid) == ("block_sum", "keys")
    assert (step.n, step.k) == (71, 71 if make is gram_kernel else 1)


def test_a_payload_that_is_not_a_product_keeps_the_per_entry_path():
    scaled, small = scaled_kernel(20, ROWS), gram_kernel(4, ROWS)
    assert not any(isinstance(s, BlockSumWrite) for s in scaled.sinks + small.sinks)
    assert len(small.sinks) == 16 and stages("gram", small.sinks) is None


def test_the_sidecar_names_the_block_sum_stage_its_keys_and_its_scratch(cache):
    kernel = gram_kernel(40, ROWS)
    side = sidecar_of(_build_sum(kernel, cache / "blk_sidecar", "host"), kernel.name)
    (team,) = [s["team"] for s in side["stages"] if "team" in s]
    keys = {"width": 1600, "axes": [["key", None]], "payload": [40, 40]}
    assert team == {"threads": 128, "grid": "keys", "keyed": keys}
    assert side["team_planes"] == {"tmb0_u": {"dtype": "real", "per_lane": 40}}      # stored only: no block scratch
    read = stored_and_read_kernel(40, ROWS)
    both = sidecar_of(_build_sum(read, cache / "blk_sidecar_read", "host"), read.name)["team_planes"]
    assert both["tmb0_g"] == {"dtype": "real", "per_lane": 1600, "keys": keys}


def test_a_block_sum_is_refused_under_the_thread_form_by_name(cache):
    with pytest.raises(HawkError, match=r"linalg_form='thread' keeps every sum in per-entry form"):
        _build_sum(gram_kernel(40, ROWS), cache / "blk_thread", "host", "thread")


@pytest.mark.parametrize("target", [pytest.param("host", id="host"), pytest.param("cuda", marks=DEVICE, id="cuda")])
def test_a_block_sum_read_back_feeds_a_cholesky_solve(cache, target):
    """The sums read in the body (a per-lane read of the key's block), then a team Cholesky of the
    Gram and a solve: the stage chain of one damped step per key."""
    p, rows = 33, 40
    n = KEYS * rows
    rng = np.random.default_rng(3)
    j, r = rng.standard_normal((n, p)), rng.standard_normal(n)
    jk, rk = j.reshape(KEYS, rows, p), r.reshape(KEYS, rows)
    kernel = step_kernel(p, rows)
    bundle = _build_sum(kernel, cache / f"blk_step_{target}", target)
    assert [bool(s.get("team")) for s in sidecar_of(bundle, kernel.name)["stages"]] == [
        False, True, True, False, True, False]
    out = _run_sum_n(bundle, kernel, {"j": np.ascontiguousarray(j.T), "r": r}, {"x": np.zeros((p, n))}, target, n)
    for k in range(KEYS):
        a = jk[k].T @ jk[k] 
        want = np.linalg.solve(a, jk[k].T @ rk[k])
        for lane in range(k * rows, (k + 1) * rows):
            assert np.abs(out["x"][:, lane] - want).max() <= 100 * p * EPS * np.abs(want).max() * np.linalg.cond(a)


@pytest.mark.parametrize("target", [pytest.param("host", id="host"), pytest.param("cuda", marks=DEVICE, id="cuda")])
def test_entries_of_a_block_sum_are_read_back_per_lane(cache, target):
    p = 40
    readback = readback_kernel(p, ROWS)
    j, r = _rows(p)
    gram, grad, _ = _exact(j, r)
    got = _run_sum(_build_sum(readback, cache / f"blk_read_{target}", target), readback,
                   _gram_planes(j, r), {"out": np.zeros((2, KEYS * ROWS))}, target)["out"]
    assert np.allclose(got[0], np.repeat(gram[:, 1, 2], ROWS), rtol=0, atol=SUM_BOUND * EPS * 10 * ROWS)
    assert np.allclose(got[1], np.repeat(grad[:, 3], ROWS), rtol=0, atol=SUM_BOUND * EPS * 10 * ROWS)


def _run_sum_n(bundle, kernel, planes, outs, target, lanes):
    if target == "host":
        rt.run(rt.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name)), **planes, **outs)
        return outs
    import cupy as cp
    from _team_stages import run_device

    dev = {k: cp.asarray(v) for k, v in {**planes, **outs}.items()}
    run_device(bundle, kernel.name, dev, lanes)
    return {k: dev[k].get() for k in outs}


@pytest.mark.parametrize("target", [pytest.param("host", id="host"), pytest.param("cuda", marks=DEVICE, id="cuda")])
def test_a_block_sum_never_changes_a_bit_with_the_team_size(cache, target):
    j, r = _rows(71)
    runs = []
    for threads in ((1, 8) if target == "host" else (64, 128, 256)):
        os.environ["HAWK_TEAM_THREADS"] = str(threads)
        try:
            runs.append([_sums_of(make(71, ROWS), cache, f"blkbit_{make.__name__}_{target}_{threads}", target,
                                  "auto", j, r, 71) for make in (gram_kernel, grad_kernel)])
        finally:
            os.environ.pop("HAWK_TEAM_THREADS")
    for other in runs[1:]:
        for a, b in zip(runs[0], other):
            assert a.tobytes() == b.tobytes()


def test_a_block_sum_under_an_index_set_sums_the_live_rows_of_each_key(cache):
    """A lane set lists the live lanes: the rows of a key are the listed lanes in its range."""
    from hawk.ext import IndexSet, Kind

    p = 24
    kind = Kind("lane_set", lanes=IndexSet())
    j, r = _rows(p)
    live = np.array([0, 2, 3, 5, 6, 7, 8, 12, 14], dtype=np.int32)       # keys of 5 rows: 0 | 1 | 2
    kernel = gram_kernel(p, ROWS, kind=kind)
    bundle = build_bundle([kernel], cache / "blk_set", targets=("host",), host_profile="native")
    out = {"out": np.zeros((p, p, KEYS))}
    rt.run(rt.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name)),
           j=np.ascontiguousarray(j.T), **out, active_map=live, active_count=np.array([len(live)], np.uint32))
    want = np.zeros((p, p, KEYS))
    for k in range(KEYS):
        rows = [i for i in live if i // ROWS == k]
        want[:, :, k] = j[rows].T @ j[rows]
    assert np.abs(out["out"] - want).max() <= SUM_BOUND * EPS * np.abs(want).max() * ROWS * 4


# --- the same sums over every lane (one key) -----------------------------------------

TOTAL_ROWS = 3                     # rows of the per-lane matrix of the transpose(M) @ M spellings
SPELLINGS = ("outer", "scaled", "gram", "moment")


def _totals(spelling, p, seed=11):
    """``(planes, exact, scale)``: the inputs of ``total_kernel(spelling, p)`` over ``KEYS * ROWS`` lanes, the
    exact sum (``math.fsum`` over every lane and row) and the sum of the absolute terms."""
    lanes, rng = KEYS * ROWS, np.random.default_rng(seed)
    if spelling in ("outer", "scaled"):
        j, r = rng.standard_normal((lanes, p)), rng.standard_normal(lanes)
        terms = np.einsum("la,lb->lab", j, j) if spelling == "outer" else j * r[:, None]
        planes = {"j": np.ascontiguousarray(j.T), "r": r}
    else:
        mat, r = rng.standard_normal((lanes, TOTAL_ROWS, p)), rng.standard_normal((lanes, TOTAL_ROWS))
        terms = (np.einsum("lta,ltb->ltab", mat, mat) if spelling == "gram"
                 else mat * r[:, :, None]).reshape(-1, *((p, p) if spelling == "gram" else (p,)))
        planes = {"mat": _plane_of(mat), "r": _plane_of(r)}
    flat = terms.reshape(len(terms), -1)
    exact = np.array([math.fsum(flat[:, e]) for e in range(flat.shape[1])])
    return planes, exact, np.abs(flat).sum(axis=0)


def _plane_of(a):
    return np.ascontiguousarray(a.reshape(len(a), -1).T)


def _total_run(cache, tag, spelling, p, target, form="auto"):
    kernel = total_kernel(spelling, p, TOTAL_ROWS, form)
    bundle = _build_sum(kernel, cache / f"tot_{tag}_{spelling}_{target}", target, form)
    planes, _, _ = _totals(spelling, p)
    size = p * p if spelling in ("outer", "gram") else p
    return _run_sum(bundle, kernel, _only(kernel, planes), {"out": np.zeros((size, 1))}, target)["out"][:, 0]


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_a_product_sum_over_every_lane_at_p71_lowers_to_one_block_sum_of_one_key(spelling):
    kernel = total_kernel(spelling, 71, TOTAL_ROWS)
    assert [type(s) for s in kernel.sinks] == [BlockSumWrite]
    found = stages(kernel.name, kernel.sinks)
    assert [s.team is not None for s in found] == [False, True]
    (step,) = found[1].team.steps
    assert (step.op, step.n, step.k, step.m) == (
        "block_sum", 71, 71 if spelling in ("outer", "gram") else 1, TOTAL_ROWS if spelling in ("gram", "moment") else 1)
    assert found[1].team.keyed.kept == ()


def test_a_small_or_thread_form_sum_over_every_lane_keeps_the_per_entry_path():
    assert not any(isinstance(s, BlockSumWrite) for s in total_kernel("outer", 4).sinks)
    assert not any(isinstance(s, BlockSumWrite) for s in total_kernel("gram", 71, TOTAL_ROWS, "thread").sinks)


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("target", [pytest.param("host", id="host"), pytest.param("cuda", marks=DEVICE, id="cuda")])
def test_a_sum_over_every_lane_matches_the_exact_sum(cache, spelling, target):
    _, exact, scale = _totals(spelling, 71)
    got = _total_run(cache, "val", spelling, 71, target)
    assert np.all(np.abs(got - exact) <= SUM_BOUND * EPS * scale + 1e-300)


@pytest.mark.parametrize("target", [pytest.param("host", id="host"), pytest.param("cuda", marks=DEVICE, id="cuda")])
def test_a_sum_over_every_lane_never_changes_a_bit_with_the_team_size(cache, target):
    for spelling in ("outer", "gram"):
        runs = []
        for threads in ((1, 8) if target == "host" else (64, 128, 256)):
            os.environ["HAWK_TEAM_THREADS"] = str(threads)
            try:
                runs.append(_total_run(cache, f"bit{threads}", spelling, 71, target))
            finally:
                os.environ.pop("HAWK_TEAM_THREADS")
        assert all(run.tobytes() == runs[0].tobytes() for run in runs[1:])


@pytest.mark.parametrize("split", (True, False))
def test_the_host_runtime_refuses_a_plane_name_the_kernel_does_not_declare(cache, split):
    """As eagle's bind does: a plane under a wrong name is never written, so it is refused by name."""
    p = 24
    kernel = stored_and_read_kernel(p, ROWS) if split else gram_kernel(p, ROWS)
    name = "gram" if split else "out"
    bundle = _build_sum(kernel, cache / f"names_{split}", "host")
    j, _ = _rows(p)
    loaded = rt.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name))
    outs = {name: np.zeros((p, p, KEYS)), **({"x": np.zeros(KEYS * ROWS)} if split else {})}
    with pytest.raises(HawkError, match=r"\['grams'\] are not planes it declares"):
        rt.run(loaded, j=np.ascontiguousarray(j.T), grams=np.zeros((p, p, KEYS)), **outs)
    rt.run(loaded, j=np.ascontiguousarray(j.T), **outs)                     # the declared names run


@pytest.mark.parametrize("target", [pytest.param("host", id="host"), pytest.param("cuda", marks=DEVICE, id="cuda")])
def test_a_transpose_product_sum_under_an_index_set_sums_the_live_lanes_rows(cache, target):
    """Each live lane owns its ``TOTAL_ROWS`` rows: the sum over the listed lanes of a key equals the dense one over them."""
    from hawk.ext import IndexSet, Kind

    p = 24
    lanes, rng = KEYS * ROWS, np.random.default_rng(13)
    mat = rng.standard_normal((lanes, TOTAL_ROWS, p))
    live = np.array([0, 2, 3, 5, 6, 7, 8, 12, 14], dtype=np.int32)
    kernel = keyed_transpose_kernel(p, TOTAL_ROWS, ROWS, Kind("lane_set", lanes=IndexSet()))
    bundle = build_bundle([kernel], cache / f"tgram_set_{target}", targets=(target if target == "host" else "cuda",),
                          host_profile="native")
    planes = {"mat": _plane_of(mat), "active_map": live, "active_count": np.array([len(live)], np.uint32)}
    out = {"out": np.zeros((p, p, KEYS))}
    if target == "host":
        rt.run(rt.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name)), **planes, **out)
    else:
        import cupy as cp
        from _team_stages import run_device

        dev = {k: cp.asarray(v) for k, v in {**planes, **out}.items()}
        run_device(bundle, kernel.name, dev, lanes)
        out = {"out": dev["out"].get()}
    want = np.zeros((p, p, KEYS))
    for k in range(KEYS):
        rows = mat[[i for i in live if i // ROWS == k]].reshape(-1, p)
        want[:, :, k] = rows.T @ rows
    assert np.abs(out["out"] - want).max() <= SUM_BOUND * EPS * np.abs(want).max() * ROWS * TOTAL_ROWS


@pytest.mark.parametrize("target", [pytest.param("host", id="host"), pytest.param("cuda", marks=DEVICE, id="cuda")])
def test_a_team_stage_chain_is_captured_and_replayed_through_eagle_bitwise(cache, target):
    """An eagle plan launches a team stage with one block per work item, captures the stage chain in
    one graph and replays it: three replays equal three fresh runs, bit for bit."""
    import eagle
    import eagle.plan as eplan

    p, rows, keys = 33, 40, 2
    n = keys * rows
    rng = np.random.default_rng(3)
    j, r = rng.standard_normal((n, p)), rng.standard_normal(n)
    (plan,) = eplan.auto([step_kernel(p, rows)], targets=(target,), cache_dir=str(cache / f"a5_{target}"))
    if target == "host":
        xp = np
    else:
        import cupy as xp

    def planes():           # lane-shaped (width, key, row): a two-axis Grid reads it unambiguously
        return {"j": xp.asarray(np.ascontiguousarray(j.T).reshape(p, keys, rows)), "r": xp.asarray(r),
                "x": xp.zeros((p, keys, rows))}

    def got(plane):
        return np.array(plane.get() if hasattr(plane, "get") else plane)

    fresh = []
    for _ in range(3):
        run = planes()
        plan.bind(**run).launch()
        fresh.append(got(run["x"]))
    assert fresh[0].tobytes() == fresh[1].tobytes() == fresh[2].tobytes()
    run = planes()
    graph = eagle.Graph().add(plan.bind(**run), name="step").build()
    for _ in range(3):
        run["x"][...] = 0
        graph.launch()
        assert got(run["x"]).tobytes() == fresh[0].tobytes()
    for k in range(keys):
        jk, rk = j[k * rows:(k + 1) * rows], r[k * rows:(k + 1) * rows]
        want = np.linalg.solve(jk.T @ jk, jk.T @ rk)
        assert np.abs(fresh[0][:, k, 0] - want).max() <= 100 * p * EPS * np.abs(want).max() * np.linalg.cond(jk.T @ jk)
