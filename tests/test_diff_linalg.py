# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The derivatives of the per-lane linear-algebra kinds.

Checked four ways: against central differences (``test_diff_finite_differences.py``
holds one row per kind; here the sizes and the hard matrices), against the
algebraic identity ``<bar, jvp(E)> == <vjp(bar), E>`` that a reverse rule must
meet to rounding, against closed forms, and as COMPILED derivative kernels run
on the host. A failed lane has the defined output zero, so its derivatives are
exactly zero, never NaN; a derivative holds one factorisation per call site.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from _eval import evaluate
from _linalg_kernels import chol_from_kernel, chol_kernel, eigh_kernel, lu_kernel
from conftest import sidecar_of

from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp
from hawk.ir import Assign, Leaf
from hawk.ir import make as mk
from hawk.ir.linalg import FACTORISATIONS, as_matrix
from hawk.ir.walk import canonical_nodes
from hawk.trace.kernel import Kernel
from hawk.types import TensorType

# A failed or degenerate lane still evaluates the branch its mask discards.
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

F64 = "f64"
RNG = np.random.default_rng(20261025)


def _t(*shape):
    return TensorType(shape, F64)


def _leaf(name, *shape):
    return Leaf("vocab_read", {0: "per_sample", 1: "vec_in", 2: "mat_in"}[len(shape)], name,
                _t(*shape))


def _sym(m):
    return (m + m.T) / 2


def _spd(n, eigs):
    q, _ = np.linalg.qr(RNG.standard_normal((n, n)))
    return q @ np.diag(eigs) @ q.T


def _factor_kernel(n):
    """``L = chol(A)`` as the only output."""
    a = _leaf("A", n, n)
    return Kernel(f"la_chol_factor_{n}", (Assign("L", as_matrix(mk("chol", (a,))), _t(n, n)),))


def _inputs(kernel, n, a):
    env = {"A": a, "b": RNG.standard_normal(n), "B": RNG.standard_normal((n, 2))}
    return {name: env[name] for name in ("A", "b", "B")}, env


def _general(n):
    return RNG.standard_normal((n, n)) + 2.0 * np.sqrt(n) * np.eye(n)


def _outputs(kernel):
    return [(s.name, s.ttype.shape) for s in kernel.sinks if s.ttype.dtype not in ("int32",)]


def _random_like(shape):
    return RNG.standard_normal(shape) if shape else float(RNG.standard_normal())


def _inner(x, y):
    return float(np.sum(np.asarray(x) * np.asarray(y)))


def _built_factor(n):
    """A lower triangular ``L`` with a positive diagonal, and garbage above it."""
    return np.tril(RNG.standard_normal((n, n)), -1) + np.diag(1.0 + RNG.random(n)) \
        + 0.5 * np.triu(RNG.standard_normal((n, n)), 1)


KERNELS = {"lu": (lu_kernel, _general), "chol": (chol_kernel, lambda n: _spd(n, np.linspace(1, 4, n))),
           "eigh": (eigh_kernel, _general), "chol_from": (chol_from_kernel, _built_factor)}


# --- the reverse rule is the transpose of the forward rule ----------------------

@pytest.mark.parametrize("n", [1, 2, 3, 8, 16])
def test_the_cholesky_factor_vjp_is_the_adjoint_of_its_jvp(n):
    kernel = _factor_kernel(n)
    a = _spd(n, np.linspace(1, 50, n))
    e = RNG.standard_normal((n, n))                     # NOT symmetric
    lbar = np.tril(RNG.standard_normal((n, n)))
    forward = evaluate(jvp(kernel), {"A": a, "dot_A": e})["dot_L"]
    reverse = evaluate(vjp(kernel), {"A": a, "bar_L": lbar})["bar_A"]
    assert _inner(lbar, forward) == pytest.approx(_inner(reverse, e), rel=1e-13, abs=1e-14)


@pytest.mark.parametrize("which", sorted(KERNELS))
@pytest.mark.parametrize("n", [1, 3, 5])
def test_every_kernels_vjp_is_the_adjoint_of_its_jvp(which, n):
    make, matrix = KERNELS[which]
    kernel = make(n)
    inputs, env = _inputs(kernel, n, matrix(n))
    tangents = {f"dot_{k}": _random_like(np.shape(v)) for k, v in inputs.items()}
    cotangents = {f"bar_{name}": _random_like(shape) for name, shape in _outputs(kernel)}
    forward = evaluate(jvp(kernel), {**env, **tangents})
    reverse = evaluate(vjp(kernel), {**env, **cotangents})
    lhs = sum(_inner(c, forward[f"dot_{name}"]) for name, c in
              ((k[4:], v) for k, v in cotangents.items()))
    rhs = sum(_inner(reverse[f"bar_{k}"], tangents[f"dot_{k}"]) for k in inputs
              if f"bar_{k}" in reverse)
    assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("which", sorted(KERNELS))
@pytest.mark.parametrize("n", [1, 2, 5])
def test_every_kernels_jvp_matches_central_differences(which, n):
    make, matrix = KERNELS[which]
    kernel = make(n)
    inputs, env = _inputs(kernel, n, matrix(n))
    tangents = {k: _random_like(np.shape(v)) for k, v in inputs.items()}
    got = evaluate(jvp(kernel), {**env, **{f"dot_{k}": t for k, t in tangents.items()}})
    h = 1e-6
    plus = evaluate(kernel.sinks, {**env, **{k: v + h * tangents[k] for k, v in inputs.items()}})
    minus = evaluate(kernel.sinks, {**env, **{k: v - h * tangents[k] for k, v in inputs.items()}})
    for name, _ in _outputs(kernel):
        expect = (np.asarray(plus[name]) - np.asarray(minus[name])) / (2 * h)
        np.testing.assert_allclose(got[f"dot_{name}"], expect, rtol=1e-6, atol=1e-7)


# --- the Cholesky factor where it is hard ---------------------------------------

def _steps(eigs, e):
    """The central-difference step along ``e``: 1e-3 lambda_min, so both sides stay
    inside the SPD cone (a side that leaves it returns the 0-fill), and no more
    than 1e-6."""
    return min(1e-6, 1e-3 * eigs.min() / np.abs(e).max())


@pytest.mark.parametrize("n", [3, 8])
@pytest.mark.parametrize("name,eigs,tol", [
    ("well", lambda n: np.linspace(1, 10, n), 1e-8),
    ("cond1e8", lambda n: np.logspace(0, -8, n), 1e-5),
    ("near_semidefinite", lambda n: np.r_[np.ones(n - 1), 1e-10], 1e-2)])
def test_the_cholesky_factor_jvp_matches_fd_with_h_from_lambda_min(n, name, eigs, tol):
    spectrum = eigs(n)
    a = _spd(n, spectrum)
    e = RNG.standard_normal((n, n))
    h = _steps(spectrum, e)
    kernel = _factor_kernel(n)
    sides = [evaluate(kernel.sinks, {"A": a + s * h * e}) for s in (1, -1)]
    assert all(np.any(side["L"]) for side in sides), "a side left the cone"
    expect = (sides[0]["L"] - sides[1]["L"]) / (2 * h)
    got = evaluate(jvp(kernel), {"A": a, "dot_A": e})["dot_L"]
    assert np.abs(got - expect).max() <= tol * np.abs(expect).max()
    assert np.isfinite(got).all()


@pytest.mark.parametrize("n", [1, 3, 8, 32])
@pytest.mark.parametrize("spectrum", ["linear", "near_semidefinite"])
def test_the_cholesky_factor_closed_form_cotangent_is_half_q(n, spectrum):
    """``Lbar = Q L`` gives ``Abar = Q / 2`` exactly: the deterministic row of
    the pathwise Gaussian gradient ``d E[f] / d Sigma`` for ``f = x^T Q x / 2``."""
    rng = np.random.default_rng(20261026 + n)           # this row's own draw, whatever ran before
    eigs = np.linspace(1, 3, n) if spectrum == "linear" else np.r_[np.ones(n - 1), 1e-10]
    basis, _ = np.linalg.qr(rng.standard_normal((n, n)))
    sigma = basis @ np.diag(eigs) @ basis.T
    q = _sym(rng.standard_normal((n, n)))
    lower = np.linalg.cholesky(sigma)
    got = evaluate(vjp(_factor_kernel(n)), {"A": sigma, "bar_L": q @ lower})["bar_A"]
    # The rounding in Sigma and in L is amplified by the condition number of Sigma, so a
    # spectrum down to 1e-10 (cond 1e10) can only be met to ~ 10 eps cond; the other
    # spectrum is well conditioned and keeps the absolute 1e-12.
    cond = eigs.max() / eigs.min()
    tol = 1e-12 if spectrum == "linear" else max(1e-12, 10 * np.finfo(float).eps * cond)
    assert np.abs(got - q / 2).max() <= tol * max(1.0, np.abs(q).max())


@pytest.mark.parametrize("k", [3, 9, 15, 21, 27, 33])
def test_the_derivative_grows_as_the_boundary_is_approached_and_stays_finite(k):
    """``A = [[1, 1], [1, 1 + d]]`` has ``L22 = sqrt(d)``, so ``dL22/dA22`` is
    ``1 / (2 sqrt(d))``: large, finite, with ``info == 0``; no clamping."""
    d = 2.0 ** -k                                       # exact in 1 + d
    a = np.array([[1.0, 1.0], [1.0, 1.0 + d]])
    kernel = _factor_kernel(2)
    got = evaluate(jvp(kernel), {"A": a, "dot_A": np.array([[0.0, 0.0], [0.0, 1.0]])})["dot_L"]
    assert np.isfinite(got).all()
    assert got[1, 1] == pytest.approx(0.5 / np.sqrt(d), rel=1e-9)
    assert evaluate(chol_kernel(2).sinks, {"A": a, "b": np.zeros(2), "B": np.zeros((2, 2))})["info"] == 0


def test_a_non_symmetric_input_is_the_factor_of_its_symmetric_part():
    n = 4
    a = _spd(n, np.linspace(1, 3, n)) + 0.3 * RNG.standard_normal((n, n))
    kernel = _factor_kernel(n)
    both = evaluate(kernel.sinks, {"A": a})["L"], evaluate(kernel.sinks, {"A": _sym(a)})["L"]
    assert np.array_equal(*both)


# --- failed lanes: the derivative of a defined constant is zero ------------------

def _all_zero(values):
    return all(np.all(np.asarray(v) == 0) and not np.any(np.isnan(v)) for v in values)


def test_a_failed_cholesky_lane_has_zero_tangents_and_adjoints():
    n = 3
    kernel = chol_kernel(n)
    a = np.diag([1.0, -1.0, 1.0])
    env = {"A": a, "b": RNG.standard_normal(n), "B": RNG.standard_normal((n, 2))}
    assert evaluate(kernel.sinks, env)["info"] == 2
    forward = evaluate(jvp(kernel), {**env, "dot_A": RNG.standard_normal((n, n)),
                                     "dot_b": RNG.standard_normal(n),
                                     "dot_B": RNG.standard_normal((n, 2))})
    assert _all_zero(forward[f"dot_{k}"] for k in ("x", "X", "L"))
    reverse = evaluate(vjp(kernel), {**env, "bar_x": RNG.standard_normal(n),
                                     "bar_X": RNG.standard_normal((n, 2)),
                                     "bar_L": RNG.standard_normal((n, n))})
    assert _all_zero(reverse[f"bar_{k}"] for k in ("A", "b", "B"))


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_a_failed_built_factor_lane_has_zero_tangents_and_adjoints(bad):
    n = 3
    kernel = chol_from_kernel(n)
    low = _built_factor(n)
    low[1, 1] = bad
    env = {"A": low, "b": RNG.standard_normal(n), "B": RNG.standard_normal((n, 2))}
    got = evaluate(kernel.sinks, env)
    assert got["info"] == 2 and _all_zero([got["x"], got["X"], got["L"]])
    forward = evaluate(jvp(kernel), {**env, "dot_A": RNG.standard_normal((n, n)),
                                     "dot_b": RNG.standard_normal(n),
                                     "dot_B": RNG.standard_normal((n, 2))})
    assert _all_zero(forward[f"dot_{k}"] for k in ("x", "X", "L"))
    reverse = evaluate(vjp(kernel), {**env, "bar_x": RNG.standard_normal(n),
                                     "bar_X": RNG.standard_normal((n, 2)),
                                     "bar_L": RNG.standard_normal((n, n))})
    assert _all_zero(reverse[f"bar_{k}"] for k in ("A", "b", "B"))


def test_a_built_factor_solves_with_its_product_and_ignores_above_the_diagonal():
    n = 5
    low = _built_factor(n)
    b = RNG.standard_normal(n)
    got = evaluate(chol_from_kernel(n).sinks, {"A": low, "b": b, "B": np.zeros((n, 2))})
    tril = np.tril(low)
    np.testing.assert_allclose(got["x"], np.linalg.solve(tril @ tril.T, b), rtol=1e-12)
    np.testing.assert_array_equal(got["L"], tril)
    assert got["info"] == 0


def test_a_failed_lu_lane_has_zero_tangents_and_adjoints():
    n = 3
    kernel = lu_kernel(n)
    a = RNG.standard_normal((n, n))
    a[1] = 0.0
    env = {"A": a, "b": RNG.standard_normal(n), "B": RNG.standard_normal((n, 2))}
    assert evaluate(kernel.sinks, env)["info"] > 0
    forward = evaluate(jvp(kernel), {**env, "dot_A": RNG.standard_normal((n, n)),
                                     "dot_b": RNG.standard_normal(n),
                                     "dot_B": RNG.standard_normal((n, 2))})
    assert _all_zero(forward[f"dot_{k}"] for k in ("x", "X", "det"))
    reverse = evaluate(vjp(kernel), {**env, "bar_x": RNG.standard_normal(n),
                                     "bar_X": RNG.standard_normal((n, 2)), "bar_det": 1.5})
    assert _all_zero(reverse[f"bar_{k}"] for k in ("A", "b", "B"))


def test_a_degenerate_eigh_lane_sets_bit_two_and_zeroes_only_the_vector_derivatives():
    n = 3
    kernel = eigh_kernel(n)
    a = np.diag([1.0, 1.0, 2.0])
    e = RNG.standard_normal((n, n))
    assert evaluate(kernel.sinks, {"A": a})["info"] & 2
    forward = evaluate(jvp(kernel), {"A": a, "dot_A": e})
    assert _all_zero([forward["dot_V"]])
    np.testing.assert_allclose(forward["dot_w"], np.diag(_sym(e)), atol=1e-14)   # V = I here
    reverse = evaluate(vjp(kernel), {"A": a, "bar_V": RNG.standard_normal((n, n)),
                                     "bar_w": np.zeros(n)})
    assert _all_zero([reverse["bar_A"]])


# --- one factorisation per call site --------------------------------------------

def _factorisations(sinks):
    nodes, _ = canonical_nodes(tuple(sinks))
    return sorted(n.kind for n in nodes if getattr(n, "kind", None) in FACTORISATIONS)


@pytest.mark.parametrize("which,kind", [("lu", "lu"), ("chol", "chol"), ("eigh", "eigh"),
                                        ("chol_from", "chol_from")])
@pytest.mark.parametrize("derive", [vjp, jvp])
def test_a_derivative_never_refactors(which, kind, derive):
    kernel = KERNELS[which][0](4)
    assert _factorisations(kernel.sinks) == [kind]
    assert _factorisations(derive(kernel)) == [kind]


@pytest.mark.parametrize("derive", [vjp, jvp])
def test_two_call_sites_hold_two_factorisations(derive):
    a, c, b = _leaf("A", 3, 3), _leaf("C", 3, 3), _leaf("b", 3)
    out = mk("add", (mk("lu_solve", (mk("lu", (a,)), b)), mk("cho_solve", (mk("chol", (c,)), b))))
    sinks = (Assign("out", out, _t(3)),)
    assert _factorisations(sinks) == ["chol", "lu"]
    assert _factorisations(derive(sinks)) == ["chol", "lu"]


# --- the pathwise Gaussian sample -------------------------------------------------

def _mvn(n):
    """``x = mean + chol(cov) z``: the composition ``m.mvn_sample`` is, with ``z``
    a plane standing for the counter generator's draws (independent of every
    differentiated input, so it carries no derivative)."""
    mean, cov, z = _leaf("mu", n), _leaf("S", n, n), _leaf("z", n)
    return (Assign("x", mk("add", (mean, mk("mv", (mk("chol", (cov,)), z)))), _t(n)),)


def test_the_pathwise_sample_differentiates_both_ways_to_the_closed_forms():
    n, count = 3, 4000
    sigma = _spd(n, np.array([1.0, 2.0, 3.0]))
    mu = RNG.standard_normal(n)
    q = _sym(RNG.standard_normal((n, n)))
    sinks = _mvn(n)
    reverse = vjp(sinks, wrt=("mu", "S"))
    grads_mu, grads_sigma, covariance = [], [], np.zeros((n, n))
    for z in np.random.default_rng(7).standard_normal((count, n)):
        env = {"mu": mu, "S": sigma, "z": z}
        x = evaluate(sinks, env)["x"]
        covariance += np.outer(x - mu, x - mu) / count
        got = evaluate(reverse, {**env, "bar_x": q @ x})            # f = x^T Q x / 2
        grads_mu.append(got["bar_mu"])
        grads_sigma.append(got["bar_S"])
    for samples, expect in ((np.array(grads_mu), q @ mu), (np.array(grads_sigma), q / 2)):
        error = samples.std(axis=0, ddof=1) / np.sqrt(count)
        assert np.all(np.abs(samples.mean(axis=0) - expect) <= 5 * error)
    assert np.abs(covariance - sigma).max() <= 5 * np.abs(sigma).max() / np.sqrt(count) * 2
    # the sample is a pure function of z: the tangents of mean and cov add up
    z = RNG.standard_normal(n)
    env = {"mu": mu, "S": sigma, "z": z}
    e, dmu = RNG.standard_normal((n, n)), RNG.standard_normal(n)
    got = evaluate(jvp(sinks, wrt=("mu", "S")), {**env, "dot_mu": dmu, "dot_S": e})["dot_x"]
    h = 1e-6 * np.linalg.eigvalsh(sigma).min()
    plus, minus = (evaluate(sinks, {"mu": mu + s * h * dmu, "S": sigma + s * h * e, "z": z})["x"]
                   for s in (1, -1))
    np.testing.assert_allclose(got, (plus - minus) / (2 * h), rtol=1e-6, atol=1e-7)


# --- compiled derivative kernels, run on the host ---------------------------------

LANES = 8
BAD = 3


@pytest.fixture(scope="module")
def host(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("la_diff_artifacts")

    def build(primal, derived, tag):
        home = build_bundle([primal], pathlib.Path(root) / f"{primal.name}_primal",
                            targets=("host",), cache_dir=cache_dir, host_profile="native")
        kernel = Kernel(f"{primal.name}_{tag}", derived)
        bundle = build_bundle([kernel], pathlib.Path(root) / kernel.name,
                              targets=("host",), cache_dir=cache_dir, host_profile="native",
                              derivative={kernel.name: (primal.name, home.digest)})
        return kernel, rt.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name))
    return build


def _lanes(which, n):
    """``LANES`` matrices; lane ``BAD`` is a failing one for ``which``."""
    make, matrix = KERNELS[which]
    mats = np.stack([matrix(n) for _ in range(LANES)])
    if which == "lu":
        mats[BAD, min(1, n - 1)] = 0.0
    elif which in ("chol", "chol_from"):
        mats[BAD, n // 2, n // 2] = -1.0
    elif n > 1:
        mats[BAD] = np.eye(n) * np.r_[np.ones(n - 1), 2.0]
    return mats


def _plane(a):
    plane = np.ascontiguousarray(a.reshape(LANES, -1).T)
    return plane[0] if plane.shape[0] == 1 else plane


def _run(compiled, planes, outs):
    sink = {name: np.zeros(LANES if width == 1 else (width, LANES)) for name, width in outs.items()}
    declared = {name for _, name in compiled.arg_spec}
    rt.run(compiled, **{k: _plane(v) for k, v in planes.items() if k in declared}, **sink)
    return {name: plane.reshape(outs[name], LANES).T for name, plane in sink.items()}


@pytest.mark.parametrize("derive", [vjp, jvp], ids=["vjp", "jvp"])
@pytest.mark.parametrize("which", sorted(KERNELS))
@pytest.mark.parametrize("n", [1, 5])
def test_the_compiled_derivative_matches_the_interpreter_and_zeroes_failed_lanes(
        host, which, n, derive):
    primal = KERNELS[which][0](n)
    derived = derive(primal)
    kernel, compiled = host(primal, derived, derive.__name__)
    mats = _lanes(which, n)
    inputs = {"A": mats, "b": RNG.standard_normal((LANES, n)), "B": RNG.standard_normal((LANES, n, 2))}
    if derive is vjp:
        seeds = {f"bar_{name}": RNG.standard_normal((LANES, *shape))
                 for name, shape in _outputs(primal)}
    else:
        seeds = {f"dot_{k}": RNG.standard_normal(v.shape) for k, v in inputs.items()}
    outs = {s.name: int(np.prod(s.ttype.shape or (1,))) for s in kernel.sinks}
    got = _run(compiled, {**inputs, **seeds}, outs)
    for lane in range(LANES):
        env = {k: v[lane] for k, v in {**inputs, **seeds}.items()}
        want = evaluate(derived, env)
        for name, width in outs.items():
            reference = np.asarray(want[name], dtype=float).reshape(width)
            scale = max(1.0, np.abs(reference).max())
            np.testing.assert_allclose(got[name][lane], reference, rtol=0, atol=1e-8 * scale,
                                       err_msg=f"{name} lane {lane}")
        if lane == BAD and which != "eigh":
            assert all(np.all(got[name][lane] == 0) for name in outs), "a failed lane's derivative"
