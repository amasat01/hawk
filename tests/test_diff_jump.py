# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``@kernel(jump=...)``: the derivative through a parameter-dependent branch.

A switch ``if x < t`` passes ``K(x - t) * ybar * (a - b)`` to ``t`` (Epanechnikov
``K``, standard deviation ``h``, half-width ``sqrt(5) h``). Rows:

* **Oracle.** The compiled host derivative equals the numpy formula for every
  ordered comparison, ``and``/``or``, a nested ``if`` and a vector comparison;
  the branch values' gradients are bitwise those of the kernel without ``jump=``.
* **Smoothed loss.** The gradient matches a central difference of the
  kernel-smoothed loss, and plain gradient descent finds the switch.
* **Transpose.** ``jvp`` is ``vjp``'s transpose.
* **Off by default.** ``jump=`` leaves the primal's digest alone, and a kernel
  without a switching comparison derives exactly as without it.
* **Width Param.** ``jump="h"`` equals the same fixed width bitwise, and commits
  the per-site spread cells.
* **Refusals** and the emitted ``switch`` on ``|d| < w``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import hawk
from hawk import HawkError, Kernel, Mutable, Param, Scalar, Terminated, Vector, kernel
from hawk.diff import jvp, vjp
from hawk.math import floor, land, lor, real, tanh, where

H = 0.04
LOOP, STEP = 4, 0.25
W = math.sqrt(5.0) * H
A, B, T, TSTAR = -0.5, 0.8, 0.4, 0.3


def K(d, h=H):
    w = math.sqrt(5.0) * h
    return np.where(np.abs(d) < w, 0.75 / w * (1.0 - (d / w) ** 2), 0.0)


def F(z, h=H):
    """The CDF of ``K``: the smoothed step."""
    u = np.clip(z / (math.sqrt(5.0) * h), -1.0, 1.0)
    return 0.5 + 0.75 * u - 0.25 * u ** 3


# --------------------------------------------------------------------------- #
# -- kernels
# --------------------------------------------------------------------------- #
@kernel(jump=H)
def k_lt(x: Scalar, t: Param, a: Param, b: Param, y: Mutable[Scalar]):
    if x < t:
        y = a
    else:
        y = b


@kernel
def k_lt_off(x: Scalar, t: Param, a: Param, b: Param, y: Mutable[Scalar]):
    if x < t:
        y = a
    else:
        y = b


@kernel(jump=H)
def k_le(x: Scalar, t: Param, a: Param, b: Param, y: Mutable[Scalar]):
    y = a if x <= t else b


@kernel(jump=H)
def k_gt(x: Scalar, t: Param, a: Param, b: Param, y: Mutable[Scalar]):
    y = where(x > t, a, b)


@kernel(jump=H)
def k_ge(x: Scalar, t: Param, a: Param, b: Param, y: Mutable[Scalar]):
    if x >= t:
        y = a
    else:
        y = b


@kernel(jump=H)
def k_and(x: Scalar, t: Scalar, u: Scalar, v: Scalar, y: Mutable[Scalar]):
    if land(x < t, u < v):
        y = A * x
    else:
        y = B * t


@kernel(jump=H)
def k_or(x: Scalar, t: Scalar, u: Scalar, v: Scalar, y: Mutable[Scalar]):
    if lor(x < t, u < v):
        y = A * x
    else:
        y = B * t


@kernel(jump=H)
def k_nested(x: Scalar, t: Param, u: Scalar, v: Scalar, y: Mutable[Scalar]):
    if u < v:
        if x < t:
            y = A
        else:
            y = B
    else:
        y = 2.0


@kernel(jump=H)
def k_vec(x: Vector[3], t: Param, y: Mutable[Vector[3]]):
    y = where(x < t, x + A, x - B)


@kernel(jump=H)
def k_three(x: Scalar, t: Param, y1: Mutable[Scalar], y2: Mutable[Scalar],
            y3: Mutable[Scalar]):
    if x < t:
        y1, y2, y3 = A, 2.0 * A, 3.0 * A
    else:
        y1, y2, y3 = B, 2.0 * B, 3.0 * B


@kernel(jump="h")
def k_param(x: Scalar, t: Param, a: Param, b: Param, h: Param, y: Mutable[Scalar]):
    if x < t:
        y = a
    else:
        y = b


@kernel(jump=H)
def k_const_cmp(x: Scalar, c: Scalar, y: Mutable[Scalar]):
    if c < 0.5:
        y = 2.0 * x
    else:
        y = 3.0 * x


@kernel
def k_const_cmp_off(x: Scalar, c: Scalar, y: Mutable[Scalar]):
    if c < 0.5:
        y = 2.0 * x
    else:
        y = 3.0 * x


@kernel(jump=H)
def k_loop_carried(x: Scalar, t: Param, y: Mutable[Scalar]):
    total = 0.0
    xs = x
    for _k in range(LOOP):
        if xs < t:
            total = total + A
        else:
            total = total + B
        xs = xs + STEP
    y = total


@kernel(jump=H)
def k_loop_sum(x: Scalar, t: Param, y: Mutable[Scalar]):
    total = 0.0
    for k in range(LOOP):
        total = total + where(x + STEP * real(k) < t, A, B)
    y = total


#: a KAN edge cell as a downstream package writes it: per-sample inputs, a
#: lowered loop over edges, the level-0 basis an indicator of the knot span
N_IN, GK, KO = 3, 5, 3
KN = tuple(-1.0 + (j - KO) * 2.0 / GK for j in range(GK + 2 * KO + 1))
NB = GK + KO
THETA = hawk.Table["edge":N_IN, "basis":NB]


def _spline(xe, theta, e):
    """One edge's spline (a plain helper: its comprehensions run at trace time)."""
    level = [where(land(xe >= KN[i], xe < KN[i + 1]), 1.0, 0.0)
             for i in range(len(KN) - 1)]
    for d in range(1, KO + 1):
        level = [((xe - KN[i]) / (KN[i + d] - KN[i])) * level[i]
                 + ((KN[i + d + 1] - xe) / (KN[i + d] - KN[i])) * level[i + 1]
                 for i in range(len(level) - 1)]
    spline = level[0] * theta.at(edge=e, basis=0)
    for b in range(1, NB):
        spline = spline + level[b] * theta.at(edge=e, basis=b)
    return spline


@kernel(jump=0.02)
def k_kan(x: Vector[N_IN], theta: THETA, g: Param,
          y: Mutable[Scalar]):
    total = 0.0
    for e in range(N_IN):
        total = total + tanh(_spline(x[e] * g, theta, e))
    y = total


def kan_dual(x, theta, g, h=0.02):
    """``(y, dy/dg)`` in numpy forward mode: the piecewise derivative, every
    indicator's tangent zero."""
    tot, dtot = 0.0, 0.0
    for e in range(N_IN):
        xe, dxe = x[e] * g, x[e]
        lev = [(((xe >= KN[i]) & (xe < KN[i + 1])) * 1.0, 0.0 * xe)
               for i in range(len(KN) - 1)]
        for d in range(1, KO + 1):
            nxt = []
            for i in range(len(lev) - 1):
                den = KN[i + d] - KN[i]
                lo, hi = (xe - KN[i]) / den, (KN[i + d + 1] - xe) / den
                nxt.append((lo * lev[i][0] + hi * lev[i + 1][0],
                            dxe / den * lev[i][0] + lo * lev[i][1]
                            - dxe / den * lev[i + 1][0] + hi * lev[i + 1][1]))
            lev = nxt
        s = sum(lev[b][0] * theta[e * NB + b] for b in range(NB))
        ds = sum(lev[b][1] * theta[e * NB + b] for b in range(NB))
        tot, dtot = tot + np.tanh(s), dtot + (1.0 - np.tanh(s) ** 2) * ds
    return tot, dtot


def kan_flips(x, theta, g, h=0.02):
    """The per-lane EXPECTATION of the alternate-path derivative: the piecewise
    derivative plus, for each of an edge's comparisons, ``K(d) sigma d' (L(c=1)
    - L(c=0))`` with the loss ``tanh`` of the spline recomputed with that one
    comparison forced (the other indicator comparisons keep their outcome)."""
    _, total = kan_dual(x, theta, g, h)
    for e in range(N_IN):
        xe, th = x[e] * g, theta[e * NB:(e + 1) * NB]
        ge = [xe >= KN[i] for i in range(len(KN) - 1)]
        lt = [xe < KN[i + 1] for i in range(len(KN) - 1)]

        def loss(ge_, lt_):
            level = [(a & b) * 1.0 for a, b in zip(ge_, lt_)]
            for d in range(1, KO + 1):
                level = [((xe - KN[i]) / (KN[i + d] - KN[i])) * level[i]
                         + ((KN[i + d + 1] - xe) / (KN[i + d] - KN[i])) * level[i + 1]
                         for i in range(len(level) - 1)]
            return np.tanh(sum(level[b] * th[b] for b in range(NB)))

        on, off = np.ones_like(xe, dtype=bool), np.zeros_like(xe, dtype=bool)
        for i in range(len(KN) - 1):
            for side, knot, flags in ((1.0, KN[i], ge), (-1.0, KN[i + 1], lt)):
                forced = []
                for value in (on, off):
                    held = list(flags)
                    held[i] = value
                    forced.append(loss(held, lt) if flags is ge else loss(ge, held))
                total = total + K(xe - knot, h) * side * x[e] * (forced[0] - forced[1])
    return total


# --------------------------------------------------------------------------- #
# -- the host route
# --------------------------------------------------------------------------- #
def _declared(kernel, arrays: dict) -> dict:
    """``arrays`` cut to the planes ``kernel`` declares: a derivative binds what its body reads (a
    primal input it does not read is no slot of it), and a run refuses a name that is none."""
    names = {n for _, n in kernel.arg_spec}
    return {k: v for k, v in arrays.items() if k in names}


@pytest.fixture(scope="module")
def rig(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("jump")
    built: dict = {}

    def unit(primal, direction, wrt):
        key = (primal.name, direction, wrt)
        if key not in built:
            d = root / f"{primal.name}_{direction}_{'_'.join(wrt)}"
            d.mkdir(exist_ok=True)
            derive = vjp if direction == "vjp" else jvp
            hawk.build([primal, Kernel("g", derive(primal, wrt=wrt))], d,
                       targets=("host",), cache_dir=cache_dir)
            built[key] = d
        return built[key]

    def grad(primal, wrt, seeds, **inputs):
        """``{name: bar}`` for ``wrt``, seeds ``{bar_<sink>: array}``."""
        d = unit(primal, "vjp", wrt)
        n = np.shape(next(iter(seeds.values())))[-1]
        bars = {f"bar_{w}": np.zeros(n) for w in wrt}
        arrs = {k: (float(v) if np.ndim(v) == 0 else np.asarray(v, dtype=float))
                for k, v in inputs.items()}
        kernel = hawk.load(d, "g")
        arrs = _declared(kernel, arrs)
        hawk.run(kernel, **seeds, **bars, **arrs)
        return {w: bars[f"bar_{w}"] for w in wrt}

    def tangent(primal, wrt, dots, out, **inputs):
        d = unit(primal, "jvp", wrt)
        n = next(np.shape(v)[-1] for v in dots.values() if np.ndim(v))
        res = {f"dot_{o}": np.zeros(n) for o in out}
        arrs = {k: (float(v) if np.ndim(v) == 0 else np.asarray(v, dtype=float))
                for k, v in inputs.items()}
        tans = {f"dot_{k}": (float(v) if np.ndim(v) == 0 else np.asarray(v, dtype=float))
                for k, v in dots.items()}
        kernel = hawk.load(d, "g")
        hawk.run(kernel, **tans, **res, **_declared(kernel, arrs))
        return {o: res[f"dot_{o}"] for o in out}

    def sidecar(primal, direction, wrt):
        d = unit(primal, direction, wrt)
        return json.loads((d / "g.json").read_text())

    def text(primal, direction, wrt):
        return (unit(primal, direction, wrt) / "g.cpp").read_text()

    return type("Rig", (), {"grad": staticmethod(grad), "tangent": staticmethod(tangent),
                            "sidecar": staticmethod(sidecar), "text": staticmethod(text),
                            "unit": staticmethod(unit)})


def _data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    return x


def _ybar(y, x, n):
    ystar = np.where(x < TSTAR, A, B)
    return 2.0 * (y - ystar) / n


def _rel(got, want):
    return abs(got - want) / max(abs(want), 1e-300)


# --------------------------------------------------------------------------- #
# -- the oracle
# --------------------------------------------------------------------------- #
#: comparison -> (primal, numpy predicate, sign of bar_t per K*ybar*(a-b))
_TABLE = {"lt": (k_lt, np.less, 1.0), "le": (k_le, np.less_equal, 1.0),
          "gt": (k_gt, np.greater, -1.0), "ge": (k_ge, np.greater_equal, -1.0)}


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n", (256, 4096))
@pytest.mark.parametrize("op", sorted(_TABLE))
def test_bar_t_is_the_kernel_weighted_jump(rig, op, n, seed):
    primal, pred, sign = _TABLE[op]
    x = _data(n, seed)
    y = np.where(pred(x, T), A, B)
    ybar = _ybar(y, x, n)
    g = rig.grad(primal, ("t", "a", "b"), {"bar_y": ybar}, x=x, t=T, a=A, b=B)
    want = sign * ybar * (A - B) * K(x - T)
    np.testing.assert_allclose(g["t"], want, rtol=1e-12, atol=1e-300)
    assert _rel(g["t"].sum(), want.sum()) <= 1e-12
    off = rig.grad(k_lt_off, ("t", "a", "b"), {"bar_y": ybar}, x=x, t=T, a=A, b=B)
    if op == "lt":
        assert np.array_equal(g["a"], off["a"]) and np.array_equal(g["b"], off["b"])
        assert not off["t"].any()        # jump-off: a switch has derivative zero
    else:
        assert np.array_equal(g["a"], np.where(pred(x, T), ybar, 0.0))
        assert np.array_equal(g["b"], np.where(pred(x, T), 0.0, ybar))


def test_bar_t_matches_the_smoothed_loss_difference(rig):
    n = 4096
    x = _data(n, 7)
    ystar = np.where(x < TSTAR, A, B)

    def loss(t):                     # L_h(t): the switch replaced by its CDF
        y = B + (A - B) * F(t - x)
        return np.mean((y - ystar) ** 2)

    y = np.where(x < T, A, B)
    g = rig.grad(k_lt, ("t",), {"bar_y": _ybar(y, x, n)}, x=x, t=T, a=A, b=B)
    step = 1e-5
    fd = (loss(T + step) - loss(T - step)) / (2 * step)
    assert _rel(g["t"].sum(), fd) < 0.02


def test_plain_descent_finds_the_switch(rig):
    n, t = 4096, -0.4
    x = _data(n, 11)
    for _ in range(200):
        y = np.where(x < t, A, B)
        t -= 0.1 * rig.grad(k_lt, ("t",), {"bar_y": _ybar(y, x, n)},
                            x=x, t=t, a=A, b=B)["t"].sum()
    assert abs(t - TSTAR) < 0.02


@pytest.mark.parametrize("seed", range(3))
def test_and_or_nested_and_vector_conditions(rig, seed):
    n = 512
    rng = np.random.default_rng(100 + seed)
    x = rng.uniform(-1, 1, n)
    t = x + rng.uniform(-0.1, 0.1, n)            # most samples near the switch
    u, v = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    ybar = rng.normal(size=n)
    jumped = (A * x - B * t) * K(x - t)            # (a - b) K, a = A x, b = B t
    g = rig.grad(k_and, ("t",), {"bar_y": ybar}, x=x, t=t, u=u, v=v)["t"]
    plain = np.where((x < t) & (u < v), 0.0, B * ybar)
    np.testing.assert_allclose(g, plain + ybar * jumped * (u < v), rtol=1e-12, atol=1e-15)
    g = rig.grad(k_or, ("t",), {"bar_y": ybar}, x=x, t=t, u=u, v=v)["t"]
    plain = np.where((x < t) | (u < v), 0.0, B * ybar)
    np.testing.assert_allclose(g, plain + ybar * jumped * (u >= v), rtol=1e-12, atol=1e-15)
    tp = 0.1
    g = rig.grad(k_nested, ("t",), {"bar_y": ybar}, x=x, t=tp, u=u, v=v)["t"]
    np.testing.assert_allclose(g, (u < v) * ybar * (A - B) * K(x - tp),
                               rtol=1e-12, atol=1e-300)
    xv = rng.uniform(-0.2, 0.2, (3, n))
    yb = rng.normal(size=(3, n))
    g = rig.grad(k_vec, ("t",), {"bar_y": yb}, x=xv, t=0.05)["t"]
    want = (yb * (A + B) * K(xv - 0.05)).sum(0)
    np.testing.assert_allclose(g, want, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("primal", (k_and, k_or))
def test_jvp_is_the_transpose_of_vjp(rig, primal, seed):
    n = 512
    rng = np.random.default_rng(200 + seed)
    x = rng.uniform(-1, 1, n)
    t = x + rng.uniform(-0.1, 0.1, n)            # most samples near the switch
    u, v = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    ybar = rng.normal(size=n)
    dx, dt = rng.normal(size=n), rng.normal(size=n)
    g = rig.grad(primal, ("x", "t"), {"bar_y": ybar}, x=x, t=t, u=u, v=v)
    dy = rig.tangent(primal, ("x", "t"), {"x": dx, "t": dt}, ("y",),
                     x=x, t=t, u=u, v=v)["y"]
    lhs, rhs = ybar * dy, g["x"] * dx + g["t"] * dt
    scale = np.abs(ybar * dy).sum() + np.abs(rhs).sum()
    assert np.all(np.abs(lhs - rhs) <= 1e-10 * (np.abs(lhs) + np.abs(rhs)) + 1e-300)
    assert abs(lhs.sum() - rhs.sum()) <= 1e-10 * scale


def test_vector_jvp_is_the_transpose_of_vjp(rig):
    n = 256
    rng = np.random.default_rng(300)
    x = rng.uniform(-0.1, 0.1, (3, n))
    yb = rng.normal(size=(3, n))
    bar_t = rig.grad(k_vec, ("t",), {"bar_y": yb}, x=x, t=0.01)["t"]
    d = rig.unit(k_vec, "jvp", ("t",))
    dot_y = np.zeros((3, n))
    hawk.run(hawk.load(d, "g"), dot_t=1.0, dot_y=dot_y, x=x, t=0.01)
    lhs = (yb * dot_y).sum(0)
    np.testing.assert_allclose(lhs, bar_t, rtol=1e-10, atol=1e-300)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("primal", (k_loop_carried, k_loop_sum))
def test_a_switch_in_a_loop_body_gets_its_jump(rig, primal, seed):
    n = 2048
    rng = np.random.default_rng(400 + seed)
    x = rng.uniform(-1.5, 0.5, n)
    ybar = rng.normal(size=n)
    g = rig.grad(primal, ("t",), {"bar_y": ybar}, x=x, t=T)["t"]
    want = ybar * sum((A - B) * K(x + STEP * k - T) for k in range(LOOP))
    np.testing.assert_allclose(g, want, rtol=1e-12, atol=1e-14)
    # the transpose: jvp in t equals vjp, and through the carried x as well
    gx = rig.grad(primal, ("x", "t"), {"bar_y": ybar}, x=x, t=T)
    np.testing.assert_allclose(gx["x"], -want, rtol=1e-12, atol=1e-14)
    dx = rng.normal(size=n)
    dy = rig.tangent(primal, ("x", "t"), {"x": dx, "t": 0.7}, ("y",), x=x, t=T)["y"]
    np.testing.assert_allclose(ybar * dy, gx["x"] * dx + gx["t"] * 0.7, rtol=1e-10,
                               atol=1e-13)


def test_a_loop_switch_matches_the_smoothed_loss_difference(rig):
    n = 4096
    x = np.random.default_rng(9).uniform(-1.5, 0.5, n)
    xs = x[None, :] + STEP * np.arange(LOOP)[:, None]
    ystar = np.where(xs < TSTAR, A, B).sum(0)

    def loss(t):
        y = (B + (A - B) * F(t - xs)).sum(0)
        return np.mean((y - ystar) ** 2)

    # seeded with the smoothed residual, the rule IS the smoothed loss's
    # derivative (four switches per sample: the staircase residual's own bias
    # is about 3% here, against about 1% for one switch)
    y = (B + (A - B) * F(T - xs)).sum(0)
    g = rig.grad(k_loop_carried, ("t",), {"bar_y": 2.0 * (y - ystar) / n},
                 x=x, t=T)["t"]
    fd = (loss(T + 1e-5) - loss(T - 1e-5)) / 2e-5
    assert _rel(g.sum(), fd) < 0.02


def _agrees_on_average(got, want):
    gap = got - want
    assert abs(gap.mean()) < 4.0 * gap.std() / math.sqrt(gap.size)


@pytest.mark.parametrize("seed", range(3))
def test_kan_edges_get_their_jump(rig, seed):
    """A switched value that meets ``tanh`` is differentiated by the alternate
    path: the lane mean is the sum of the single-comparison flips."""
    n = 4096
    rng = np.random.default_rng(500 + seed)
    x = rng.uniform(-1.0, 1.0, (N_IN, n))
    theta = rng.normal(size=N_IN * NB)
    gval = 0.9
    want = kan_flips(x, theta, gval)
    ybar = rng.normal(size=n)
    got = rig.grad(k_kan, ("g",), {"bar_y": ybar}, x=x, theta=theta, g=gval)["g"]
    _agrees_on_average(got, ybar * want)
    d = rig.unit(k_kan, "jvp", ("g",))
    dot_y = np.zeros(n)
    hawk.run(hawk.load(d, "g"), dot_g=1.0, dot_y=dot_y, x=x, theta=theta, g=gval)
    _agrees_on_average(dot_y, want)
    side = rig.sidecar(k_kan, "vjp", ("g",))["derivative"]["jump"]
    assert side["rule"] == "altpath_v1" and side["seed_param"] == "jump_seed"
    assert len(side["sites"]) == 2 * (len(KN) - 1)
    assert {s["loop"] for s in side["sites"]} == {"e"}


def test_vector_comparison_sites_share_one_group(rig):
    sites = vjp(k_vec, wrt=("t",)).jump["sites"]
    assert [s["name"] for s in sites] == ["jump0", "jump1", "jump2"]
    assert {s["group"] for s in sites} == {0}


# --------------------------------------------------------------------------- #
# -- "auto": the width from the batch's spread, one stage before the derivative
# --------------------------------------------------------------------------- #
SPREAD = 0.02 * math.sqrt(12.0)


@kernel(jump="auto")
def k_auto(x: Scalar, t: Param, a: Param, b: Param, y: Mutable[Scalar]):
    if x < t:
        y = a
    else:
        y = b


@kernel(jump="auto")
def k_auto_vec(x: Vector[3], t: Param, y: Mutable[Vector[3]]):
    y = where(x < t, x + A, x - B)


def _sd(d):
    return math.sqrt(np.mean(d * d) - np.mean(d) ** 2)


def _auto(rig, primal, wrt, **inputs):
    """``(gradients, cells)`` of one host run of ``primal``'s auto vjp."""
    d = rig.unit(primal, "vjp", wrt)
    side = json.loads((d / "g.json").read_text())
    seeds = {k: inputs.pop(k) for k in list(inputs) if k.startswith("bar_")}
    n = np.shape(next(iter(seeds.values())))[-1]
    bars = {f"bar_{w}": np.zeros(n) for w in wrt}
    cells = {c: np.zeros(1) for c in side["cells"] if c.startswith("jump_")}
    arrs = {k: (float(v) if np.ndim(v) == 0 else np.asarray(v, dtype=float))
            for k, v in inputs.items()}
    hawk.run(hawk.load(d, "g"), **seeds, **bars, **arrs, **cells)
    return {w: bars[f"bar_{w}"] for w in wrt}, {c: v[0] for c, v in cells.items()}


def test_auto_is_two_stages_and_its_sidecar_says_so(rig):
    side = rig.sidecar(k_auto, "vjp", ("t", "a", "b"))
    assert [s["kernel"] for s in side["stages"]] == ["g", "g_2"]
    assert side["stages"][0]["cells_out"] == ["jump_sd0_0", "jump_sd0_1", "jump_sd0_2"]
    assert side["stages"][1]["cells_out"] == ["jump_h0"]
    jump = side["derivative"]["jump"]
    assert (jump["rule"], jump["h"], jump["constant"], jump["param"]) == \
        ("spread_v1", None, SPREAD, None)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("n", (256, 4096))
def test_auto_width_is_the_batch_spread_and_the_oracle_holds(rig, n, seed):
    x = _data(n, seed)
    ybar = _ybar(np.where(x < T, A, B), x, n)
    g, cells = _auto(rig, k_auto, ("t", "a", "b"), bar_y=ybar, x=x, t=T, a=A, b=B)
    h = SPREAD * _sd(x - T)
    assert _rel(cells["jump_h0"], h) <= 1e-12
    assert cells["jump_sd0_2"] == n
    np.testing.assert_allclose(g["t"], ybar * (A - B) * K(x - T, h), rtol=1e-12,
                               atol=1e-300)
    assert np.array_equal(g["a"], np.where(x < T, ybar, 0.0))


def test_auto_pools_a_vector_comparison_into_one_width(rig):
    n = 512
    rng = np.random.default_rng(31)
    x = rng.uniform(-0.3, 0.3, (3, n))
    yb = rng.normal(size=(3, n))
    g, cells = _auto(rig, k_auto_vec, ("t",), bar_y=yb, x=x, t=0.05)
    h = SPREAD * _sd((x - 0.05).ravel())
    assert sorted(c for c in cells if c.startswith("jump_h")) == ["jump_h0"]
    assert _rel(cells["jump_h0"], h) <= 1e-12 and cells["jump_sd0_2"] == 3 * n
    np.testing.assert_allclose(g["t"], (yb * (A + B) * K(x - 0.05, h)).sum(0),
                               rtol=1e-12, atol=1e-15)


def test_auto_digest_is_batch_independent(rig):
    d = rig.unit(k_auto, "vjp", ("t",))
    digest = json.loads((d / "g.json").read_text())["digest"]
    for n, seed in ((300, 1), (5000, 2)):
        x = _data(n, seed)
        ybar = _ybar(np.where(x < T, A, B), x, n)
        g, cells = _auto(rig, k_auto, ("t",), bar_y=ybar, x=x, t=T, a=A, b=B)
        np.testing.assert_allclose(g["t"], ybar * (A - B) * K(x - T, SPREAD * _sd(x - T)),
                                   rtol=1e-12, atol=1e-300)
    assert json.loads((d / "g.json").read_text())["digest"] == digest
    again = rig.unit(k_auto, "vjp", ("t",))           # the same unit, whatever the batch
    assert json.loads((again / "g.json").read_text())["digest"] == digest
    from hawk.ir.split import stages
    first = stages("g", vjp(k_auto, wrt=("t",)))[0]
    assert Kernel("g", first.sinks).walk.digest == digest


class _Adam:
    def __init__(self, lr):
        self.lr, self.m, self.v, self.k = lr, 0.0, 0.0, 0

    def step(self, g):
        self.k += 1
        self.m = 0.9 * self.m + 0.1 * g
        self.v = 0.999 * self.v + 0.001 * g * g
        return self.lr * (self.m / (1 - 0.9 ** self.k)) / (
            math.sqrt(self.v / (1 - 0.999 ** self.k)) + 1e-8)


@pytest.mark.parametrize("seed", range(3))
def test_auto_trains_the_switch_on_concentrated_data(rig, seed):
    """Beta(2,5) mapped to [-1, 1], t* = .3: |t - t*| < .02 within 70 Adam steps."""
    n, ts = 4096, 0.3
    r = np.random.default_rng(seed)
    x = 2 * r.beta(2, 5, n) - 1
    f = np.where(x < ts, A, B)
    y = f + 0.02 * np.sqrt(np.mean(f ** 2)) * r.standard_normal(n)
    t, a, b = -0.4, 0.0, 0.0
    opt = {k: _Adam(0.01) for k in "tab"}
    first = None
    for k in range(70):
        ybar = 2 * (np.where(x < t, a, b) - y) / n
        g, _ = _auto(rig, k_auto, ("t", "a", "b"), bar_y=ybar, x=x, t=t, a=a, b=b)
        a -= opt["a"].step(g["a"].sum())
        b -= opt["b"].step(g["b"].sum())
        t -= opt["t"].step(g["t"].sum())
        if abs(t - ts) < 0.02:
            first = k
            break
    assert first is not None


def test_auto_with_parameter_planes_sums_the_gradient_beside_the_width(tmp_path,
                                                                        cache_dir):
    """A trainable Param's gradient (a sum cell) and the width (a mean cell) are
    stored by one stage; ``bar_t`` matches numpy, and the width, a mean of one
    value repeated n times, is within 4 ULP of ``0.02 sqrt(12) sd``."""
    from hawk.diff import param_planes

    planes = param_planes(k_auto)
    derived = Kernel("g", vjp(planes, wrt=("t",), primal_unit="0" * 64))
    hawk.build([derived], tmp_path, targets=("host",), cache_dir=cache_dir)
    side = json.loads((tmp_path / "g.json").read_text())
    stage2 = side["stages"][1]["cells_out"]
    assert {"bar_t", "jump_h0"} <= set(stage2)
    assert side["cells"]["jump_h0"]["op"] == "sum"          # a mean: the sum of h / n
    n = 4096
    x = _data(n, 21)
    ybar = _ybar(np.where(x < T, A, B), x, n)
    cells = {c: np.zeros(1) for c in side["cells"] if not c.startswith("cut")}
    hawk.run(hawk.load(tmp_path, "g"), bar_y=ybar, x=x, t=np.full(1, T),
             a=np.full(1, A), b=np.full(1, B), **cells)
    h = SPREAD * _sd(x - T)
    assert abs(cells["jump_h0"][0] - h) <= 4 * np.spacing(h)
    want = (ybar * (A - B) * K(x - T, cells["jump_h0"][0])).sum()
    assert _rel(cells["bar_t"][0], want) <= 1e-12


def test_auto_plain_descent_finds_the_switch(rig):
    n, t = 4096, -0.4
    x = _data(n, 11)
    for _ in range(200):
        y = np.where(x < t, A, B)
        t -= 0.1 * _auto(rig, k_auto, ("t",), bar_y=_ybar(y, x, n), x=x, t=t, a=A,
                         b=B)[0]["t"].sum()
    assert abs(t - TSTAR) < 0.02


def test_auto_host_run_equals_the_eagle_plan_run(rig, cache_dir):
    """The derived kernel's two stages through eagle's plan, which runs the
    finish between them, give hawk.run's gradient and width bitwise."""
    pytest.importorskip("eagle.exec")
    from eagle import plan as eplan

    n = 4096
    x = _data(n, 5)
    ybar = _ybar(np.where(x < T, A, B), x, n)
    g, cells = _auto(rig, k_auto, ("t",), bar_y=ybar, x=x, t=T, a=A, b=B)
    derived = Kernel("g", vjp(k_auto, wrt=("t",), primal_unit="0" * 64))
    got = eplan.auto(derived, targets=("host",), cache_dir=cache_dir).run(
        bar_y=ybar, bar_t=np.zeros(n), x=x, t=T, a=A, b=B)
    assert np.asarray(got["bar_t"]).tobytes() == g["t"].tobytes()
    assert float(np.asarray(got["jump_h0"]).reshape(-1)[0]) == cells["jump_h0"]


# --------------------------------------------------------------------------- #
# -- off by default
# --------------------------------------------------------------------------- #
def test_jump_leaves_the_primal_digest_alone():
    assert k_lt.walk.digest == k_lt_off.walk.digest
    assert k_lt.jump == H and k_lt_off.jump is None


def test_a_comparison_off_every_wrt_path_derives_as_without_jump():
    on, off = vjp(k_const_cmp, wrt=("x",)), vjp(k_const_cmp_off, wrt=("x",))
    assert Kernel("g", on).walk.digest == Kernel("g", off).walk.digest
    assert on.jump is None and off.jump is None
    on, off = vjp(k_lt, wrt=("a", "b")), vjp(k_lt_off, wrt=("a", "b"))
    assert Kernel("g", on).walk.digest == Kernel("g", off).walk.digest


def test_sidecar_jump_block_on_the_derived_unit_only(rig):
    jump = rig.sidecar(k_lt, "vjp", ("t", "a", "b"))["derivative"]["jump"]
    assert jump == {"kernel": "epanechnikov", "rule": "fixed", "h": H,
                    "constant": None, "param": None,
                    "sites": [{"name": "jump0", "cond": "(x < t)", "reaches": ["t"],
                               "group": 0}]}
    assert "jump" not in rig.sidecar(k_lt_off, "vjp", ("t",))["derivative"]


# --------------------------------------------------------------------------- #
# -- the width as a Param
# --------------------------------------------------------------------------- #
def test_width_param_is_bitwise_the_fixed_width(rig):
    n = 4096
    x = _data(n, 3)
    ybar = _ybar(np.where(x < T, A, B), x, n)
    fixed = rig.grad(k_lt, ("t",), {"bar_y": ybar}, x=x, t=T, a=A, b=B)["t"]
    d = rig.unit(k_param, "vjp", ("t",))
    side = json.loads((d / "g.json").read_text())
    assert side["derivative"]["jump"]["rule"] == "param"
    assert side["derivative"]["jump"]["param"] == "h"
    assert side["derivative"]["wrt"] == ["t"]
    assert sorted(side["cells"]) == ["jump_sd0_0", "jump_sd0_1", "jump_sd0_2"]
    for h in (H, 0.01):
        bar_t = np.zeros(n)
        cells = {name: np.zeros(1) for name in side["cells"]}
        hawk.run(hawk.load(d, "g"), bar_y=ybar, bar_t=bar_t, x=x, t=T, a=A, b=B, h=h,
                 **cells)
        if h == H:
            assert np.array_equal(bar_t, fixed)
        else:
            np.testing.assert_allclose(bar_t, ybar * (A - B) * K(x - T, h), rtol=1e-12,
                                       atol=1e-300)
        s = {j: cells[f"jump_sd0_{j}"][0] for j in range(3)}
        sd = math.sqrt(s[1] / s[2] - (s[0] / s[2]) ** 2)
        dd = x - T
        assert _rel(sd, math.sqrt(np.mean(dd * dd) - np.mean(dd) ** 2)) <= 1e-12


def test_width_param_derives_one_ir():
    assert vjp(k_param, wrt=("t",)).wrt == ("t",)
    assert vjp(k_param).wrt == ("x", "t", "a", "b")


# --------------------------------------------------------------------------- #
# -- refusals
# --------------------------------------------------------------------------- #
def test_refuses_equality_on_a_differentiated_input():
    @kernel(jump=H)
    def k_eq(x: Scalar, t: Param, y: Mutable[Scalar]):
        y = 1.0 if x == t else 2.0
    with pytest.raises(HawkError, match="a jump needs an ordered comparison; `==` "
                                        "has no side to cross"):
        vjp(k_eq, wrt=("t",))


def test_refuses_a_loop_exit_through_a_differentiated_input():
    @kernel(jump=H)
    def k_exit(x: Scalar, t: Scalar, y: Mutable[Scalar]):
        s = x
        for _k in range(4):
            if s > t:
                break
            s = s + 0.5
        y = s
    with pytest.raises(HawkError, match="a loop exit through a differentiated input "
                                        "is an event, not a jump"):
        vjp(k_exit, wrt=("t",))


def test_refuses_a_rounding_on_the_compared_path():
    @kernel(jump=H)
    def k_floor(x: Scalar, t: Param, y: Mutable[Scalar]):
        y = 1.0 if floor(x * t) < 1.0 else 2.0
    with pytest.raises(HawkError, match="a rounding, sign or integer cast sits "
                                        "between a differentiated input"):
        vjp(k_floor, wrt=("t",))


def test_refuses_a_second_order():
    once = Kernel("k_lt_vjp", vjp(k_lt, wrt=("t",)))
    with pytest.raises(HawkError, match="differentiating it again"):
        vjp(once)
    with pytest.raises(HawkError, match="differentiating it again"):
        jvp(once)


@pytest.mark.parametrize("bad", (-0.1, 0.0, math.inf, math.nan, True, "nope", [1]))
def test_refuses_a_bad_width(bad):
    with pytest.raises(HawkError, match="jump="):
        @kernel(jump=bad)
        def k_bad(x: Scalar, t: Param, nope: Scalar, y: Mutable[Scalar]):
            y = 1.0 if x < t else 2.0


def test_refuses_a_width_param_that_is_not_one():
    with pytest.raises(HawkError, match="names no declared Param"):
        @kernel(jump="h")
        def k_plane(x: Scalar, t: Param, h: Scalar, y: Mutable[Scalar]):
            y = 1.0 if x < t else 2.0
    with pytest.raises(HawkError, match="a Param named 'auto'"):
        @kernel(jump=H)
        def k_auto(x: Scalar, auto: Param, y: Mutable[Scalar]):
            y = 1.0 if x < auto else 2.0
    with pytest.raises(HawkError, match="the jump width is not differentiable"):
        vjp(k_param, wrt=("t", "h"))
    with pytest.raises(HawkError, match="the jump width is not differentiable"):
        jvp(k_param, wrt=("h",))


def test_auto_on_a_switch_inside_a_loop_is_refused():
    @kernel(jump="auto")
    def k_auto_loop(x: Scalar, t: Param, y: Mutable[Scalar]):
        total = 0.0
        for k in range(LOOP):
            total = total + where(x + STEP * real(k) < t, A, B)
        y = total
    for derive in (vjp, jvp):
        with pytest.raises(HawkError, match="'auto' needs a width from statistics "
                                            "over samples; switches inside a `for` "
                                            "take a number in this release"):
            derive(k_auto_loop, wrt=("t",))


@kernel(jump="h")
def k_loop_param(x: Scalar, t: Param, h: Param, y: Mutable[Scalar]):
    total = 0.0
    for k in range(LOOP):
        total = total + where(x + STEP * real(k) < t, A, B)
    y = total


def test_a_param_width_inside_a_loop_needs_no_statistics(rig):
    n = 512
    x = np.random.default_rng(12).uniform(-1.5, 0.5, n)
    ybar = np.random.default_rng(13).normal(size=n)
    fixed = rig.grad(k_loop_sum, ("t",), {"bar_y": ybar}, x=x, t=T)["t"]
    got = rig.grad(k_loop_param, ("t",), {"bar_y": ybar}, x=x, t=T, h=H)["t"]
    assert np.array_equal(got, fixed)
    side = rig.sidecar(k_loop_param, "vjp", ("t",))
    assert side["derivative"]["jump"]["sites"][0]["loop"] == "k"
    assert not any(name.startswith("jump_sd") for _, name in side["arg_spec"])


def test_a_finishing_step_with_jump_builds():
    @kernel(jump=H)
    def k_stop(x: Scalar, t: Param, terminated: Terminated, y: Mutable[Scalar]):
        y = x + (1.0 if x < t else 2.0)
        terminated = y > 3.0
    assert vjp(k_stop, wrt=("t",)) is not None


# --------------------------------------------------------------------------- #
# -- emission
# --------------------------------------------------------------------------- #
def test_the_jump_term_is_emitted_inside_one_support_switch(rig):
    text = rig.text(k_three, "vjp", ("t",))
    assert text.count("switch (") == 1
    body = text[text.index("switch ("):]
    case, rest = body.split("default:", 1)
    before = text[:text.index("switch (")]
    for const in (0.75 / W, 0.75 / W / (W * W)):     # K = 3/(4w) - 3/(4w^3) d^2
        assert repr(const) in case
        assert repr(const) not in rest and repr(const) not in before
    assert repr(W * W) in before                     # the support test, d^2 < w^2
    assert "< 0 ? 0" not in before                   # a 0/1 selector is not clamped


# --------------------------------------------------------------------------- #
# -- device twins (run by the GPU gate)
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
@pytest.mark.parametrize("op", sorted(_TABLE))
def test_device_matches_host(rig, tmp_path, cache_dir, op):
    import _deploy as L
    import cupy
    import eagle.exec as eexec
    from eagle import plan as eplan

    from hawk.artifact import build_bundle

    primal, pred, _ = _TABLE[op]
    n = 4096
    x = _data(n, 0)
    ybar = _ybar(np.where(pred(x, T), A, B), x, n)
    host = rig.grad(primal, ("t",), {"bar_y": ybar}, x=x, t=T, a=A, b=B)["t"]
    bundle = build_bundle([primal, Kernel("g", vjp(primal, wrt=("t",)))], tmp_path,
                          targets=("cuda",), cache_dir=cache_dir)
    side = json.loads((tmp_path / "g.json").read_text())
    runs = []
    for _ in range(3):
        planes = {"bar_y": cupy.asarray(ybar), "bar_t": cupy.zeros(n),
                  "x": cupy.asarray(x), "t": T, "a": A, "b": B}
        eplan.plan(L.device_plugin(bundle.directory, "g", side),
                   structure=eexec.DeviceKernel).bind(**planes).launch()
        runs.append(cupy.asnumpy(planes["bar_t"]))
    np.testing.assert_allclose(runs[0], host, rtol=1e-12, atol=1e-300)
    assert all(np.array_equal(r, runs[0]) for r in runs[1:])


@pytest.mark.gpu
def test_device_matches_host_through_a_loop(rig, tmp_path, cache_dir):
    import _deploy as L
    import cupy
    import eagle.exec as eexec
    from eagle import plan as eplan

    from hawk.artifact import build_bundle

    n = 2048
    x = np.random.default_rng(1).uniform(-1.5, 0.5, n)
    ybar = np.random.default_rng(2).normal(size=n)
    host = rig.grad(k_loop_carried, ("t",), {"bar_y": ybar}, x=x, t=T)["t"]
    bundle = build_bundle([k_loop_carried, Kernel("g", vjp(k_loop_carried, wrt=("t",)))],
                          tmp_path, targets=("cuda",), cache_dir=cache_dir)
    side = json.loads((tmp_path / "g.json").read_text())
    runs = []
    for _ in range(3):
        planes = {"bar_y": cupy.asarray(ybar), "bar_t": cupy.zeros(n),
                  "x": cupy.asarray(x), "t": T}
        eplan.plan(L.device_plugin(bundle.directory, "g", side),
                   structure=eexec.DeviceKernel).bind(**planes).launch()
        runs.append(cupy.asnumpy(planes["bar_t"]))
    np.testing.assert_allclose(runs[0], host, rtol=1e-12, atol=1e-300)
    assert all(np.array_equal(r, runs[0]) for r in runs[1:])
