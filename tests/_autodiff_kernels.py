# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The kernels of ``test_autodiff_matrix.py``: one small representative per row."""

from __future__ import annotations

import hawk
import hawk.math as m
from hawk import (
    Grid,
    Index,
    Matrix,
    Mutable,
    Param,
    Scalar,
    Table,
    Terminated,
    Vector,
    kernel,
)

A, NB, NC = 3, 4, 6
LANE = (A, NB, NC)


@kernel
def bcast(lane: Grid["a", "b":4, "c":6], x: Scalar, w: Scalar) -> Scalar:
    return m.sin(x) * w + x * x


@kernel
def vbcast(lane: Grid["a", "b":4, "c":6], v: Vector[3], w: Scalar) -> Scalar:
    return v[0] * v[1] + m.sin(v[2]) * w


@kernel
def grid_kw(lane: Grid["a", "b":4, "c":6], p: Scalar, x: Scalar, k: Param,
            y: Mutable[Scalar]):
    y = m.exp(0.3 * p) * x + k * p * x * x


@kernel
def node_gather(lane: Grid["e"], w: Table["elem", "c":3], x: Scalar) -> Scalar:
    return m.sin(x) * (w.at(elem=lane.e, c=1) + 2.0 * w.at(elem=lane.e, c=2))


@kernel
def node_scatter(lane: Grid["e"], x: Scalar, b: Mutable["node"]):
    hawk.scatter(b)[lane.e % 4] += m.sin(x) * x


@kernel
def switch_if(x: Scalar, g: Scalar, y: Mutable[Scalar]):
    if x < 0.3:
        y = x * x * g
    elif x < 0.7:
        y = 3.0 * x + g
    else:
        y = m.sin(x) * g


@kernel
def switch_dispatch(x: Scalar, g: Scalar, y: Mutable[Scalar]):
    k = m.int32(x * 3.0)
    y = g * m.dispatch(k, [x * x, 3.0 * x + 1.0, m.sin(x)], policy="switch")


CAP = 12


@kernel
def stop_break(x: Scalar, g: Scalar, tol: Scalar, terminated: Terminated,
               n: Mutable[Scalar], e_out: Mutable[Scalar]):
    e = x
    count = -1.0
    for k in range(CAP):
        e = e * g
        c = e < tol
        count = m.where(c, 1.0 * k, count)
        if c:
            break
    n = count
    e_out = e


@kernel
def exit_s_break(x: Scalar, g: Scalar, lim: Scalar, tau: Mutable[Scalar],
                 y: Mutable[Scalar]):
    s = x
    t = 0.0
    r = -1.0
    for _k in range(CAP):
        s = m.sin(g * s) + s * 1.2
        t = t + 0.1
        if s > lim:
            r = t + 0.25 * s
            break
    tau = r
    y = s


@kernel
def keyed_sum(lane: Grid["key":2, "k2":3, "sample":16], x: Scalar,
              tb: Mutable["key":2, "k2":3]):
    tb = hawk.sum(x * x, over="sample")


@kernel
def keyed_sum_c(lane: Grid["key":2, "k2":3, "sample":16], x: Scalar,
                tb: Mutable["key":2, "k2":3]):
    tb = hawk.sum(m.sin(x) * x, over="sample", policy="compensated")


@kernel
def readback(lane: Grid["row", "col":16], x: Scalar, y: Mutable[Scalar]):
    y = x / hawk.sum(x, over="col")


@kernel
def fill_curve(a: Scalar, b: Scalar, curve: Mutable[Vector[4]]):
    for j in range(4):
        curve[j] = m.sin(a * (j + 1.0)) * b


@kernel(steps=1)
def pend(dt: Param, c: Param, t_final: Scalar, terminated: Terminated,
         x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    a = -m.sin(x) - c * v * v
    vn = v + dt * a
    x = x + dt * v
    v = vn
    tn = t + dt
    t = tn
    terminated = tn >= t_final


@kernel
def la_solve(A: Matrix[3, 3], b: Vector[3], x: Mutable[Vector[3]]):
    x, _ = m.solve(A, b)


def _newton2(x0, b, steps=3):
    """Newton on ``F(x) = (x0^2 + x1 - b0, x0 x1 - b1)``, the Jacobian assembled with ``m.mat``."""
    x = x0
    for _ in range(steps):
        f = m.vec(x[0] * x[0] + x[1] - b[0], x[0] * x[1] - b[1])
        step, _info = m.solve(m.mat([[2.0 * x[0], 1.0], [x[1], x[0]]]), f)
        x = x - step
    return x


@kernel
def la_mat_newton(x0: Vector[2], b: Vector[2], x: Mutable[Vector[2]]):
    x = _newton2(x0, b)


@kernel
def la_mat_rows(u: Vector[3], v: Vector[3], w: Vector[3], b: Vector[3], x: Mutable[Vector[3]]):
    x, _ = m.solve(m.mat([u, v, w]), b)


@kernel
def la_chol(S: Matrix[3, 3], b: Vector[3], x: Mutable[Vector[3]], L: Mutable[Matrix[3, 3]]):
    chol, _ = m.cholesky(S)
    L = chol
    x = m.cho_solve(chol, b)


@kernel
def la_chol_from(L: Matrix[3, 3], b: Vector[3], x: Mutable[Vector[3]], f: Mutable[Matrix[3, 3]]):
    factor, _ = m.chol_from(L)
    f = factor
    x = m.cho_solve(factor, b)


@kernel
def la_chol_from_rows(u: Vector[3], v: Vector[3], w: Vector[3], b: Vector[3],
                      x: Mutable[Vector[3]]):
    factor, _ = m.chol_from(m.mat([u, v, w]))
    x = m.cho_solve(factor, b)


@kernel
def la_det(A: Matrix[3, 3], d: Mutable[Scalar]):
    d = m.det(A)


@kernel
def la_inv(A: Matrix[3, 3], Ainv: Mutable[Matrix[3, 3]]):
    Ainv, _ = m.inv(A)


@kernel
def la_eigh(S: Matrix[3, 3], w: Mutable[Vector[3]]):
    w, _, _ = m.eigh(S)


@kernel
def la_lstsq(J: Matrix[4, 3], r: Vector[4], x: Mutable[Vector[3]]):
    x, _ = m.lstsq(J, r)


@kernel
def la_mvn(seed: Index, counter: Index, mu: Vector[3], S: Matrix[3, 3], y: Mutable[Vector[3]]):
    y = m.mvn_sample(seed, counter, mu, S)


# jump= : lanes carry their own s; the exact derivative of each counting game is known
H_JUMP, S_JUMP, STEPS_JUMP, JLANES = 0.01, 0.3, 6, 200_000


@kernel(jump=H_JUMP)
def jgate(u: Vector[6], s: Scalar, tally: Mutable[Scalar], alive_out: Mutable[Scalar]):
    alive = 1.0
    c = 0.0
    for i in range(6):
        if alive > 0.5:
            if u[i] < s:
                alive = 0.0
            else:
                c = c + 1.0
    tally = c
    alive_out = alive


@kernel(jump=H_JUMP)
def jchain(u: Vector[6], s: Scalar, tally: Mutable[Scalar], alive_out: Mutable[Scalar]):
    y = 0.0
    if u[0] < s:
        y = 1.0
    z = 0.0
    if y > 0.5:
        z = 1.0
    tally = z
    alive_out = y


_Q = 1.0 - S_JUMP
TRUTH = {"jgate": (-sum(k * _Q ** (k - 1) for k in range(1, STEPS_JUMP + 1)),
                   -STEPS_JUMP * _Q ** (STEPS_JUMP - 1)),
         "jchain": (1.0, 1.0)}
