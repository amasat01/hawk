# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A KAN edge function in two spellings, at HAWK's own authoring surface — TEST
SCAFFOLD for ``tests/test_spline_local.py`` and the register probe.

The edge is ``phi(x) = w_b * silu(u) + w_s * sum_g c_g B_g(u)`` with
``u = (x - x0) / xs`` on a uniform grid of ``G`` intervals and degree ``k``
(``G + k`` basis functions, knots extending ``[-1, 1]`` by ``k`` intervals each
side), lanes ``[w_b, w_s, c_0 .. c_{G+k-1}, x0, xs]``.

* DENSE: the indicator Cox-de Boor over every degree-0 interval, then the
  sum over all ``G + k`` coefficients, as a downstream package's reference
  spline writes it (reproduced here, never imported).
* LOCAL: the interval index by ``floor``, only the ``k + 1`` basis values
  that can be non-zero, by the SAME recursion formulas on the local knots, and
  the coefficients read at the runtime index with ``component_at``.
"""

from __future__ import annotations

import hawk
import hawk.math as m

G, K = 10, 3
N_BASIS = G + K
W = 2 + N_BASIS + 2
ROWS = 2
STEP = 2.0 / G


def knot(q):
    """Knot ``q`` (``0 .. G + 2K``): the one formula both forms evaluate."""
    return -1.0 + (q - K) * STEP


def silu(u):
    return u * (1.0 / (1.0 + m.exp(-u)))


def dense_edge(x, th, off=0):
    """Indicator recursion over all intervals, then the full coefficient sum."""
    u = (x - th[off + W - 2]) / th[off + W - 1]
    knots = [knot(q) for q in range(G + 2 * K + 1)]
    level = [m.where(m.land(u >= knots[i], u < knots[i + 1]), 1.0, 0.0)
             for i in range(len(knots) - 1)]
    for d in range(1, K + 1):
        level = [((u - knots[i]) / (knots[i + d] - knots[i])) * level[i]
                 + ((knots[i + d + 1] - u) / (knots[i + d] - knots[i])) * level[i + 1]
                 for i in range(len(level) - 1)]
    spline = level[0] * th[off + 2]
    for g in range(1, N_BASIS):
        spline = spline + level[g] * th[off + 2 + g]
    return th[off] * silu(u) + th[off + 1] * spline


def local_edge(x, th, off=0, exact_index=True):
    """The ``K + 1`` live basis values at the interval ``j`` the input falls in.

    ``j = clamp(floor((u + 1) / step), 0, G - 1)`` is the first live coefficient
    (the active knot interval is ``j + K``). With ``exact_index`` the floor is
    corrected once against the knots themselves, so a value ON a knot lands in
    the interval the indicator form picks (``u >= knot`` opens the next one)."""
    u = (x - th[off + W - 2]) / th[off + W - 1]
    jf = m.minimum(m.maximum(m.floor((u + 1.0) / STEP), 0.0), G - 1.0)
    if exact_index:
        up = m.where(m.land(u >= knot(jf + (K + 1.0)), jf < G - 1.0), 1.0, 0.0)
        down = m.where(m.land(u < knot(jf + float(K)), jf > 0.0), 1.0, 0.0)
        jf = jf + up - down
    j = m.index(jf, G)
    t = [knot(jf + float(s)) for s in range(2 * K + 2)]       # t[s] is knot j + s
    live = [1.0]
    for d in range(1, K + 1):
        nxt = []
        for r in range(d + 1):
            p = K - d + r                           # knot offset of this basis function
            # the dense form divides by a literal, which HAWK spells as a
            # multiplication by the literal's reciprocal; the same here, so the
            # two round alike
            inv = 1.0 / (t[p + d] - t[p])
            hi = ((t[p + d + 1] - u) * inv) * live[r] if r < d else None
            lo = ((u - t[p]) * inv) * live[r - 1] if r > 0 else None
            nxt.append(hi if lo is None else lo if hi is None else lo + hi)
        live = nxt
    spline = live[0] * th[j + (off + 2)]
    for r in range(1, K + 1):
        spline = spline + live[r] * th[j + (off + 2 + r)]
    return th[off] * silu(u) + th[off + 1] * spline


def edge_cell(form: str, exact_index: bool = True):
    """One edge as a kernel: ``y = phi(x)`` with the lanes a ``Vector[W]``."""
    fn = dense_edge if form == "dense" else (
        lambda x, th: local_edge(x, th, 0, exact_index))

    @hawk.kernel
    def cell(x: hawk.Scalar, theta: hawk.Vector[W], y: hawk.Mutable[hawk.Scalar]):
        y = fn(x, theta)

    return cell


def _sum_edges(fn, x, theta, n_in):
    """The edges unrolled in plain python (a body ``for`` would be a lowered loop)."""
    total = None
    for e in range(n_in):
        v = fn(x[e * ROWS], theta, e * W)
        total = v if total is None else total + v
    return total


def kan_layer(form: str, n_in: int):
    """``n_in`` spline edges summed, in nowl's ``kan_routed`` shape: ``x`` carries
    ``ROWS`` rows per input, ``theta`` the lanes of every edge, ``y`` row 0 the sum."""
    fn = dense_edge if form == "dense" else local_edge

    @hawk.kernel
    def kan_routed(x: hawk.Vector[n_in * ROWS],
                   theta: hawk.Vector[n_in * W],
                   terminated: hawk.Terminated,
                   y: hawk.Mutable[hawk.Vector[2]]):
        y = m.vec(_sum_edges(fn, x, theta, n_in), 0.0)

    return kan_routed


def _switch_edges(x, theta, kinds, n_in):
    """Each edge a three-way switch on its kind: the local spline, a polynomial,
    a power; the kind is data (nowl's shape)."""
    total = None
    for e in range(n_in):
        xe = x[e * ROWS]
        base = e * W
        cases = [local_edge(xe, theta, base),
                 theta[base] * xe * xe + theta[base + 1],
                 theta[base] * m.exp(theta[base + 2] * m.log(m.sqrt(xe * xe) + 1e-6))]
        v = m.dispatch(m.int32(kinds[e]), cases, policy="switch")
        total = v if total is None else total + v
    return total


def kan_switch(n_in: int):
    """:func:`kan_layer` with a kind switch per edge."""
    @hawk.kernel
    def kan_routed(x: hawk.Vector[n_in * ROWS],
                   theta: hawk.Vector[n_in * W],
                   kinds: hawk.Vector[n_in],
                   terminated: hawk.Terminated,
                   y: hawk.Mutable[hawk.Vector[2]]):
        y = m.vec(_switch_edges(x, theta, kinds, n_in), 0.0)

    return kan_routed


def kan_rolled(form: str, n_in: int):
    """The same layer with the edges as ONE lowered ``for`` (``plain``: the local
    spline; ``switch``: a kind switch per edge), so its reverse is a derived loop
    that commits the runtime-indexed gradient once per iteration."""
    if form == "plain":
        @hawk.kernel
        def kan_loop(x: hawk.Vector[n_in * ROWS], theta: hawk.Vector[n_in * W],
                     terminated: hawk.Terminated, y: hawk.Mutable[hawk.Vector[2]]):
            total = 0.0
            for e in range(n_in):
                total = total + local_edge(x[e * ROWS], theta, e * W)
            y = m.vec(total, 0.0)
        return kan_loop

    @hawk.kernel
    def kan_loop(x: hawk.Vector[n_in * ROWS], theta: hawk.Vector[n_in * W],
                 kinds: hawk.Vector[n_in], terminated: hawk.Terminated,
                 y: hawk.Mutable[hawk.Vector[2]]):
        total = 0.0
        for e in range(n_in):
            xe = x[e * ROWS]
            base = e * W
            cases = [local_edge(xe, theta, base),
                     theta[base] * xe * xe + theta[base + 1],
                     theta[base] * m.exp(theta[base + 2] * m.log(m.sqrt(xe * xe) + 1e-6))]
            total = total + m.dispatch(m.int32(kinds[e]), cases, policy="switch")
        y = m.vec(total, 0.0)
    return kan_loop
