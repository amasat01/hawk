# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Loop-invariant hoisting across a ``dispatch`` arm: a row index gathered once
and read both inside and outside the arms is hoisted out of the loop, and so is
a plane read in an arm when the selector is loop-invariant (gated by its arm, so
an untaken arm's out-of-range row is never read). A read guarded by a
per-iteration condition stays in the body."""

from __future__ import annotations
from _specs import HOST, default_kind
from _specs import aether_writer

import numpy as np
from _eval import evaluate

import hawk
import hawk.math as m
from raptor.ir import Loop

T = 6
W = 3


@hawk.kernel
def shared_row(x: hawk.Scalar, kind: hawk.Scalar, perm: hawk.Table["i":4],
               th: hawk.Table["r":4, "c":W],
               xs: hawk.Table[hawk.Scalar], y: hawk.Mutable[hawk.Scalar]):
    q = m.index(perm.at(i=1), 4)
    s = x * th.at(r=q, c=0)
    for t in range(T):
        arms = [s * th.at(r=q, c=1), s + th.at(r=q, c=2) * 0.5,
                m.sin(s) * th.at(r=q, c=0)]
        s = m.dispatch(m.int32(kind), arms, strategy="switch") * 0.9 + 0.1 + xs.at(q + t) * 0.01
    y = s


@hawk.kernel
def per_iteration_guard(x: hawk.Scalar, n: hawk.Scalar, a: hawk.Table["k":T],
                        y: hawk.Mutable[hawk.Scalar]):
    s = x
    for k in range(T):
        s = s + m.where(m.real(k) < n, a.at(m.index(n, T)), 0.0)
    y = s


def _loop(kernel):
    return next(n for n in kernel.walk.order if isinstance(n, Loop))


@hawk.kernel
def far_arm(x: hawk.Scalar, kind: hawk.Scalar, tab: hawk.Table[hawk.Scalar],
            y: hawk.Mutable[hawk.Scalar]):
    s = x
    for t in range(T):
        arms = [s * tab.at(m.int32(x * 100.0)), s * 2.0]
        s = m.dispatch(m.int32(kind), arms, strategy="switch") + 0.1
    y = s


def test_the_hoisted_kernel_gives_the_same_answer():
    perm = np.array([0.0, 2.0, 1.0, 3.0])
    th = np.linspace(0.5, 1.5, 4 * W)
    xs = np.linspace(-1.0, 1.0, 4 + T)
    for kind in (0.0, 1.0, 2.0):
        got = evaluate(shared_row.sinks, {"x": 0.4, "kind": kind, "perm": perm, "th": th,
                                    "xs": xs, "terminated": False}, lanes=1)["y"]
        s = 0.4 * th[6]
        for t in range(T):
            arms = [s * th[7], s + th[8] * 0.5, np.sin(s) * th[6]]
            s = arms[int(kind)] * 0.9 + 0.1 + xs[2 + t] * 0.01
        assert abs(got - s) < 1e-12


