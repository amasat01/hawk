# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The derivative of a BROADCAST parameter (``Param[Vector[W]]``, ``Param[Matrix[R,C]]``):
its cells are one ``lookup`` plane read at constant cell indices (a ``Table`` read), so
the reverse pass scatter-adds the gradient of every lane into ``bar_<name>`` and the forward pass reads ONE tangent plane
shared by all lanes."""

from __future__ import annotations

import numpy as np
from _eval import evaluate

from hawk.diff import jvp, vjp
from hawk.ir import Assign, At, Const, Leaf
from hawk.ir import make as mk
from hawk.types import TensorType

F64 = TensorType((), "f64")
LANES = 5
H = 1e-6


def _primal():
    """``y = w[0] * x + w[1] * x * x`` over per-lane ``x`` and ONE two-cell ``w``."""
    x = Leaf("vocab_read", "per_sample", "x", F64)
    w = Leaf("table_read", "lookup", "w", F64)
    w0, w1 = (At(w, Const(k, TensorType((), "i32")), F64) for k in (0, 1))
    y = mk("add", (mk("mul", (w0, x)), mk("mul", (w1, mk("mul", (x, x))))))
    return (Assign("y", y, F64),)


def test_the_reverse_pass_sums_a_broadcast_parameter_over_the_lanes():
    sinks = _primal()
    x = np.linspace(0.5, 2.0, LANES)
    w = np.array([1.5, -0.7])
    cot = np.linspace(1.0, 2.0, LANES)
    derived = vjp(sinks)
    # one lane at a time, each scatter-adding into the ONE shared plane
    got = sum(evaluate(derived, {"x": x[i], "w": w, "bar_y": cot[i]}, lanes=2)["bar_w"]
              for i in range(LANES))

    def y(wv):
        return wv[0] * x + wv[1] * x * x

    expect = np.zeros(2)
    for k in range(2):
        up, dn = w.copy(), w.copy()
        up[k] += H
        dn[k] -= H
        expect[k] = np.sum(cot * (y(up) - y(dn)) / (2 * H))
    assert np.allclose(got[:2], expect, rtol=1e-6, atol=1e-8), f"{got} vs {expect}"
    assert np.allclose(got[:2], [np.sum(cot * x), np.sum(cot * x * x)])


def test_the_forward_pass_reads_one_tangent_plane_for_all_lanes():
    sinks = _primal()
    x = np.linspace(0.5, 2.0, LANES)
    w = np.array([1.5, -0.7])
    tangent = np.array([0.3, 1.1])
    derived = jvp(sinks)
    got = np.array([evaluate(derived, {"x": x[i], "w": w, "dot_w": tangent, "dot_x": 0.0},
                             lanes=2)["dot_y"] for i in range(LANES)])
    expect = tangent[0] * x + tangent[1] * x * x
    assert np.allclose(got, expect, rtol=1e-12)
