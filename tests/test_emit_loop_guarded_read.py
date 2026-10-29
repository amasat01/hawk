# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A plane read the loop leaves inside one switch arm, also read after the loop.

The read sits under a nested arm, so the loop does not hoist it; the same value
is read after the loop, so the flat walk reaches it, and it may first render
AFTER the loop. The arm that reads it inside the loop renders it itself rather
than naming a value that does not exist yet. The rows build the kernel through
``eagle.plan.auto`` and compare it with the same kernel carrying the value in its
state, which never leaves the loop."""

from __future__ import annotations

import numpy as np
import pytest

import hawk
import hawk.math as m

STEPS, ROWS, WIDTH = 4, 4, 8
LANES = ROWS * WIDTH


def _kernel(carried: bool):
    def arm(st, v, scale):
        damp = m.exp(-m.abs(v - st[1]) / scale)
        return m.vec(m.where(v > st[1], st[0] * damp + 1.0, st[0] + damp),
                     m.maximum(st[1], v), st[2], st[3], st[4])

    @hawk.kernel
    def chain(lane: hawk.Index, a: hawk.Table[hawk.Scalar], kind: hawk.Table[hawk.Scalar],
              mode: hawk.Table[hawk.Scalar], scale_t: hawk.Table[hawk.Scalar],
              out: hawk.Mutable[hawk.Vector[3]]):
        row, _col = m.split_index(WIDTH)
        case = m.int32(mode.at(row) + 0.5)
        scale = scale_t.at(row)
        zero = 0.0 * m.real(lane)
        st = m.vec(m.where(case == 1, 1.0, 0.0) + zero, -1e30 + zero, zero, zero,
                   (scale + zero) if carried else zero)
        for i in range(STEPS):
            x = a.at(lane * STEPS + i)
            which = m.int32(kind.at(row * STEPS + i) + 0.5)
            v = 0.0
            if which == 0:
                v = m.sin(x)
            elif which == 1:
                v = m.exp(x)
            elif which == 2:
                v = m.tanh(x)
            else:
                v = m.cos(x)
            if case == 0:
                st = m.vec(st[0] + v, st[1], st[2], st[3], st[4])
            elif case == 1:
                st = m.vec(st[0] * v, st[1], st[2] + 1.0, st[3], st[4])
            elif case == 2:
                st = m.vec(st[0], m.maximum(st[1], v), st[2], st[3], st[4])
            else:
                st = arm(st, v, st[4] if carried else scale)
        y = st[0]
        w = st[2]
        if case == 3:                          # two names, so each case's work is shared, not lazy
            y = st[1] + scale * m.log(st[0])
            w = st[2] + scale * m.exp(st[1])
        out = m.vec(y, w, st[3])

    return chain


def _run(kernel, cache_dir):
    import eagle.plan
    rng = np.random.default_rng(3)
    planes = {"lane": np.arange(LANES, dtype=np.int32), "a": rng.random(LANES * STEPS) + 0.5,
              "kind": np.tile([0.0, 1.0, 2.0, 3.0], ROWS),
              "mode": np.arange(ROWS, dtype=float),
              "scale_t": rng.random(ROWS) + 1.0,
              "out": np.zeros((3, LANES))}
    plan = eagle.plan.auto([kernel], targets=("host",), cache_dir=cache_dir,
                           scalar_type="float64")[0]
    plan.run(**planes)
    return planes["out"]


def test_a_loop_arm_reading_a_value_also_read_after_the_loop_builds_and_agrees(cache_dir):
    pytest.importorskip("eagle")
    got = _run(_kernel(False), cache_dir)
    np.testing.assert_array_equal(got, _run(_kernel(True), cache_dir))
    assert np.all(np.isfinite(got))
