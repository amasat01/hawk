# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.diff.step_jvp(kernel)`` — the forward-mode twin of one step of an
in-place step kernel: one kernel that commits each state plane and its tangent
``dot_<plane>``, so a tangent rides beside the state across launches.

Rows: central differences of a nonlinear step; the analytic derivative of a
decay after many launches; a fused ``steps=4`` kernel equals four single
launches; a sample that terminates keeps its state and tangent; a
``time_step=`` kernel carries its time plane; ``jump=`` and parameter cells are
refused; the reverse twin is tagged so the broadcast door can rebuild it."""

from __future__ import annotations

import numpy as np
import pytest
from _eval import evaluate

import hawk
from hawk import Mutable, Param, Scalar, Terminated, kernel
from hawk import math as hm
from hawk.diff import step_jvp, step_vjp
from hawk.ir import HawkError


@kernel(steps=1)
def decay(a: Param, terminated: Terminated, x: Mutable[Scalar]):
    x = x * a
    terminated = x < 0.5


@kernel(steps=1)
def pend(dt: Param, terminated: Terminated, x: Mutable[Scalar], v: Mutable[Scalar]):
    vn = v - dt * hm.sin(x)
    x = x + dt * v
    v = vn
    terminated = x > 10.0


def test_matches_central_differences():
    env = dict(dt=0.05, terminated=False, x=0.7, v=-0.4)
    fwd = step_jvp(pend)
    got = evaluate(fwd.sinks, {**env, "dot_dt": 1.0, "dot_x": 0.3, "dot_v": -0.2})
    h = 1e-6

    def after(**kw):
        out = evaluate(pend.sinks, {**env, **kw})
        return np.array([float(out["x"]), float(out["v"])])

    d = (after(x=0.7 + h * 0.3, v=-0.4 - h * 0.2, dt=0.05 + h)
         - after(x=0.7 - h * 0.3, v=-0.4 + h * 0.2, dt=0.05 - h)) / (2 * h)
    np.testing.assert_allclose([float(got["dot_x"]), float(got["dot_v"])], d, rtol=1e-6)


def _launch(k, launches, **state):
    for _ in range(launches):
        hawk.run(k, **state)
    return state


def _built(tmp_path, *kernels):
    hawk.build(list(kernels), tmp_path, targets=("host",))
    return [hawk.load(tmp_path, k.name) for k in kernels]


def _state():
    return dict(x=np.array([1.0, 2.0]), dot_x=np.array([0.0, 1.0]),
                terminated=np.zeros(2, bool))


def test_the_analytic_decay_derivative_and_k4_equals_k1(tmp_path):
    fwd = step_jvp(kernel(_slow, steps=1))
    fused = hawk.steps(fwd, 4)
    single, four = _built(tmp_path, fwd, fused)
    a, x0 = 0.9, np.array([1.0, 2.0])
    one = _launch(single, 12, a=a, dot_a=1.0, **_state())
    quad = _launch(four, 3, a=a, dot_a=1.0, **_state())
    exact = 12 * x0 * a ** 11 + np.array([0.0, 1.0]) * a ** 12
    np.testing.assert_allclose(one["x"], x0 * a ** 12, rtol=1e-14)
    np.testing.assert_allclose(one["dot_x"], exact, rtol=1e-13)
    np.testing.assert_array_equal(quad["x"], one["x"])
    np.testing.assert_allclose(quad["dot_x"], one["dot_x"], rtol=1e-14)


def _slow(a: Param, terminated: Terminated, x: Mutable[Scalar]):
    x = x * a
    terminated = x < 0.0


def test_a_terminated_sample_keeps_its_state_and_tangent(tmp_path):
    fwd = step_jvp(decay)
    (run,) = _built(tmp_path, fwd)
    a, x0 = 0.9, np.array([1.0, 2.0])
    out = _launch(run, 20, a=a, dot_a=1.0, **_state())
    n = np.array([7, 14])                                  # first a^n x0 < 0.5
    assert out["terminated"].all()
    np.testing.assert_allclose(out["x"], x0 * a ** n, rtol=1e-14)
    np.testing.assert_allclose(out["dot_x"], n * x0 * a ** (n - 1) + np.array([0.0, 1.0]) * a ** n,
                               rtol=1e-13)


def test_a_time_step_kernel_carries_its_time_plane():
    @kernel(steps=1, time="t", time_step="dt")
    def timed(dt: Param, terminated: Terminated, x: Mutable[Scalar], t: Mutable[Scalar]):
        x = x + dt * t
        t = t + dt
        terminated = t > 5.0

    fwd = step_jvp(timed)
    assert (fwd.time, fwd.time_step) == ("t", "dt")
    got = evaluate(fwd.sinks, dict(dt=0.1, terminated=False, x=1.0, t=0.5, dot_dt=1.0,
                                   dot_x=0.0, dot_t=0.0))
    assert float(got["dot_t"]) == 1.0 and float(got["dot_x"]) == pytest.approx(0.5)


def test_jump_and_parameter_cells_are_refused():
    @kernel(steps=1, jump=0.1)
    def jumpy(a: Param, terminated: Terminated, x: Mutable[Scalar]):
        x = hm.where(x < a, x * 2.0, x)
        terminated = x > 100.0

    with pytest.raises(HawkError, match="jump="):
        step_jvp(jumpy)
    cells = hawk.diff.param_planes(decay, ("a",))
    with pytest.raises(HawkError, match="parameter cells"):
        step_jvp(cells)


def test_a_step_vjp_is_tagged_for_the_broadcast_door():
    back = step_vjp(decay)
    assert back.source is decay and back.lane_space == getattr(decay, "lane_space", ((None, None),))
    assert back.rebuild(decay).primal == back.primal
