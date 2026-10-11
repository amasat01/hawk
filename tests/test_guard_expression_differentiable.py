# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A step kernel's stop rule read as a smooth guard (``hawk.diff.event``).

Rows:

* **sign convention** — for every comparison form, ``g <= 0`` exactly where
  the finish condition holds (evaluated on random states, both sides of the
  surface, plus a point ON it);
* **or** — one guard per comparison, in operand order;
* **differentiable** — the guard kernel's ``jvp`` w.r.t. the state AND a
  parameter equals central differences of ``g``; a moving surface
  ``g = x - a*t`` keeps its ``t`` and ``a`` dependence;
* **refusals** — ``and``, ``not``, ``==``, an integer count, a class-test
  flag and a rounding on the guard's path each refuse with the pinned
  message naming the kernel; a kernel that never finishes, and a fused
  ``steps=K`` kernel, refuse too.
"""

from __future__ import annotations

import numpy as np
import pytest
from _eval import evaluate

import hawk
from hawk import Index, Mutable, Param, Scalar, Terminated, kernel
from hawk.diff import jvp
from hawk.diff.event import event_guard, guard_kernel
from hawk.ir import HawkError

_H = 1e-6
_REFUSAL = (r'events="refine": the stop rule of {name!r} is not a smooth crossing '
            r"\(.*\); refine needs g\(x\) that changes sign, not a count or a flag")


@kernel(steps=1)
def falls(dt: Param, terminated: Terminated, y: Mutable[Scalar], v: Mutable[Scalar]):
    v_start = v
    v = v_start - 9.81 * dt
    y = y + dt * v_start
    terminated = y < 0.0


@kernel(steps=1)
def horizon(dt: Param, t_final: Scalar, terminated: Terminated,
            x: Mutable[Scalar], t: Mutable[Scalar]):
    x = x - dt * x
    tn = t + dt
    t = tn
    terminated = tn >= t_final


@kernel(steps=1)
def moving(dt: Param, a: Param, terminated: Terminated,
           x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    v_start = v
    v = v_start - dt * x
    x = x + dt * v_start
    tn = t + dt
    t = tn
    terminated = x <= a * tn


@kernel(steps=1)
def either(dt: Param, terminated: Terminated, x: Mutable[Scalar], t: Mutable[Scalar]):
    x = x + dt
    tn = t + dt
    t = tn
    terminated = hawk.math.lor(x > 1.0, tn >= 2.0)


def _env(rng, names):
    return {n: float(rng.uniform(-1.5, 1.5)) for n in names}


@pytest.mark.parametrize("k,inputs", [
    (falls, ("dt", "y", "v")),
    (horizon, ("dt", "t_final", "x", "t")),
    (moving, ("dt", "a", "x", "v", "t")),
])
def test_g_is_nonpositive_exactly_where_the_sample_stops(k, inputs):
    eg = event_guard(k)
    assert len(eg.guards) == 1 and eg.mask == "terminated"
    finish = next(s for s in k.sinks if type(s).__name__ == "Finish")
    gk = guard_kernel(k)
    rng = np.random.default_rng(7)
    flips = set()
    for _ in range(200):
        env = _env(rng, inputs)
        stops = bool(evaluate((finish,), env)["terminated"])
        g = evaluate(gk.sinks, env)["event_g0"]
        assert (g <= 0.0) == stops, (k.name, env, g, stops)
        flips.add(stops)
    assert flips == {True, False}, f"{k.name}: the sample never crossed"


def test_a_point_on_the_surface_has_g_zero_and_stops():
    env = {"dt": 0.25, "t_final": 1.0, "x": 0.3, "t": 0.75}
    assert evaluate(guard_kernel(horizon).sinks, env)["event_g0"] == 0.0


def test_an_or_is_one_guard_per_comparison_in_operand_order():
    eg = event_guard(either)
    assert len(eg.guards) == 2
    assert "x" in eg.rules[0] and ">" in eg.rules[0]
    assert ">=" in eg.rules[1]
    got = evaluate(guard_kernel(either).sinks, {"dt": 0.5, "x": 0.25, "t": 1.0})
    assert got["event_g0"] == pytest.approx(1.0 - 0.75)
    assert got["event_g1"] == pytest.approx(2.0 - 1.5)


@pytest.mark.parametrize("k,inputs", [
    (falls, ("dt", "y", "v")),
    (horizon, ("dt", "t_final", "x", "t")),
    (moving, ("dt", "a", "x", "v", "t")),
])
def test_the_guard_s_jvp_matches_central_differences(k, inputs):
    gk = guard_kernel(k)
    derived = jvp(gk)
    rng = np.random.default_rng(11)
    env = _env(rng, inputs)
    env["dt"] = 0.1
    dirs = {n: float(rng.normal()) for n in inputs}
    got = evaluate(derived, {**env, "terminated": False,
                             **{f"dot_{n}": dirs[n] for n in inputs}})
    plus = {n: env[n] + _H * dirs[n] for n in inputs}
    minus = {n: env[n] - _H * dirs[n] for n in inputs}
    fd = (evaluate(gk.sinks, plus)["event_g0"]
          - evaluate(gk.sinks, minus)["event_g0"]) / (2 * _H)
    assert got["dot_event_g0"] == pytest.approx(fd, rel=1e-7, abs=1e-9)


def test_a_moving_surface_keeps_its_time_and_parameter_dependence():
    gk = guard_kernel(moving)
    derived = jvp(gk, wrt=("a", "t"))
    env = {"dt": 0.1, "a": 0.4, "x": 0.2, "v": -0.3, "t": 1.0}
    got = evaluate(derived, {**env, "terminated": False, "dot_a": 1.0, "dot_t": 0.0})
    assert got["dot_event_g0"] == pytest.approx(-(env["t"] + env["dt"]))
    got = evaluate(derived, {**env, "terminated": False, "dot_a": 0.0, "dot_t": 1.0})
    assert got["dot_event_g0"] == pytest.approx(-env["a"])


@kernel(steps=1)
def both(dt: Param, terminated: Terminated, x: Mutable[Scalar]):
    x = x + dt
    terminated = hawk.math.land(x > 1.0, x < 2.0)


@kernel(steps=1)
def negated(dt: Param, terminated: Terminated, x: Mutable[Scalar]):
    x = x + dt
    terminated = hawk.math.lnot(x < 1.0)


@kernel(steps=1)
def equal(dt: Param, terminated: Terminated, x: Mutable[Scalar]):
    x = x + dt
    terminated = x == 1.0


@kernel(steps=1)
def counted(dt: Param, limit: Index, n: Index, terminated: Terminated,
            x: Mutable[Scalar]):
    x = x + dt
    terminated = n >= limit


@kernel(steps=1)
def flagged(dt: Param, terminated: Terminated, x: Mutable[Scalar]):
    x = x / dt
    terminated = hawk.math.isinf(x)


@kernel(steps=1)
def stepwise(dt: Param, terminated: Terminated, x: Mutable[Scalar]):
    x = x + dt
    terminated = hawk.math.floor(x) >= 3.0


@pytest.mark.parametrize("k", [both, negated, equal, counted, flagged, stepwise],
                         ids=lambda k: k.name)
def test_a_guard_with_no_smooth_crossing_refuses_with_the_pinned_message(k):
    with pytest.raises(HawkError, match=_REFUSAL.format(name=k.name)):
        event_guard(k)
    with pytest.raises(HawkError, match=_REFUSAL.format(name=k.name)):
        guard_kernel(k)


def test_a_kernel_that_never_finishes_refuses():
    @kernel(steps=1)
    def drift(dt: Param, x: Mutable[Scalar]):
        x = x + dt

    with pytest.raises(HawkError, match=r"never finishes its sample"):
        event_guard(drift)


def test_a_fused_kernel_refuses_and_the_default_reads_its_single_step():
    with pytest.raises(HawkError, match=r"takes 4 steps per launch"):
        event_guard(hawk.steps(falls, 4))

    @kernel
    def falls_default(dt: Param, terminated: Terminated, y: Mutable[Scalar],
                      v: Mutable[Scalar]):
        v_start = v
        v = v_start - 9.81 * dt
        y = y + dt * v_start
        terminated = y < 0.0

    assert falls_default.step is not None
    assert len(event_guard(falls_default).guards) == 1
