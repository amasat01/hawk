# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.diff.step_vjp(kernel)`` — the vjp of one step of an in-place step
kernel.

Rows:

* **central differences** — on a nonlinear step, ``bar_<plane>`` from the
  vjp (seeded ``bar_<plane>_next``) equals central differences of
  ``J = a·x' + b·v'`` through one step, for the state and a parameter;
* **naming contract** — reads the planes it needs under their own names and
  ``bar_<plane>_next``, writes ``bar_<plane>``; no stop rule, no tape;
  tagged as the vjp of the single step; a ``steps=K`` and a default
  automatic kernel give the same planes as the single step; a
  ``time_step=`` kernel reads the twin's index-form planes;
* **plain vjp still refuses** an in-place step.
"""

from __future__ import annotations

import numpy as np
import pytest
from _eval import evaluate

import hawk
from hawk import Mutable, Param, Scalar, Terminated, kernel
from hawk import math as hm
from hawk.ir import HawkError
from hawk.diff import step_vjp, vjp


def _pend(dt: Param, c: Param, t_final: Scalar, terminated: Terminated,
          x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    a = -hm.sin(x) - c * v * v
    vn = v + dt * a
    x = x + dt * v
    v = vn
    tn = t + dt
    t = tn
    terminated = tn >= t_final


pend = kernel(_pend, steps=1)


def _after(env):
    """``(x', v')`` of one step at ``env``."""
    out = evaluate(pend.sinks, env)
    return float(out["x"]), float(out["v"])


def test_matches_central_differences():
    env = dict(dt=0.05, c=0.3, t_final=9.0, terminated=False, x=0.7, v=-0.4, t=0.0)
    a, b = 1.3, -0.6
    grads = evaluate(step_vjp(pend), {**env, "bar_x_next": a, "bar_v_next": b,
                                      "bar_t_next": 0.0})
    h = 1e-6
    for name in ("x", "v", "c", "dt"):
        up, down = dict(env), dict(env)
        up[name] += h
        down[name] -= h
        (xu, vu), (xd, vd) = _after(up), _after(down)
        fd = (a * (xu - xd) + b * (vu - vd)) / (2 * h)
        assert float(grads[f"bar_{name}"]) == pytest.approx(fd, rel=1e-7, abs=1e-9), name


def test_naming_contract():
    back = hawk.Kernel("pend_step_vjp", step_vjp(pend))
    names = {n for _role, n in back.walk.arg_spec}
    # reads only the values its derivative needs (t enters linearly, so it is not read)
    assert {"x", "v", "dt", "c", "bar_x_next", "bar_v_next", "bar_t_next"} <= names
    assert {"bar_x", "bar_v", "bar_t", "bar_dt", "bar_c"} <= names
    assert not names & {"terminated", "tape", "tape_on", "applied_steps", "finished_count"}
    tag = step_vjp(pend)
    assert tag.kind == "vjp" and tag.primal == "_pend"
    for other in (hawk.steps(pend, 4), kernel(_pend)):
        again = hawk.Kernel("other_step_vjp", step_vjp(other))
        assert again.walk.arg_spec == back.walk.arg_spec
    timed = kernel(_pend, steps=1, time="t", time_step="dt")
    names = {n for _r, n in hawk.Kernel("timed_step_vjp", step_vjp(timed)).walk.arg_spec}
    assert {"applied_steps", "bar_t_0"} <= names       # the index form t_0 + k dt


def test_plain_vjp_still_refuses_an_in_place_step():
    with pytest.raises(HawkError, match="read before it is assigned"):
        vjp(pend)
