# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.diff.tape(kernel, window, events="refine")`` — the closed, tolerant
stop test — and the ``spacing`` op it is built on.

Rows:

* **spacing** — ``hawk.math.spacing`` equals ``np.spacing`` bitwise (signed,
  zero, subnormal, huge), compiled on the host (``cuda`` arm marked ``gpu``);
  its derivative is exactly zero;
* **the tolerance** — on the moving surface ``x <= a*t`` the twin stops a
  sample whose guard is 3 tolerance-ulps above the surface and not one 5 above,
  while the kernel's own test (``events="step"``) stops neither: the
  tolerance is ``4 * spacing(|t|) * |dg/dt|`` with ``dg/dt = -a`` read
  from the guard itself;
* **the grid** — a fixed horizon ``T = t_0 + m*h`` on the step grid, written
  with a STRICT ``t > T``: over 1194 ``(h, m)`` pairs the refine twin stops at
  step ``m`` every time, while the kernel's own strict test, run through the
  same twin, stops one step late wherever ``t`` lands exactly on ``T`` (the
  non-vacuity half);
* **refusals** — an unknown ``events``, and ``"refine"`` without
  ``time_step=``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from _eval import evaluate

import hawk
import hawk.math as M
from hawk import Mutable, Param, Scalar, Terminated, kernel
from hawk.diff import jvp, tape
from hawk.ir import HawkError


@kernel(steps=1)
def spaced(x: Scalar, y: Mutable[Scalar]):
    y = M.spacing(x)


@kernel(steps=1, time="t", time_step="dt")
def moving(dt: Param, a: Param, terminated: Terminated,
           x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    v_start = v
    v = v_start - dt * x
    x = x + dt * v_start
    tn = t + dt
    t = tn
    terminated = x <= a * tn


@kernel(steps=1, time="t", time_step="dt")
def horizon(dt: Param, T: Scalar, terminated: Terminated, t: Mutable[Scalar]):
    tn = t + dt
    t = tn
    terminated = tn > T          # STRICT on purpose: the non-vacuity half


_VALUES = np.array([0.0, -0.0, 1.0, -1.0, 1.5, -6.0, 1e300, -1e-300, 5e-324,
                    np.finfo(float).tiny, 0.1, 123456.789])


@pytest.fixture(scope="module")
def built(tmp_path_factory, cache_dir):
    root = pathlib.Path(tmp_path_factory.mktemp("closed_stop"))
    hawk.build([spaced, tape(horizon, 4, events="refine")], root,
               targets=("cuda", "host"), cache_dir=cache_dir)
    return root


def test_spacing_is_np_spacing_compiled_on_the_host(built):
    y = np.zeros(len(_VALUES))
    hawk.run(hawk.load(built, "spaced"), x=_VALUES.copy(), y=y)
    np.testing.assert_array_equal(y, np.spacing(_VALUES))


@pytest.mark.gpu
def test_spacing_is_np_spacing_compiled_on_the_device(built):
    import json

    import _deploy as L
    import cupy as cp
    import eagle.exec as eexec
    from eagle import plan as eplan

    sidecar = json.loads((built / "spaced.json").read_text())
    x, y = cp.asarray(_VALUES), cp.zeros(len(_VALUES))
    eplan.plan(L.device_plugin(built, "spaced", sidecar),
               structure=eexec.DeviceKernel).bind(x=x, y=y).launch()
    np.testing.assert_array_equal(cp.asnumpy(y), np.spacing(_VALUES))


def test_spacing_carries_a_zero_derivative():
    (sink,) = jvp(spaced, wrt=("x",))
    assert evaluate((sink,), {"x": 1.5, "dot_x": 1.0})["dot_y"] == 0.0


def _stops(twin, x_next, a):
    """The twin's stop test at a state whose post-step ``x`` is ``x_next``
    and post-step time 1.5 (``t_0 = 1``, ``k = 0``, ``dt = 0.5``, ``v = 0``)."""
    finish = next(s for s in twin.sinks if type(s).__name__ == "Finish")
    env = {"x": x_next, "v": 0.0, "t": 1.0, "dt": 0.5, "t_0": 1.0,
           "applied_steps": 0, "a": a, "terminated": False}
    return bool(evaluate((finish,), env)["terminated"])


def test_the_tolerance_is_four_ulps_of_t_times_dg_dt():
    a = 4.0
    surface = a * 1.5                       # 6.0, so one ulp of x is 4*spacing(1.5)
    tol_ulp = np.spacing(surface)
    assert tol_ulp == a * np.spacing(1.5)
    above3, above5 = surface + 3 * tol_ulp, surface + 5 * tol_ulp
    refine, own = tape(moving, 4, events="refine"), tape(moving, 4)
    assert _stops(refine, above3, a) and not _stops(refine, above5, a)
    assert not _stops(own, above3, a) and not _stops(own, above5, a)
    assert _stops(own, surface, a) and _stops(refine, surface, a)


def _grid():
    hs, ms = [], []
    for h in (0.1, 0.01, 1e-3, 0.05, 0.2, 0.3):
        for m in range(1, 200):
            hs.append(h)
            ms.append(m)
    return np.array(hs), np.array(ms)


def _run(root, name, h, T):
    n = len(h)
    p = dict(t=np.zeros(n), terminated=np.zeros(n, dtype=bool),
             applied_steps=np.zeros(n, dtype=np.int64), tape=np.zeros((4, n)),
             tape_on=0, dt=h, t_0=np.zeros(n), T=T)
    k = hawk.load(root, name)
    for _ in range(205):
        if p["terminated"].all():
            break
        hawk.run(k, **p)
    return p["applied_steps"]


def test_a_horizon_on_the_grid_stops_at_its_step(built):
    h, m = _grid()
    assert len(h) == 1194
    T = m * h
    refine = _run(built, "horizon_tape", h, T)
    np.testing.assert_array_equal(refine, m)
    # the same twin build under events="step": the kernel's own strict test
    step_root = built.parent / "closed_stop_step"
    hawk.build([tape(horizon, 4)], step_root, targets=("host",))
    late = _run(step_root, "horizon_tape", h, T)
    assert (late >= m).all() and (late == m + 1).sum() > 0


def test_refusals():
    with pytest.raises(HawkError, match=r"events is 'step' .* or 'refine'"):
        tape(moving, 4, events="exact")

    @kernel(steps=1)
    def untimed(dt: Param, terminated: Terminated, x: Mutable[Scalar]):
        x = x - dt
        terminated = x < 0.0

    with pytest.raises(HawkError, match=r"declare it with @hawk.kernel\(time_step="):
        tape(untimed, 4, events="refine")
