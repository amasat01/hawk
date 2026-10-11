# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``@hawk.kernel(time="t", time_step="dt")``: the time plane in the taping
twin, and the ``<name>_refine`` entry.

Rows:

* **declaration** — the plain kernel is unchanged by the declaration (same
  digest, same planes); the structural check ``t = t + dt`` refuses any other
  update, a ``time=`` without ``time_step=``, a ``time_step`` the body never
  reads, and a time plane read but never updated;
* **index form in the twin** — the step size is a per-sample plane, the time
  plane is ``t_0 + k*dt`` with ``k = applied_steps``: after a run, ``t`` equals
  ``t_0 + k*dt`` BITWISE per sample (one multiply, one add), for a per-sample
  ``dt``, compiled on the host (``cuda`` arm marked ``gpu``); a body that
  declares no time plane gets one minted, and ``t`` is taped;
* **refine entry** — ``d t_refined / d tau == 1`` EXACTLY (symbolically and
  compiled), the guard ``event_g0`` is ``g`` at the refined state; at
  ``tau = dt`` the refined state equals the stepping path's (bitwise for the
  state, within one ulp for ``t``); a kernel with no ``time_step``, or one that
  never finishes, refuses.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from _eval import evaluate

import hawk
from hawk import Kernel, Mutable, Param, Scalar, Terminated, kernel
from hawk.diff import jvp, tape
from hawk.diff.tape import refine_entry, taped_planes
from hawk.ir import HawkError

N = 29


def osc(dt: Param, a: Param, terminated: Terminated,
              x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    v_start = v
    v = v_start - dt * x
    x = x + dt * v_start
    tn = t + dt
    t = tn
    terminated = x <= a * tn


osc_plain = kernel(osc, steps=1)
osc = kernel(osc, steps=1, time="t", time_step="dt")


@kernel(steps=1, time="t", time_step="h")
def fall(h: Param, terminated: Terminated, y: Mutable[Scalar], v: Mutable[Scalar]):
    """No time plane in the body: the twin mints ``t``."""
    v_start = v
    v = v_start - 9.81 * h
    y = y + h * v_start
    terminated = y < 0.0


def test_the_declaration_leaves_the_plain_kernel_unchanged():
    assert osc.walk.digest == osc_plain.walk.digest
    assert osc.walk.arg_spec == osc_plain.walk.arg_spec
    assert (osc.time, osc.time_step) == ("t", "dt")
    assert osc_plain.time is None


def test_the_structural_check_refuses_other_updates():
    with pytest.raises(HawkError, match=r"must advance as `t = t \+ dt`"):
        @kernel(steps=1, time="t", time_step="dt")
        def doubled(dt: Param, x: Mutable[Scalar], t: Mutable[Scalar]):
            x = x + dt
            t = t + 2.0 * dt

    with pytest.raises(HawkError, match=r"needs time_step="):
        @kernel(steps=1, time="t")
        def no_step(dt: Param, x: Mutable[Scalar], t: Mutable[Scalar]):
            x = x + dt
            t = t + dt

    with pytest.raises(HawkError, match=r"must name a scalar floating-point input"):
        @kernel(steps=1, time="t", time_step="h")
        def unread(dt: Param, x: Mutable[Scalar], t: Mutable[Scalar]):
            x = x + dt
            t = t + dt

    with pytest.raises(HawkError, match=r"is read but never updated"):
        @kernel(steps=1, time="t", time_step="dt")
        def frozen(dt: Param, t: Scalar, x: Mutable[Scalar]):
            x = x + dt * t


def test_the_twin_promotes_the_step_and_writes_index_form_time():
    tw = tape(osc, 8)
    slots = dict(tw.walk.arg_spec)
    spec = set(tw.walk.arg_spec)
    assert ("per_sample", "dt") in spec and ("uniform", "dt") not in spec
    assert ("per_sample", "t_0") in spec
    assert [n for n, _ in taped_planes(tw.step or tw)] == ["t", "v", "x"]
    del slots
    minted = tape(fall, 4)
    assert ("mutable", "t") in set(minted.walk.arg_spec)
    assert "t" in [n for n, _ in taped_planes(minted.step or minted)]


@pytest.fixture(scope="module")
def built(tmp_path_factory, cache_dir):
    root = pathlib.Path(tmp_path_factory.mktemp("time_plane"))
    kernels = [tape(osc, 64), tape(fall, 64), refine_entry(osc),
               Kernel("osc_refine_jvp", jvp(refine_entry(osc), wrt=("dt_tau",)))]
    hawk.build(kernels, root, targets=("cuda", "host"), cache_dir=cache_dir,
               host_profile="native")
    return root


def _run_twin(root, name, planes, dt, t0, *, a=None):
    n = len(dt)
    planes = dict(planes, terminated=np.zeros(n, dtype=bool),
                  applied_steps=np.zeros(n, dtype=np.int64),
                  tape=np.zeros((64 * 3, n)), tape_on=0)
    kw = {"dt": dt, "t_0": t0} if name == "osc_tape" else {"h": dt, "t_0": t0}
    if a is not None:
        kw["a"] = a
    k = hawk.load(root, name)
    for _ in range(200):
        if planes["terminated"].all():
            break
        hawk.run(k, **kw, **planes)
    return planes


def test_twin_time_is_index_form_bitwise_per_sample(built):
    rng = np.random.default_rng(5)
    dt = rng.uniform(0.01, 0.1, N)
    t0 = rng.uniform(0.0, 3.0, N)
    planes = {"x": rng.uniform(0.2, 1.0, N), "v": rng.uniform(-1.0, 1.0, N),
              "t": t0.copy()}
    got = _run_twin(built, "osc_tape", planes, dt, t0, a=0.05)
    k = got["applied_steps"]
    assert (k > 1).all() and len(set(k.tolist())) > 1
    np.testing.assert_array_equal(got["t"], t0 + k * dt)
    accumulated = t0.copy()
    for n in range(N):
        for _ in range(int(k[n]) - 1):
            accumulated[n] += dt[n]
    # non-vacuity: accumulating t + dt step by step rounds differently somewhere
    assert not np.array_equal(accumulated + dt, got["t"])


def test_a_minted_time_plane_is_index_form_too(built):
    rng = np.random.default_rng(6)
    h = rng.uniform(0.01, 0.05, N)
    t0 = rng.uniform(0.0, 1.0, N)
    planes = {"y": rng.uniform(0.5, 2.0, N), "v": np.zeros(N), "t": np.zeros(N)}
    got = _run_twin(built, "fall_tape", planes, h, t0)
    np.testing.assert_array_equal(got["t"], t0 + got["applied_steps"] * h)


def test_refine_entry_dt_dtau_is_one_symbolically():
    derived = jvp(refine_entry(osc), wrt=("dt_tau",))
    env = {"x": 0.3, "v": -0.7, "dt": 0.05, "dt_tau": 0.021, "t_0": 1.5,
           "applied_steps": 12, "a": 0.4, "dot_dt_tau": 1.0}
    got = evaluate(derived, env)
    assert got["dot_t_refined"] == 1.0
    assert got["dot_x_refined"] == env["v"]
    assert got["dot_event_g0"] == pytest.approx(env["v"] - env["a"])


def test_refine_entry_compiled_dt_dtau_is_one_and_tau_h_is_the_step(built):
    rng = np.random.default_rng(8)
    x, v = rng.uniform(0.2, 1.0, N), rng.uniform(-1.0, 1.0, N)
    dt = rng.uniform(0.01, 0.1, N)
    t0 = rng.uniform(0.0, 3.0, N)
    k = rng.integers(0, 50, N).astype(np.int64)
    out = {n: np.zeros(N) for n in ("x_refined", "v_refined", "t_refined",
                                    "event_g0")}
    dots = {f"dot_{n}": np.zeros(N) for n in out}
    common = dict(x=x, v=v, dt=dt, t_0=t0, applied_steps=k, a=0.4)
    hawk.run(hawk.load(built, "osc_refine_jvp"), dt_tau=0.5 * dt,
             dot_dt_tau=np.ones(N), **common, **dots)
    assert (dots["dot_t_refined"] == 1.0).all()
    hawk.run(hawk.load(built, "osc_refine"), dt_tau=dt, **common, **out)
    # the stepping path, one step from the same state at k
    np.testing.assert_array_equal(out["x_refined"], x + dt * v)
    np.testing.assert_array_equal(out["v_refined"], v - dt * x)
    stepped_t = t0 + (k + 1) * dt
    assert (np.abs(out["t_refined"] - stepped_t) <= np.spacing(stepped_t)).all()
    np.testing.assert_array_equal(out["event_g0"],
                                  out["x_refined"] - 0.4 * out["t_refined"])


def test_refine_entry_refusals():
    with pytest.raises(HawkError, match=r"declares no time_step="):
        refine_entry(osc_plain)

    @kernel(steps=1, time="t", time_step="dt")
    def drift(dt: Param, x: Mutable[Scalar], t: Mutable[Scalar]):
        x = x + dt
        t = t + dt

    with pytest.raises(HawkError, match=r"never finishes its sample"):
        refine_entry(drift)
    assert refine_entry(osc).name == "osc_refine"


@pytest.mark.gpu
def test_refine_entry_dt_dtau_is_one_on_the_device(built):
    """The device arm of the compiled ``d t / d tau == 1`` row."""
    import _deploy as L
    import cupy as cp
    import eagle.exec as eexec
    from eagle import plan as eplan

    name = "osc_refine_jvp"
    sidecar = json.loads((built / f"{name}.json").read_text())
    plugin = L.device_plugin(built, name, sidecar)
    rng = np.random.default_rng(9)
    planes = {"x": cp.asarray(rng.uniform(0.2, 1.0, N)),
              "v": cp.asarray(rng.uniform(-1.0, 1.0, N)),
              "dt": cp.asarray(rng.uniform(0.01, 0.1, N)),
              "t_0": cp.asarray(rng.uniform(0.0, 3.0, N)),
              "applied_steps": cp.asarray(rng.integers(0, 50, N).astype(np.int64)),
              "dt_tau": cp.asarray(rng.uniform(0.0, 0.01, N)),
              "dot_dt_tau": cp.ones(N)}
    outs = {f"dot_{n}": cp.zeros(N) for n in ("x_refined", "v_refined", "t_refined",
                                              "event_g0")}
    # A derivative binds only the planes it reads: d t / d tau needs no t_0,
    # applied_steps or dt, so the derived kernel does not declare them.
    declared = {n for _, n in sidecar["arg_spec"]}
    planes = {k: v for k, v in planes.items() if k in declared}
    eplan.plan(plugin, structure=eexec.DeviceKernel).bind(a=0.4, **planes, **outs).launch()
    assert (cp.asnumpy(outs["dot_t_refined"]) == 1.0).all()
