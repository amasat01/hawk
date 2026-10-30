# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.diff.step_vjp(hawk.steps(k, K), steps_reverse="fused")``: the reverse
of K steps in ONE launch, and ``jvp`` of a fused kernel.

Rows (oracle: K calls of the per-step ``step_vjp`` over the recorded states):

* **state adjoints** — ``bar_x``/``bar_v``/``bar_t`` equal K per-step reverses bit
  for bit on the host, for every stop class (stop at trip 0..K-1, never,
  terminated on entry), plain and ``time_step=`` kernels; on the device within a
  few ULP.
* **Param gradients** — ``bar_c``/``bar_dt``/``bar_t_0`` sum the K terms in another
  order: within ``8 K eps`` of the sum of the terms' magnitudes; and against
  central differences of the K-step function.
* **entry-terminated lane** — its adjoint passes through (a caller presets
  ``bar_<plane>`` to ``bar_<plane>_next``), its Param gradients stay 0.
* **sidecar / refusals** — the derivative block records ``steps_reverse``; a
  single step, an ``"auto"`` kernel and a bad value refuse.
* **jvp** — ``jvp`` of ``hawk.steps(k, K)`` is ``hawk.steps(step_jvp(k), K)``.
* **embedded C** — the fused reverse builds as a standalone C unit.
"""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import Mutable, Param, Scalar, Terminated, kernel
from hawk import math as hm
from hawk.diff import jvp, step_jvp, step_vjp, vjp
from hawk.ir import HawkError
from hawk.trace.steps import _fused_function

from _eval import evaluate

K = 8
N = 4 * (K + 2)
DT, C = 0.05, 0.3
EPS = np.finfo(float).eps


@kernel(steps=1)
def pend(dt: Param, c: Param, t_final: Scalar, terminated: Terminated,
         x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    a = -hm.sin(x) - c * v * v
    vn = v + dt * a
    x = x + dt * v
    v = vn
    tn = t + dt
    t = tn
    terminated = tn >= t_final


@kernel(steps=1, time="t", time_step="dt")
def timed(dt: Param, c: Param, t_final: Scalar, terminated: Terminated,
          x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    a = -hm.sin(x) - c * v * v + 0.2 * hm.sin(t)
    vn = v + dt * a
    x = x + dt * v
    v = vn
    tn = t + dt
    t = tn
    terminated = tn >= t_final


PRIMALS = {"pend": pend, "timed": timed}
WRT = {"pend": ("x", "v", "t", "dt", "c"), "timed": ("x", "v", "t", "dt", "c", "t_0")}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("steps_reverse")
    kernels, derivative = [], {}
    for name, k in PRIMALS.items():
        fused = hawk.steps(k, K)
        fz = hawk.Kernel(f"{name}_fz", step_vjp(fused, wrt=WRT[name], steps_reverse="fused"))
        sv = hawk.Kernel(f"{name}_sv", step_vjp(k, wrt=WRT[name]))
        kernels += [k, fused, fz, sv]
        derivative[fz.name] = (fused.name, "0" * 64)
        derivative[sv.name] = (k.name, "0" * 64)
    hawk.build(kernels, out, targets=("host",), derivative=derivative)
    return {n: hawk.load(out, n) for n in [k.name for k in kernels]}, out


def _decl(art):
    return {n for _, n in art.arg_spec}


def _host_runner(arts):
    def run(name, **arrays):
        hawk.run(arts[name], **{k: v for k, v in arrays.items() if k in _decl(arts[name])})
    return run


def _reverse(run, arts, name, cls, *, timed_kernel):
    """``(per-step reference, fused)`` bar planes of one run of ``N`` lanes whose
    lane ``i`` stops by class ``cls[i]``: ``0..K-1`` after that trip, ``K`` never,
    ``K+1`` terminated on entry."""
    primal, fz, sv = arts[name], arts[f"{name}_fz"], arts[f"{name}_sv"]
    rng = np.random.default_rng(3)
    x0, v0 = rng.uniform(.3, 1, N), rng.uniform(-.5, .5, N)
    t00 = rng.uniform(0, 1, N) if timed_kernel else np.zeros(N)
    k0 = (np.arange(N) % 4).astype(np.int32) if timed_kernel else np.zeros(N, np.int32)
    tf = np.where(cls < K, t00 + (k0 + cls + 0.5) * DT, 1e9)
    term0 = cls == K + 1
    tf = np.where(term0, t00, tf)
    x, v, term = x0.copy(), v0.copy(), term0.copy()
    hist, applied = [], []
    for i in range(K):
        t = t00 + (k0 + i) * DT
        hist.append((x.copy(), v.copy(), t.copy(), (k0 + i).astype(np.int32)))
        applied.append(~term.copy())
        st = dict(x=x, v=v, t=t.copy(), terminated=term, dt=DT, c=C, t_final=tf)
        run(name, **st)
    rs = np.random.default_rng(7)
    bx, bv, bt = (rs.normal(size=N) for _ in range(3))
    names = ("bar_x", "bar_v", "bar_t", "bar_dt", "bar_c") + (("bar_t_0",) if timed_kernel else ())
    cur = dict(bar_x=bx.copy(), bar_v=bv.copy(), bar_t=bt.copy())
    acc = {n: np.zeros(N) for n in names[3:]}
    mag = {n: np.zeros(N) for n in names[3:]}
    for i in reversed(range(K)):
        hx, hv, ht, hk = hist[i]
        o = {n: np.zeros(N) for n in names}
        a = dict(x=hx, v=hv, t=ht, applied_steps=hk, dt=np.full(N, DT) if timed_kernel else DT,
                 c=C, t_final=tf, terminated=term0,
                 bar_x_next=cur["bar_x"], bar_v_next=cur["bar_v"], bar_t_next=cur["bar_t"])
        run(f"{name}_sv", **o, **a)
        live = applied[i]
        for n in cur:
            cur[n] = np.where(live, o[n], cur[n])
        for n in acc:
            acc[n] = acc[n] + np.where(live, o[n], 0.0)
            mag[n] = mag[n] + np.where(live, np.abs(o[n]), 0.0)
    ref = {**cur, **acc}
    o = {n: np.zeros(N) for n in names}
    o["bar_x"][:], o["bar_v"][:], o["bar_t"][:] = bx, bv, bt
    a = dict(x=x0, v=v0, t=t00 + k0 * DT, applied_steps=k0, t_0=t00,
             dt=np.full(N, DT) if timed_kernel else DT, c=C, t_final=tf, terminated=term0,
             bar_x_next=bx, bar_v_next=bv, bar_t_next=bt)
    run(f"{name}_fz", **o, **a)
    return ref, o, mag, term0


@pytest.mark.parametrize("name", ["pend", "timed"])
@pytest.mark.parametrize("stops", ["never", "every_class"])
def test_state_adjoints_bit_equal_and_param_gradients_within_bound(built, name, stops):
    arts, _ = built
    cls = np.full(N, K) if stops == "never" else np.arange(N) % (K + 2)
    ref, got, mag, term0 = _reverse(_host_runner(arts), arts, name, cls, timed_kernel=name == "timed")
    for n in ("bar_x", "bar_v", "bar_t"):
        np.testing.assert_array_equal(got[n], ref[n], err_msg=n)
    for n in mag:
        bound = 8 * K * EPS * mag[n]
        assert (np.abs(got[n] - ref[n]) <= bound + 1e-300).all(), (
            n, float(np.max(np.abs(got[n] - ref[n]) - bound)))
        if term0.any():
            assert (got[n][term0] == 0.0).all(), n


def test_param_gradients_match_central_differences():
    step = pend
    Kd = 4
    fn = _fused_function(step, Kd)
    env = dict(dt=0.05, c=0.3, t_final=9.0, terminated=False, x=0.7, v=-0.4, t=0.0)
    wa, wb, wc = 1.3, -0.6, 0.4
    grads = evaluate(vjp(fn, wrt=("x", "v", "dt", "c")),
                     {**env, "bar_x_next": wa, "bar_v_next": wb, "bar_t_next": wc})

    def after(**kw):
        out = evaluate(fn.sinks, {**env, **kw})
        return wa * float(out["x_next"]) + wb * float(out["v_next"]) + wc * float(out["t_next"])

    h = 1e-6
    for name in ("x", "v", "dt", "c"):
        fd = (after(**{name: env[name] + h}) - after(**{name: env[name] - h})) / (2 * h)
        assert float(grads[f"bar_{name}"]) == pytest.approx(fd, rel=1e-6, abs=1e-9), name


def test_the_derivative_records_its_choice_and_refusals(built):
    _, out = built
    import json
    block = json.loads((out / "pend_fz.json").read_text())["derivative"]
    assert block["steps_reverse"] == {"choice": "fused", "steps": K}
    assert block["primal"] == f"{pend.name}_x{K}"
    plain = json.loads((out / "pend_sv.json").read_text())["derivative"]
    assert "steps_reverse" not in plain
    with pytest.raises(HawkError, match="fixed integer K"):
        step_vjp(pend, steps_reverse="fused")
    with pytest.raises(HawkError, match="fixed integer K"):
        step_vjp(hawk.steps(pend, "auto"), steps_reverse="fused")
    with pytest.raises(HawkError, match="'per_step' or 'fused'"):
        step_vjp(hawk.steps(pend, K), steps_reverse="fast")


def test_jvp_of_a_fused_kernel_is_the_fused_step_jvp():
    fused = hawk.steps(pend, 4)
    got = jvp(fused)
    want = hawk.steps(step_jvp(pend), 4)
    assert got.name == want.name == f"{pend.name}_jvp_x4"
    assert [type(s) for s in got.sinks] == [type(s) for s in want.sinks]
    assert got.walk.arg_spec == want.walk.arg_spec


def test_the_fused_reverse_builds_as_a_standalone_c_unit(tmp_path):
    fused = hawk.steps(pend, K)
    fz = hawk.Kernel("pend_fz", step_vjp(fused, steps_reverse="fused"))
    hawk.build([fz], tmp_path, targets=("c",), derivative={fz.name: (fused.name, "0" * 64)})
    assert any(tmp_path.glob("pend_fz*.c")) or any(tmp_path.glob("pend_fz*"))


def _device_runner(bundle, arts):
    import cupy as cp
    import eagle.exec as eexec
    from conftest import sidecar_of
    from eagle import plan as eplan

    import _deploy as L

    def run(name, **arrays):
        arrays = {k: v for k, v in arrays.items() if k in _decl(arts[name])}
        dev = {k: cp.asarray(v.astype(np.int64) if v.dtype == np.int32 else v)   # the wire is 64-bit
               if isinstance(v, np.ndarray) else v for k, v in arrays.items()}
        sidecar = sidecar_of(bundle, name)
        if sidecar.get("finish"):
            dev["finished_count"] = cp.zeros(1, dtype=cp.uint32)
        plugin = L.device_plugin(bundle.directory, name, sidecar)
        eplan.plan(plugin, structure=eexec.DeviceKernel).bind(**dev).launch()
        for k, v in arrays.items():
            if isinstance(v, np.ndarray) and v.dtype != np.int32:
                v[...] = dev[k].get()

    return run


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["pend", "timed"])
def test_the_fused_reverse_on_the_device_matches_the_per_step_reverse(tmp_path_factory, name):
    from hawk.artifact import build_bundle

    k = PRIMALS[name]
    fused = hawk.steps(k, K)
    fz = hawk.Kernel(f"{name}_fz", step_vjp(fused, wrt=WRT[name], steps_reverse="fused"))
    sv = hawk.Kernel(f"{name}_sv", step_vjp(k, wrt=WRT[name]))
    bundle = build_bundle([k, fused, fz, sv], tmp_path_factory.mktemp("steps_reverse_dev"),
                          targets=("cuda", "host"),
                          derivative={fz.name: (fused.name, "0" * 64), sv.name: (k.name, "0" * 64)})
    from hawk.runtime import load
    arts = {a.name: load(bundle.directory, a.name, a.sidecar) for a in bundle.artifacts}
    cls = np.arange(N) % (K + 2)
    ref, got, mag, _ = _reverse(_device_runner(bundle, arts), arts, name, cls,
                                timed_kernel=name == "timed")
    for n, r in ref.items():
        # nvcc keeps FMA contraction: a few ULP per step; Param sums within their magnitude
        scale = mag[n] if n in mag else np.abs(r)
        np.testing.assert_allclose(got[n], r, rtol=1e-11, atol=1e-11 * (1 + scale.max()),
                                   err_msg=n)
