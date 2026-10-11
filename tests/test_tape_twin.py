# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.diff.tape(kernel, window)`` — the taping twin of a step kernel.

Rows (host; the ``cuda`` arm of each compiled row is marked ``gpu``):

* **tape rows == forward state** — single-step twin, ``tape_on = 1``, window
  ``W`` >= the longest run: row ``j`` of every sample equals that sample's
  state BEFORE its step ``j`` in a plain-kernel run, ``applied_steps`` equals
  its stop step (the finishing step counted), rows past it untouched;
* **tape off vs on, bitwise** — the SAME twin entry with ``tape_on = 0`` vs
  ``1``: every state plane, mask, counter and ``applied_steps`` ``==``
  bitwise (one compiled entry), and the ``tape_on = 0`` tape is never written;
* **plain vs twin** — the twin's state equals the plain kernel's (host,
  exact profile);
* **window wrap** — ``W`` smaller than the run: row ``applied_steps mod W``;
* **fused == single** — the ``steps="auto"`` twin: one launch of ``k`` steps
  vs ``k`` one-step launches (word = 1) of the same artifact, tape rows and
  state ``==`` (host bitwise);
* **width derived** — a plane written without being read, and an integer
  state plane, are not taped; the tape is ``(W*w,)`` with ``w`` the taped
  components, ``arg_spec`` order;
* **sample_local** — a taped kernel keeps the plain kernel's access class;
* **plain untouched** — the plain kernel's digest is unchanged by taping;
* **refusals** — a reserved plane name, a bad window, nothing to tape.
"""

from __future__ import annotations

import _deploy as L
import numpy as np
import pytest
from conftest import sidecar_of

import hawk
from hawk import Index, Mutable, Param, Scalar, Terminated, Vector, kernel
from hawk.artifact import build_bundle
from hawk.diff.tape import tape, tape_width, taped_planes
from hawk.ir import HawkError

N = 37
DT = 0.0625
W_BIG = 64


@kernel(steps=1)
def osc(dt: Param, t_final: Scalar, terminated: Terminated,
        x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    v_start = v
    a = -x
    v = v_start + dt * a
    x = x + dt * v_start
    tn = t + dt
    t = tn
    terminated = tn >= t_final


@kernel
def osc_auto(dt: Param, t_final: Scalar, terminated: Terminated,
             x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    v_start = v
    a = -x
    v = v_start + dt * a
    x = x + dt * v_start
    tn = t + dt
    t = tn
    terminated = tn >= t_final


@kernel(steps=1)
def spin(dt: Param, limit: Scalar, terminated: Terminated, r: Mutable[Vector[3]],
         s: Mutable[Scalar], e: Mutable[Scalar], n: Mutable[Index]):
    """A rank-1 state, a scalar state, a write-only plane ``e`` and an
    integer counter ``n`` — only ``r`` and ``s`` are taped."""
    w = hawk.math.vec(0.3, -0.2, 0.7)
    rn = r + hawk.math.cross(w, r) * dt
    r = rn
    sn = s + hawk.math.dot(rn, rn) * dt
    s = sn
    e = sn * sn
    n = n + 1
    terminated = sn >= limit


@kernel(time="t", time_step="dt")
def osc_timed(dt: Param, t_final: Scalar, terminated: Terminated,
              x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    v_start = v
    a = -x
    v = v_start + dt * a
    x = x + dt * v_start
    tn = t + dt
    t = tn
    terminated = tn >= t_final


TWINS = {"osc": tape(osc, W_BIG), "osc_w8": tape(osc, 8),
         "osc_timed": tape(osc_timed, W_BIG),
         "osc_auto": tape(osc_auto, W_BIG), "spin": tape(spin, W_BIG)}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("tape")
    # distinct names: the W=8 twin is the same kernel name, so it gets its own unit
    return {
        "main": build_bundle([osc, TWINS["osc"], TWINS["spin"], spin],
                             root / "main", targets=("cuda", "host"),
                             cache_dir=cache_dir, host_profile="native"),
        "w8": build_bundle([TWINS["osc_w8"]], root / "w8", targets=("cuda", "host"),
                           cache_dir=cache_dir, host_profile="native"),
        "auto": build_bundle([TWINS["osc_auto"], TWINS["osc_timed"]], root / "auto",
                             targets=("cuda", "host"), cache_dir=cache_dir,
                             host_profile="native"),
    }


def _initial(which, xp):
    rng = np.random.default_rng(3)
    if which == "osc":
        stop = rng.integers(3, 40, N)
        return {"t_final": xp.array(DT * stop.astype(float)),
                "x": xp.array(rng.uniform(-1, 1, N)),
                "v": xp.array(rng.uniform(-1, 1, N)), "t": xp.zeros(N)}
    return {"limit": xp.array(rng.uniform(0.1, 0.8, N)),
            "r": xp.array(rng.uniform(0.4, 1.0, (3, N))), "s": xp.zeros(N),
            "e": xp.zeros(N), "n": xp.zeros(N, dtype=xp.int64)}


def _host(a):
    return np.array(a.get() if hasattr(a, "get") else a, copy=True)


class _Run:
    """One bound kernel and its planes, launched by hand."""

    def __init__(self, bundle, name, which, target, *, twin_width=None,
                 window=W_BIG, tape_on=1, timed=False):
        import eagle.exec as eexec
        from eagle import plan as eplan

        sidecar = sidecar_of(bundle, name)
        if target == "host":
            plugin, structure, xp = (L.host_plugin(bundle.directory, name, sidecar),
                                     eexec.HostTeam, np)
        else:
            import cupy as cp

            plugin, structure, xp = (L.device_plugin(bundle.directory, name, sidecar),
                                     eexec.DeviceKernel, cp)
        self.xp = xp
        self.planes = _initial(which, xp)
        self.planes["terminated"] = xp.zeros(N, dtype=bool)
        self.counter = xp.zeros(1, dtype=xp.uint32)
        extra, uniforms = {}, {"dt": DT}
        if timed:   # the time declaration: a per-sample step and the start time
            del uniforms["dt"]
            rng = np.random.default_rng(17)
            self.planes["dt"] = xp.asarray(rng.uniform(0.5, 1.5, N) * DT)
            self.planes["t_0"] = xp.zeros(N)
        if twin_width is not None:
            self.planes["applied_steps"] = xp.zeros(N, dtype=xp.int64)  # Int wire
            self.planes["tape"] = xp.full((window * twin_width, N), -7.0)
            uniforms["tape_on"] = tape_on
        if sidecar.get("finish", {}).get("steps") == "auto":
            extra["fused_steps"] = xp.zeros(1, dtype=xp.int64)
        self.extra = extra
        self.bound = eplan.plan(plugin, structure=structure).bind(
            finished_count=self.counter.view(xp.int32), **uniforms,
            **self.planes, **extra)

    def launch(self, k=None):
        if k is not None:
            self.extra["fused_steps"][0] = k
        self.bound.launch()

    def done(self):
        return int(self.counter[0]) >= N

    def snap(self):
        return {k: _host(a) for k, a in self.planes.items()}


def _plain_states(bundle, which, name, target):
    """Per launch of the plain kernel: the state planes before it."""
    run = _Run(bundle, name, which, target)
    before = []
    while not run.done():
        before.append(run.snap())
        run.launch()
        assert len(before) < 200
    return before, run.snap()


def _state_names(twin):
    return [n for n, _ in taped_planes(twin.step or twin)]


def _row_of(snap, names):
    return np.concatenate([np.atleast_2d(snap[n]).reshape(-1, N) for n in names],
                          axis=0)


def _assert_tape_rows(tape_plane, applied, before, names, window, width):
    for n in range(N):
        stop = int(applied[n])
        for j in range(stop):
            want = _row_of(before[j], names)[:, n]
            row = j % window
            got = tape_plane[row * width:(row + 1) * width, n]
            if j + window >= stop:      # the last write to this row
                np.testing.assert_array_equal(got, want, err_msg=f"sample {n} step {j}")
        if stop < window:
            assert (tape_plane[stop * width:, n] == -7.0).all(), n


@pytest.mark.parametrize("which,plain,twin", [("osc", "osc", "osc"),
                                              ("spin", "spin", "spin")])
@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_tape_rows_are_the_pre_step_state_and_applied_steps_the_stop_step(
        bundle, which, plain, twin, target):
    b = bundle["main"]
    before, final = _plain_states(b, which, plain, target)
    tw = TWINS[twin]
    names, width = _state_names(tw), tape_width(tw)
    run = _Run(b, tw.name, which, target, twin_width=width)
    for _ in range(200):
        if run.done():
            break
        run.launch()
    got = run.snap()
    stops = np.array([sum(1 for s in before if not s["terminated"][n])
                      for n in range(N)])
    np.testing.assert_array_equal(got["applied_steps"], stops)
    _assert_tape_rows(got["tape"], got["applied_steps"], before, names, W_BIG, width)
    if target == "host":
        for plane, want in final.items():
            np.testing.assert_array_equal(got[plane], want, err_msg=plane)


@pytest.mark.parametrize("name,key,which", [("osc_tape", "main", "osc"),
                                            ("spin_tape", "main", "spin"),
                                            ("osc_auto_tape", "auto", "osc")])
@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_tape_off_and_on_run_the_same_entry_bitwise(bundle, name, key, which, target):
    tw = {"osc_tape": TWINS["osc"], "spin_tape": TWINS["spin"],
          "osc_auto_tape": TWINS["osc_auto"]}[name]
    width = tape_width(tw)
    runs = {on: _Run(bundle[key], name, which, target, twin_width=width, tape_on=on)
            for on in (0, 1)}
    k_seq = (1, 5, 2, 9)
    launches = 0
    while not runs[1].done() and launches < 200:
        for run in runs.values():
            run.launch(k_seq[launches % 4] if run.extra else None)
        launches += 1
    off, on = runs[0].snap(), runs[1].snap()
    assert int(runs[0].counter[0]) == N
    for plane in off:
        if plane != "tape":
            np.testing.assert_array_equal(off[plane], on[plane], err_msg=plane)
    assert (off["tape"] == -7.0).all()
    assert not (on["tape"] == -7.0).all()


@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_a_small_window_wraps_rows_mod_w(bundle, target):
    before, _ = _plain_states(bundle["main"], "osc", "osc", target)
    tw = TWINS["osc_w8"]
    run = _Run(bundle["w8"], tw.name, "osc", target, twin_width=3, window=8)
    for _ in range(200):
        if run.done():
            break
        run.launch()
    got = run.snap()
    _assert_tape_rows(got["tape"], got["applied_steps"], before, _state_names(tw), 8, 3)


@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_a_fused_launch_of_k_steps_equals_k_single_step_launches(bundle, target):
    tw = TWINS["osc_auto"]
    width = tape_width(tw)
    fused = _Run(bundle["auto"], tw.name, "osc", target, twin_width=width)
    single = _Run(bundle["auto"], tw.name, "osc", target, twin_width=width)
    for k in (3, 7, 1, 12, 64):
        fused.launch(k)
        for _ in range(k):
            single.launch(1)
        a, b = fused.snap(), single.snap()
        for plane in a:
            if target == "host":
                np.testing.assert_array_equal(a[plane], b[plane], err_msg=f"{plane} k={k}")
            else:
                np.testing.assert_allclose(a[plane], b[plane], rtol=1e-12, atol=1e-14,
                                           err_msg=f"{plane} k={k}")
    assert fused.done()


def test_the_width_is_derived_from_the_read_and_overwritten_state():
    assert [n for n, _ in taped_planes(spin)] == ["r", "s"]
    assert tape_width(spin) == 4
    tw = TWINS["spin"]
    assert tw.walk.slot_types[("out", "tape")].shape == (W_BIG * 4,)
    assert [n for n, _ in taped_planes(osc)] == ["t", "v", "x"]
    assert ("mutable", "applied_steps") in tw.walk.arg_spec
    assert ("uniform", "tape_on") in tw.walk.arg_spec


def test_a_taped_kernel_stays_sample_local_and_the_plain_kernel_is_untouched():
    before = osc.walk.digest
    for tw in TWINS.values():
        assert tw.walk.access.cls == "sample_local", tw.name
    assert tape(osc, 4).walk.access.cls == "sample_local"
    assert osc.walk.digest == before
    assert TWINS["osc"].name == "osc_tape" and TWINS["osc_auto"].name == "osc_auto_tape"
    assert TWINS["osc_auto"].walk.finish == ("terminated", "auto")
    assert TWINS["osc_auto"].step.walk.finish == ("terminated", 1)
    assert TWINS["osc"].walk.finish == ("terminated", 1)


def test_refusals_name_the_kernel():
    @kernel(steps=1)
    def reserved(dt: Param, tape_on: Scalar, x: Mutable[Scalar]):
        x = x + dt * tape_on

    with pytest.raises(HawkError, match=r"reserved for the taping twin"):
        tape(reserved, 4)
    for bad in (0, -1, 2.5, True):
        with pytest.raises(HawkError, match=r"window is a positive int"):
            tape(osc, bad)

    @kernel(steps=1)
    def write_only(dt: Param, x: Scalar, y: Mutable[Scalar]):
        y = x * dt

    with pytest.raises(HawkError, match=r"nothing to tape"):
        tape(write_only, 4)


def test_a_fixed_k_kernel_tapes_through_its_single_step():
    """``steps=K`` keeps its single step as ``.step``: the twin is the fused
    twin of that step (a runtime sets the trip count, at most K), the same
    twin an automatic kernel of the same body gets."""
    fixed = hawk.steps(osc, 4)
    assert fixed.step is osc and fixed.name == "osc_x4"
    twin, auto = tape(fixed, 8), tape(osc_auto, 8)
    assert ("lookup", "fused_steps") in twin.walk.arg_spec
    assert twin.walk.arg_spec == auto.walk.arg_spec
    assert twin.one_step.walk.arg_spec == tape(osc, 8).walk.arg_spec


@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_the_fused_timed_twin_writes_index_form_time_every_trip(bundle, target):
    """The time declaration through the fused loop: after launches of mixed
    ``k``, ``t == t_0 + applied_steps * dt`` per sample (bitwise on the host,
    exact profile; within one ulp on the device, where the multiply-add may
    contract into ONE rounding), and a fused launch equals ``k`` one-step
    launches."""
    tw = TWINS["osc_timed"]
    width = tape_width(tw.step)
    fused = _Run(bundle["auto"], tw.name, "osc", target, twin_width=width, timed=True)
    single = _Run(bundle["auto"], tw.name, "osc", target, twin_width=width, timed=True)
    for k in (5, 1, 9, 64):
        fused.launch(k)
        for _ in range(k):
            single.launch(1)
    a, b = fused.snap(), single.snap()
    want = a["t_0"] + a["applied_steps"] * a["dt"]
    if target == "host":
        np.testing.assert_array_equal(a["t"], want)
        for plane in a:
            np.testing.assert_array_equal(a[plane], b[plane], err_msg=plane)
    else:
        assert (np.abs(a["t"] - want) <= np.spacing(np.abs(want))).all()
    assert fused.done()
