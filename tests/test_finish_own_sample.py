# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A kernel FINISHES its own sample by assigning its declared mask.

``terminated = cond`` — one statement, a traced rank-0 bool — makes the
sample's mask ``mask | cond`` (set-only, monotone) and adds one to the reserved
``finished_count`` plane per NEWLY finished sample. The epilogue is the last
text of the guarded scope, after every other commit, on both targets.

Rows:

* **sidecar** — ``finish == {mask, counter, steps: 1}``, ``terminated_readonly``
  is ``False`` and ``("lookup", "finished_count")`` is bound; a derived
  (vjp/jvp) kernel carries none of it.
* **refusals** — a second finish, a Python ``bool`` literal, a mask the guard
  does not gate on and an undeclared ``Terminated`` each refuse NAMING the kernel.
* **structure** — the epilogue is the last statement inside the guard, after
  every commit; non-finishing kernels never see the helper or the twin binding.
* **oracle, host AND device** — an oscillator stepped until its counter reaches
  ``n``: every plane bit-equal to a NumPy reference with the same stop rule, the
  mask all set, the counter exactly ``n`` and the launch count exactly the
  longest sample's stop step. Pre-set mask entries are skipped, never counted.
* **non-vacuity** — the same run against an epilogue with the count dropped
  leaves the counter at zero while the mask still sets: the counter row is
  what would catch a guard that never fires.
"""

from __future__ import annotations

import _deploy as L
import numpy as np
import pytest
from conftest import sidecar_of

from hawk import Mutable, Param, Scalar, Terminated, kernel
from hawk.artifact import build_bundle
from hawk.artifact.sidecar import sidecar_meta
from hawk.diff import jvp, vjp
from hawk.emit import render_body
from hawk.ir import HawkError, canonical

N = 40
DT = 0.125
MAX_LAUNCHES = 200
EPILOGUE = ("if (!trm_terminated[i].eval() && (t11 >= psc_t_final[i].eval())) { "
            "trm_terminated_w[i] = true; "
            "hawk_abi::finish_count(lut_finished_count); }")


@kernel(steps=1)
def osc_finish(dt: Param, t_final: Scalar, terminated: Terminated,
               x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    v_start = v
    a = -x
    v = v_start + dt * a
    x = x + dt * v_start
    tn = t + dt
    t = tn
    terminated = tn >= t_final


@kernel
def stop_only(t: Scalar, t_final: Param, terminated: Terminated):
    """A kernel that ONLY finishes — the stop rule on its own."""
    terminated = t >= t_final


@kernel(steps=1)
def branch_finish(x: Scalar, terminated: Terminated, y: Mutable[Scalar]):
    """A finish inside one side of an ``if`` folds to a select."""
    y = x * 2.0
    if x > 0.0:
        terminated = x > 1.0


@kernel
def smooth_finish(x: Scalar, terminated: Terminated, y: Mutable[Scalar]):
    """A differentiable primal (no launch-start read) that finishes."""
    y = x * x + 1.0
    terminated = x > 2.0


# --------------------------------------------------------------------------- #
# -- the IR and the sidecar.
# --------------------------------------------------------------------------- #
def test_the_walk_records_the_finish_and_binds_the_counter():
    assert osc_finish.walk.finish == ("terminated", 1)
    assert ("lookup", "finished_count") in osc_finish.arg_spec
    assert osc_finish.walk.slot_types[("lookup", "finished_count")].dtype == "i32"
    assert stop_only.walk.finish == ("terminated", 1)


def test_a_finishing_kernels_sidecar_declares_the_finish():
    meta = sidecar_meta("osc_finish", osc_finish.walk, fmt="ptx",
                        scalar_type="float64")
    assert meta["finish"] == {"mask": "terminated", "counter": "finished_count",
                              "steps": 1}
    assert meta["terminated_readonly"] is False
    assert ["lookup", "finished_count"] in meta["arg_spec"]
    keys = list(meta)
    assert keys.index("finish") == keys.index("terminated_readonly") + 1


@pytest.mark.parametrize("transform", [vjp, jvp], ids=["vjp", "jvp"])
def test_a_derived_kernel_drops_the_finish(transform):
    sinks = transform(smooth_finish, wrt=("x",))
    assert not any(getattr(s, "kind", None) == "finish" for s in sinks)
    walk = canonical(sinks)
    assert walk.finish is None
    assert ("lookup", "finished_count") not in walk.arg_spec
    meta = sidecar_meta("d", walk, fmt="ptx", scalar_type="float64")
    assert "finish" not in meta and meta["terminated_readonly"] is True
    assert "finish_count" not in render_body(sinks, walk).text


def test_a_non_finishing_kernel_keeps_its_readonly_sidecar():
    import _deployable as D

    plain = sidecar_meta("d", D.diagnostic.walk, fmt="ptx", scalar_type="float64")
    assert "finish" not in plain and plain["terminated_readonly"] is True
    assert ["lookup", "finished_count"] not in plain["arg_spec"]


# --------------------------------------------------------------------------- #
# -- refusals: each names the kernel.
# --------------------------------------------------------------------------- #
def test_a_second_finish_of_one_mask_refuses():
    def twice(x: Scalar, terminated: Terminated, y: Mutable[Scalar]):
        y = x
        terminated = x > 1.0
        terminated = x > 2.0
    with pytest.raises(HawkError, match=r"'twice'.*finished twice"):
        kernel(twice)


def test_a_finish_on_both_sides_of_an_if_is_two_statements_and_refuses():
    def both(x: Scalar, terminated: Terminated, y: Mutable[Scalar]):
        y = x
        if x > 0.0:
            terminated = x > 1.0
        else:
            terminated = x < -1.0
    with pytest.raises(HawkError, match=r"'both'.*finished twice"):
        kernel(both)


@pytest.mark.parametrize("literal", ["True", "False"])
def test_a_bool_literal_refuses(literal, tmp_path):
    src = (f"def lit(x: Scalar, terminated: Terminated, y: Mutable[Scalar]):\n"
           f"    y = x\n"
           f"    terminated = {literal}\n")
    scope = {"Scalar": Scalar, "Terminated": Terminated, "Mutable": Mutable}
    path = tmp_path / f"lit_{literal}.py"     # the trace reads the source file
    path.write_text(src)
    exec(compile(src, str(path), "exec"), scope)
    with pytest.raises(HawkError, match=r"'lit'.*Python literal"):
        kernel(scope["lit"])


def test_a_mask_the_guard_does_not_gate_on_refuses():
    def other(x: Scalar, terminated: Terminated, rejected: Terminated,
              y: Mutable[Scalar]):
        y = x
        rejected = x > 1.0
    with pytest.raises(HawkError, match=r"'other'.*'rejected'.*guard"):
        kernel(other)


def test_an_undeclared_terminated_refuses():
    def undeclared(x: Scalar, y: Mutable[Scalar]):
        y = x
        Terminated[...] = x > 1.0
    with pytest.raises(HawkError, match=r"'undeclared'.*did not declare"):
        kernel(undeclared)


def test_a_finish_inside_a_for_refuses():
    def looped(x: Scalar, terminated: Terminated, y: Mutable[Scalar]):
        y = x
        for _ in range(3):
            terminated = x > 1.0
    with pytest.raises(HawkError, match=r"'looped'.*inside a `for`"):
        kernel(looped)


# --------------------------------------------------------------------------- #
# -- structure.
# --------------------------------------------------------------------------- #
def test_the_epilogue_is_the_last_text_of_the_guarded_scope():
    lines = render_body(osc_finish.sinks, osc_finish.walk).text.splitlines()
    assert lines[0].strip() == "if (!trm_terminated[i].eval()) {"
    assert lines[-1].strip() == "}"
    assert lines[-2].strip() == EPILOGUE
    commits = [k for k, line in enumerate(lines) if "] = " in line
               and "trm_terminated_w" not in line]
    assert commits and max(commits) < len(lines) - 2


def test_a_one_sided_finish_folds_to_a_select():
    text = render_body(branch_finish.sinks, branch_finish.walk).text
    assert "? " in text and ": false)" in text, text


# --------------------------------------------------------------------------- #
# -- the oracle: host and device.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def bundles(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("finish")
    return build_bundle([osc_finish, stop_only], root / "finish",
                        targets=("cuda", "host"), cache_dir=cache_dir)


def _inputs():
    rng = np.random.default_rng(7)
    t_final = DT * rng.integers(3, 30, N).astype(float)
    x = rng.uniform(-1.0, 1.0, N)
    v = rng.uniform(-1.0, 1.0, N)
    return t_final, x, v


def _reference(t_final, x, v, mask):
    """NumPy, the same step and the same stop rule, launch by launch."""
    x, v, t, mask = x.copy(), v.copy(), np.zeros(N), mask.copy()
    launches = 0
    while not mask.all():
        live = ~mask
        xp, vp = x[live].copy(), v[live].copy()
        v[live] = vp + DT * (-xp)
        x[live] = xp + DT * vp
        t[live] = t[live] + DT
        mask |= live & (t >= t_final)
        launches += 1
    return x, v, t, mask, launches


def _runner(bundle, name, target):
    import eagle.exec as eexec
    from eagle import plan as eplan

    sidecar = sidecar_of(bundle, name)
    if target == "host":
        plugin = L.host_plugin(bundle.directory, name, sidecar)
        return eplan.plan(plugin, structure=eexec.HostTeam), np
    import cupy as cp

    plugin = L.device_plugin(bundle.directory, name, sidecar)
    return eplan.plan(plugin, structure=eexec.DeviceKernel), cp


def _step_until_done(bundle, target, preset=None):
    plan, xp = _runner(bundle, "osc_finish", target)
    t_final, x, v = _inputs()
    mask = np.zeros(N, dtype=bool) if preset is None else preset.copy()
    planes = {"t_final": xp.array(t_final), "x": xp.array(x),
              "v": xp.array(v), "t": xp.zeros(N), "terminated": xp.array(mask)}
    counter = xp.array(np.array([mask.sum()], dtype=np.uint32))
    bound = plan.bind(dt=DT, finished_count=counter.view(xp.int32), **planes)
    launches = 0
    while int(counter[0]) < N and launches < MAX_LAUNCHES:
        bound.launch()      # the counter read is the stream-ordered host read
        launches += 1

    def host(a):
        return np.asarray(a.get() if hasattr(a, "get") else a)

    got = {k: host(a) for k, a in planes.items()}
    return got, int(host(counter)[0]), launches, (t_final, x, v, mask)


def _check(bundle, target):
    got, counted, launches, (t_final, x, v, mask) = _step_until_done(bundle, target)
    rx, rv, rt, rmask, rlaunches = _reference(t_final, x, v, mask)
    assert counted == N, f"{target}: the counter reached {counted}, not {N}"
    assert got["terminated"].all() and rmask.all()
    assert launches == rlaunches == int(round(t_final.max() / DT)), (
        launches, rlaunches)
    for name, ref in (("x", rx), ("v", rv), ("t", rt)):
        np.testing.assert_array_equal(got[name], ref, err_msg=f"{target}:{name}")


def test_compiled_host_finishes_counts_and_matches_the_reference(bundles):
    _check(bundles, "host")


@pytest.mark.gpu
def test_compiled_device_finishes_counts_and_matches_the_reference(bundles):
    _check(bundles, "cuda")


@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_a_preset_mask_is_skipped_and_never_counted(bundles, target):
    preset = np.zeros(N, dtype=bool)
    preset[[0, 3, 7, 11]] = True
    got, counted, _launches, (t_final, x, v, mask) = _step_until_done(
        bundles, target, preset=preset)
    assert counted == N, f"{target}: a pre-set sample was counted again ({counted})"
    np.testing.assert_array_equal(got["x"][preset], x[preset])
    np.testing.assert_array_equal(got["t"][preset], np.zeros(preset.sum()))


@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_a_stop_only_kernel_finishes_and_counts(bundles, target):
    plan, xp = _runner(bundles, "stop_only", target)
    t = np.linspace(0.0, 1.0, N)
    mask = xp.zeros(N, dtype=bool)
    counter = xp.zeros(1, dtype=xp.uint32)
    bound = plan.bind(t=xp.array(t), t_final=0.5, terminated=mask,
                      finished_count=counter.view(xp.int32))
    bound.launch()
    host_mask = np.asarray(mask.get() if hasattr(mask, "get") else mask)
    np.testing.assert_array_equal(host_mask, t >= 0.5)
    assert int(counter[0]) == int((t >= 0.5).sum())
    bound.launch()
    assert int(counter[0]) == int((t >= 0.5).sum()), "a set mask was re-counted"


# --------------------------------------------------------------------------- #
# -- non-vacuity: drop the count, the counter row goes red.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_dropping_the_count_from_the_epilogue_is_caught(tmp_path, cache_dir,
                                                        monkeypatch, target):
    from hawk.emit import aether

    original = aether._Renderer._emit

    def no_count(self, line):
        original(self, line.replace(" hawk_abi::finish_count(lut_finished_count);",
                                    ""))
    monkeypatch.setattr(aether._Renderer, "_emit", no_count)
    stripped = build_bundle([osc_finish], tmp_path / "stripped",
                            targets=("cuda", "host"), cache_dir=cache_dir)
    monkeypatch.undo()
    text = (tmp_path / "stripped" / "osc_finish.cpp").read_text() \
        if (tmp_path / "stripped" / "osc_finish.cpp").exists() else None
    if text is not None:
        assert "finish_count(lut_finished_count)" not in text
    got, counted, launches, _ = _step_until_done(stripped, target)
    assert got["terminated"].all(), "the mask still sets"
    assert counted == 0 and launches == MAX_LAUNCHES, (
        "with no count the guard never fires: the loop runs to its cap")


def test_an_active_set_build_of_a_finishing_kernel_keeps_the_finish(tmp_path,
                                                                    cache_dir):
    """The compaction build (``Guard(active_set=True)``) binds the counter
    beside the map planes and marks the mask through the map's sample."""
    from hawk.ext import Guard, Kind

    active = Kind("finish_active", guard=Guard(active_set=True))
    bundle = build_bundle([osc_finish], tmp_path / "active", kind=active,
                          targets=("cuda", "host"), cache_dir=cache_dir)
    side = sidecar_of(bundle, "osc_finish")
    assert side["finish"] == {"mask": "terminated", "counter": "finished_count",
                              "steps": 1}
    for name in ("finished_count", "active_map", "active_count"):
        assert ["lookup", name] in side["arg_spec"]
