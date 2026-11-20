# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The runner verbs: ``hawk.until_done``, ``hawk.simulate`` and ``hawk.ActiveSet``. Every value is held against a NumPy loop
that steps the same recurrence, and a run under a team of workers against the serial reference."""

from __future__ import annotations

import _osprey
import numpy as np
import pytest

import hawk
from hawk import HawkError, Mutable, Param, Scalar, Terminated
from hawk.artifact import build_bundle
from hawk.ext import Guard, Kind

N = 100
GOAL = 3.0
RATE = np.linspace(0.2, 1.0, N)
MAPPED = Kind("runner_rows", guard=Guard(active_set=True))
GROUP = 32


def lead_step(rate: Scalar, goal: Param, terminated: Terminated, x: Mutable[Scalar]):
    xn = x + rate
    x = xn
    terminated = xn >= goal


def mapped_step(rate: Scalar, goal: Param, terminated: Terminated, x: Mutable[Scalar]):
    xn = x + rate
    x = xn
    terminated = xn >= goal


def grouped_step(rate: Scalar, goal: Param, terminated: Terminated, x: Mutable[Scalar]):
    xn = x + rate
    x = xn
    terminated = xn >= goal


@hawk.kernel
def ramp(rate: Scalar, goal: Param, terminated: Terminated, x: Mutable[Scalar]):
    xn = x + rate
    x = xn
    terminated = xn >= goal


@hawk.kernel
def tail(x: Scalar, z: Mutable[Scalar]):
    z = z + x


# one step per launch, so a step cap and a compaction cadence count single steps
lead = hawk.steps(hawk.kernel(lead_step), 1)
mapped = hawk.steps(hawk.kernel(mapped_step, kind=MAPPED), 1)
grouped = hawk.steps(hawk.kernel(grouped_step, kind=MAPPED, group=GROUP), 1)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory, cache_dir):
    """``{kernel name: loaded kernel}`` of the plain kernels (one bundle) and of the two over an active set."""
    out = {}
    for name, kernels in (("plain", [ramp, lead, tail]), ("mapped", [mapped]), ("grouped", [grouped])):
        bundle = build_bundle(kernels, tmp_path_factory.mktemp(name), targets=("host",), cache_dir=cache_dir)
        out |= {k.name: hawk.load(bundle.directory, k.name) for k in kernels}
    return out


def _stepped(steps: int, n: int = N, rate=RATE):
    """The NumPy loop of the ramp kernels: ``(x, terminated, steps taken)`` after at most ``steps`` steps, a finished sample resting."""
    x, done = np.zeros(n), np.zeros(n, dtype=bool)
    taken = 0
    while taken < steps and not done.all():
        stepped = x + rate
        x = np.where(done, x, stepped)
        done |= stepped >= GOAL
        taken += 1
    return x, done, taken


def test_simulate_runs_every_sample_to_done_and_equals_the_numpy_loop(loaded):
    res = hawk.simulate(hawk.plan(loaded["ramp"]), rate=RATE, goal=GOAL, x=np.zeros(N), max_steps=1000)
    want, _, taken = _stepped(1000)
    assert res.status == "done" and res.done and res.finished == res.n == N
    assert res.x.tobytes() == want.tobytes()
    assert res.steps >= taken


def test_simulate_deploys_a_kernel_it_is_given_and_runs_a_plan_it_is_given(loaded):
    by_kernel = hawk.simulate(ramp, rate=RATE, goal=GOAL, x=np.zeros(N), max_steps=1000)
    by_plan = hawk.simulate(hawk.plan(loaded["ramp"]), rate=RATE, goal=GOAL, x=np.zeros(N), max_steps=1000)
    assert by_kernel.x.tobytes() == by_plan.x.tobytes()


def test_reaching_max_steps_is_a_result_with_its_status(loaded):
    res = hawk.simulate(hawk.plan(loaded["ramp"]), rate=RATE, goal=GOAL, x=np.zeros(N), max_steps=3)
    want, done, _ = _stepped(3)
    assert res.status == "max_steps" and not res.done and res.steps == 3
    assert res.finished == int(done.sum())
    assert res.x.tobytes() == want.tobytes()


def test_the_state_is_the_callers_own_arrays_by_name(loaded):
    x = np.zeros(N)
    res = hawk.simulate(hawk.plan(loaded["ramp"]), rate=RATE, goal=GOAL, x=x, max_steps=1000)
    assert res.x is x and res.state["x"] is x and set(res.state) == {"x"}
    assert res.report.n == N and res.wall_s > 0.0
    with pytest.raises(AttributeError):
        res.nothing


def test_an_omitted_state_plane_and_mask_are_made_and_start_at_zero(loaded):
    res = hawk.simulate(hawk.plan(loaded["ramp"]), rate=RATE, goal=GOAL, max_steps=1000)
    want, _, _ = _stepped(1000)
    assert res.x.tobytes() == want.tobytes() and res.done


def test_the_team_and_the_serial_structure_run_to_the_same_state(loaded):
    team = hawk.simulate(hawk.plan(loaded["ramp"], hawk.HostTeam), rate=RATE, goal=GOAL, max_steps=1000)
    serial = hawk.simulate(hawk.plan(loaded["ramp"], hawk.HostSerial), rate=RATE, goal=GOAL, max_steps=1000)
    assert team.x.tobytes() == serial.x.tobytes() and team.finished == serial.finished


def test_a_list_of_plans_runs_as_one_step_in_order_over_one_namespace(loaded):
    plans = [hawk.plan(loaded[lead.name]), hawk.plan(loaded["tail"])]
    res = hawk.simulate(plans, rate=RATE, goal=GOAL, max_steps=4)
    want_x = _stepped(4)[0]
    want_z = sum(_stepped(k)[0] for k in range(1, 5))   # the tail adds the state the lead just wrote, every step
    assert res.x.tobytes() == want_x.tobytes() and res.z.tobytes() == want_z.tobytes()
    assert res.steps == 4 and set(res.state) == {"x", "z"}


def test_a_counter_array_receives_what_each_run_finished_and_a_reset_starts_over(loaded):
    finished = np.zeros(1, dtype=np.uint32)
    x = np.zeros(N)
    runner = hawk.until_done(hawk.plan(loaded["ramp"]), max_steps=4, rate=RATE, goal=GOAL, x=x,
                             finished_count=finished)
    first = runner.run()
    assert first.finished == int(_stepped(4)[1].sum()) and finished[0] == first.finished
    second = runner.run()
    assert second.finished == int(_stepped(8)[1].sum()) and finished[0] == second.finished
    runner.reset()
    finished[:], x[:] = 0, 0.0
    assert runner.run().finished == first.finished and finished[0] == first.finished
    assert runner.planes["x"] is x and runner.outputs == ("x",)
    runner.close()


def test_a_runner_reports_its_figures_and_refuses_a_run_once_closed(loaded):
    runner = hawk.until_done(hawk.plan(loaded["ramp"]), max_steps=10, rate=RATE, goal=GOAL, x=np.zeros(N))
    assert runner.figures()["mode"] == _osprey.EG_MODE_HOST
    runner.close()
    runner.close()
    with pytest.raises(HawkError):
        runner.run()


@pytest.mark.parametrize("name", ["mapped", "grouped"])
def test_a_run_over_an_active_set_equals_the_run_without_one(loaded, name):
    kernel = loaded[mapped.name if name == "mapped" else grouped.name]
    res = hawk.simulate(hawk.plan(kernel), rate=RATE, goal=GOAL, max_steps=1000, active=hawk.ActiveSet())
    want, _, _ = _stepped(1000)
    assert res.done and res.x.tobytes() == want.tobytes()
    assert res.report.compactions > 0


def test_a_cadence_states_the_steps_between_compactions(loaded):
    plan = hawk.plan(loaded[mapped.name])
    res = hawk.simulate(plan, rate=RATE, goal=GOAL, max_steps=1000, active=hawk.ActiveSet(every=8))
    assert res.done and res.x.tobytes() == _stepped(1000)[0].tobytes()


def test_a_reordering_run_over_many_samples_equals_the_numpy_loop(loaded):
    n = 6000
    rate = np.random.default_rng(5).permutation(np.linspace(0.03, 0.5, n))   # thinning irregularly in sample order
    res = hawk.simulate(hawk.plan(loaded[mapped.name]), rate=rate, goal=GOAL, max_steps=1000,
                        active=hawk.ActiveSet(every=8, reorder=0.5))
    assert res.done and res.x.tobytes() == _stepped(1000, n, rate)[0].tobytes()
    assert res.report.reorders > 0


def test_the_library_refuses_a_compaction_a_model_cannot_have_and_an_out_of_range_threshold(loaded):
    with pytest.raises(HawkError, match="no active set"):
        hawk.until_done(hawk.plan(loaded["ramp"]), max_steps=10, active=hawk.ActiveSet(every=8), rate=RATE,
                        goal=GOAL, x=np.zeros(N))
    with pytest.raises(HawkError, match="theta|reorder"):
        hawk.until_done(hawk.plan(loaded[mapped.name]), max_steps=10, active=hawk.ActiveSet(reorder=0.9), rate=RATE,
                        goal=GOAL, x=np.zeros(N))


@pytest.mark.parametrize("bad", [{"every": 2.5}, {"every": True}, {"reorder": "half"}])
def test_an_active_set_states_a_whole_cadence_and_a_numeric_threshold(bad):
    with pytest.raises(HawkError, match="ActiveSet"):
        hawk.ActiveSet(**bad)


def test_the_active_set_is_the_librarys_and_a_caller_cannot_bind_it(loaded):
    kernel = loaded[mapped.name]
    planes = {"rate": RATE, "goal": GOAL, "x": np.zeros(N), "terminated": np.zeros(N, dtype=bool)}
    with pytest.raises(HawkError, match="active set"):
        hawk.plan(kernel).bind(**planes, active_map=np.arange(N, dtype=np.int32), active_count=np.full(1, N, np.uint32))
    with pytest.raises(HawkError):      # without a runner there is no active set to read
        hawk.plan(kernel).bind(**planes).launch()


def test_a_wrong_model_or_plane_is_refused_by_name(loaded):
    with pytest.raises(HawkError, match="empty"):
        hawk.until_done([], max_steps=3)
    with pytest.raises(HawkError, match="plan it first"):
        hawk.until_done(loaded["ramp"], max_steps=3)
    with pytest.raises(HawkError, match=r"\['velocity'\] are not planes"):
        hawk.until_done(hawk.plan(loaded["ramp"]), max_steps=3, rate=RATE, goal=GOAL, velocity=np.zeros(N))
    with pytest.raises(HawkError, match="max_steps|steps"):
        hawk.until_done(hawk.plan(loaded["ramp"]), max_steps=0, rate=RATE, goal=GOAL, x=np.zeros(N))


def test_a_state_plane_named_as_a_field_of_the_result_is_refused(tmp_path, cache_dir):
    @hawk.kernel
    def odd(rate: Scalar, goal: Param, terminated: Terminated, steps: Mutable[Scalar]):
        xn = steps + rate
        steps = xn
        terminated = xn >= goal

    bundle = build_bundle([odd], tmp_path, targets=("host",), cache_dir=cache_dir)
    with pytest.raises(HawkError, match=r"\['steps'\]"):
        hawk.simulate(hawk.plan(hawk.load(bundle.directory, "odd")), rate=RATE, goal=GOAL, max_steps=10)
