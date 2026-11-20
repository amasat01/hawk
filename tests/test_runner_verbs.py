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
grouped = hawk.kernel(grouped_step, kind=MAPPED, group=GROUP, steps=1)


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


def test_a_step_whose_finisher_is_automatic_runs_one_step_per_launch_to_done(loaded):
    plans = [hawk.plan(loaded["ramp"]), hawk.plan(loaded["tail"])]
    res = hawk.simulate(plans, rate=RATE, goal=GOAL, max_steps=1000)
    taken = _stepped(1000)[2]
    want_z = sum(_stepped(k)[0] for k in range(1, taken + 1))     # the tail adds the state the finisher just wrote
    assert res.done and res.steps == taken and res.x.tobytes() == _stepped(1000)[0].tobytes()
    assert res.z.tobytes() == want_z.tobytes()


def test_the_library_refuses_a_step_word_bound_to_an_automatic_finisher_in_a_several_plan_step(loaded):
    plans = [hawk.plan(loaded["ramp"]), hawk.plan(loaded["tail"])]
    with pytest.raises(HawkError, match="steps word"):
        hawk.until_done(plans, max_steps=5, rate=RATE, goal=GOAL, fused_steps=np.ones(1, dtype=np.int64))


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


def _plain_planes(live=None):
    """The planes of one plain launch of a ramp kernel over ``N`` samples, ``live`` being the mask's complement."""
    dead = np.zeros(N, dtype=bool) if live is None else ~np.asarray(live)
    return {"rate": RATE, "goal": GOAL, "x": np.zeros(N), "terminated": dead}


def _one_step(live_positions):
    """The NumPy oracle of one launch: every sample at ``live_positions`` takes one step."""
    want = np.zeros(N)
    want[live_positions] = RATE[live_positions]
    return want


@pytest.mark.parametrize("structure", [hawk.HostTeam, hawk.HostSerial], ids=["team", "serial"])
@pytest.mark.parametrize("name", ["mapped", "grouped"])
def test_a_plain_launch_runs_the_samples_of_a_set_made_from_a_mask(loaded, name, structure):
    kernel = loaded[(mapped if name == "mapped" else grouped).name]
    rng = np.random.default_rng(3)
    live = rng.random(N) < 0.4
    planes = _plain_planes(live)
    stated = hawk.ActiveSet(planes["terminated"])
    hawk.plan(kernel, structure).bind(**planes, **stated.planes(kernel)).launch()
    # a live lane is in a live group, and a dead lane of a live group is skipped by the mask: the live lanes step
    np.testing.assert_array_equal(planes["x"], _one_step(np.flatnonzero(live)))


def test_a_set_from_positions_runs_them_in_any_order_and_team_equals_serial(loaded):
    kernel = loaded[mapped.name]
    ends_only = np.arange(70, dtype=np.int32)       # the identity at both ends of the tile, not in between
    ends_only[35] = N - 1
    positions = {"reversed": np.arange(N - 1, 59, -1), "scattered": np.array([97, 3, 50, 4, 99, 0, 61]),
                 "ends_only": ends_only}
    for label, entries in positions.items():
        results = []
        for structure in (hawk.HostTeam, hawk.HostSerial):
            planes = _plain_planes()
            hawk.plan(kernel, structure).bind(**planes, **hawk.ActiveSet(positions=entries).planes(kernel)).launch()
            results.append(planes["x"])
        np.testing.assert_array_equal(results[0], _one_step(np.unique(entries)), err_msg=label)
        assert results[0].tobytes() == results[1].tobytes(), label


def test_a_set_from_group_positions_runs_whole_groups(loaded):
    kernel = loaded[grouped.name]
    planes = _plain_planes()
    hawk.plan(kernel).bind(**planes, **hawk.ActiveSet(positions=[3, 1]).planes(kernel)).launch()
    np.testing.assert_array_equal(planes["x"], _one_step(np.r_[GROUP:2 * GROUP, 3 * GROUP:N]))


def test_planes_called_again_rewrite_the_same_arrays_from_the_mask_as_it_is_then(loaded):
    kernel = loaded[mapped.name]
    planes = _plain_planes()
    stated = hawk.ActiveSet(planes["terminated"])
    made = stated.planes(kernel)
    bound = hawk.plan(kernel).bind(**planes, **made)
    bound.launch()
    np.testing.assert_array_equal(planes["x"], RATE)
    planes["terminated"][::2] = True
    planes["x"][:] = 0.0
    again = stated.planes(kernel)
    assert all(again[name] is made[name] for name in made)
    bound.launch()
    np.testing.assert_array_equal(planes["x"], _one_step(np.arange(1, N, 2)))


def test_a_count_above_the_map_is_refused_by_name_at_the_launch(loaded):
    kernel = loaded[mapped.name]
    bound = hawk.plan(kernel).bind(**_plain_planes(), active_map=np.arange(4, dtype=np.int32),
                                   active_count=np.array([9], dtype=np.uint32))
    with pytest.raises(HawkError, match="count 9 is above the map's 4 entries"):
        bound.launch()


def test_a_plain_plan_over_an_active_set_needs_the_planes_and_a_runner_refuses_a_plan_that_holds_them(loaded):
    kernel = loaded[mapped.name]
    with pytest.raises(HawkError, match="active_map"):
        hawk.plan(kernel).bind(**_plain_planes())
    held = hawk.ActiveSet(positions=[0, 1]).planes(kernel)
    with pytest.raises(HawkError, match="caller"):
        hawk.until_done(hawk.plan(kernel), max_steps=3, **_plain_planes(), **held)


def test_an_active_set_states_one_thing_and_its_planes_need_a_kernel_with_a_set(loaded):
    with pytest.raises(HawkError, match="not both"):
        hawk.ActiveSet(np.zeros(N, dtype=bool), positions=[1])
    with pytest.raises(HawkError, match="one or the other"):
        hawk.ActiveSet(np.zeros(N, dtype=bool), every=8)
    with pytest.raises(HawkError, match="reads no active set"):
        hawk.ActiveSet(positions=[0]).planes(loaded["ramp"])
    with pytest.raises(HawkError, match="runner's compaction"):
        hawk.ActiveSet(every=8).planes(loaded[mapped.name])
    with pytest.raises(HawkError, match="makes its own active set"):
        hawk.until_done(hawk.plan(loaded[mapped.name]), max_steps=3, active=hawk.ActiveSet(positions=[0]),
                        **_plain_planes())


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
