# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Recording a run and reversing it: ``hawk.deploy(..., record=True)``, ``hawk.until_done(..., record=True)``,
``hawk.simulate(..., record=True)`` and ``hawk.reverse``.

The model is ``x <- a * x + b`` one step per launch, a sample stopping after ``T`` steps, so that after ``T`` steps
``x_T = a**T * x0 + b * S`` with ``S = (a**T - 1) / (a - 1)``. The gradients of ``J = sum(g * x_T)`` have closed forms in ``x0``,
``a`` and ``b``, and each is also held against central differences of the forward run itself.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import hawk
from hawk import HawkError, Mutable, Param, Scalar, Terminated
from hawk import runtime as rt
from hawk.diff.recording import DerivedUnit, recording

N = 40
A, B = 0.5, 0.25
T = 1.0 + (np.arange(N) % 7)
X0 = 1.0 + np.arange(N) / 8
G = np.linspace(0.5, 1.5, N)
EPS = np.finfo(np.float64).eps


def ramp_one(a: Param, b: Param, t_final: Scalar, terminated: Terminated, x: Mutable[Scalar], t: Mutable[Scalar]):
    x = a * x + b
    tn = t + 1.0
    t = tn
    terminated = tn >= t_final


def ramp_four(a: Param, b: Param, t_final: Scalar, terminated: Terminated, x: Mutable[Scalar], t: Mutable[Scalar]):
    x = a * x + b
    tn = t + 1.0
    t = tn
    terminated = tn >= t_final


def ramp_timed(dt: Param, t_final: Scalar, terminated: Terminated, x: Mutable[Scalar], t: Mutable[Scalar]):
    x = x + dt
    tn = t + dt
    t = tn
    terminated = tn >= t_final


def ramp_plain(rate: Scalar, goal: Param, terminated: Terminated, x: Mutable[Scalar]):
    xn = x + rate
    x = xn
    terminated = xn >= goal


def tail_step(x: Scalar, z: Mutable[Scalar]):
    z = z + x


one = hawk.kernel(ramp_one, steps=1)
four = hawk.steps(hawk.kernel(ramp_four, steps=1), 4)
automatic = hawk.kernel(ramp_plain)
timed = hawk.kernel(ramp_timed, steps=1, time="t", time_step="dt")
tail = hawk.kernel(tail_step)


def _closed_form(a: float = A, b: float = B):
    """``(x_T, dJ/dx0 per sample, dJ/da, dJ/db)`` of ``J = sum(G * x_T)`` for the model, from its closed form."""
    power = a**T
    total = (power - 1.0) / (a - 1.0)
    d_total = (T * a ** (T - 1.0) * (a - 1.0) - (power - 1.0)) / (a - 1.0) ** 2
    return (power * X0 + b * total, G * power, float(np.sum(G * (T * a ** (T - 1.0) * X0 + b * d_total))),
            float(np.sum(G * total)))


def _planes(**more):
    return dict(x=X0.copy(), t=np.zeros(N), t_final=T, a=A, b=B) | more


def _recorded_run(plan, **options):
    """``(runner, gradients)`` of one recorded run of ``plan`` and its reverse with the cotangent ``G`` on ``x``."""
    runner = hawk.until_done(plan, max_steps=40, record=True, **options, **_planes())
    runner.run()
    return runner, runner.reverse({"x": G})


@pytest.fixture(scope="module")
def plan(cache_dir):
    """``one`` deployed recordable with a window of three steps, shorter than the longest run."""
    return hawk.deploy(one, record=True, window=3, cache_dir=cache_dir)


@pytest.fixture(scope="module")
def plan_four(cache_dir):
    """``four`` (four steps per launch) deployed recordable with a window of four, so the reverse may take the four-step unit."""
    return hawk.deploy(four, record=True, window=4, cache_dir=cache_dir)


def test_the_reverse_of_a_recorded_run_equals_the_closed_form(plan):
    runner, grads = _recorded_run(plan)
    x_t, d_x0, d_a, d_b = _closed_form()
    assert np.array_equal(runner.planes["x"], x_t) or np.allclose(runner.planes["x"], x_t, rtol=4 * EPS, atol=0.0)
    np.testing.assert_allclose(grads["x"], d_x0, rtol=8 * EPS, atol=0.0)
    np.testing.assert_allclose(grads["a"], d_a, rtol=64 * EPS, atol=0.0)
    np.testing.assert_allclose(grads["b"], d_b, rtol=64 * EPS, atol=0.0)
    assert grads["a"].shape == () and grads["x"].shape == (N,)


def test_a_recorded_run_reports_its_gradients_bit_exact_and_strict_refuses_nothing(plan):
    _, grads = _recorded_run(plan)
    assert isinstance(grads, hawk.runner.Gradients)
    assert grads.report == hawk.runner.GradientReport(bitwise=dict.fromkeys(grads, True), chunks=grads.report.chunks)
    assert sorted(grads) == ["a", "b", "t", "x"] and grads.report.chunks >= 1
    runner = hawk.until_done(plan, max_steps=40, record=True, **_planes())     # a record is reversed once: run again
    runner.run()
    strict = hawk.reverse(runner.record, {"x": G}, strict=True)
    assert strict["x"].tobytes() == grads["x"].tobytes() and strict.report.bitwise == grads.report.bitwise


def test_a_reverse_over_a_memory_budget_reports_the_passes_it_made(plan):
    _, split = _recorded_run(plan, memory_budget=400)
    assert split.report.chunks > 1


def _objective(plan, **changes):
    """``J = sum(G * x_T)`` of a plain forward run of ``plan`` with ``changes`` to the bound values."""
    planes = _planes() | changes
    return float(np.sum(G * hawk.simulate(plan, max_steps=40, **planes).x))


def test_the_reverse_equals_central_differences_of_the_forward_run(plan):
    _, grads = _recorded_run(plan)
    h = 1e-5
    for name, held in (("a", A), ("b", B)):
        slope = (_objective(plan, **{name: held + h}) - _objective(plan, **{name: held - h})) / (2 * h)
        assert grads[name] == pytest.approx(slope, rel=1e-7)
    for i in (0, 9, N - 1):
        up, down = X0.copy(), X0.copy()
        up[i] += h
        down[i] -= h
        slope = (_objective(plan, x=up) - _objective(plan, x=down)) / (2 * h)
        assert grads["x"][i] == pytest.approx(slope, rel=1e-7)


def test_a_team_and_a_serial_run_reverse_to_the_same_bytes(plan):
    _, team = _recorded_run(plan)
    _, serial = _recorded_run(hawk.plan(plan.kernel, hawk.HostSerial))
    assert {name: value.tobytes() for name, value in team.items()} == {name: value.tobytes() for name, value in serial.items()}


@pytest.mark.parametrize("window", [1, 2, 7])
def test_the_window_a_kernel_is_built_with_does_not_change_the_gradients(cache_dir, plan, window):
    other = hawk.deploy(one, record=True, window=window, cache_dir=cache_dir)
    _, grads = _recorded_run(other)
    _, want = _recorded_run(plan)
    assert grads["x"].tobytes() == want["x"].tobytes()
    for name in ("a", "b"):
        np.testing.assert_allclose(grads[name], want[name], rtol=64 * EPS)


def test_a_run_over_another_window_than_the_one_built_is_refused_by_name(plan):
    assert plan.kernel.recorded_window() == 3
    with pytest.raises(HawkError, match=r"records windows of 3 steps \(built in\)"):
        hawk.until_done(plan, max_steps=40, record=True, window=4, **_planes())
    runner = hawk.until_done(plan, max_steps=40, record=True, window=3, **_planes())
    runner.close()


def test_a_memory_budget_that_splits_the_reverse_in_passes_keeps_the_gradients(plan):
    _, whole = _recorded_run(plan)
    _, passes = _recorded_run(plan, memory_budget=400)
    assert passes["x"].tobytes() == whole["x"].tobytes()
    for name in ("a", "b"):
        np.testing.assert_allclose(passes[name], whole[name], rtol=64 * EPS)


def test_a_memory_budget_below_what_one_sample_needs_is_refused_by_name(plan):
    with pytest.raises(HawkError, match=r"memory_budget 100: one sample alone needs \d+ bytes"):
        _recorded_run(plan, memory_budget=100)


def test_the_per_step_and_the_fused_reverse_agree_within_a_bound_in_the_machine_epsilon(plan_four):
    k = 4
    per_step = _recorded_run(plan_four, steps_reverse="per_step")[1]
    fused_runner, fused = _recorded_run(plan_four, steps_reverse="fused")
    assert fused_runner.figures()["steps_reverse"] == hawk_steps_reverse("fused")
    # a state adjoint is the same product of the same factors; a parameter gradient sums the K terms in another order: within
    # 8 K eps of the magnitudes of the terms it sums
    np.testing.assert_allclose(fused["x"], per_step["x"], rtol=8 * k * EPS, atol=0.0)
    _, d_x0, _, _ = _closed_form()
    magnitude = np.sum(np.abs(G) * (T + 1.0) * X0)
    for name in ("a", "b"):
        assert abs(fused[name] - per_step[name]) <= 8 * k * EPS * max(magnitude, abs(per_step[name]))
    auto_runner, auto = _recorded_run(plan_four, steps_reverse="auto")
    assert auto_runner.figures()["steps_reverse"] in (hawk_steps_reverse("per_step"), hawk_steps_reverse("fused"))
    np.testing.assert_allclose(auto["x"], d_x0, rtol=8 * k * EPS)


def hawk_steps_reverse(word: str) -> int:
    """The library's code of a ``steps_reverse`` word."""
    from hawk.runner import _STEPS_REVERSE

    return _STEPS_REVERSE[word]


def test_a_kernel_with_an_automatic_step_count_records_and_reverses(cache_dir):
    auto_plan = hawk.deploy(automatic, record=True, window=4, cache_dir=cache_dir)
    assert [r["role"] for r in auto_plan.kernel.sidecar["derived"]] == ["tape_fused", "step_vjp"]
    rate = np.linspace(0.2, 1.0, N)
    runner = hawk.until_done(auto_plan, max_steps=40, record=True, rate=rate, goal=3.0, x=np.zeros(N))
    runner.run()
    grads = runner.reverse({"x": np.ones(N)})
    np.testing.assert_array_equal(grads["x"], np.ones(N))      # x is added to, so a unit cotangent passes through unchanged
    taken, x = np.zeros(N), np.zeros(N)
    for _ in range(40):                                          # the NumPy loop of the model: the steps each sample takes
        live = x < 3.0
        x = np.where(live, x + rate, x)
        taken += live
    np.testing.assert_array_equal(grads["rate"], taken)


def test_a_kernel_passed_to_a_recorded_run_is_deployed_recordable(cache_dir, monkeypatch):
    monkeypatch.setenv("HAWK_CACHE_DIR", cache_dir)
    result = hawk.simulate(one, max_steps=40, record=True, window=5, **_planes())
    grads = hawk.reverse(result.record, {"x": G})
    np.testing.assert_allclose(grads["x"], _closed_form()[1], rtol=8 * EPS)
    assert result.record.runner.planes["x"] is result.x


def test_the_library_refuses_a_reverse_without_a_recorded_run_by_name(plan):
    runner = hawk.until_done(plan, max_steps=40, record=True, **_planes())
    with pytest.raises(HawkError, match="no recorded run to reverse"):
        runner.reverse({"x": G})
    runner.run()
    runner.reverse({"x": G})
    runner.reset()
    with pytest.raises(HawkError, match="no recorded run to reverse"):
        runner.reverse({"x": G})


def test_the_library_refuses_a_recording_of_a_several_plan_step_by_name(plan, cache_dir):
    follower = hawk.deploy(tail, cache_dir=cache_dir)
    with pytest.raises(HawkError, match="of a several-plan step is not written"):
        hawk.until_done([plan, follower], max_steps=40, record=True, **_planes(), z=np.zeros(N))


def test_a_plain_runner_keeps_no_record_and_refuses_a_reverse_by_name(cache_dir):
    plain = hawk.deploy(hawk.kernel(ramp_plain), cache_dir=cache_dir)
    rate = np.linspace(0.2, 1.0, N)
    runner = hawk.until_done(plain, max_steps=40, rate=rate, goal=3.0, x=np.zeros(N))
    runner.run()
    assert runner.record is None
    with pytest.raises(HawkError, match="this runner records nothing"):
        runner.reverse({"x": np.ones(N)})
    assert hawk.simulate(plain, max_steps=40, rate=rate, goal=3.0, x=np.zeros(N)).record is None


def test_a_recorded_run_of_a_kernel_built_without_its_derived_units_is_refused_by_name(cache_dir):
    plain = hawk.deploy(one, cache_dir=cache_dir)
    with pytest.raises(HawkError, match=r"its build records no derived unit for \['tape', 'step_vjp'\]"):
        hawk.until_done(plain, max_steps=40, record=True, **_planes())


@pytest.mark.parametrize("options, text", [
    ({"record": 1}, "record= is True or False"),
    ({"window": 0}, "window= is a positive whole number"),
    ({"window": 1.5}, "window= is a positive whole number"),
    ({"memory_budget": -3}, "memory_budget= is a positive whole number"),
    ({"steps_reverse": "sideways"}, "steps_reverse= is one of"),
])
def test_a_recording_option_that_is_not_of_its_kind_is_refused_by_name(plan, options, text):
    with pytest.raises(HawkError, match=text):
        hawk.until_done(plan, max_steps=40, **_planes(), **options)


@pytest.mark.parametrize("options, text", [
    ({"window": 4}, "without EG_RUN_RECORD"),
    ({"memory_budget": 1 << 20}, "without EG_RUN_RECORD"),
    ({"steps_reverse": "fused"}, "FUSED needs EG_RUN_RECORD"),
])
def test_the_library_refuses_a_recording_option_without_record(plan, options, text):
    with pytest.raises(HawkError, match=text):
        hawk.until_done(plan, max_steps=40, **_planes(), **options)


def test_a_kernel_with_a_time_step_records_and_reverses_with_the_start_copy_of_its_time(cache_dir):
    """``x <- x + dt`` with ``t <- t + dt``: ``x_T = x0 + T * dt``, so ``dJ/dx0 = g`` and ``dJ/d dt = sum(g * T)``."""
    dt, steps = 0.25, 1.0 + (np.arange(N) % 7)
    timed_plan = hawk.deploy(timed, record=True, window=3, cache_dir=cache_dir)
    assert [r["role"] for r in timed_plan.kernel.sidecar["derived"]] == ["tape", "step_vjp"]
    runner = hawk.until_done(timed_plan, max_steps=40, record=True, x=X0.copy(), t=np.zeros(N), t_final=steps * dt, dt=dt)
    runner.run()
    grads = runner.reverse({"x": G})
    np.testing.assert_allclose(runner.planes["x"], X0 + steps * dt, rtol=8 * EPS)
    np.testing.assert_allclose(grads["x"], G, rtol=8 * EPS)
    np.testing.assert_allclose(grads["dt"], np.sum(G * steps), rtol=64 * EPS)
    assert "t_0" not in grads


def test_a_recording_of_a_time_stepped_kernel_states_the_start_copy_and_its_gradient_link():
    twin, vjp = recording(timed, 4)
    assert twin.facts["time0"] == "t_0"
    assert {"slot": "t_0", "what": "Time0", "form": "Value", "of": "t", "reduced": False} in twin.facts["links"]
    assert {"slot": "bar_t_0", "what": "Time0", "form": "Grad", "of": "t", "reduced": False} in vjp.facts["links"]
    assert {"slot": "applied_steps", "what": "Steps", "form": "Value", "of": "applied_steps", "reduced": False} in vjp.facts["links"]


def test_a_build_refuses_what_it_cannot_record_by_name(cache_dir):
    with pytest.raises(HawkError, match="window= is the steps a recorded run keeps"):
        hawk.deploy(one, window=4, cache_dir=cache_dir)
    with pytest.raises(HawkError, match="record= is a window of steps"):
        hawk.deploy(one, record=True, window=0, cache_dir=cache_dir)


def test_the_build_states_the_derived_units_with_their_digests_and_window(plan, plan_four):
    directory = plan.kernel.path.parent
    derived = plan.kernel.sidecar["derived"]
    assert [(d["role"], d["kernel"], d.get("window")) for d in derived] == [
        ("tape", "ramp_one_tape", 3), ("step_vjp", "ramp_one_step_vjp", None)]
    for record in derived:
        own = json.loads((directory / f"{record['kernel']}.json").read_text())
        assert own["digest"] == record["digest"] and own["kernel"] == record["kernel"]
    vjp = json.loads((directory / "ramp_one_step_vjp.json").read_text())
    assert vjp["derivative"]["kind"] == "vjp" and vjp["derivative"]["primal"] == "ramp_one"
    assert [r["role"] for r in plan_four.kernel.sidecar["derived"]] == ["tape_fused", "step_vjp", "fused_vjp"]
    assert sorted(plan.kernel.gradient_names()) == ["a", "b", "t", "x"]


def test_a_recording_states_the_slots_and_links_its_differentiation_made():
    twin, vjp = recording(one, 4)
    assert isinstance(twin, DerivedUnit)
    link = lambda slot, what, form, of, reduced=False: {"slot": slot, "what": what, "form": form, "of": of, "reduced": reduced}  # noqa: E731
    assert (twin.role, twin.window, twin.kernel.name) == ("tape", 4, "ramp_one_tape")
    assert twin.facts == {
        "tape": "tape", "tape_rows": 2, "taped": [{"plane": "t", "rows": 1}, {"plane": "x", "rows": 1}],
        "steps_slot": "applied_steps", "tape_on": "tape_on",
        "links": [link(name, "PrimalSlot", "Value", name) for name in ("t", "x", "t_final", "terminated", "a", "b")]}
    assert (vjp.role, vjp.window, vjp.kernel.name) == ("step_vjp", None, "ramp_one_step_vjp")
    assert vjp.facts == {"links": [
        link("bar_a", "PrimalSlot", "Grad", "a", True), link("bar_b", "PrimalSlot", "Grad", "b", True),
        link("bar_t", "PrimalSlot", "AdjointOut", "t"), link("bar_x", "PrimalSlot", "AdjointOut", "x"),
        link("bar_t_next", "PrimalSlot", "AdjointIn", "t"), link("bar_x_next", "PrimalSlot", "AdjointIn", "x"),
        link("x", "TapeRow", "Value", "x"), link("a", "PrimalSlot", "Value", "a")]}
    assert [unit.role for unit in recording(four, 4)] == ["tape_fused", "step_vjp", "fused_vjp"]


def test_the_roles_a_recording_takes_follow_the_steps_of_the_kernel_and_the_reverse_asked_for(plan, plan_four):
    assert plan.kernel.derived_roles() == ["tape", "step_vjp"]
    assert plan.kernel.derived_roles("auto") == ["tape", "step_vjp"]
    assert plan_four.kernel.derived_roles("auto") == ["tape_fused", "step_vjp", "fused_vjp"]
    assert plan_four.kernel.derived_roles("fused") == ["tape_fused", "step_vjp", "fused_vjp"]
    with pytest.raises(HawkError, match=r"no derived unit for \['fused_vjp'\]"):
        plan.kernel.derived_units(plan.kernel.derived_roles("fused"))


def test_hawk_reverse_takes_the_record_of_a_run_or_of_a_graph_and_nothing_else(plan, cache_dir):
    with pytest.raises(HawkError, match="takes the record of a run made with record=True, got dict"):
        hawk.reverse({}, {"x": np.ones(N)})
    graph = hawk.Graph(record=True)
    graph.add(hawk.deploy(tail, cache_dir=cache_dir).bind(x=np.ones(N), z=np.zeros(N)))
    graph.build()
    with pytest.raises(HawkError, match="the graph was never launched"):      # the library's reason, carried
        hawk.reverse(graph.record, {"z": np.ones(N)})


def test_a_result_field_named_record_cannot_be_a_state_plane(cache_dir):
    @hawk.kernel
    def odd(rate: Scalar, goal: Param, terminated: Terminated, record: Mutable[Scalar]):
        xn = record + rate
        record = xn
        terminated = xn >= goal

    with pytest.raises(HawkError, match=r"\['record'\]"):
        hawk.simulate(hawk.deploy(odd, cache_dir=cache_dir), rate=np.ones(N), goal=3.0, max_steps=10)


def test_a_kernel_names_the_slot_and_the_plane_a_cotangent_binds_to(plan):
    loaded = plan.kernel
    assert [loaded.slot(name) for name in loaded.sidecar["slots"]] == list(range(len(loaded.sidecar["slots"])))
    seed = np.ones(N)
    assert np.shares_memory(loaded.plane_view("x", seed), seed)
    assert isinstance(loaded, rt.LoadedKernel)
