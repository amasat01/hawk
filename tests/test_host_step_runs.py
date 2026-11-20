# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Host steps in a run: ``hawk.run`` and the stepping run (``hawk.until_done``, ``hawk.simulate``) take a kernel or plan, a bound
host step, a built graph, or any iterable of them. The model is ``x <- 3 x`` (a host step) then ``x <- a x + b`` (a kernel),
every sample stopping after ``T`` steps, so the oracle is the loop in NumPy."""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import HawkError, Mutable, Param, Scalar, Terminated

N = 12
T = 5
A, B = 0.5, 0.25
X0 = 1.0 + np.arange(N) / 8
G = np.linspace(0.5, 1.5, N)


def ramp(
    a: Param,
    b: Param,
    t_final: Scalar,
    terminated: Terminated,
    x: Mutable[Scalar],
    t: Mutable[Scalar],
):
    x = a * x + b
    tn = t + 1.0
    t = tn
    terminated = tn >= t_final


def plus_one(x: Scalar, y: Mutable[Scalar]):
    y = x + 1.0


one = hawk.kernel(ramp, steps=1)
add_one = hawk.kernel(plus_one)


@hawk.host_step
def triple(x: Mutable):
    x *= 3.0


@triple.reverse
def triple_reverse(value, cotangent, gradient):
    gradient.x += 3.0 * cotangent.x


def planes() -> dict:
    return dict(x=X0.copy(), t=np.zeros(N), t_final=np.full(N, float(T)), a=A, b=B)


def hand_loop() -> np.ndarray:
    """The oracle: ``T`` times, triple then ``a x + b``."""
    x = X0.copy()
    for _ in range(T):
        x = A * (3.0 * x) + B
    return x


def run_model(model, **more):
    arrays = planes() | more
    hawk.until_done(model, max_steps=40, **arrays).run()
    return arrays["x"]


def test_a_list_of_a_host_step_and_a_kernel_equals_the_hand_loop():
    x = planes()["x"]
    np.testing.assert_allclose(
        run_model([triple(x=x), one], x=x), hand_loop(), rtol=1e-12
    )


def test_a_tuple_and_a_generator_run_as_the_list_does():
    x = X0.copy()
    np.testing.assert_allclose(
        run_model((triple(x=x), one), x=x), hand_loop(), rtol=1e-12
    )
    x = X0.copy()
    np.testing.assert_allclose(
        run_model((item for item in [triple(x=x), one]), x=x), hand_loop(), rtol=1e-12
    )


def test_a_graph_made_by_a_builder_function_runs_once_per_step():
    @hawk.Graph.builder
    def step(x) -> hawk.Graph:
        g = hawk.Graph()
        triple(x)
        return g

    x = X0.copy()
    np.testing.assert_allclose(run_model([step(x), one], x=x), hand_loop(), rtol=1e-12)


def test_simulate_takes_the_same_model_and_a_host_step_then_two_kernels_runs_in_order():
    x = X0.copy()
    result = hawk.simulate([triple(x=x), one], max_steps=40, **(planes() | dict(x=x)))
    assert result.status == "done" and result.steps == T
    np.testing.assert_allclose(result.x, hand_loop(), rtol=1e-12)


def test_a_recorded_run_reverses_through_the_kernel_and_the_step_to_the_closed_form(
    cache_dir,
):
    x = X0.copy()
    plan = hawk.deploy(one, record=True, window=3, cache_dir=cache_dir)
    runner = hawk.until_done(
        [triple(x=x), plan], max_steps=40, record=True, **(planes() | dict(x=x))
    )
    runner.run()
    gradients = hawk.reverse(runner.record, {"x": G})
    np.testing.assert_allclose(gradients["x"], G * (3.0 * A) ** T, rtol=1e-12)


def test_hawk_run_takes_a_step_alone_a_list_in_order_and_a_built_graph():
    x = X0.copy()
    hawk.run(triple(x=x))
    np.testing.assert_array_equal(x, 3.0 * X0)
    y, z = np.zeros(N), np.zeros(N)
    x = X0.copy()
    hawk.run([triple(x=x), add_one], x=x, y=y)  # the step, then the kernel
    np.testing.assert_allclose(y, 3.0 * X0 + 1.0, rtol=1e-12)
    hawk.run([add_one, triple(x=y)], x=X0.copy(), y=y)  # the kernel, then the step
    np.testing.assert_allclose(y, 3.0 * (X0 + 1.0), rtol=1e-12)
    g = hawk.Graph().add(triple(x=z)).build()
    z[:] = 1.0
    hawk.run(g)
    np.testing.assert_array_equal(z, 3.0)


def test_the_refusals_are_by_name():
    x = X0.copy()
    with pytest.raises(HawkError, match="the model is empty"):
        hawk.until_done([], max_steps=3, **planes())
    with pytest.raises(
        HawkError, match="holds a str: a run takes plans, kernels, bound host steps"
    ):
        hawk.until_done([triple(x=x), "ramp"], max_steps=3, **planes())
    with pytest.raises(HawkError, match="holds a HostStep"):
        hawk.until_done([triple, one], max_steps=3, **planes())
    with pytest.raises(HawkError, match="before_plan"):
        arrays = planes()
        hawk.until_done(
            [one, triple(x=arrays["x"])], max_steps=3, record=True, **arrays
        )
    with pytest.raises(HawkError, match="no plan or kernel"):
        hawk.until_done([triple(x=x)], max_steps=3, **planes())
    with pytest.raises(HawkError, match="not built"):
        hawk.until_done([hawk.Graph().add(triple(x=x)), one], max_steps=3, **planes())
    with pytest.raises(
        HawkError, match="base= and count= are the range of a loaded kernel"
    ):
        hawk.run(triple(x=x), base=1)


def test_a_host_step_after_a_kernel_runs_after_it_in_every_step():
    arrays = planes()
    hawk.until_done([one, triple(x=arrays["x"])], max_steps=40, **arrays).run()
    expected = X0.copy()
    for _ in range(T):
        expected = 3.0 * (A * expected + B)
    np.testing.assert_allclose(arrays["x"], expected, rtol=1e-14)


def test_host_steps_on_both_sides_of_a_kernel_run_in_the_order_written():
    arrays = planes()
    x = arrays["x"]
    hawk.until_done([triple(x=x), one, triple(x=x)], max_steps=40, **arrays).run()
    expected = X0.copy()
    for _ in range(T):
        expected = 3.0 * (A * (3.0 * expected) + B)
    np.testing.assert_allclose(x, expected, rtol=1e-14)
