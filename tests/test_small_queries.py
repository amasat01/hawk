# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The small queries: device properties, host thread control, the bound plan's ``names`` and ``n``, and ``returns="numpy"`` on a
call and on a stepping run. A figure is held against the library's own answer; a conversion against NumPy."""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import HawkError, Mutable, Param, Scalar, Terminated
from hawk import plans as plans_module

PROPS = dict(
    name="Test Device",
    cc_major=8,
    cc_minor=6,
    sm_count=16,
    clock_rate_khz=1500000,
    memory_clock_rate_khz=5000000,
    memory_bus_width_bits=256,
    shared_mem_per_block=49152,
    shared_mem_per_sm=102400,
    regs_per_block=65536,
    regs_per_sm=65536,
    warp_size=32,
    peak_bytes_per_s=3.2e11,
    peak_flops_sp=1.2e13,
    peak_flops_dp=1.9e11,
    fp64_ratio=0.0156,
    ridge_flops_per_byte_sp=37.5,
    ridge_flops_per_byte_dp=0.59,
)


class _Door:
    """A door that answers ``device_props`` with a fixed dict, as the library's door does for a device it sees."""

    def device_props(self, ordinal):
        assert ordinal == 1
        return dict(PROPS)


def test_device_props_hands_over_what_the_door_states_and_selects_the_ridge_by_dtype(
    monkeypatch,
):
    served = next(t.id for t in plans_module.TARGETS.values() if t.device)
    monkeypatch.setattr(
        plans_module._place, "open_door", lambda target, ordinal: _Door()
    )
    props = hawk.device_props(f"{served}:1")
    assert isinstance(props, hawk.DeviceProps) and props == hawk.DeviceProps(**PROPS)
    assert props.ridge_flops_per_byte("float32") == PROPS["ridge_flops_per_byte_sp"]
    assert props.ridge_flops_per_byte("float64") == PROPS["ridge_flops_per_byte_dp"]
    with pytest.raises(Exception, match="cannot assign"):
        props.name = "other"


def test_device_props_refuses_a_device_that_is_not_a_served_device_target_by_name():
    with pytest.raises(
        HawkError,
        match=r"hawk.device_props: device='nowhere:0' is not '<target>:<ordinal>' of a device target",
    ):
        hawk.device_props("nowhere:0")


def test_the_host_thread_count_is_set_read_back_and_restored():
    before = hawk.host_threads()
    try:
        hawk.set_host_threads(2)
        assert hawk.host_threads() == 2
        hawk.set_host_threads(None)
        assert hawk.host_threads() == before
    finally:
        hawk.set_host_threads(None)


def test_a_thread_count_that_is_not_a_whole_number_or_that_the_library_refuses_is_refused():
    with pytest.raises(HawkError, match="takes a whole number of threads or None"):
        hawk.set_host_threads(2.5)
    with pytest.raises(HawkError, match="takes a whole number of threads or None"):
        hawk.set_host_threads(True)
    with pytest.raises(HawkError):
        hawk.set_host_threads(-1)


@hawk.kernel
def affine(x: Scalar, a: Param, y: Mutable[Scalar]):
    y = a * x


def test_a_bound_plan_states_its_names_in_declared_order_and_its_sample_count():
    x, y = np.arange(7.0), np.zeros(7)
    bound = hawk.deploy(affine).bind(x=x, a=2.0, y=y)
    assert bound.n == 7
    assert bound.names == tuple(name for _role, name in bound.kernel.arg_spec)
    assert sorted(bound.names) == ["a", "x", "y"]
    bound.close()


def test_returns_numpy_on_a_call_converts_every_array_and_refuses_another_word():
    x = np.arange(5.0)
    plan = hawk.deploy(affine)
    out = plan(x, a=3.0, returns="numpy")
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, 3.0 * x)
    np.testing.assert_array_equal(
        affine(x, a=3.0, returns="numpy"), 3.0 * x
    )  # the direct call is the plan's
    with pytest.raises(HawkError, match="'numpy' is the one conversion this build has"):
        plan(x, a=3.0, returns="torch")


def ramp(rate: Scalar, goal: Param, terminated: Terminated, x: Mutable[Scalar]):
    xn = x + rate
    x = xn
    terminated = xn >= goal


def test_simulate_returns_numpy_state_and_returned_keeps_or_converts():
    rate = np.linspace(0.2, 1.0, 9)
    kernel = hawk.steps(hawk.kernel(ramp), 1)
    result = hawk.simulate(
        kernel, max_steps=40, rate=rate, goal=3.0, x=np.zeros(9), returns="numpy"
    )
    assert all(isinstance(v, np.ndarray) for v in result.state.values())
    assert (
        result.returned(None) is result
        and result.returned("numpy").state["x"] is result.state["x"]
    )
    steps = np.ceil(3.0 / rate)
    np.testing.assert_allclose(result.x, steps * rate, rtol=1e-12)
    with pytest.raises(HawkError, match="'numpy' is the one conversion"):
        hawk.simulate(
            kernel, max_steps=40, rate=rate, goal=3.0, x=np.zeros(9), returns="jax"
        )
