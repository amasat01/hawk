# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.diff.step_function(kernel)`` — one step of an in-place step kernel as a pure map.

The function reads every carried plane as it was before the step and writes ``<plane>_next``. Rows: its planes and name; its
values equal a NumPy transcription of the step and the in-place kernel's own step; its jvp equals the forward twin
``step_jvp`` of the in-place kernel at the same point; a kernel with no carried plane is refused by name."""

from __future__ import annotations

import numpy as np
import pytest
from _eval import evaluate

from hawk import Mutable, Param, Scalar, Terminated, kernel
from hawk import math as hm
from hawk.diff import jvp, step_function, step_jvp
from hawk.errors import HawkError


@kernel(steps=1)
def pend(dt: Param, terminated: Terminated, x: Mutable[Scalar], v: Mutable[Scalar]):
    vn = v - dt * hm.sin(x)
    x = x + dt * v
    v = vn
    terminated = x > 10.0


@kernel
def stateless(a: Scalar, out: Mutable[Scalar]):
    out = a * 2.0


POINT = dict(dt=0.05, terminated=False, x=0.7, v=-0.4)


def _names(sinks):
    return sorted(s.field_text("name") for s in sinks)


def test_a_step_function_is_named_for_its_kernel_and_writes_one_next_plane_per_carried_state_plane():
    function = step_function(pend)
    assert function.name == "pend_step"
    assert _names(function.sinks) == ["v_next", "x_next"]


def test_the_values_of_a_step_function_are_one_step_of_the_in_place_kernel_and_of_the_numpy_transcription():
    got = evaluate(step_function(pend).sinks, POINT)
    in_place = evaluate(pend.sinks, POINT)
    x, v, dt = POINT["x"], POINT["v"], POINT["dt"]
    np.testing.assert_allclose(float(got["x_next"]), x + dt * v, rtol=1e-15)
    np.testing.assert_allclose(float(got["v_next"]), v - dt * np.sin(x), rtol=1e-15)
    np.testing.assert_allclose(
        [float(got["x_next"]), float(got["v_next"])],
        [float(in_place["x"]), float(in_place["v"])],
        rtol=1e-15,
    )


def test_the_jvp_of_a_step_function_equals_the_forward_twin_of_the_in_place_kernel():
    seeds = {"dot_dt": 1.0, "dot_x": 0.3, "dot_v": -0.2}
    tangent = evaluate(
        tuple(jvp(step_function(pend), wrt=("dt", "x", "v"))), {**POINT, **seeds}
    )
    twin = evaluate(step_jvp(pend).sinks, {**POINT, **seeds})
    np.testing.assert_allclose(
        [float(tangent["dot_x_next"]), float(tangent["dot_v_next"])],
        [float(twin["dot_x"]), float(twin["dot_v"])],
        rtol=1e-14,
    )


def test_a_kernel_with_no_carried_plane_is_refused_by_name():
    with pytest.raises(HawkError, match=r"'stateless'.*no floating-point state plane"):
        step_function(stateless)
