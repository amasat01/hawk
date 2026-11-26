# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Rules of a function that hold a fixed-count loop, checked against central differences on the host.

A function's rule is an ordinary traced function of the function table whose body may read nothing but its parameters, so the
properties of a loop in a rule (free leaves, loop locals) are those of a function whose body holds a loop or a call inside a
loop, held by test_function_derivatives.py."""

import pytest
from _eval import evaluate

import hawk
from hawk import _ir
from hawk.diff import jvp, vjp

H = 1e-5


def _vjp_rule(x, bar):
    acc = 0.0 * bar
    for k in range(3):
        acc = acc + bar * x
    return acc


def _jvp_rule(x, dx):
    acc = 0.0 * dx
    for k in range(3):
        acc = acc + x * dx
    return acc


def _make(name):
    @hawk.function(name=name, vjp=_vjp_rule, jvp=_jvp_rule)
    def cube_half(
        x,
    ):  # 1.5 x^2: derivative 3 x, which both rules sum as three terms of x
        return 1.5 * x * x

    return cube_half


looped = _make("w1_loop_rule")


@hawk.kernel
def uses_looped(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    y = looped(x)


@pytest.mark.parametrize("x0", [-1.3, 0.4, 2.6])
def test_the_loop_rules_match_finite_differences_on_the_host(x0):
    sinks = uses_looped.sinks
    h = H * max(1.0, abs(x0))
    ref = (
        float(evaluate(sinks, {"x": x0 + h})["y"])
        - float(evaluate(sinks, {"x": x0 - h})["y"])
    ) / (2 * h)
    assert ref == pytest.approx(3.0 * x0, rel=1e-6)
    assert float(
        evaluate(vjp(sinks), {"x": x0, "bar_y": 1.0})["bar_x"]
    ) == pytest.approx(ref, rel=1e-6, abs=1e-9)
    assert float(
        evaluate(jvp(sinks), {"x": x0, "dot_x": 1.0})["dot_y"]
    ) == pytest.approx(ref, rel=1e-6, abs=1e-9)


def test_the_digest_of_a_kernel_using_the_function_is_stable_across_two_declarations():
    """RESTATED: a second declaration is a new object (there is no registry to return the first), so the property is that the
    kernel's digest is the same."""
    first = _ir.digest(uses_looped.walk)
    again = _make("w1_loop_rule")

    @hawk.kernel
    def uses_looped2(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = again(x)

    vjp(uses_looped), vjp(uses_looped2)
    assert _ir.digest(uses_looped2.walk) == first == _ir.digest(uses_looped.walk)
