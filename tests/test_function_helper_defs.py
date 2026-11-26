# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``@hawk.function`` beside the plain helpers of test_helper_defs.py: a bare decorator names the function after its ``def``, rules
are taken beside a helper that needs the control-flow rewrite, a notebook cell run twice gives one digest, and a taken name is
refused by the function table."""

import math
import textwrap

import pytest
from _eval import evaluate
from test_helper_defs import side

import hawk
import hawk.math as m
from hawk import HawkError, _ir
from hawk.diff import vjp


def test_a_bare_function_is_named_after_its_def_and_differentiates_through_it():
    @hawk.function
    def hd_softplus(x):
        return m.log(1.0 + m.exp(x))

    @hawk.kernel
    def k(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = hd_softplus(x)

    got = evaluate(vjp(k.sinks), {"x": 0.3, "bar_y": 1.0})["bar_x"]
    assert got == pytest.approx(1.0 / (1.0 + math.exp(-0.3)), rel=1e-12)


def test_a_function_with_rules_still_takes_them_and_a_helper_inside_it_is_rewritten():
    @hawk.function(name="hd_branchy", vjp=lambda x, bar: 7.0 * bar)
    def branchy(x):
        return side(x)

    @hawk.kernel
    def k(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = branchy(x)

    assert evaluate(k.sinks, {"x": 0.8})["y"] == pytest.approx(0.64)
    assert evaluate(vjp(k.sinks), {"x": 0.8, "bar_y": 1.0})["bar_x"] == pytest.approx(
        7.0
    )


def test_running_a_notebook_cell_twice_declares_the_same_function_text():
    """RESTATED: the second run is a new object; what must hold is that a kernel calling either has one digest."""
    cell = textwrap.dedent("""
        import hawk

        @hawk.function
        def hd_cell_square(x):
            return x * x
    """)
    first, second = {}, {}
    exec(compile(cell, "<cell>", "exec"), first)
    exec(compile(cell, "<cell>", "exec"), second)
    one, two = first["hd_cell_square"], second["hd_cell_square"]

    @hawk.kernel
    def uses_one(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = one(x)

    @hawk.kernel
    def uses_two(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = two(x)

    assert _ir.digest(uses_one.walk) == _ir.digest(uses_two.walk)


def test_a_different_function_under_a_taken_name_is_refused():
    """RESTATED: refused by the function table when one kernel holds both."""

    @hawk.function(name="hd_taken")
    def one(x):
        return x + 1.0

    @hawk.function(name="hd_taken")
    def other(x):
        return x + 2.0

    with pytest.raises(HawkError, match="already declared with another body"):

        @hawk.kernel
        def k(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
            y = one(x) + other(x)
