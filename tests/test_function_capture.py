# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A function sees its parameters only: a traced value of the enclosing kernel it captures is refused by name.

A ``def`` cannot be written inside a kernel body, so a value reaches a function's body (or a rule) through module state; the
tests stash the kernel's value in a module list. Plain helpers are inline and may use the kernel's values; one test pins that
difference. A Python or NumPy constant is not a traced value and stays allowed."""

from __future__ import annotations

import numpy as np
import pytest
from _eval import evaluate

import hawk
from hawk import HawkError
from hawk.trace.value import Value

SCALE = 3.0
SHIFT = np.float64(0.5)
STASH: list = []


def keep(value):
    """Hold ``value`` where a function's body can reach it, and give it back."""
    STASH.append(value)
    return value


@hawk.function
def typed_body(a: hawk.Scalar) -> hawk.Scalar:
    return a + STASH[-1]


@hawk.function
def untyped_body(a):
    return a + STASH[-1]


@hawk.function(jvp=lambda a, da: da * STASH[-1])
def jvp_rule(a: hawk.Scalar) -> hawk.Scalar:
    return a * 1.0


@hawk.function(vjp=lambda a, bar: bar * STASH[-1])
def vjp_rule(a: hawk.Scalar) -> hawk.Scalar:
    return a * 1.0


@hawk.function(vjp=lambda a, bar: bar * STASH[-1])
def untyped_rule(a):
    return a * 1.0


@hawk.function
def constants(a: hawk.Scalar) -> hawk.Scalar:
    return a * SCALE + SHIFT


def plain(a):
    return a + STASH[-1]


FUNCTIONS = [typed_body, untyped_body, jvp_rule, vjp_rule, untyped_rule]
IDS = ["typed body", "untyped body", "jvp rule", "vjp rule", "untyped rule"]


def _kernel_holding(source, function):
    """Trace a kernel whose body stashes ``source(x, w, i)`` from inside a loop, then calls ``function``."""

    @hawk.kernel
    def k(x: hawk.Scalar, w: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        acc = 0.0
        for i in range(3):
            acc = acc + keep(source(x, w, i)) * 0.0
        y = function(x)

    return k


def _declaring(function):
    """The name ``function`` and its rules are refused by: the rule functions carry a suffix."""
    return function.__name__


@pytest.mark.parametrize("function", FUNCTIONS, ids=IDS)
@pytest.mark.parametrize(
    "source", [lambda x, w, i: x * 2.0, lambda x, w, i: w + 0.0], ids=["local", "plane"]
)
def test_a_read_of_a_kernel_local_or_plane_is_refused_naming_the_function(
    source, function
):
    STASH.clear()
    with pytest.raises(
        HawkError,
        match=_declaring(function)
        + r".*a traced body is pure and its body holds a read of the plane",
    ):
        _kernel_holding(source, function)


@pytest.mark.parametrize("function", FUNCTIONS, ids=IDS)
def test_the_index_of_a_kernel_loop_is_refused_naming_the_function(function):
    STASH.clear()
    with pytest.raises(
        HawkError,
        match=_declaring(function)
        + r".*a traced body is pure and its body holds the index or a carry of a loop it is not in",
    ):
        _kernel_holding(lambda x, w, i: i + 0.0, function)


@pytest.mark.parametrize("function", FUNCTIONS, ids=IDS)
def test_a_loop_of_the_kernel_that_reads_a_plane_is_refused_naming_the_function(
    function,
):
    STASH.clear()

    with pytest.raises(
        HawkError, match=_declaring(function) + r".*a read of the plane 'w'"
    ):

        @hawk.kernel
        def k(x: hawk.Scalar, w: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
            acc = 0.0
            for _ in range(3):
                acc = acc + w
            keep(acc)
            y = function(x)


@pytest.mark.parametrize("function", FUNCTIONS, ids=IDS)
def test_a_value_left_over_from_another_kernel_is_refused_naming_the_function(function):
    @hawk.kernel
    def other(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = x * 3.0

    STASH.clear()
    STASH.append(Value(other.sinks[0].operands[0]))
    with pytest.raises(
        HawkError,
        match=_declaring(function) + r".*its body holds a value of another trace",
    ):

        @hawk.kernel
        def k(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
            y = function(x)


def test_a_python_or_numpy_constant_of_the_module_is_not_a_capture():
    @hawk.kernel
    def k(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = constants(x)

    assert evaluate(k.sinks, {"x": 2.0})["y"] == pytest.approx(
        2.0 * SCALE + 0.5, rel=1e-15
    )


def test_a_plain_helper_is_inline_and_may_use_the_values_of_the_kernel():
    STASH.clear()

    @hawk.kernel
    def k(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        t = keep(x * 2.0)
        y = plain(x) + t * 0.0

    assert evaluate(k.sinks, {"x": 1.5})["y"] == pytest.approx(1.5 + 3.0, rel=1e-15)
