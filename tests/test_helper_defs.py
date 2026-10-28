# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Plain ``def`` helpers, ``@hawk.primitive`` and the complex-pair helpers.

* **Helpers.** A user function called from a kernel with traced values gets the
  kernel body's control-flow rewrite once, is inlined, and is differentiated
  through. Library functions and calls with untraced arguments are left alone.
* **``@hawk.primitive``.** Named after the function, bare or with rules;
  registering the same function again is a no-op.
* **Complex pairs.** ``cmul``, ``conj``, ``abs2`` and ``arg`` on ``Vector[2]``."""

from __future__ import annotations

import math
import textwrap

import numpy as np
import pytest
from _eval import evaluate

import hawk
import hawk.math as m
from hawk import HawkError
from hawk.diff import jvp, vjp
from hawk.emit import render_body
from hawk.ir import canonical

SCALE = 3.0


def clamp_scaled(x, lo):
    if x < lo:
        r = lo
    elif x > 1.0:
        r = 1.0
    else:
        r = x
    return r * SCALE                                  # a closure over a constant


def running(x):
    acc = 0.0
    for _i in range(4):
        acc = acc * x + 1.0
        if acc > 5.0:
            break
    return acc


def nested(x):
    return clamp_scaled(running(x), 0.5)


def side(x):
    if x > 0.5:
        r = x * x
    else:
        r = x
    return r


def forever(x):
    return forever(x) + 1.0 if x > 0.0 else x


def library_only(x):
    return x * math.sqrt(4.0)


@hawk.kernel
def uses_nested(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    y = nested(x)


@hawk.kernel
def uses_condition(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    if side(x) > 0.4:                                 # a helper that branches, in a condition
        y = m.exp(x)
    else:
        y = x


@hawk.kernel
def uses_library(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    y = library_only(x)


def _text(kernel) -> str:
    return render_body(kernel.sinks, canonical(kernel.sinks)).text


def _ref_nested(x):
    acc = 0.0
    for _ in range(4):
        acc = acc * x + 1.0
        if acc > 5.0:
            break
    return (0.5 if acc < 0.5 else 1.0 if acc > 1.0 else acc) * SCALE


def _value(kernel, x):
    return evaluate(kernel.sinks, {"x": x})["y"]


def test_a_helper_with_if_for_and_break_is_rewritten_and_inlined():
    text = _text(uses_nested)
    assert "for (" in text and "break;" in text and "?" in text
    for x in (0.1, 0.4, 0.9, 1.7, 2.5):
        assert _value(uses_nested, x) == pytest.approx(_ref_nested(x), rel=1e-14)


def test_a_helper_that_branches_inside_a_branch_condition():
    for x in (0.2, 0.45, 0.6, 0.9):
        want = math.exp(x) if (x * x if x > 0.5 else x) > 0.4 else x
        assert _value(uses_condition, x) == pytest.approx(want, rel=1e-14)


def test_derivatives_go_through_the_helpers_against_central_differences():
    h = 1e-6
    for kernel in (uses_nested, uses_condition):
        for x in (0.3, 0.7, 1.4):
            slope = (_value(kernel, x + h) - _value(kernel, x - h)) / (2 * h)
            rev = evaluate(vjp(kernel.sinks), {"x": x, "bar_y": 1.0})["bar_x"]
            fwd = evaluate(jvp(kernel.sinks), {"x": x, "dot_x": 1.0})["dot_y"]
            assert rev == pytest.approx(slope, rel=1e-5, abs=1e-8)
            assert fwd == pytest.approx(slope, rel=1e-5, abs=1e-8)


def test_a_library_call_and_an_untraced_call_are_left_alone():
    assert _value(uses_library, 1.5) == pytest.approx(3.0)
    assert clamp_scaled(0.25, 0.5) == 1.5             # a plain call is plain Python


def test_recursion_is_refused_by_name():
    with pytest.raises(HawkError, match="helper 'forever' calls itself"):
        @hawk.kernel
        def loops_forever(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
            y = forever(x)


def test_the_rewrite_runs_once_per_function(monkeypatch):
    from hawk.trace import astpass, helpers

    def once(x):
        if x > 0.0:
            r = x
        else:
            r = -x
        return r

    seen = []
    real = astpass.transform
    monkeypatch.setattr(astpass, "transform", lambda *a, **k: seen.append(a[0].name) or real(*a, **k))
    helpers._REWRITTEN.pop(once.__code__, None)
    for _ in range(3):
        @hawk.kernel
        def twice(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
            y = once(x) + once(x * 2.0)
    assert seen.count("__hawk_helper__") == 1       # the kernel is the other transform


# --------------------------------------------------------------------------- #
# @hawk.primitive
# --------------------------------------------------------------------------- #
def test_a_bare_primitive_is_named_after_its_function_and_differentiates_through_it():
    @hawk.primitive
    def hd_softplus(x):
        return m.log(1.0 + m.exp(x))

    assert hawk.primitive is hawk.ext.primitive
    assert hawk.ext.primitives()["hd_softplus"] is hd_softplus

    @hawk.kernel
    def k(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = hd_softplus(x)

    got = evaluate(vjp(k.sinks), {"x": 0.3, "bar_y": 1.0})["bar_x"]
    assert got == pytest.approx(1.0 / (1.0 + math.exp(-0.3)), rel=1e-12)


def test_a_primitive_with_rules_still_takes_them_and_a_helper_inside_it_is_rewritten():
    @hawk.primitive("hd_branchy", vjp=lambda x, bar: 7.0 * bar)
    def branchy(x):
        return side(x)

    @hawk.kernel
    def k(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = branchy(x)

    assert evaluate(k.sinks, {"x": 0.8})["y"] == pytest.approx(0.64)
    assert evaluate(vjp(k.sinks), {"x": 0.8, "bar_y": 1.0})["bar_x"] == pytest.approx(7.0)


def test_running_a_notebook_cell_twice_registers_once():
    cell = textwrap.dedent('''
        import hawk, hawk.math as m

        @hawk.primitive
        def hd_cell_square(x):
            return x * x
    ''')
    first, second = {}, {}
    exec(compile(cell, "<cell>", "exec"), first)
    exec(compile(cell, "<cell>", "exec"), second)
    assert second["hd_cell_square"] is first["hd_cell_square"]


def test_a_different_function_under_a_taken_name_is_refused():
    @hawk.primitive("hd_taken")
    def one(x):
        return x + 1.0

    with pytest.raises(HawkError, match="already registered"):
        @hawk.primitive("hd_taken")
        def other(x):
            return x + 2.0


# --------------------------------------------------------------------------- #
# complex pairs
# --------------------------------------------------------------------------- #
def _pairs(dtype):
    @hawk.kernel
    def k(a: hawk.Vector[2], b: hawk.Vector[2], p: hawk.Mutable[hawk.Vector[2]],
          c: hawk.Mutable[hawk.Vector[2]], n2: hawk.Mutable[hawk.Scalar],
          ph: hawk.Mutable[hawk.Scalar]):
        p = m.cmul(a, b)
        c = m.conj(a)
        n2 = m.abs2(a)
        ph = m.arg(a)
    return k


def test_the_pair_helpers_equal_numpy_complex():
    k = _pairs("f64")
    rng = np.random.default_rng(5)
    for _ in range(6):
        a, b = rng.normal(size=2), rng.normal(size=2)
        got = evaluate(k.sinks, {"a": a, "b": b})
        za, zb = complex(*a), complex(*b)
        np.testing.assert_allclose(got["p"], [(za * zb).real, (za * zb).imag], rtol=1e-14)
        np.testing.assert_allclose(got["c"], [za.real, -za.imag], rtol=1e-14)
        assert got["n2"] == pytest.approx(abs(za) ** 2, rel=1e-14)
        assert got["ph"] == pytest.approx(np.angle(za), rel=1e-14)


def test_the_pair_helpers_differentiate_against_central_differences():
    @hawk.kernel
    def k(a: hawk.Vector[2], b: hawk.Vector[2], y: hawk.Mutable[hawk.Scalar]):
        z = m.cmul(a, b)
        y = m.arg(z) + m.abs2(m.conj(z))

    rng = np.random.default_rng(9)
    a, b = rng.normal(size=2) + 1.0, rng.normal(size=2) + 1.0
    h = 1e-6

    def f(a_, b_):
        return evaluate(k.sinks, {"a": a_, "b": b_})["y"]

    got = evaluate(vjp(k.sinks), {"a": a, "b": b, "bar_y": 1.0})
    for i in range(2):
        e = np.eye(2)[i] * h
        np.testing.assert_allclose(got["bar_a"][i], (f(a + e, b) - f(a - e, b)) / (2 * h), rtol=1e-5)
        np.testing.assert_allclose(got["bar_b"][i], (f(a, b + e) - f(a, b - e)) / (2 * h), rtol=1e-5)
    da = np.array([0.3, -0.2])
    fwd = evaluate(jvp(k.sinks), {"a": a, "b": b, "dot_a": da, "dot_b": np.zeros(2)})["dot_y"]
    assert fwd == pytest.approx((f(a + h * da, b) - f(a - h * da, b)) / (2 * h), rel=1e-5)


def test_a_pair_helper_refuses_a_non_pair():
    with pytest.raises(HawkError, match=r"cmul: a complex value is a Vector\[2\]"):
        @hawk.kernel
        def k(a: hawk.Vector[3], y: hawk.Mutable[hawk.Vector[2]]):
            y = m.cmul(a, a)


def test_the_pair_helpers_render_in_both_precisions():
    from hawk.emit import FLOAT32, FLOAT64, render_source
    from hawk.targets import TARGETS

    k = _pairs("both")
    host = TARGETS["host"]
    for mode in (FLOAT32, FLOAT64):
        text = render_source("pairs", k.sinks, canonical(k.sinks), host, mode=mode).text
        assert "atan2" in text
