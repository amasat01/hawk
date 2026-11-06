# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Conditions combine with ``& | ^ ~`` and ``and or not``; arithmetic on a bool is refused.

The logical forms lower to ``land`` / ``lor`` / ``ne`` / ``lnot`` and are spelled so the C++ is a bool (the bitwise kinds are
integer-only; a bool ``*`` is an int in C++, which a derivative's bool lane row refuses). Each row runs the primal and its
forward derivative, per-sample and interchanged, against NumPy. ``+ - * / ** // %``, unary ``-``, ``abs``, ``min`` / ``max``
and the ``hawk.math`` numeric functions on a bool are refused with the remedy.
"""

import _oracle as O
import numpy as np
import pytest
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import HawkError, Mutable, Scalar
from hawk.artifact import build_bundle

N = 41


@hawk.kernel
def both(x: Scalar, y: Mutable[Scalar]):
    y = m.select((x >= 0.0) & (x <= 3.0), x * x, -x)


@hawk.kernel
def either(x: Scalar, y: Mutable[Scalar]):
    y = m.select((x <= 1.0) | (x >= 2.0), x * x, -x)


@hawk.kernel
def differ(x: Scalar, y: Mutable[Scalar]):
    y = m.select((x >= 0.0) ^ (x >= 2.0), x * x, -x)


@hawk.kernel
def negated(x: Scalar, y: Mutable[Scalar]):
    y = m.select(~((x >= 0.0) & (x <= 3.0)), -x, x * x)


@hawk.kernel
def worded(x: Scalar, y: Mutable[Scalar]):
    y = m.select(x >= 0.0 and x <= 3.0 or not x <= 100.0, x * x, -x)


@hawk.kernel
def looped(x: Scalar, y: Mutable[Scalar]):
    inside = (x >= 0.0) & (x <= 3.0)             # held across the loop: a bool lane row on the host interchange
    acc = x
    for _ in range(20):
        acc = m.select(inside, acc * 1.5, acc + x)
    y = acc


def _looped(x):
    return np.where(_inside(x), x * 1.5**20, x * 21.0)


def _inside(x):
    return (x >= 0.0) & (x <= 3.0)


def _outside(x):
    return (x <= 1.0) | (x >= 2.0)


def _differ(x):
    return (x >= 0.0) ^ (x >= 2.0)


CASES = {
    "both": (both, lambda x: np.where(_inside(x), x * x, -x), lambda x: np.where(_inside(x), 2 * x, -1.0)),
    "either": (either, lambda x: np.where(_outside(x), x * x, -x), lambda x: np.where(_outside(x), 2 * x, -1.0)),
    "differ": (differ, lambda x: np.where(_differ(x), x * x, -x), lambda x: np.where(_differ(x), 2 * x, -1.0)),
    "negated": (negated, lambda x: np.where(~_inside(x), -x, x * x), lambda x: np.where(~_inside(x), -1.0, 2 * x)),
    "worded": (worded, lambda x: np.where(_inside(x), x * x, -x), lambda x: np.where(_inside(x), 2 * x, -1.0)),
    "looped": (looped, _looped, lambda x: np.where(_inside(x), 1.5**20, 21.0)),
}


def _derived(kernel):
    from hawk.diff import jvp
    from hawk.trace.kernel import Kernel

    return Kernel(kernel.name + "_jvp", jvp(kernel, wrt=("x",), primal_unit="0" * 64))


def _run(kernel, planes, tmp_path, cache_dir, defines=()):
    b = build_bundle([kernel], tmp_path, targets=("host",), cache_dir=cache_dir, host_profile="native", wide_form="thread",
                     defines=defines)
    loaded = O.load(b.directory, kernel.name, sidecar_of(b, kernel.name))
    return np.asarray(O.run_kernel(loaded, N, **planes))


@pytest.mark.parametrize("name", CASES)
def test_bool_arithmetic_is_numpys_in_the_primal_and_its_derivative(name, tmp_path, cache_dir):
    kernel, want, want_d = CASES[name]
    x = np.linspace(-1.0, 4.0, N)
    for defines in ((), ("HAWK_HOST_INTERCHANGE=0",)):
        assert np.array_equal(_run(kernel, {"x": x.copy(), "y": np.zeros(N)}, tmp_path / "p", cache_dir, defines), want(x))
        got = _run(_derived(kernel), {"x": x.copy(), "dot_x": np.ones(N), "dot_y": np.zeros(N)}, tmp_path / "d", cache_dir,
                   defines)
        assert np.array_equal(got, want_d(x)), f"{name}: the derivative differs from the analytic one"


def _refused(body, match="takes numbers, not a bool"):
    with pytest.raises(HawkError, match=match):
        hawk.kernel(body)


def test_the_dfnx_shaped_product_of_two_conditions_is_refused():
    def body(x: Scalar, y: Mutable[Scalar]):
        y = m.select((x >= 0.0) * (x <= 3.0), x * x, -x)

    _refused(body, r"op 'mul' takes numbers.*use & \| \^ ~.*m.select\(cond, x, 0.0\)")


def test_a_sum_of_conditions_is_refused():
    def body(x: Scalar, y: Mutable[Scalar]):
        y = m.select((x >= 0.0) + (x <= 3.0), x * x, -x)

    _refused(body)


def test_a_condition_times_a_number_is_refused():
    def body(x: Scalar, y: Mutable[Scalar]):
        y = (x >= 0.0) * (x * x)

    _refused(body)


def test_a_negated_condition_is_refused():
    def body(x: Scalar, y: Mutable[Scalar]):
        y = m.select(-(x >= 0.0), x, -x)

    _refused(body)


def test_min_of_two_conditions_is_refused():
    def body(x: Scalar, y: Mutable[Scalar]):
        y = m.select(m.min(x >= 0.0, x <= 3.0), x, -x)

    _refused(body)


def test_a_precedence_slip_names_the_parenthesised_spelling():
    def body(x: Scalar, y: Mutable[Scalar]):
        y = m.select(x >= 0.0 & x <= 3.0, x, -x)

    _refused(body, r"parenthesise each comparison around & and \|.*\(x >= 0\) & \(x <= 3\)")


def test_a_traced_value_in_a_python_truth_test_names_the_same_remedy():
    def body(x: Scalar, y: Mutable[Scalar]):
        y = m.select(bool(x >= 0.0), x, -x)

    _refused(body, r"no truth value.*parenthesise each comparison around & and \|")


@pytest.mark.gpu
@pytest.mark.parametrize("name", CASES)
def test_the_device_returns_numpys_values_for_the_primal_and_its_derivative(name, cache_dir):
    import eagle.plan
    from eagle import arrays
    from eagle._alloc import synchronize

    kernel, want, want_d = CASES[name]
    x = np.linspace(-1.0, 4.0, N)

    def run(k, **planes):
        dev = {n: arrays.asarray(np.ascontiguousarray(v), device="cuda:0") for n, v in planes.items()}
        eagle.plan.auto(k, targets=("cuda",), scalar_type="float64", cache_dir=cache_dir).bind(**dev).launch()
        synchronize()
        return np.asarray(dev[[n for n in planes if n.startswith(("y", "dot_y"))][0]].get())

    assert np.array_equal(run(kernel, x=x, y=np.zeros(N)), want(x))
    assert np.array_equal(run(_derived(kernel), x=x, dot_x=np.ones(N), dot_y=np.zeros(N)), want_d(x))
