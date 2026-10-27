# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A Mutable vector filled element by element differentiates when the body stores
every element before it reads one (:mod:`hawk.trace.cover`): zeros are then the
start value, and the gradient matches finite differences. Anything weaker keeps the
launch-start value and the derivative refuses BY NAME ("read before it is
assigned"): a partial fill, a read of an element not yet stored, a store under a
one-sided ``if``, an index the trace cannot compute."""

from __future__ import annotations

import numpy as np
import pytest
from _eval import evaluate

import hawk
import hawk.math as m
from hawk.diff import jvp, vjp
from hawk.ir import HawkError

V3 = hawk.Vector[3]
OUT3 = hawk.Mutable[hawk.Vector[3]]
DAYS = 6


@hawk.kernel
def straight(x: V3, g: hawk.Scalar, out: OUT3):
    out[0] = x[0] * g
    out[1] = x[1] * x[0]
    out[2] = x[2] + g


@hawk.kernel
def reversed_order(x: V3, g: hawk.Scalar, out: OUT3):
    out[2] = x[2] * g
    out[0] = x[0] + 1.0
    out[1] = out[0] * out[2]


@hawk.kernel
def both_arms(x: V3, g: hawk.Scalar, out: OUT3):
    for d in range(3):
        if x[d] > 0.0:
            out[d] = x[d] * g
        else:
            out[d] = x[d] * x[d]


@hawk.kernel
def fill_then_stencil(x: V3, g: hawk.Scalar, out: OUT3):
    out[0] = x[0]
    for d in range(1, 3):
        out[d] = out[d - 1] * g + x[d]


@hawk.kernel
def duplicated_store(x: V3, g: hawk.Scalar, out: OUT3):
    for d in range(3):
        out[d] = x[d] * g
    for d in range(3):
        out[d] = out[d] + 1.0


@hawk.kernel
def named_bound(x: hawk.Vector[DAYS], g: hawk.Scalar, curve: hawk.Mutable[hawk.Vector[DAYS]]):
    """The epidemiology shape: one value per day from a module-level bound, the
    state carried between days."""
    s = 1.0
    for day in range(DAYS):
        s = s * (1.0 - 0.1 * g) + x[day]
        curve[day] = s * s
        if day == 2:
            s = s + 0.5


@hawk.kernel
def nested_index(x: hawk.Vector[6], g: hawk.Scalar, out: hawk.Mutable[hawk.Vector[6]]):
    for i in range(2):
        for d in range(3):
            out[3 * i + d] = x[3 * i + d] * g + i


def _fd_check(kernel, env, wrt, outname, width):
    """vjp and jvp against central differences, one-hot on the output."""
    prim = lambda e: np.asarray(evaluate(kernel.sinks, {**e, "terminated": False})[outname], dtype=float)
    for k in range(width):
        bar = np.eye(width)[k]
        rev = evaluate(vjp(kernel, wrt=wrt), {**env, "terminated": False, "bar_" + outname: bar})
        for w in wrt:
            base = np.asarray(env[w], dtype=float)
            for i in range(max(base.size, 1)):
                hi, lo = base.copy().reshape(-1), base.copy().reshape(-1)
                hi[i] += 1e-6
                lo[i] -= 1e-6
                shape = base.shape
                want = (prim({**env, w: hi.reshape(shape)})[k] - prim({**env, w: lo.reshape(shape)})[k]) / 2e-6
                got = np.asarray(rev["bar_" + w], dtype=float).reshape(-1)[i]
                np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-8)
    t = {w: np.asarray(env[w], dtype=float) * 0 + 0.3 for w in wrt}
    fwd = evaluate(jvp(kernel, wrt=wrt), {**env, "terminated": False,
                                         **{"dot_" + w: t[w] for w in wrt}})
    base = {w: np.asarray(env[w], dtype=float) for w in wrt}
    hi = {**env, **{w: base[w] + 1e-6 * t[w] for w in wrt}}
    lo = {**env, **{w: base[w] - 1e-6 * t[w] for w in wrt}}
    np.testing.assert_allclose(np.asarray(fwd["dot_" + outname]), (prim(hi) - prim(lo)) / 2e-6,
                               rtol=1e-6, atol=1e-8)


X3 = np.array([0.7, -0.4, 0.3])


@pytest.mark.parametrize("kernel", [straight, reversed_order, both_arms, fill_then_stencil,
                                    duplicated_store],
                         ids=lambda k: k.name)
def test_a_complete_fill_differentiates(kernel):
    _fd_check(kernel, {"x": X3, "g": 1.3}, ("x", "g"), "out", 3)


def test_a_fill_from_a_module_level_bound_with_a_carried_state():
    _fd_check(named_bound, {"x": np.linspace(0.1, 0.6, DAYS), "g": 0.8}, ("x", "g"), "curve", DAYS)


def test_a_nested_index_fill():
    _fd_check(nested_index, {"x": np.linspace(-0.3, 0.5, 6), "g": 1.1}, ("x", "g"), "out", 6)


# -- what keeps the launch-start value, and so stays refused ------------------------

def _refused(kernel):
    with pytest.raises(HawkError, match="read before it is assigned"):
        vjp(kernel, wrt=("x",))
    with pytest.raises(HawkError, match="read before it is assigned"):
        jvp(kernel, wrt=("x",))


def test_partial_cover_stays_refused():
    @hawk.kernel
    def partial(x: V3, g: hawk.Scalar, out: OUT3):
        for d in range(2):
            out[d] = x[d] * g
    _refused(partial)


def test_a_read_of_an_element_not_yet_stored_stays_refused():
    @hawk.kernel
    def ahead(x: V3, g: hawk.Scalar, out: OUT3):
        for d in range(2):
            out[d] = x[d] * g + out[d + 1]
        out[2] = x[2]
    _refused(ahead)

    @hawk.kernel
    def whole_read_early(x: V3, g: hawk.Scalar, out: OUT3):
        out[0] = x[0]
        s = out[1] + out[2]
        out[1] = s
        out[2] = s
    _refused(whole_read_early)


def test_a_store_under_a_one_sided_if_stays_refused():
    @hawk.kernel
    def one_sided(x: V3, g: hawk.Scalar, out: OUT3):
        for d in range(3):
            if x[d] > 0.0:
                out[d] = x[d] * g
    _refused(one_sided)


def test_an_index_the_trace_cannot_compute_stays_refused():
    @hawk.kernel
    def runtime_index(x: V3, g: hawk.Scalar, out: OUT3):
        out[m.index(g, 3)] = x[0]
        out[1] = x[1]
        out[2] = x[2]
    _refused(runtime_index)


def test_a_cover_is_decided_by_the_plane_width_not_the_loop_bound():
    @hawk.kernel
    def wider(x: hawk.Vector[4], g: hawk.Scalar, out: hawk.Mutable[hawk.Vector[4]]):
        for d in range(3):
            out[d] = x[d] * g
    with pytest.raises(HawkError, match="read before it is assigned"):
        vjp(wider, wrt=("x",))


# -- adversarial rows: a rebound loop variable, a re-used loop name -------------------

def test_a_loop_variable_assigned_in_the_body_is_not_a_cover():
    """``for i in range(2): i = 0; out[i] = ...`` stores ``out[0]`` twice and leaves
    ``out[1]`` at its launch-start value: not a cover (it used to start from zeros)."""
    @hawk.kernel
    def reassigned(x: V3, out: hawk.Mutable[hawk.Vector[2]]):
        for i in range(2):
            i = 0
            out[i] = x[1]
    with pytest.raises(HawkError, match="read before it is assigned"):
        vjp(reassigned, wrt=("x",))

    @hawk.kernel
    def scaled(x: V3, out: hawk.Mutable[hawk.Vector[2]]):
        for i in range(2):
            i *= 0
            out[i] = x[1]
    with pytest.raises(HawkError, match="read before it is assigned"):
        vjp(scaled, wrt=("x",))


def test_an_inner_loop_reusing_the_outer_name_still_defines_and_is_not_a_cover():
    @hawk.kernel
    def reused(x: V3, out: OUT3):
        for i in range(3):
            for i in range(3):
                out[i] = x[i]
    assert reused.sinks                     # it traces (a SyntaxError before this fix)
    with pytest.raises(HawkError, match="read before it is assigned"):
        vjp(reused, wrt=("x",))
