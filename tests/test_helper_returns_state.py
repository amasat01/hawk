# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A helper that returns the new state, ``x, v = step(x, v, dt)``, works for
Mutable and Terminated state: the kernel it builds emits the same body, and
the same digest, as the one with the update written inline."""

from __future__ import annotations

import numpy as np

import hawk
from hawk import Mutable, Param, Scalar, Terminated
from hawk.emit import render_body


def step(x, v, dt):
    """Semi-implicit Euler for ``x'' = -x``, returning the new state."""
    v2 = v - dt * x
    return x + dt * v2, v2


@hawk.kernel
def by_helper(dt: Param, t_final: Scalar, terminated: Terminated,
              x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    x, v = step(x, v, dt)
    tn = t + dt
    t = tn
    terminated = tn >= t_final


@hawk.kernel
def inline(dt: Param, t_final: Scalar, terminated: Terminated,
           x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    v2 = v - dt * x
    xn = x + dt * v2
    x = xn
    v = v2
    tn = t + dt
    t = tn
    terminated = tn >= t_final


@hawk.kernel
def by_helper_mutable(dt: Param, x: Mutable[Scalar], v: Mutable[Scalar]):
    x, v = step(x, v, dt)


def test_the_tuple_returning_helper_emits_the_inline_body():
    a = render_body(by_helper.sinks, by_helper.walk).text
    b = render_body(inline.sinks, inline.walk).text
    assert a == b


def test_the_tuple_returning_helper_runs_on_mutable_state():
    x = np.array([1.0, 0.5, -2.0])
    v = np.zeros(3)
    out = by_helper_mutable(0.1, x.copy(), v.copy())
    v2 = v - 0.1 * x
    np.testing.assert_array_equal(np.asarray(out["x"]), x + 0.1 * v2)
    np.testing.assert_array_equal(np.asarray(out["v"]), v2)


def early_return(x):
    if x > 1.0:
        return x * 2.0
    return x


def test_a_helper_the_rewrite_refuses_is_named_and_the_fix_given():
    import pytest

    with pytest.raises(hawk.HawkError, match=r"helper 'early_return'.*inlined like a kernel body.*m\.where"):
        @hawk.kernel
        def uses(x: Scalar, y: Mutable[Scalar]):
            y = early_return(x)
