# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""What a derivative does EXACTLY AT a branch point, pinned.

A derived (``vjp``/``jvp``) kernel is piecewise: away from a switch it is the
slope of whichever branch is taken, and at the switch itself a CONVENTION
decides. These rows pin each convention on the compiled host route, at the tie
and a step to either side, with exact expected values:

* ``minimum``/``maximum``: a tie goes to the FIRST operand (``le(lo, hi)``).
* ``abs``: ``+1`` at ``x >= 0`` (zero included).
* ``clip``: ``x`` passes on the CLOSED interval; ``lo > hi`` selects ``hi``;
  a NaN ``x`` passes nothing.
* ``copysign``: ``abs``'s rule for the first operand, nothing to the second.
* a Python ``if x < y:``: the traced condition becomes a select, so a tie
  takes the ``else`` branch (the condition is false at ``x == y``).
* ``floor``/``round``/``sign``/comparisons: exactly zero, everywhere.

The termination mask (a stopped sample contributes exactly zero) is pinned
here with one row; the fuller set -- scattered sums, device route, finishing
primals -- is ``test_diff_terminated.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import Kernel, Mutable, Scalar, Terminated, kernel
from hawk.diff import jvp, vjp
from hawk.math import clip, copysign, floor, maximum, minimum, where
from hawk.math import round as hround
from hawk.math import sign


# --------------------------------------------------------------------------- #
# -- primals: one tiny kernel per rule
# --------------------------------------------------------------------------- #
@kernel
def k_min(a: Scalar, b: Scalar, out: Mutable[Scalar]):
    out = minimum(a, b)


@kernel
def k_max(a: Scalar, b: Scalar, out: Mutable[Scalar]):
    out = maximum(a, b)


@kernel
def k_abs(x: Scalar, out: Mutable[Scalar]):
    out = abs(x)


@kernel
def k_clip(x: Scalar, lo: Scalar, hi: Scalar, out: Mutable[Scalar]):
    out = clip(x, lo, hi)


@kernel
def k_copysign(a: Scalar, b: Scalar, out: Mutable[Scalar]):
    out = copysign(a, b)


@kernel
def k_if(x: Scalar, y: Scalar, out: Mutable[Scalar]):
    if x < y:
        out = 3.0 * x
    else:
        out = 5.0 * y


@kernel
def k_floor(x: Scalar, out: Mutable[Scalar]):
    out = floor(x)


@kernel
def k_round(x: Scalar, out: Mutable[Scalar]):
    out = hround(x)


@kernel
def k_sign(x: Scalar, out: Mutable[Scalar]):
    out = sign(x)


@kernel
def k_compare(x: Scalar, y: Scalar, out: Mutable[Scalar]):
    out = where(x > y, 1.0, 0.0)


@kernel
def k_stop(x: Scalar, terminated: Terminated, out: Mutable[Scalar]):
    out = x * x


# --------------------------------------------------------------------------- #
# -- host-route helpers: build primal + vjp + jvp once per kernel
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def derive(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("tie_conventions")
    built: dict = {}

    def get(primal, wrt):
        key = (primal.name, wrt)
        if key not in built:
            d = root / f"{primal.name}_{'_'.join(wrt)}"
            d.mkdir()
            hawk.build([primal,
                        Kernel("g_vjp", vjp(primal, wrt=wrt)),
                        Kernel("g_jvp", jvp(primal, wrt=wrt))],
                       d, targets=("host",), cache_dir=cache_dir)
            built[key] = d
        return built[key]

    def grad(primal, wrt, seed=1.0, **inputs):
        """Reverse: ``{wrt name: d(out)/d(name)}`` per sample, seed 1."""
        d = get(primal, wrt)
        arrs = {k: np.asarray(v, dtype=bool if k == "terminated" else float)
                for k, v in inputs.items()}
        n = len(next(iter(arrs.values())))
        bars = {f"bar_{w}": np.zeros(n) for w in wrt}
        hawk.run(hawk.load(d, "g_vjp"), bar_out=np.full(n, seed),
                 **bars, **arrs)
        return {w: bars[f"bar_{w}"] for w in wrt}

    def tangent(primal, wrt, dots, **inputs):
        """Forward: ``d(out)`` per sample for tangents ``dots`` (name -> value)."""
        d = get(primal, wrt)
        arrs = {k: np.asarray(v, dtype=float) for k, v in inputs.items()}
        n = len(next(iter(arrs.values())))
        tans = {f"dot_{w}": np.full(n, float(dots[w])) for w in wrt}
        dot_out = np.zeros(n)
        hawk.run(hawk.load(d, "g_jvp"), dot_out=dot_out, **tans, **arrs)
        return dot_out

    return type("Rig", (), {"grad": staticmethod(grad), "tangent": staticmethod(tangent)})


def _eq(actual, expected):
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=float))


# --------------------------------------------------------------------------- #
# -- min / max: the FIRST operand takes a tie
# --------------------------------------------------------------------------- #
# samples: a < b, a == b (the tie), a > b
A = [0.0, 1.0, 2.0]
B = [1.0, 1.0, 1.0]


def test_min_tie_goes_to_the_first_operand():
    g = _RIG.grad(k_min, ("a", "b"), a=A, b=B)
    _eq(g["a"], [1.0, 1.0, 0.0])
    _eq(g["b"], [0.0, 0.0, 1.0])


def test_max_tie_goes_to_the_first_operand():
    g = _RIG.grad(k_max, ("a", "b"), a=A, b=B)
    _eq(g["a"], [0.0, 1.0, 1.0])
    _eq(g["b"], [1.0, 0.0, 0.0])


def test_min_and_max_jvp_follow_the_same_tie_rule():
    # tangents 2 on a and 7 on b: the tie carries a's tangent, never b's
    _eq(_RIG.tangent(k_min, ("a", "b"), {"a": 2.0, "b": 7.0}, a=A, b=B), [2.0, 2.0, 7.0])
    _eq(_RIG.tangent(k_max, ("a", "b"), {"a": 2.0, "b": 7.0}, a=A, b=B), [7.0, 2.0, 2.0])


# --------------------------------------------------------------------------- #
# -- abs: +1 at zero
# --------------------------------------------------------------------------- #
def test_abs_is_plus_one_at_zero_and_signed_either_side():
    x = [-1.0, -0.0, 0.0, 1.0]
    _eq(_RIG.grad(k_abs, ("x",), x=x)["x"], [-1.0, 1.0, 1.0, 1.0])
    _eq(_RIG.tangent(k_abs, ("x",), {"x": 3.0}, x=x), [-3.0, 3.0, 3.0, 3.0])


# --------------------------------------------------------------------------- #
# -- clip: closed interval passes x; lo > hi -> hi; NaN passes nothing
# --------------------------------------------------------------------------- #
#            below  at lo  inside  at hi  above  lo>hi   NaN x
CX = [0.0, 1.0, 2.0, 3.0, 4.0, 0.5, np.nan]
CLO = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0]
CHI = [3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 3.0]


def test_clip_vjp_at_lo_at_hi_inside_outside_inverted_and_nan():
    g = _RIG.grad(k_clip, ("x", "lo", "hi"), x=CX, lo=CLO, hi=CHI)
    _eq(g["x"], [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    _eq(g["lo"], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _eq(g["hi"], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0])


def test_clip_jvp_follows_the_same_masks():
    t = _RIG.tangent(k_clip, ("x", "lo", "hi"), {"x": 2.0, "lo": 5.0, "hi": 11.0},
                     x=CX, lo=CLO, hi=CHI)
    _eq(t, [5.0, 2.0, 2.0, 2.0, 11.0, 11.0, 0.0])


# --------------------------------------------------------------------------- #
# -- copysign: abs's rule on the first operand, nothing to the second
# --------------------------------------------------------------------------- #
def test_copysign_at_zero_uses_abs_rule_and_gives_the_sign_operand_nothing():
    a = [-1.0, 0.0, 0.0, 1.0]
    b = [2.0, 2.0, -2.0, -2.0]
    g = _RIG.grad(k_copysign, ("a", "b"), a=a, b=b)
    _eq(g["a"], [-1.0, 1.0, -1.0, -1.0])
    _eq(g["b"], [0.0, 0.0, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# -- Python `if x < y:` becomes a select; the tie takes the else branch
# --------------------------------------------------------------------------- #
def test_python_if_at_the_tie_takes_the_else_branch():
    x = [0.0, 1.0, 2.0]
    y = [1.0, 1.0, 1.0]
    g = _RIG.grad(k_if, ("x", "y"), x=x, y=y)
    # x < y: out = 3x. Tie (x < y false): out = 5y. Above: out = 5y.
    _eq(g["x"], [3.0, 0.0, 0.0])
    _eq(g["y"], [0.0, 5.0, 5.0])
    _eq(_RIG.tangent(k_if, ("x", "y"), {"x": 2.0, "y": 7.0}, x=x, y=y), [6.0, 35.0, 35.0])


# --------------------------------------------------------------------------- #
# -- the zero rules: piecewise-constant ops have exactly zero slope
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("primal", [k_floor, k_round, k_sign])
def test_step_functions_have_exactly_zero_gradient(primal):
    x = [-1.5, -0.5, 0.0, 0.5, 1.0, 2.5]
    _eq(_RIG.grad(primal, ("x",), x=x)["x"], np.zeros(len(x)))
    _eq(_RIG.tangent(primal, ("x",), {"x": 4.0}, x=x), np.zeros(len(x)))


def test_comparison_selected_constants_have_exactly_zero_gradient():
    g = _RIG.grad(k_compare, ("x", "y"), x=A, y=B)
    _eq(g["x"], [0.0, 0.0, 0.0])
    _eq(g["y"], [0.0, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# -- termination: a stopped sample contributes exactly 0, its neighbour not
# --------------------------------------------------------------------------- #
def test_terminated_sample_contributes_exactly_zero_while_its_neighbour_does_not():
    # fuller coverage (scatters, device route, finishing primals): test_diff_terminated.py
    g = _RIG.grad(k_stop, ("x",), seed=2.0, x=[3.0, 3.0], terminated=[False, True])
    _eq(g["x"], [12.0, 0.0])      # live: 2 * d(x*x)/dx = 12; stopped: 0


# the fixture is a function of pytest state; bind it once for the module so the
# rows above read as plain calls
@pytest.fixture(autouse=True, scope="module")
def _bind(derive):
    global _RIG
    _RIG = derive
    yield


_RIG = None
