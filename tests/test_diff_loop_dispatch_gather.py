# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A plane gathered with ``at()`` inside a ``dispatch`` branch of a loop body,
differentiated in reverse: the branches' contributions merge into ONE scatter
(masked value, taken-branch row), checked against central finite differences,
with an untaken branch's row never committed out of range."""

from __future__ import annotations

import numpy as np
import pytest
from _eval import evaluate

import hawk
import hawk.math as m
from hawk.diff import vjp
from hawk.emit import render_body
from hawk.ir import HawkError, canonical

N = 5


@hawk.kernel
def switch_gather(x: hawk.Scalar, kind: hawk.Scalar,
                  tab: hawk.Table["row":N, "col":2], out: hawk.Mutable[hawk.Scalar]):
    s = x
    for k in range(N):
        cases = [s * tab.at(row=k, col=0), tab.at(row=k, col=1) * m.sin(s), s + 0.5]
        s = m.dispatch(m.int32(kind), cases, policy="switch") * 0.9 + 0.1
    out = s


def _primal(x, kind, tab):
    s = x
    for k in range(N):
        c = [s * tab[2 * k], tab[2 * k + 1] * np.sin(s), s + 0.5]
        s = c[int(kind)] * 0.9 + 0.1
    return s


def _fd(f, v, h=1e-6):
    return np.array([(f(v + h * e) - f(v - h * e)) / (2 * h) for e in np.eye(len(v))])


@pytest.mark.parametrize("kind", [0, 1, 2])
def test_the_gradient_matches_finite_differences(kind):
    tab = np.random.default_rng(1).uniform(0.5, 1.2, 2 * N)
    got = evaluate(vjp(switch_gather, wrt=("x", "tab")),
                   {"x": 0.4, "kind": float(kind), "tab": tab, "terminated": False,
                    "bar_out": 1.0}, lanes=2 * N)
    want_x = _fd(lambda v: _primal(v[0], kind, tab), np.array([0.4]))[0]
    want_t = _fd(lambda v: _primal(0.4, kind, v), tab)
    assert abs(got["bar_x"] - want_x) < 1e-8
    np.testing.assert_allclose(got["bar_tab"], want_t, atol=1e-8)
    if kind == 2:                         # the branch with no gather
        assert not np.any(got["bar_tab"])


@hawk.kernel
def far_row(x: hawk.Scalar, kind: hawk.Scalar,
            tab: hawk.Table["row":N], out: hawk.Mutable[hawk.Scalar]):
    s = x
    for k in range(2):
        # branch 0 reads a row only valid when x is small; branch 1 never reads tab
        cases = [s * tab.at(m.int32(x * 100.0 * (1.0 + 0.0 * k))), s * 2.0]
        s = m.dispatch(m.int32(kind), cases, policy="switch") + 0.1
    out = s


def test_an_untaken_branchs_row_is_never_committed_out_of_range():
    """x = 0.6 makes branch 0's row 60 (a row that depends on the loop index, so it stays inside the branch) (the plane has 5): with kind = 1 the lane
    takes branch 1, so the scatter must not write at row 60 (the evaluator raises
    IndexError on any out-of-range commit)."""
    d = vjp(far_row, wrt=("x", "tab"))
    tab = np.arange(1.0, N + 1)
    taken = evaluate(d, {"x": 0.6, "kind": 1.0, "tab": tab, "terminated": False,
                         "bar_out": 1.0}, lanes=N)
    assert not np.any(taken["bar_tab"])
    ok = evaluate(d, {"x": 0.04, "kind": 0.0, "tab": tab, "terminated": False,
                      "bar_out": 1.0}, lanes=N)
    assert ok["bar_tab"][4] != 0.0 and not np.any(ok["bar_tab"][:4])
    text = render_body(tuple(d), canonical(tuple(d))).text
    assert "bar_tab" in text


def test_a_dispatch_inside_a_dispatch_branch_with_a_gather_is_refused_by_name():
    @hawk.kernel
    def nested(x: hawk.Scalar, kind: hawk.Scalar,
               tab: hawk.Table["row":N], out: hawk.Mutable[hawk.Scalar]):
        s = x
        for k in range(2):
            inner = m.dispatch(m.int32(kind), [tab.at(row=k), s], policy="switch")
            s = m.dispatch(m.int32(kind), [inner * s, s + 1.0], policy="switch")
        out = s

    with pytest.raises(HawkError, match="inside a `dispatch` inside a `dispatch` branch|dispatch"):
        vjp(nested, wrt=("x", "tab"))


# -- every gather guarded by a branch or a condition stays inside it -------------------

@hawk.kernel
def straight_far(x: hawk.Scalar, kind: hawk.Scalar, tab: hawk.Table["row":N],
                 out: hawk.Mutable[hawk.Scalar]):
    cases = [x * tab.at(m.int32(x * 100.0)), x * 2.0]
    out = m.dispatch(m.int32(kind), cases, policy="switch")


@hawk.kernel
def where_raw(x: hawk.Scalar, tab: hawk.Table["row":N], out: hawk.Mutable[hawk.Scalar]):
    i = m.int32(x * 100.0)
    out = m.where(i < N, tab.at(i) * x, x * 2.0)


@hawk.kernel
def invariant_far(x: hawk.Scalar, kind: hawk.Scalar, tab: hawk.Table["row":N],
                  out: hawk.Mutable[hawk.Scalar]):
    s = x
    for _k in range(2):
        cases = [s * tab.at(m.int32(x * 100.0)), s * 2.0]
        s = m.dispatch(m.int32(kind), cases, policy="switch") + 0.1
    out = s


TAB = np.arange(1.0, N + 1)


def test_a_straight_line_dispatch_never_scatters_out_of_range():
    d = vjp(straight_far, wrt=("x", "tab"))
    far = evaluate(d, {"x": 0.6, "kind": 1.0, "tab": TAB, "terminated": False, "bar_out": 1.0}, lanes=N)
    assert not np.any(far["bar_tab"]) and abs(far["bar_x"] - 2.0) < 1e-12
    near = evaluate(d, {"x": 0.04, "kind": 0.0, "tab": TAB, "terminated": False, "bar_out": 1.0}, lanes=N)
    assert near["bar_tab"][4] == pytest.approx(0.04) and not np.any(near["bar_tab"][:4])


def test_the_reverse_of_a_where_reads_and_scatters_only_where_its_condition_holds():
    d = vjp(where_raw, wrt=("x", "tab"))
    far = evaluate(d, {"x": 0.6, "tab": TAB, "terminated": False, "bar_out": 1.0}, lanes=N)
    assert not np.any(far["bar_tab"]) and abs(far["bar_x"] - 2.0) < 1e-12     # the else arm
    near = evaluate(d, {"x": 0.04, "tab": TAB, "terminated": False, "bar_out": 1.0}, lanes=N)
    assert near["bar_x"] == pytest.approx(TAB[4]) and near["bar_tab"][4] == pytest.approx(0.04)
    assert not np.any(near["bar_tab"][:4])


def test_a_loop_invariant_gather_inside_a_branch_is_not_hoisted_out_of_it():
    """Primal and reverse, kind 1 with a row (60) that only branch 0 may read."""
    env = {"x": 0.6, "kind": 1.0, "tab": TAB, "terminated": False}
    assert float(evaluate(invariant_far.sinks, env, lanes=N)["out"]) == pytest.approx(
        ((0.6 * 2 + 0.1) * 2 + 0.1))
    r = evaluate(vjp(invariant_far, wrt=("x", "tab")), {**env, "bar_out": 1.0}, lanes=N)
    assert not np.any(r["bar_tab"]) and r["bar_x"] == pytest.approx(4.0)
    # branch 0: the hoisted-in gather still differentiates
    near = {"x": 0.04, "kind": 0.0, "tab": TAB, "terminated": False}
    r0 = evaluate(vjp(invariant_far, wrt=("tab",)), {**near, "bar_out": 1.0}, lanes=N)
    assert r0["bar_tab"][4] != 0.0 and not np.any(r0["bar_tab"][:4])
