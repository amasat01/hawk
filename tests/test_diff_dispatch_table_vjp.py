# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The VJP of a ``Table[Scalar]`` read reached from inside a ``dispatch``'s
branches must not be double-counted, including a lane read by only one
branch and a read used as an operand of further arithmetic inside the
branch (not just as a branch's own value, which
``tests/test_dispatch.py``'s rows already cover)."""

from __future__ import annotations

import numpy as np
import pytest
from _eval import evaluate

from hawk import _ir
from hawk.diff import vjp

S = _ir.TensorType((), "f64")
I32 = _ir.TensorType((), "i32")
_H = 1e-5


def _leaf(context, name: str, t, role: str):
    return _ir.leaf(context, "vocab_read", role, name, t)


def _assign(name: str, value):
    return _ir.make_node("Assign", S, [value], name=name)


def _gather(plane, index):
    return _ir.make_node("At", S, [plane, index])


def _table_dispatch_primal():
    """K=2 ``dispatch``: branch 0 reads ``x0`` only; branch 1 reads ``x1``
    AND ``tab.at(idx)`` as an OPERAND of its own arithmetic (``x1 *
    tab[idx]``)."""
    context = _ir.new_context()
    k = _leaf(context, "k", I32, "plane_in")
    x0 = _leaf(context, "x0", S, "plane_in")
    x1 = _leaf(context, "x1", S, "plane_in")
    tab = _ir.leaf(context, "table_read", "lookup", "tab", S)
    idx = _leaf(context, "idx", I32, "plane_in")
    branch0 = _ir.make("mul", (x0, x0))
    branch1 = _ir.make("mul", (x1, _gather(tab, idx)))
    d = _ir.make_node("Dispatch", S, [k, branch0, branch1])
    return (_assign("out", d),)


def test_table_read_vjp_inside_dispatch_matches_fd_not_doubled():
    """``bar_tab[idx]`` must match central FD on the ONE branch that reads
    the table; the defect returns exactly 2.0x that value. The off-branch
    leaf gradient (``bar_x0``) stays exactly zero either way (the
    off-branch-zero half is not what this row is pinned on)."""
    sinks = _table_dispatch_primal()
    derived = vjp(sinks)
    rng = np.random.default_rng(20260911)
    x0_val = float(rng.uniform(0.5, 2.0))
    x1_val = float(rng.uniform(0.5, 2.0))
    tab_val = float(rng.uniform(0.5, 2.0))
    idx_val = 0
    seed = float(rng.normal())
    tab_arr = np.array([tab_val, tab_val + 3.0])

    env = {
        "k": 1,
        "x0": x0_val,
        "x1": x1_val,
        "tab": tab_arr,
        "idx": idx_val,
        "bar_out": seed,
    }
    got = evaluate(derived, env)
    analytic = float(np.asarray(got["bar_tab"])[idx_val])

    plus = dict(env)
    plus["tab"] = tab_arr.copy()
    plus["tab"][idx_val] += _H
    minus = dict(env)
    minus["tab"] = tab_arr.copy()
    minus["tab"][idx_val] -= _H
    fplus = evaluate(sinks, plus)["out"]
    fminus = evaluate(sinks, minus)["out"]
    fd = float((fplus - fminus) / (2 * _H) * seed)

    ratio = analytic / fd
    assert analytic == pytest.approx(fd, rel=1e-5, abs=1e-7), (
        f"dispatch+Table[Scalar] VJP: bar_tab[idx]={analytic} vs central FD "
        f"{fd} (ratio {ratio:.6f} -- the HAWK GAP is exactly 2.0x)"
    )
    assert float(got["bar_x0"]) == 0.0, (
        f"off-branch leaf gradient bar_x0 must stay exactly zero, got {got['bar_x0']}"
    )


def test_table_read_vjp_outside_dispatch_is_exact_control():
    """Control: the SAME ``x1 * tab.at(idx)`` construct, OUTSIDE any
    dispatch, is exact (ratio 1.0) -- isolates the defect to dispatch's own
    reverse rule, not ``At``'s ordinary (non-local-pass) scatter rule."""
    context = _ir.new_context()
    x1 = _leaf(context, "x1", S, "plane_in")
    tab = _ir.leaf(context, "table_read", "lookup", "tab", S)
    idx = _leaf(context, "idx", I32, "plane_in")
    y = _ir.make("mul", (x1, _gather(tab, idx)))
    sinks = (_assign("out", y),)
    derived = vjp(sinks)

    x1_val, tab_val, idx_val, seed = 1.7, 0.9, 0, -0.6
    tab_arr = np.array([tab_val, tab_val + 3.0])
    env = {"x1": x1_val, "tab": tab_arr, "idx": idx_val, "bar_out": seed}
    got = evaluate(derived, env)
    analytic = float(np.asarray(got["bar_tab"])[idx_val])
    expected = x1_val * seed
    assert analytic == pytest.approx(expected, rel=1e-9), (
        f"table read OUTSIDE dispatch: bar_tab[idx]={analytic} vs exact "
        f"{expected} (ratio {analytic / expected:.6f})"
    )
