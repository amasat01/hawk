# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""If/elif chains inside a loop and in scalar code: hoisting, helper calls and costly arms.

* A loop-invariant value hoisted above a ``for`` and read in ONE arm of a vector-valued chain that
  lowers to a switch inside the loop is rendered where the arm reads it.
* A branch condition may call a plain function that itself branches on traced values.
* A scalar chain whose arms each read a transcendental evaluates each arm's once, under its arm.
Each row compares with a numpy reference, on the host and on GPU 1.
"""

from __future__ import annotations

import numpy as np
import pytest

import hawk
import hawk.math as m

LANES, STEPS = 12, 4
RNG = np.random.default_rng(17)


def _hoisted_arm():
    @hawk.kernel
    def chain(lane: hawk.Index, taus: hawk.Table[hawk.Scalar], kind: hawk.Table[hawk.Scalar],
              a: hawk.Table[hawk.Scalar], out: hawk.Mutable[hawk.Vector[3]]):
        st = m.vec(a.at(lane), 0.0, 0.0)
        for j in range(2):
            tau = taus.at(lane * 2 + j)
            for i in range(STEPS):
                which = m.int32(kind.at(lane * STEPS + i) + 0.5)
                if which == 0:
                    st = m.vec(st[0] + m.exp(tau * st[0]), st[1], st[2])
                elif which == 1:
                    st = m.vec(st[0] * 0.5, st[1] + m.sin(st[0]), st[2])
                else:
                    st = m.vec(st[0], st[1], st[2] + m.cos(st[0]))
        out = st
    return chain


def _planes():
    return {"lane": np.arange(LANES, dtype=np.int32), "taus": RNG.random(LANES * 2) * 0.1,
            "kind": RNG.integers(0, 3, LANES * STEPS).astype(float), "a": RNG.random(LANES) + 1.0}


def _hoisted_want(p):
    out = np.zeros((3, LANES))
    for lane in range(LANES):
        s = [p["a"][lane], 0.0, 0.0]
        for j in range(2):
            tau = p["taus"][lane * 2 + j]
            for i in range(STEPS):
                w = int(p["kind"][lane * STEPS + i])
                s = ([s[0] + np.exp(tau * s[0]), s[1], s[2]] if w == 0 else
                     [s[0] * 0.5, s[1] + np.sin(s[0]), s[2]] if w == 1 else [s[0], s[1], s[2] + np.cos(s[0])])
        out[:, lane] = s
    return out


def _run(kernel, planes, out_shape, cache_dir, target):
    import eagle.plan
    sink = np.zeros(out_shape)
    args = {**{k: v for k, v in planes.items() if k in {n for _, n in kernel.arg_spec}}, "out": sink}
    if target == "cuda":
        import cupy as cp

        dev = {k: cp.asarray(v) for k, v in args.items()}
        eagle.plan.auto([kernel], targets=("cuda",), cache_dir=cache_dir, scalar_type="float64")[0].run(**dev)
        return dev["out"].get()
    eagle.plan.auto([kernel], targets=("host",), cache_dir=cache_dir, scalar_type="float64")[0].run(**args)
    return sink


TARGETS = [pytest.param("host", id="host"), pytest.param("cuda", marks=pytest.mark.gpu, id="cuda")]


@pytest.mark.parametrize("target", TARGETS)
def test_a_hoisted_value_read_in_one_arm_of_a_loop_switch_builds_and_agrees(cache_dir, target):
    pytest.importorskip("eagle")
    p = _planes()
    got = _run(_hoisted_arm(), p, (3, LANES), cache_dir, target)
    np.testing.assert_allclose(got, _hoisted_want(p), rtol=1e-12)


def _band(x):
    """A plain function that branches on a traced value (rewritten at the call, not decorated)."""
    r = 0.0
    if x > 1.0:
        r = 2.0
    elif x > 0.0:
        r = 1.0
    return r


def _outside(x):
    """A helper whose RESULT is the traced bool the caller branches on, itself chosen by a branch."""
    r = x > 1.0
    if x < -1.0:
        r = x < -1.5
    return r


def _helper_condition():
    @hawk.kernel
    def pick(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        if _band(x) > 1.5:
            y = m.exp(x)
        elif _band(x) > 0.5:
            y = m.sin(x)
        else:
            y = m.cos(x)
        if _outside(x):
            y = y + 10.0
        out = y
    return pick


@pytest.mark.parametrize("target", TARGETS)
def test_a_branch_condition_calling_a_plain_helper_that_branches_builds_and_agrees(cache_dir, target):
    pytest.importorskip("eagle")
    x = np.linspace(-2.0, 2.5, 41)
    got = _run(_helper_condition(), {"x": x}, (41,), cache_dir, target)
    want = np.where(x > 1.0, np.exp(x), np.where(x > 0.0, np.sin(x), np.cos(x)))
    want = want + 10.0 * np.where(x < -1.0, x < -1.5, x > 1.0)
    np.testing.assert_allclose(got, want, rtol=1e-12)


def _two_names():
    @hawk.kernel
    def two(x: hawk.Scalar, out: hawk.Mutable[hawk.Vector[2]]):
        if x > 1.0:
            a = m.exp(x)
            b = a * 2.0
        elif x > 0.0:
            a = m.sin(x)
            b = a + 1.0
        else:
            a = m.cos(x)
            b = 0.5 * a
        out = m.vec(a, b)
    return two


def test_a_scalar_chain_evaluates_each_arms_transcendental_only_under_its_arm():
    """Both names read each arm's transcendental, which a ``?:`` would name ahead of the chain and
    evaluate on every path: every call sits in a case of a switch."""
    from hawk.emit import render_body
    from hawk.ir import canonical

    kernel = _two_names()
    text = render_body(kernel.sinks, canonical(kernel.sinks)).text
    assert text.count("switch (") == 2
    outside = text.split("switch (")[0]
    assert not any(f"math::{fn}(" in outside for fn in ("exp", "sin", "cos"))
    assert all(text.count(f"math::{fn}(") == 2 for fn in ("exp", "sin", "cos"))    # once per name, in its case


@pytest.mark.parametrize("target", TARGETS)
def test_a_scalar_chain_over_two_names_keeps_its_values(cache_dir, target):
    pytest.importorskip("eagle")
    x = np.linspace(-2.0, 2.5, 41)
    got = _run(_two_names(), {"x": x}, (2, 41), cache_dir, target)
    a = np.where(x > 1.0, np.exp(x), np.where(x > 0.0, np.sin(x), np.cos(x)))
    b = np.where(x > 1.0, 2.0 * np.exp(x), np.where(x > 0.0, np.sin(x) + 1.0, 0.5 * np.cos(x)))
    np.testing.assert_allclose(got, np.stack([a, b]), rtol=1e-12)

