# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``vjp(..., strategy={group: ...})``, ``"auto"`` and the selector classification.

Rows:

* **override** — a derived dispatch inherits the primal's strategy unless
  ``strategy=`` names its group; overridden, every derived dispatch carries the
  new strategy, and the gradient equals central differences EITHER way (and the
  two gradients are equal); an unknown group or strategy refuses by name.
* **auto** — accepted by the node, a distinct digest, rendered as ``switch``
  plus one branch-mix count line until a chooser resolves it.
* **static_selector** — a selector reading only inputs is static; one reading
  a plane the kernel writes (its own commit, or its finished mask) is not; the
  STEP-scoped verdict turns a kernel's static group state-dependent when
  another kernel of the step writes the selector's plane; the sidecar record
  and a segmented unit's record carry the classification.
"""

from __future__ import annotations

import numpy as np
import pytest
from _eval import evaluate

import hawk.math as M
from hawk import Index, Mutable, Param, Scalar, Terminated, kernel
from hawk.diff import vjp
from hawk import HawkError
from hawk import _ir
from hawk.trace.value import dispatch_node, dispatch_strategy

S = _ir.TensorType((), "f64")
I32 = _ir.TensorType((), "i32")
_H = 1e-6


def _primal(strategy: str, K: int = 3):
    context = _ir.new_context()
    k = _ir.leaf(context, "vocab_read", "plane_in", "k", I32)
    xs = [_ir.leaf(context, "vocab_read", "plane_in", f"x{i}", S) for i in range(K)]
    branches = [
        _ir.make("mul", (xs[i], _ir.make("sin", (xs[(i + 1) % K],)))) for i in range(K)
    ]
    return (
        _ir.make_node("Assign", S, [dispatch_node(k, branches, strategy, S)], name="out"),
    )


def _strategies(sinks) -> set:
    order = _ir.canonical_nodes(tuple(sinks))[0]
    return {dispatch_strategy(n) for n in order if _ir.is_a(n, "dispatch")}


@pytest.mark.parametrize(
    "primal_strategy,override",
    [
        ("switch", "predicated"),
        ("predicated", "switch"),
        ("segmented", "switch"),
        ("switch", "segmented"),
        ("auto", "predicated"),
    ],
)
def test_an_overridden_reverse_strategy_differs_and_both_match_fd(
    primal_strategy, override
):
    sinks = _primal(primal_strategy)
    inherited = vjp(sinks)
    overridden = vjp(sinks, strategy={"dispatch0": override})
    assert _strategies(inherited) == {primal_strategy}
    assert _strategies(overridden) == {override}
    rng = np.random.default_rng(31)
    for kind in range(3):
        env = {"k": kind, **{f"x{i}": float(rng.uniform(0.5, 2.0)) for i in range(3)}}
        seed = float(rng.normal())
        a = evaluate(inherited, {**env, "bar_out": seed})
        b = evaluate(overridden, {**env, "bar_out": seed})
        for i in range(3):
            assert a[f"bar_x{i}"] == b[f"bar_x{i}"]
            plus, minus = dict(env), dict(env)
            plus[f"x{i}"] += _H
            minus[f"x{i}"] -= _H
            fd = (evaluate(sinks, plus)["out"] - evaluate(sinks, minus)["out"]) / (
                2 * _H
            )
            assert b[f"bar_x{i}"] == pytest.approx(fd * seed, rel=1e-6, abs=1e-8)


def test_an_unknown_group_or_strategy_refuses_by_name():
    sinks = _primal("switch")
    with pytest.raises(HawkError, match=r"'dispatch7' is not a dispatch group"):
        vjp(sinks, strategy={"dispatch7": "switch"})
    with pytest.raises(HawkError, match=r"unknown strategy 'fused'"):
        vjp(sinks, strategy={"dispatch0": "fused"})


# -- the selector classification --------------------------------------------- #
@kernel(steps=1)
def by_input(dt: Param, kind: Index, x: Mutable[Scalar]):
    x = x + dt * M.dispatch(kind, [x, -x], strategy="auto")


@kernel(steps=1)
def by_state(dt: Param, kind: Index, x: Mutable[Scalar], c: Mutable[Index]):
    x = x + dt * M.dispatch(c, [x, -x], strategy="auto")
    c = M.select(x > 0.0, kind, kind + 1)


@kernel(steps=1)
def by_mask(dt: Param, kind: Index, terminated: Terminated, x: Mutable[Scalar]):
    sel = M.select(terminated, kind, kind + 1)
    x = x + dt * M.dispatch(sel, [x, -x])
    terminated = x > 3.0


def test_a_selector_reading_only_inputs_is_static():
    (d,) = _ir.dispatch_groups(by_input.walk)
    assert _ir.static_selector(by_input.walk, 0) is True
    assert ("plane_in", "kind") in _ir.selector_reads(by_input.walk, 0)


@pytest.mark.parametrize("k", [by_state, by_mask], ids=lambda k: k.name)
def test_a_selector_reading_a_plane_the_kernel_writes_is_not(k):
    (d,) = _ir.dispatch_groups(k.walk)
    assert _ir.static_selector(k.walk, 0) is False


@kernel(steps=1)
def by_uniform(dt: Param, kind: Index, x: Mutable[Scalar]):
    x = x + dt * M.dispatch(M.select(dt > 0.5, kind, kind + 1), [x, -x])


def test_a_uniform_the_selector_reads_is_listed_and_stays_static():
    """A host-set uniform is not state (the class it picks changes only on a
    rebind, which the launching runtime watches by these names): listed in
    ``selector_reads``, and the group stays static."""
    (d,) = _ir.dispatch_groups(by_uniform.walk)
    assert ("uniform", "dt") in _ir.selector_reads(by_uniform.walk, 0)
    assert _ir.static_selector(by_uniform.walk, 0) is True
