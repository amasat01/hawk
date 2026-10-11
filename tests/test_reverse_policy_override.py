# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``vjp(..., policy={group: ...})``, ``"auto"`` and the selector classification.

Rows:

* **override** — a derived dispatch inherits the primal's policy unless
  ``policy=`` names its group; overridden, every derived dispatch carries the
  new policy, and the gradient equals central differences EITHER way (and the
  two gradients are equal); an unknown group or policy refuses by name.
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
from hawk.artifact.sidecar import _dispatches
from hawk.diff import vjp
from hawk.emit import render_body
from hawk.ir import Assign, Dispatch, HawkError, Leaf, canonical
from hawk.ir import make as mk
from hawk.ir.nodes import DISPATCH_POLICIES
from hawk.ir.segment import split_segmented
from hawk.ir.walk import static_selectors
from hawk.types import TensorType

S = TensorType((), "f64")
I32 = TensorType((), "i32")
_H = 1e-6


def _primal(policy: str, K: int = 3):
    k = Leaf("vocab_read", "per_sample", "k", I32)
    xs = [Leaf("vocab_read", "per_sample", f"x{i}", S) for i in range(K)]
    branches = [mk("mul", (xs[i], mk("sin", (xs[(i + 1) % K],)))) for i in range(K)]
    return (Assign("out", Dispatch(k, branches, policy, S), S),)


def _policies(sinks) -> set:
    return {n.policy for n in canonical(tuple(sinks)).order if isinstance(n, Dispatch)}


@pytest.mark.parametrize("primal_policy,override", [
    ("switch", "predicated"), ("predicated", "switch"), ("segmented", "switch"),
    ("switch", "segmented"), ("auto", "predicated"),
])
def test_an_overridden_reverse_policy_differs_and_both_match_fd(primal_policy, override):
    sinks = _primal(primal_policy)
    inherited = vjp(sinks)
    overridden = vjp(sinks, policy={"dispatch0": override})
    assert _policies(inherited) == {primal_policy}
    assert _policies(overridden) == {override}
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
            fd = (evaluate(sinks, plus)["out"] - evaluate(sinks, minus)["out"]) / (2 * _H)
            assert b[f"bar_x{i}"] == pytest.approx(fd * seed, rel=1e-6, abs=1e-8)


def test_an_unknown_group_or_policy_refuses_by_name():
    sinks = _primal("switch")
    with pytest.raises(HawkError, match=r"'dispatch7' is not a dispatch group"):
        vjp(sinks, policy={"dispatch7": "switch"})
    with pytest.raises(HawkError, match=r"unknown policy 'fused'"):
        vjp(sinks, policy={"dispatch0": "fused"})


def test_auto_is_accepted_distinct_and_renders_as_switch():
    assert "auto" in DISPATCH_POLICIES
    auto, switch = _primal("auto"), _primal("switch")
    assert canonical(auto).digest != canonical(switch).digest
    auto_text = render_body(auto, canonical(auto)).text
    counted = [ln for ln in auto_text.split("\n") if "hawk_abi::mix_count(" not in ln]
    assert len(counted) == len(auto_text.split("\n")) - 1      # one count line
    assert "\n".join(counted) == render_body(switch, canonical(switch)).text


# -- the selector classification --------------------------------------------- #
@kernel(steps=1)
def by_input(dt: Param, kind: Index, x: Mutable[Scalar]):
    x = x + dt * M.dispatch(kind, [x, -x], policy="auto")


@kernel(steps=1)
def by_state(dt: Param, kind: Index, x: Mutable[Scalar], c: Mutable[Index]):
    x = x + dt * M.dispatch(c, [x, -x], policy="auto")
    c = M.select(x > 0.0, kind, kind + 1)


@kernel(steps=1)
def by_mask(dt: Param, kind: Index, terminated: Terminated, x: Mutable[Scalar]):
    sel = M.select(terminated, kind, kind + 1)
    x = x + dt * M.dispatch(sel, [x, -x])
    terminated = x > 3.0


@kernel(steps=1)
def writes_kind(dt: Param, kind: Mutable[Index]):
    kind = kind + 1


def test_a_selector_reading_only_inputs_is_static():
    (d,) = by_input.walk.dispatches
    assert d.static_selector is True
    assert ("per_sample", "kind") in d.selector_reads


@pytest.mark.parametrize("k", [by_state, by_mask], ids=lambda k: k.name)
def test_a_selector_reading_a_plane_the_kernel_writes_is_not(k):
    (d,) = k.walk.dispatches
    assert d.static_selector is False


def test_the_step_scoped_verdict_sees_another_kernel_s_writes():
    assert static_selectors([by_input]) == {"by_input": {"dispatch0": True}}
    got = static_selectors([by_input, writes_kind])
    assert got["by_input"] == {"dispatch0": False}
    assert got["writes_kind"] == {}


def test_the_sidecar_and_a_segmented_unit_carry_the_classification():
    (entry,) = _dispatches(by_input.walk.dispatches)
    assert entry["static_selector"] is True
    assert ["per_sample", "kind"] in entry["selector_reads"]
    *units, classify = split_segmented("seg", _primal("segmented"))
    assert all(u.segment.static_selector for u in units)
    assert units[0].segment.selector_reads == (("per_sample", "k"),)
    assert classify.name == "seg__cls"      # a static selector's classes come from here


@kernel(steps=1)
def by_uniform(dt: Param, kind: Index, x: Mutable[Scalar]):
    x = x + dt * M.dispatch(M.select(dt > 0.5, kind, kind + 1), [x, -x])


def test_a_uniform_the_selector_reads_is_listed_and_stays_static():
    """A host-set uniform is not state (the class it picks changes only on a
    rebind, which the launching runtime watches by these names): listed in
    ``selector_reads``, and the group stays static."""
    (d,) = by_uniform.walk.dispatches
    assert ("uniform", "dt") in d.selector_reads
    assert d.static_selector is True
    (entry,) = _dispatches(by_uniform.walk.dispatches)
    assert ["uniform", "dt"] in entry["selector_reads"]
