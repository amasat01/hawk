# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

""" cross-sample forms under both transforms: a gather's reverse is a
scatter-add, a scatter's reverse is a gather.

``at(expr)`` and ``scatter(expr)`` are transposes of each other, so this is the
pair of rules the transform owns itself rather than the table (the ``at`` entry
is registered ``mode="walker"``). The numeric arms evaluate the derived IR over
a small multi-lane table and compare against a central difference on the table
entry; the classification arms are the direct consequence -- the
VJP of a gather MUST come out ``cross_sample_write``, because it writes a lane
that is not ``i``, and eagle refuses an illegal placement on that
declaration.
"""

from __future__ import annotations

import numpy as np
from _eval import evaluate

from hawk.diff import jvp, vjp
from hawk.ir import AccumWrite, Assign, At, Const, Leaf, canonical
from hawk.ir import make as mk
from hawk.types import TensorType

F64 = TensorType((), "f64")
I32 = TensorType((), "i32")
LANES = 4
H = 1e-5


def _gather_primal():
    """``out = tab.at(k) ** 3`` -- one gather, one sample-local write."""
    idx = Leaf("vocab_read", "per_sample", "idx", I32)
    tab = Leaf("table_read", "lookup", "tab", F64)
    read = At(tab, idx, F64)
    return (idx, tab), (Assign("out", mk("mul", (read, mk("mul", (read, read)))), F64),)


def test_the_reverse_of_a_gather_is_a_scatter_add_classified_cross_sample_write():
    _, sinks = _gather_primal()
    derived = vjp(sinks)
    scatter = [s for s in derived if isinstance(s, AccumWrite)]
    assert len(scatter) == 1 and scatter[0].name == "bar_tab"
    assert scatter[0].index is not None, "the scatter targets the gathered lane"
    assert canonical(derived).access.cls == "cross_sample_write", (
        "a write whose target lane is not `i` is cross_sample_write -- the "
        "class eagle checks the placement against"
    )


def test_the_gather_vjp_matches_a_finite_difference_on_the_table_entry():
    _, sinks = _gather_primal()
    table = np.array([0.7, -1.3, 2.1, 0.4])
    env = {"idx": 2, "tab": table}
    got = evaluate(vjp(sinks), {**env, "bar_out": 1.0}, lanes=LANES)["bar_tab"]
    shifted = np.array(table)
    shifted[2] += H
    plus = evaluate(sinks, {**env, "tab": shifted})["out"]
    shifted[2] -= 2 * H
    minus = evaluate(sinks, {**env, "tab": shifted})["out"]
    expect = np.zeros(LANES)
    expect[2] = (plus - minus) / (2 * H)
    assert np.allclose(got, expect, rtol=1e-6, atol=1e-8), f"{got} vs {expect}"


def test_the_gather_jvp_reads_a_tangent_lookup_plane():
    _, sinks = _gather_primal()
    derived = jvp(sinks)
    walk = canonical(derived)
    assert ("lookup", "dot_tab") in walk.slot_of
    assert walk.access.cls == "cross_sample_read"
    table = np.array([0.7, -1.3, 2.1, 0.4])
    tangent = np.array([0.0, 0.0, 1.0, 0.0])
    env = {"idx": 2, "tab": table}
    got = evaluate(derived, {**env, "dot_tab": tangent})["dot_out"]
    plus = evaluate(sinks, {**env, "tab": table + H * tangent})["out"]
    minus = evaluate(sinks, {**env, "tab": table - H * tangent})["out"]
    assert abs(got - (plus - minus) / (2 * H)) < 1e-6


def test_the_reverse_of_a_scattered_accumulate_is_a_gather():
    idx = Leaf("vocab_read", "per_sample", "idx", I32)
    x = Leaf("vocab_read", "per_sample", "x", F64)
    sinks = (AccumWrite("energy", mk("mul", (x, x)), idx, F64),)
    derived = vjp(sinks)
    walk = canonical(derived)
    assert ("lookup", "bar_energy") in walk.slot_of, (
        "the transpose of a scatter-add is a GATHER of its adjoint plane"
    )
    assert walk.access.cls == "cross_sample_read"
    table = np.array([0.0, 0.0, 3.0, 0.0])
    got = evaluate(derived, {"idx": 2, "x": 1.7, "bar_energy": table})["bar_x"]
    assert abs(got - 2 * 1.7 * 3.0) < 1e-9


def test_the_seed_of_an_own_column_accumulate_is_a_plain_read():
    x = Leaf("vocab_read", "per_sample", "x", F64)
    sinks = (AccumWrite("energy", mk("mul", (x, Const(3.0, F64))), None, F64),)
    walk = canonical(vjp(sinks))
    assert ("per_sample", "bar_energy") in walk.slot_of
    assert walk.access.cls == "sample_local"
