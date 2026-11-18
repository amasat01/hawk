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
from hawk import _ir

F64 = _ir.TensorType((), "f64")
I32 = _ir.TensorType((), "i32")
#: One context holds every node these rows build.
CTX = _ir.new_context()
LANES = 4
H = 1e-5


def _leaf(kind, role, name, ttype):
    return _ir.leaf(CTX, kind, role, name, ttype)


def _assign(name, value):
    return _ir.make_node("Assign", _ir.type_of(value), (value,), name=name)


def _gather_primal():
    """``out = tab.at(k) ** 3`` -- one gather, one sample-local write."""
    idx = _leaf("vocab_read", "per_sample", "idx", I32)
    tab = _leaf("table_read", "lookup", "tab", F64)
    read = _ir.make_node("At", F64, (tab, idx))
    return (idx, tab), (_assign("out", _ir.make("mul", (read, _ir.make("mul", (read, read))))),)


def test_the_reverse_of_a_gather_is_a_scatter_add_classified_cross_sample_write():
    _, sinks = _gather_primal()
    derived = vjp(sinks)
    scatter = [s for s in derived if _ir.is_a(s, "accum_write")]
    assert len(scatter) == 1 and scatter[0].field_text("name") == "bar_tab"
    assert scatter[0].field_kind("index"), "the scatter targets the gathered lane"
    assert _ir.access_class(_ir.canonical(derived)) == "cross_sample_write", (
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
    walk = _ir.canonical(derived)
    assert ("lookup", "dot_tab") in _ir.slot_of(walk)
    assert _ir.access_class(walk) == "cross_sample_read"
    table = np.array([0.7, -1.3, 2.1, 0.4])
    tangent = np.array([0.0, 0.0, 1.0, 0.0])
    env = {"idx": 2, "tab": table}
    got = evaluate(derived, {**env, "dot_tab": tangent})["dot_out"]
    plus = evaluate(sinks, {**env, "tab": table + H * tangent})["out"]
    minus = evaluate(sinks, {**env, "tab": table - H * tangent})["out"]
    assert abs(got - (plus - minus) / (2 * H)) < 1e-6


def test_the_reverse_of_a_scattered_accumulate_is_a_gather():
    idx = _leaf("vocab_read", "per_sample", "idx", I32)
    x = _leaf("vocab_read", "per_sample", "x", F64)
    sinks = (_ir.accum_write("energy", _ir.make("mul", (x, x)), index=idx),)
    derived = vjp(sinks)
    walk = _ir.canonical(derived)
    assert ("lookup", "bar_energy") in _ir.slot_of(walk), (
        "the transpose of a scatter-add is a GATHER of its adjoint plane"
    )
    assert _ir.access_class(walk) == "cross_sample_read"
    table = np.array([0.0, 0.0, 3.0, 0.0])
    got = evaluate(derived, {"idx": 2, "x": 1.7, "bar_energy": table})["bar_x"]
    assert abs(got - 2 * 1.7 * 3.0) < 1e-9


def test_the_seed_of_an_own_column_accumulate_is_a_plain_read():
    x = _leaf("vocab_read", "per_sample", "x", F64)
    sinks = (_ir.accum_write("energy", _ir.make("mul", (x, _ir.const(CTX, 3.0, F64)))),)
    walk = _ir.canonical(vjp(sinks))
    assert ("per_sample", "bar_energy") in _ir.slot_of(walk)
    assert _ir.access_class(walk) == "sample_local"
