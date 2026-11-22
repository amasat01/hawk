# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Access-class inference over the walk-20, …).

The classifier reads the DAG's ACCESS FORMS, never node classes: ``own(i)``
(no index operand) is ``sample_local``, an ``at(expr)`` read of a ``lookup``
plane is ``cross_sample_read``, a wide/accum sink carrying a scatter index is
``cross_sample_write``; among those three the MOST RESTRICTIVE wins,
because over-restriction costs only a placement while under-restriction is a
wrong answer under ``RankPartition``. ``mapreduce`` is disjoint and is
covered by ``test_sink_exclusivity.py``. HAWK only DECLARES — eagle's
``check_placement`` refuses; the class names are the execution
contract's.

This file also pins the canonical role order for ``arg_spec``.
"""

from __future__ import annotations

from _dags import IDX, V3, drag_dag

from hawk import _ir

CTX = _ir.new_context()
_X = _ir.leaf(CTX, "vocab_read", "plane_in", "x", V3)
_TABLE = _ir.leaf(CTX, "table_read", "lookup", "grav", V3)
_THREE = _ir.const(CTX, 3, IDX)


def test_own_column_only_is_sample_local():
    walk = _ir.canonical((_ir.make_node("Assign", V3, (_ir.make("neg", (_X,)),), name="acc"),))
    assert _ir.access_class(walk) == "sample_local"


def test_lookup_at_read_is_cross_sample_read():
    walk = _ir.canonical((_ir.make_node("Assign", V3, (_ir.make_node("At", V3, (_TABLE, _THREE)),), name="acc"),))
    assert _ir.access_class(walk) == "cross_sample_read"


def test_scattered_add_update_is_cross_sample_write():
    walk = _ir.canonical((_ir.update("e", _ir.make("norm", (_X,)), indices=(_THREE,)),))
    assert _ir.access_class(walk) == "cross_sample_write"


def test_own_column_add_update_is_not_cross_sample():
    """ classifies on the FORM: an accum whose target lane is ``i`` is not
    a scatter."""
    walk = _ir.canonical((_ir.update("e", _ir.make("norm", (_X,))),))
    assert _ir.access_class(walk) == "sample_local"


def test_own_column_store_update_is_not_cross_sample():
    walk = _ir.canonical((_ir.make_node("Update", V3, (_ir.make("neg", (_X,)),), name="g", op="store"),))
    assert _ir.access_class(walk) == "sample_local"


def test_the_most_restrictive_class_wins():
    """A DAG carrying BOTH a lookup read and a scatter declares the write class
    (the soundness-directed ranking)."""
    walk = _ir.canonical(drag_dag(CTX))
    assert _ir.access_class(walk) == "cross_sample_write"


def test_arg_spec_follows_hawks_canonical_role_order():
    walk = _ir.canonical(drag_dag(CTX))
    ranks = [_ir.ROLE_ORDER.index(role) for role, _ in _ir.arg_spec(walk)]
    assert ranks == sorted(ranks), _ir.arg_spec(walk)
    assert _ir.arg_spec(walk) == (
        ("mutable", "acc"),
        ("plane_out", "energy"),
        ("plane_in", "sc__state__pos"),
        ("plane_in", "sc__state__vel"),
        ("lookup", "grav"),
        ("uniform", "rho"),
    ), _ir.arg_spec(walk)


def test_slot_of_is_the_inverse_of_arg_spec():
    walk = _ir.canonical(drag_dag(CTX))
    assert all(_ir.arg_spec(walk)[i] == key for key, i in _ir.slot_of(walk).items())
