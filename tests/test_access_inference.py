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

from _dags import F64, IDX, V3, drag_dag

from hawk.ir import (
    AccumWrite,
    Assign,
    At,
    Const,
    Leaf,
    Op,
    WideWrite,
    canonical,
    infer,
)
from hawk.ir.nodes import ROLE_ORDER

_X = Leaf("vocab_read", "vec_in", "x", V3)
_TABLE = Leaf("table_read", "lookup", "grav", V3)


def test_own_column_only_is_sample_local():
    walk = canonical((Assign("acc", Op("neg", (_X,), V3), V3),))
    assert walk.access == ("sample_local", None)
    assert infer(walk) == walk.access


def test_lookup_at_read_is_cross_sample_read():
    walk = canonical((Assign("acc", At(_TABLE, Const(3, IDX), V3), V3),))
    assert walk.access.cls == "cross_sample_read"


def test_scattered_accum_write_is_cross_sample_write():
    walk = canonical((AccumWrite("e", Op("norm", (_X,), F64), Const(3, IDX), F64),))
    assert walk.access.cls == "cross_sample_write"


def test_own_column_accum_write_is_not_cross_sample():
    """ classifies on the FORM: an accum whose target lane is ``i`` is not
    a scatter."""
    walk = canonical((AccumWrite("e", Op("norm", (_X,), F64), None, F64),))
    assert walk.access.cls == "sample_local"


def test_own_column_wide_write_is_not_cross_sample():
    walk = canonical((WideWrite("g", Op("neg", (_X,), V3), None, V3),))
    assert walk.access.cls == "sample_local"


def test_the_most_restrictive_class_wins():
    """A DAG carrying BOTH a lookup read and a scatter declares the write class
    (the soundness-directed ranking)."""
    walk = canonical(drag_dag())
    assert walk.access == ("cross_sample_write", None)


def test_arg_spec_follows_hawks_canonical_role_order():
    walk = canonical(drag_dag())
    ranks = [ROLE_ORDER.index(role) for role, _ in walk.arg_spec]
    assert ranks == sorted(ranks), walk.arg_spec
    assert walk.arg_spec == (
        ("mutable", "acc"),
        ("accum_out", "energy"),
        ("vec_in", "sc__state__pos"),
        ("vec_in", "sc__state__vel"),
        ("lookup", "grav"),
        ("uniform", "rho"),
    ), walk.arg_spec


def test_slot_of_is_the_inverse_of_arg_spec():
    walk = canonical(drag_dag())
    assert all(walk.arg_spec[i] == key for key, i in walk.slot_of.items())
