# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Walk determinism.

Two ``_ir.canonical()`` calls on structurally equal DAGs produce an equal
``digest`` and an equal ``arg_spec``. "Structurally equal" is the load-bearing
word: the second DAG below is built in a different CONSTRUCTION order and
duplicates the shared subexpression instead of sharing the object, so the
interior dedup (``kind, dtype, shape, tag, child_ids, literal``) has to fold
it back onto the same emission order. Nothing in the digest may depend on
object identity — child references are POSITIONS in the emitted order.

This test previously failed when run against a digest computed from
``id(node)`` values instead of structural keys — the two calls returned
different digests.
"""

from __future__ import annotations

from _dags import V3, drag_dag

from hawk import _ir

#: One context holds the DAGs these rows build.
CTX = _ir.new_context()


def test_two_walks_of_the_same_dag_agree():
    dag = drag_dag(CTX)
    a, b = _ir.canonical(dag), _ir.canonical(dag)
    assert _ir.digest(a) == _ir.digest(b)
    assert _ir.arg_spec(a) == _ir.arg_spec(b)


def test_permuted_construction_order_gives_the_same_walk():
    a = _ir.canonical(drag_dag(CTX))
    b = _ir.canonical(drag_dag(CTX, permuted=True))
    assert _ir.arg_spec(a) == _ir.arg_spec(b), (_ir.arg_spec(a), _ir.arg_spec(b))
    assert _ir.digest(a) == _ir.digest(b), (
        "structurally equal DAGs must hash identically: construction order "
        "and subexpression sharing are not part of a DAG's identity"
    )


def test_digest_is_sensitive_to_a_real_change():
    """Non-vacuity: the digest must not be constant. Renaming one leaf moves it."""
    a = _ir.canonical(drag_dag(CTX))
    ctx = _ir.new_context()
    other = _ir.leaf(ctx, "vocab_read", "plane_in", "other", V3)
    b = _ir.canonical((_ir.make_node("Assign", V3, (other,), name="acc"),))
    assert _ir.digest(a) != _ir.digest(b)


def test_shared_subexpression_is_emitted_once():
    walk = _ir.canonical(drag_dag(CTX))
    kinds = [n.field_text("kind") for n in _ir.walk_order(walk)]
    assert kinds.count("mul") == 1, kinds
