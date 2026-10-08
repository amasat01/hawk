# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Walk determinism.

Two ``canonical()`` calls on structurally equal DAGs produce an equal
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

from hawk.ir import Assign, Leaf, canonical


def test_two_walks_of_the_same_dag_agree():
    dag = drag_dag()
    a, b = canonical(dag), canonical(dag)
    assert a.digest == b.digest
    assert a.arg_spec == b.arg_spec


def test_permuted_construction_order_gives_the_same_walk():
    a = canonical(drag_dag())
    b = canonical(drag_dag(permuted=True))
    assert a.arg_spec == b.arg_spec, (a.arg_spec, b.arg_spec)
    assert a.digest == b.digest, (
        "structurally equal DAGs must hash identically: construction order "
        "and subexpression sharing are not part of a DAG's identity"
    )


def test_digest_is_sensitive_to_a_real_change():
    """Non-vacuity: the digest must not be constant. Renaming one leaf moves it."""
    a = canonical(drag_dag())
    b = canonical((Assign("acc", Leaf("vocab_read", "vec_in", "other", V3), V3),))
    assert a.digest != b.digest


def test_shared_subexpression_is_emitted_once():
    walk = canonical(drag_dag())
    kinds = [getattr(n, "kind", None) for n in walk.order]
    assert kinds.count("mul") == 1, kinds
