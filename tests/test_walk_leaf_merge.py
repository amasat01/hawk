# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The leaf merge is total.

A slot is keyed on ``(role, name)`` ALONE. Two leaves sharing that key
whose ``(shape, dtype, tag)`` DISAGREE would therefore share one slot, so
``canonical()`` refuses them with a ``HawkError`` naming BOTH types — never a
silent collapse onto whichever leaf the traversal reached first, and never a
uniqueness assert over a list already de-duplicated on the same key (which
would measure itself). The collision is INJECTED here by constructing the pair
directly, which is what makes the row RED-able.

This test previously failed when run against a walk whose ``_merge_binding``
did not compare types (the naive dedup) — the collision merged silently onto one
slot and the ``pytest.raises(HawkError)`` below did not fire.
"""

from __future__ import annotations

import pytest
from _dags import F64, V3

from hawk.ir import Assign, HawkError, Leaf, Op, canonical
from hawk.types import TensorType


def _collision(a: TensorType, b: TensorType):
    """Two leaves sharing ``('vec_in', 'x')`` with the given types, both reached."""
    left = Leaf("vocab_read", "vec_in", "x", a)
    right = Leaf("vocab_read", "vec_in", "x", b)
    return (Assign("out", Op("add", (left, right), a), a),)


def test_shape_collision_is_refused_naming_both_types():
    with pytest.raises(HawkError) as excinfo:
        canonical(_collision(V3, TensorType((4,), "f64")))
    msg = str(excinfo.value)
    assert "'vec_in'" in msg and "'x'" in msg, msg
    assert "shape=(3,)" in msg and "shape=(4,)" in msg, (
        "the refusal must name BOTH types, got: " + msg
    )


def test_dtype_collision_is_refused():
    with pytest.raises(HawkError, match="collision"):
        canonical(_collision(V3, TensorType((3,), "f32")))


def test_tag_collision_is_refused():
    """The quaternion tag participates in type identity."""
    quat = TensorType((4,), "f64", "quaternion")
    with pytest.raises(HawkError, match="collision"):
        canonical(_collision(quat, TensorType((4,), "f64")))


def test_agreeing_leaves_merge_onto_one_slot():
    """The positive half: agreement merges, and merges ONCE."""
    walk = canonical(_collision(V3, TensorType((3,), "f64")))
    assert walk.arg_spec.count(("vec_in", "x")) == 1
    assert walk.slot_of[("vec_in", "x")] == walk.arg_spec.index(("vec_in", "x"))


def test_sink_and_leaf_do_not_collide_across_roles():
    """``('mutable','x')`` and ``('vec_in','x')`` are distinct slots: the key
    is the PAIR, not the name."""
    leaf = Leaf("vocab_read", "vec_in", "x", F64)
    walk = canonical((Assign("x", leaf, F64),))
    assert walk.arg_spec == (("mutable", "x"), ("vec_in", "x"))
