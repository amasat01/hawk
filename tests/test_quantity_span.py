# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

""" at WALK level-16…, ruling).

Quantity binding is all-or-nothing on the ACCESS, not on the read: if the DAG
REACHES a quantity, every wire it declares is bound and occupies a slot in one
contiguous, declaration-ordered span — including wires the body never reads,
which are bound and unused; if the quantity is UNREACHED, no wire is bound and
no slot exists. This is what autodiff requires (a VJP routinely reaches a
proper SUBSET of the primal's wires — a gradient w.r.t. position never reads
``vel``) and sends every derivative IR through the SAME walk.

Row is, because it needs a TRACED body; this file is the
programmatic walk-level unit test of the same rule, plus the namespacing.
The refusal half of fires only on the impossible state (a wire bound with
no owning quantity, or a non-contiguous / mis-ordered span).
"""

from __future__ import annotations

import pytest
from _dags import F64, V3, orientation_quantity, state_quantity

from hawk.ir import Assign, HawkError, Leaf, Op, canonical
from hawk.types import TensorType, Wire


def test_namespacing_follows_the_declaration():
    """The wire name is ``<kind-slug>__<instance>__<wire>``, and ``<kind-slug>__<wire>``
    when the quantity is not instanced."""
    assert [s.name for s in state_quantity().expand()] == ["sc__state__pos", "sc__state__vel"]
    from hawk import Quantity
    bare = Quantity(slug="drag", wires=[Wire("cd", "per_sample", ())])
    assert [s.name for s in bare.expand()] == ["drag__cd"]


def test_two_quantities_may_declare_the_same_wire_spelling():
    """The point: a consumer never writes a wire name, so no collision."""
    from hawk import Quantity
    a = Quantity(slug="earth", name="orientation", wires=[Wire("q", "vec_in", (4,), tag="quaternion")])
    b = Quantity(slug="moon", name="orientation", wires=[Wire("q", "vec_in", (4,), tag="quaternion")])
    assert a.expand()[0].name != b.expand()[0].name


def test_reached_quantity_binds_its_unread_wire_too():
    pos, _vel = state_quantity().read()          # only `pos` enters the DAG
    walk = canonical((Assign("acc", pos, V3),))
    assert ("vec_in", "sc__state__pos") in walk.slot_of
    assert ("vec_in", "sc__state__vel") in walk.slot_of, (
        "reaching ANY wire binds EVERY wire — dropping the unread one "
        "breaks every VJP, which reaches a proper subset of the primal's wires"
    )


def test_the_span_is_contiguous_and_declaration_ordered():
    pos, _vel = state_quantity().read()
    walk = canonical((Assign("acc", pos, V3),))
    span = walk.quantities["sc__state"]
    assert span.wires == ("sc__state__pos", "sc__state__vel")
    assert span.slots == (span.slots[0], span.slots[0] + 1), span.slots
    assert [walk.arg_spec[i][1] for i in span.slots] == list(span.wires)


def test_a_free_leaf_never_splits_the_span():
    """A free ``vec_in`` sorted by name would land between the two wires if the
    quantity block were not ordered ahead of it."""
    pos, _vel = state_quantity().read()
    free = Leaf("vocab_read", "vec_in", "sc__zzz" .replace("__", "_"), V3)
    walk = canonical((Assign("acc", Op("add", (pos, free), V3), V3),))
    span = walk.quantities["sc__state"]
    assert span.slots == (span.slots[0], span.slots[0] + 1), walk.arg_spec


def test_multi_role_quantity_is_contiguous_within_each_role():
    """The compound shape: ONE access, two wires of DIFFERENT roles."""
    q, w = orientation_quantity().read()
    walk = canonical((Assign("acc", Op("scale", (q, w), V3), V3),))
    span = walk.quantities["earth__orientation"]
    assert span.wires == ("earth__orientation__q", "earth__orientation__w")
    roles = [walk.arg_spec[i][0] for i in span.slots]
    assert roles == ["vec_in", "per_sample"], roles


def test_unreached_quantity_binds_nothing():
    state_quantity().read()                      # declared and expanded, never reached
    walk = canonical((Assign("acc", Leaf("vocab_read", "vec_in", "x", V3), V3),))
    assert walk.quantities == {}
    assert all(not n.startswith("sc__") for _, n in walk.arg_spec), walk.arg_spec


def test_a_namespaced_leaf_with_no_owning_quantity_is_refused():
    """The impossible state: the ``__`` namespace is a Quantity's to mint."""
    hand_rolled = Leaf("vocab_read", "vec_in", "sc__state__pos", V3)
    with pytest.raises(HawkError, match="no owning quantity"):
        canonical((Assign("acc", hand_rolled, V3),))


def test_wire_types_reach_the_slot():
    q, _w = orientation_quantity().read()
    walk = canonical((Assign("acc", Op("as_vec3", (q,), V3), V3),))
    slot = walk.leaves[walk.slot_of[("vec_in", "earth__orientation__q")]]
    assert slot.ttype == TensorType((4,), "f64", "quaternion")
    assert walk.leaves[walk.slot_of[("per_sample", "earth__orientation__w")]].ttype == F64
