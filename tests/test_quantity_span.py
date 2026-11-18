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

from hawk import HawkError, _ir

CTX = _ir.new_context()


def _assign(value):
    """The sink committing ``value`` to ``acc``."""
    return (_ir.make_node("Assign", V3, [value], name="acc"),)


def _bound(walk, quantity):
    """Where the wires of ``quantity`` sit in the walk's argument list, as ``(index, name)`` pairs."""
    spec = _ir.arg_spec(walk)
    wires = [(w.role, quantity.wire_name(w)) for w in quantity.wires]
    return [(spec.index(slot), slot[1]) for slot in wires if slot in spec]


def _leaf_named(sinks, role, name):
    """The leaf the graph under ``sinks`` binds to the slot ``(role, name)``."""
    return next(n for n in _ir.post_order(sinks)
                if _ir.is_a(n, "leaf") and (n.field_text("role"), n.field_text("name")) == (role, name))


def test_namespacing_follows_the_declaration():
    """The wire name is ``<kind-slug>__<instance>__<wire>``, and ``<kind-slug>__<wire>``
    when the quantity is not instanced."""
    assert [s.name for s in state_quantity().expand()] == ["sc__state__pos", "sc__state__vel"]
    bare = _ir.Quantity(slug="drag", wires=[_ir.Wire("cd", "per_sample", ())])
    assert [s.name for s in bare.expand()] == ["drag__cd"]


def test_two_quantities_may_declare_the_same_wire_spelling():
    """The point: a consumer never writes a wire name, so no collision."""
    a = _ir.Quantity(slug="earth", name="orientation", wires=[_ir.Wire("q", "vec_in", (4,), tag="quaternion")])
    b = _ir.Quantity(slug="moon", name="orientation", wires=[_ir.Wire("q", "vec_in", (4,), tag="quaternion")])
    assert a.expand()[0].name != b.expand()[0].name


def test_reached_quantity_binds_its_unread_wire_too():
    pos, _vel = state_quantity().read(CTX)          # only `pos` enters the DAG
    walk = _ir.canonical(_assign(pos))
    assert ("vec_in", "sc__state__pos") in _ir.slot_of(walk)
    assert ("vec_in", "sc__state__vel") in _ir.slot_of(walk), (
        "reaching ANY wire binds EVERY wire — dropping the unread one "
        "breaks every VJP, which reaches a proper subset of the primal's wires"
    )


def test_the_span_is_contiguous_and_declaration_ordered():
    pos, _vel = state_quantity().read(CTX)
    walk = _ir.canonical(_assign(pos))
    slots, names = zip(*_bound(walk, state_quantity()))
    assert names == ("sc__state__pos", "sc__state__vel")
    assert slots == (slots[0], slots[0] + 1), slots


def test_a_free_leaf_never_splits_the_span():
    """A free ``vec_in`` sorted by name would land between the two wires if the
    quantity block were not ordered ahead of it."""
    pos, _vel = state_quantity().read(CTX)
    free = _ir.leaf(CTX, "vocab_read", "vec_in", "sc_zzz", V3)
    walk = _ir.canonical(_assign(_ir.make("add", (pos, free))))
    slots = [index for index, _ in _bound(walk, state_quantity())]
    assert slots == [slots[0], slots[0] + 1], _ir.arg_spec(walk)


def test_multi_role_quantity_is_contiguous_within_each_role():
    """The compound shape: ONE access, two wires of DIFFERENT roles."""
    q, w = orientation_quantity().read(CTX)
    walk = _ir.canonical(_assign(_ir.make_node("Op", V3, [q, w], kind="scale", literal=None)))
    bound = _bound(walk, orientation_quantity())
    assert [name for _, name in bound] == ["earth__orientation__q", "earth__orientation__w"]
    roles = [_ir.arg_spec(walk)[i][0] for i, _ in bound]
    assert roles == ["vec_in", "per_sample"], roles


def test_unreached_quantity_binds_nothing():
    state_quantity().read(CTX)                      # declared and expanded, never reached
    walk = _ir.canonical(_assign(_ir.leaf(CTX, "vocab_read", "vec_in", "x", V3)))
    assert _bound(walk, state_quantity()) == []


def test_a_namespaced_leaf_with_no_owning_quantity_is_refused():
    """The impossible state: the ``__`` namespace is a Quantity's to mint."""
    hand_rolled = _ir.leaf(CTX, "vocab_read", "vec_in", "sc__state__pos", V3)
    with pytest.raises(HawkError, match="no owning quantity"):
        _ir.canonical(_assign(hand_rolled))


def test_wire_types_reach_the_slot():
    q, _w = orientation_quantity().read(CTX)
    sinks = _assign(_ir.make("as_vec3", (q,)))
    assert _ir.type_of(_leaf_named(sinks, "vec_in", "earth__orientation__q")) == _ir.TensorType((4,), "f64", "quaternion")
    walk = _ir.canonical(sinks)          # the unread wire has a slot but no leaf in the graph
    assert _ir.slot_type(walk, "per_sample", "earth__orientation__w") == F64
