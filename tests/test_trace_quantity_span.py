# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Quantity binding is all-or-nothing on the access, not on the read.

A traced body that reaches a two-wire quantity but reads only ONE wire must
still bind BOTH, contiguously and in declaration order, with the unread wire
bound and unused; a quantity the body never reaches must bind NEITHER. This is
what every VJP depends on (a gradient w.r.t. position never reads ``vel``), so the row is asserted on the traced primal AND on its derived IR --
which is the case that would break silently if the tracer bound only the wires
a body happened to touch.

This test previously failed when ``hawk/hawk/trace/kernel.py``'s
quantity binding changed to mint bare leaves (``Leaf(..., quantity=None)``)
instead of going through ``Quantity.read_with`` -- the unread wire
``sc__state__vel`` vanished from ``arg_spec`` on both the primal and the
derived walk.
"""

from __future__ import annotations

import pytest

from hawk import HawkError, Mutable, Param, Quantity, Scalar, _ir, kernel
from hawk.diff import vjp
from hawk.math import norm

STATE = Quantity(slug="sc", name="state",
                 wires=[_ir.Wire("pos", "vec_in", (3,)), _ir.Wire("vel", "vec_in", (3,))])
FIELD = Quantity(slug="earth", name="orientation",
                 wires=[_ir.Wire("q", "vec_in", (4,), tag="quaternion"),
                        _ir.Wire("w", "per_sample", ())])


@kernel
def reads_one_wire(sc: STATE, rho: Param, out: Mutable[Scalar]):
    pos, vel = sc
    out = rho * norm(pos)


@kernel
def reaches_neither(unused: FIELD, x: Scalar, out: Mutable[Scalar]):
    out = x * x


def _names_read(sinks) -> set[str]:
    return {n.field_text("name") for n in _ir.post_order(sinks) if _ir.is_a(n, "leaf")}


def _bound(walk, quantity) -> list:
    """Where the wires of ``quantity`` sit in the walk's argument list, as ``(index, name)`` pairs."""
    spec = _ir.arg_spec(walk)
    wires = [(w.role, quantity.wire_name(w)) for w in quantity.wires]
    return [(spec.index(slot), slot[1]) for slot in wires if slot in spec]


def test_reached_quantity_binds_every_wire_contiguously_in_declaration_order():
    slots, names = zip(*_bound(reads_one_wire.walk, STATE))
    assert names == ("sc__state__pos", "sc__state__vel")
    assert slots == tuple(range(slots[0], slots[0] + 2)), (
        "a reached quantity's wires occupy ONE consecutive, "
        f"declaration-ordered run; got {slots}"
    )


def test_the_unread_wire_is_bound_and_unused():
    read = _names_read(reads_one_wire.sinks)
    assert "sc__state__pos" in read
    assert "sc__state__vel" not in read, "the body reads only one wire"
    assert ("vec_in", "sc__state__vel") in _ir.slot_of(reads_one_wire.walk), (
        "a reached-but-UNREAD wire is bound and unused -- dropping it "
        "breaks every VJP"
    )


def test_an_unreached_quantity_binds_nothing():
    names = {name for _, name in reaches_neither.arg_spec}
    assert not any(name.startswith("earth__orientation") for name in names), (
        f"an UNREACHED quantity binds no wire and occupies no slot; got {names}"
    )
    assert _bound(reaches_neither.walk, FIELD) == []


def test_the_span_survives_the_derived_ir():
    slots, names = zip(*_bound(_ir.canonical(vjp(reads_one_wire)), STATE))
    assert names == ("sc__state__pos", "sc__state__vel")
    assert slots == tuple(range(slots[0], slots[0] + 2))
    assert "sc__state__vel" not in _names_read(vjp(reads_one_wire)), (
        "the VJP reaches a proper SUBSET of the primal's wires -- which is exactly "
        "why binds on the access and not on the read"
    )


def test_a_wire_name_outside_a_quantity_is_refused():
    ttype = STATE.expand()[0].ttype
    stray = _ir.leaf(_ir.new_context(), "vocab_read", "vec_in", "sc__state__pos", ttype)
    with pytest.raises(HawkError, match="no owning quantity"):
        _ir.canonical((_ir.make_node("Assign", ttype, [stray], name="out"),))
