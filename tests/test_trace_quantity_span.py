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

from hawk import Mutable, Param, Quantity, Scalar, kernel
from hawk.diff import vjp
from hawk.ir import Assign, HawkError, Leaf, Node, canonical
from hawk.math import norm
from hawk.types import Wire

STATE = Quantity(slug="sc", name="state",
                 wires=[Wire("pos", "vec_in", (3,)), Wire("vel", "vec_in", (3,))],
                 reconstruct=lambda p, v: (p, v))
FIELD = Quantity(slug="earth", name="orientation",
                 wires=[Wire("q", "vec_in", (4,), tag="quaternion"),
                        Wire("w", "per_sample", ())],
                 reconstruct=lambda q, w: (q, w))


@kernel
def reads_one_wire(sc: STATE, rho: Param, out: Mutable[Scalar]):
    pos, vel = sc
    out = rho * norm(pos)


@kernel
def reaches_neither(unused: FIELD, x: Scalar, out: Mutable[Scalar]):
    out = x * x


def _names_read(sinks) -> set[str]:
    seen: set[str] = set()

    def visit(node: Node) -> None:
        if isinstance(node, Leaf):
            seen.add(node.name)
        for child in node.operands:
            visit(child)

    for sink in sinks:
        visit(sink)
    return seen


def test_reached_quantity_binds_every_wire_contiguously_in_declaration_order():
    walk = reads_one_wire.walk
    span = walk.quantities["sc__state"]
    assert span.wires == ("sc__state__pos", "sc__state__vel")
    assert span.slots == tuple(range(span.slots[0], span.slots[0] + 2)), (
        "a reached quantity's wires occupy ONE consecutive, "
        f"declaration-ordered run; got {span.slots}"
    )
    assert [walk.arg_spec[i][1] for i in span.slots] == list(span.wires)


def test_the_unread_wire_is_bound_and_unused():
    read = _names_read(reads_one_wire.sinks)
    assert "sc__state__pos" in read
    assert "sc__state__vel" not in read, "the body reads only one wire"
    assert ("vec_in", "sc__state__vel") in reads_one_wire.walk.slot_of, (
        "a reached-but-UNREAD wire is bound and unused -- dropping it "
        "breaks every VJP"
    )


def test_an_unreached_quantity_binds_nothing():
    names = {name for _, name in reaches_neither.arg_spec}
    assert not any(name.startswith("earth__orientation") for name in names), (
        f"an UNREACHED quantity binds no wire and occupies no slot; got {names}"
    )
    assert reaches_neither.walk.quantities == {}


def test_the_span_survives_the_derived_ir():
    derived = canonical(vjp(reads_one_wire))
    span = derived.quantities["sc__state"]
    assert span.wires == ("sc__state__pos", "sc__state__vel")
    assert span.slots == tuple(range(span.slots[0], span.slots[0] + 2))
    assert "sc__state__vel" not in _names_read(vjp(reads_one_wire)), (
        "the VJP reaches a proper SUBSET of the primal's wires -- which is exactly "
        "why binds on the access and not on the read"
    )


def test_a_wire_name_outside_a_quantity_is_refused():
    stray = Leaf("vocab_read", "vec_in", "sc__state__pos", STATE.expand()[0].ttype)
    with pytest.raises(HawkError, match="no owning quantity"):
        canonical((Assign("out", stray, stray.ttype),))
