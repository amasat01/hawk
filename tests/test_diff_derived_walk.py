# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

""" re-run on DERIVED): a derivative is a
function of the primal's STRUCTURE.

 asserted on primal DAGs; owes the same assertion on the IR the
transforms synthesise, because a derivative that depended on how the primal
happened to be CONSTRUCTED (a shared subexpression versus the same expression
written twice) would hash differently on every re-trace and defeat the content
cache the whole pipeline is keyed on. Both transforms therefore traverse the
walk's own node sequence (``raptor.ir.walk.canonical_nodes``), so structurally
equal primals -- ``tests/_dags.drag_dag(CTX)`` and its permuted twin, which 
 already pins to one primal digest -- derive to ONE digest and ONE
``arg_spec``. The last two tests add own reading of "the derived IR goes
through the SAME walk": re-walking it is idempotent, and for a kernel whose
derivative genuinely reaches every input, the derived ``arg_spec`` is a
SUPERSET of the primal's read side, wire for wire.

This test previously failed when ``hawk/hawk/diff/transform.py``
reverted to its own private DFS instead of ``canonical_nodes`` -- the permuted
twin's duplicated subexpression accumulated two separate adjoints and the
derived digests diverged.
"""

from __future__ import annotations

import pytest
from _dags import drag_dag

from hawk import Mutable, Param, Scalar, Vector, kernel
from hawk.diff import jvp, vjp
from hawk import _ir

#: One context holds the DAGs these rows build.
CTX = _ir.new_context()
from hawk.math import dot, norm

TRANSFORMS = [("vjp", vjp), ("jvp", jvp)]


@kernel
def drag(state: Vector[3], rho: Param, out: Mutable[Vector[3]]):
    out = -rho * state * norm(state)


@pytest.mark.parametrize("name,transform", TRANSFORMS, ids=[t[0] for t in TRANSFORMS])
def test_structurally_equal_primals_derive_to_one_digest(name, transform):
    a = _ir.canonical(transform(drag_dag(CTX)))
    b = _ir.canonical(transform(drag_dag(CTX, permuted=True)))
    assert _ir.digest(a) == _ir.digest(b), (
        f"on derived IR: the {name} of two structurally equal primals must "
        "hash identically"
    )
    assert _ir.arg_spec(a) == _ir.arg_spec(b)


@pytest.mark.parametrize("name,transform", TRANSFORMS, ids=[t[0] for t in TRANSFORMS])
def test_re_walking_the_derived_ir_is_idempotent(name, transform):
    derived = transform(drag_dag(CTX))
    first, second = _ir.canonical(derived), _ir.canonical(derived)
    assert _ir.digest(first) == _ir.digest(second)
    assert _ir.arg_spec(first) == _ir.arg_spec(second) == _ir.arg_spec(_ir.canonical(transform(drag_dag(CTX))))


@pytest.mark.parametrize("name,transform", TRANSFORMS, ids=[t[0] for t in TRANSFORMS])
def test_the_derived_arg_spec_is_a_superset_of_the_primal_read_side(name, transform):
    primal = drag.walk
    derived = _ir.canonical(transform(drag))
    reads = {entry for entry in _ir.arg_spec(primal) if entry[0] not in ("mutable", "out",
                                                                    "plane_out")}
    assert reads <= set(_ir.arg_spec(derived)), (
        f"the {name} of a kernel whose derivative reaches every input must keep "
        f"every primal wire bound: {sorted(reads - set(_ir.arg_spec(derived)))} are gone"
    )


def test_the_derived_ir_of_a_quantity_kernel_keeps_the_whole_span():
    from hawk import Quantity
    Wire = _ir.Wire

    state = Quantity(slug="sc", name="state",
                     wires=[Wire("pos", "plane_in", (3,)), Wire("vel", "plane_in", (3,))])

    @kernel
    def energy(sc: state, out: Mutable[Scalar]):
        pos, vel = sc
        out = dot(pos, vel) * norm(pos)

    derived = _ir.canonical(vjp(energy))
    span = _ir.quantities(derived)["sc__state"]
    assert span.wires == ("sc__state__pos", "sc__state__vel")
    assert span.slots == tuple(range(span.slots[0], span.slots[0] + 2))
