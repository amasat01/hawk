# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

""" re-run on DERIVED): a derivative is a
function of the primal's STRUCTURE.

 asserted on primal DAGs; owes the same assertion on the IR the
transforms synthesise, because a derivative that depended on how the primal
happened to be CONSTRUCTED (a shared subexpression versus the same expression
written twice) would hash differently on every re-trace and defeat the content
cache the whole pipeline is keyed on. Both transforms therefore traverse the
walk's own node sequence (``hawk.ir.walk.canonical_nodes``), so structurally
equal primals -- ``tests/_dags.drag_dag()`` and its permuted twin, which 
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
from hawk.ir import canonical
from hawk.math import dot, norm

TRANSFORMS = [("vjp", vjp), ("jvp", jvp)]


@kernel
def drag(state: Vector[3], rho: Param, out: Mutable[Vector[3]]):
    out = -rho * state * norm(state)


@pytest.mark.parametrize("name,transform", TRANSFORMS, ids=[t[0] for t in TRANSFORMS])
def test_structurally_equal_primals_derive_to_one_digest(name, transform):
    a = canonical(transform(drag_dag()))
    b = canonical(transform(drag_dag(permuted=True)))
    assert a.digest == b.digest, (
        f"on derived IR: the {name} of two structurally equal primals must "
        "hash identically"
    )
    assert a.arg_spec == b.arg_spec


@pytest.mark.parametrize("name,transform", TRANSFORMS, ids=[t[0] for t in TRANSFORMS])
def test_re_walking_the_derived_ir_is_idempotent(name, transform):
    derived = transform(drag_dag())
    first, second = canonical(derived), canonical(derived)
    assert first.digest == second.digest
    assert first.arg_spec == second.arg_spec == canonical(transform(drag_dag())).arg_spec


@pytest.mark.parametrize("name,transform", TRANSFORMS, ids=[t[0] for t in TRANSFORMS])
def test_the_derived_arg_spec_is_a_superset_of_the_primal_read_side(name, transform):
    primal = drag.walk
    derived = canonical(transform(drag))
    reads = {entry for entry in primal.arg_spec if entry[0] not in ("mutable", "out",
                                                                    "wide_out",
                                                                    "accum_out")}
    assert reads <= set(derived.arg_spec), (
        f"the {name} of a kernel whose derivative reaches every input must keep "
        f"every primal wire bound: {sorted(reads - set(derived.arg_spec))} are gone"
    )


def test_the_derived_ir_of_a_quantity_kernel_keeps_the_whole_span():
    from hawk import Quantity
    from hawk.types import Wire

    state = Quantity(slug="sc", name="state",
                     wires=[Wire("pos", "vec_in", (3,)), Wire("vel", "vec_in", (3,))],
                     reconstruct=lambda p, v: (p, v))

    @kernel
    def energy(sc: state, out: Mutable[Scalar]):
        pos, vel = sc
        out = dot(pos, vel) * norm(pos)

    derived = canonical(vjp(energy))
    span = derived.quantities["sc__state"]
    assert span.wires == ("sc__state__pos", "sc__state__vel")
    assert span.slots == tuple(range(span.slots[0], span.slots[0] + 2))
