# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""One quantity read minting two wires, on the host path, asserting distinct
slots and that both wires' values round-trip.

This row runs the body and reads the answer, rather than only checking that
a refusal fires: the defect it guards against is a second wire silently
reading the first wire's mirror bytes, which surfaces as a pointer
reinterpreted as a double or, more quietly, as the other wire's number, and
neither of those is a refusal.

WHAT MAKES THE ANSWER DIAGNOSTIC. ``_deployable.two_wire`` reads ONE declared
quantity, whose declaration mints two wires of DIFFERENT ABI shapes — a
``plane_in`` quaternion (the 40-byte ``GRefMirror``) and a ``plane_in`` scalar
(the 32-byte ``ScalarHandle``) — and writes each wire's value to its own output
plane under a different exact scaling. The two constants are exact in binary
floating point, so the comparison is bit-for-bit: an aliased mirror cannot hide
inside a tolerance, and the two scalings mean a swap of the wires is visible as
well as a swap of the mirrors.

THE WHOLE PATH IS HAWK'S. The artifact is HAWK-emitted and HAWK-compiled, the
library is opened and self-checked by ``hawk._core``, the ``void*[]`` is
``hawk._core``'s, the mirrors are written by its ``bind``, and the call is its
serial ``run``. eagle is used only as the SECOND opinion at the bottom.

This test previously failed via two plants, both rebuilt and both then removed.

(a) ``ArgBlock::bind`` planted to write EVERY mirror into slot 0's storage: the
run segfaulted, because a 32-byte handle read out of a 40-byte mirror's bytes is
a bogus pointer. Recorded but discarded as the row's plant — a crash proves the
slots are distinct, it does not prove the row reads VALUES.

(b) the plant that does. A ``handle`` slot planted to take its data pointer from
the PRECEDING slot's mirror — literally "the second wire reads the first wire's
mirror", in bounds and without a crash, which is the shape::

    AssertionError: the plane_in wire earth__orientation__w did not round-trip:
    max |delta| = 6.0 (bit-identical required). A wire reading another wire's
    mirror is the defect; 4.65e-310-scale garbage is a pointer read as a
    double
    AssertionError: the plane_in wire is not read

Four of the seven rows went red together, including ``eagle agrees about both
wires``; the three that stayed green are the two structural ones (which read the
descriptor, not the answer) and the pointer-signature one (this plant produces
a wrong REAL number, not a pointer read as a double — the two signatures are
asserted apart on purpose).
"""

from __future__ import annotations

import _deploy as L
import _deployable as D
import _oracle as O
import numpy as np
import pytest
from conftest import sidecar_of


N = 48
QUANTITY = "earth__orientation"
WIRES = (("plane_in", f"{QUANTITY}__q"), ("plane_in", f"{QUANTITY}__w"))


@pytest.fixture(scope="module")
def kernel(built):
    bundle = built["two_wire"]
    return O.load(bundle.directory, "two_wire", sidecar_of(bundle, "two_wire"))


def test_the_two_wires_occupy_distinct_slots(kernel):
    """The structural half. Two wires of ONE quantity are two slots — never
    one merged slot — and they must land at different ``params[]`` indices,
    because the index IS what the emitted body dereferences."""
    slots = [kernel.slot_of[w] for w in WIRES]
    assert len(set(slots)) == 2, f"the two wires share a slot: {dict(zip(WIRES, slots))}"
    assert all(w in kernel.slot_of for w in WIRES), kernel.arg_spec


def test_both_wires_values_round_trip_through_the_host_path(kernel):
    """THE row. Both wires are read by the body and both answers must be exactly
    what was bound — through ``hawk._core``'s own ArgBlock and serial run."""
    q, w = D.quat(N), np.linspace(0.5, 2.0, N)
    got_q, got_w = O.run_kernel(kernel, N, **{WIRES[0][1]: q, WIRES[1][1]: w})
    want_q, want_w = D.ref_two_wire(q, w)

    assert got_w.tobytes() == want_w.tobytes(), (
        f"the plane_in wire {WIRES[1][1]} did not round-trip: max |delta| = "
        f"{float(np.max(np.abs(got_w - want_w)))} (bit-identical required). A wire "
        "reading another wire's mirror is the F56 defect; 4.65e-310-scale garbage "
        "is a pointer read as a double"
    )
    assert got_q.tobytes() == want_q.tobytes(), (
        f"the plane_in wire {WIRES[0][1]} did not round-trip: max |delta| = "
        f"{float(np.max(np.abs(got_q - want_q)))}"
    )


def test_neither_output_carries_a_pointer_read_as_a_double(kernel):
    """The SIGNATURE, named. A 64-bit pointer reinterpreted as an IEEE
    double lands in the subnormal range (~1e-310); no value this body can
    produce does. Asserted separately from the equality above so a future
    failure says WHICH defect it is."""
    q, w = D.quat(N), np.linspace(0.5, 2.0, N)
    for plane in O.run_kernel(kernel, N, **{WIRES[0][1]: q, WIRES[1][1]: w}):
        subnormal = np.abs(plane)[np.abs(plane) > 0.0]
        assert not np.any(subnormal < 1e-300), (
            f"an output plane carries subnormal values {subnormal[subnormal < 1e-300]}"
            " — the pointer-as-double signature of a wire reading another wire's "
            "mirror (F56)"
        )


def test_the_wires_are_not_interchangeable(kernel):
    """Non-vacuity for the equality above: change ONE wire's binding and the
    answer must move. Otherwise "both round-tripped" could be true of a body
    that read neither."""
    q, w = D.quat(N), np.linspace(0.5, 2.0, N)
    base_q, base_w = O.run_kernel(kernel, N, **{WIRES[0][1]: q, WIRES[1][1]: w})
    moved_q, moved_w = O.run_kernel(kernel, N,
                                    **{WIRES[0][1]: q, WIRES[1][1]: w + 1.0})
    assert moved_w.tobytes() != base_w.tobytes(), "the plane_in wire is not read"
    assert moved_q.tobytes() == base_q.tobytes(), (
        "changing the plane_in wire moved the QUATERNION output: the two wires "
        "are not independent, which is the F56 defect seen from the other side"
    )


def test_eagle_agrees_about_both_wires(built):
    """The second opinion. eagle packs the same artifact with its OWN classifier
    (``eagle.roles.classify_arg``, the far side); if HAWK's marshal and
    eagle's disagreed about which wire rides which mirror, the two answers would
    differ even though each is internally consistent."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    bundle = built["two_wire"]
    sidecar = sidecar_of(bundle, "two_wire")
    q, w = D.quat(N), np.linspace(0.5, 2.0, N)
    kw = {WIRES[0][1]: q, WIRES[1][1]: w}

    plugin = L.host_plugin(bundle.directory, "two_wire", sidecar)
    eagle_result = eplan.plan(plugin, structure=eexec.HostTeam).run(**kw)
    eagle_q, eagle_w = eagle_result["out_q"], eagle_result["out_w"]
    hawk_q, hawk_w = O.run(bundle.directory, "two_wire", N, sidecar, **kw)
    assert np.asarray(eagle_q).tobytes() == np.asarray(hawk_q).tobytes()
    assert np.asarray(eagle_w).tobytes() == np.asarray(hawk_w).tobytes()
