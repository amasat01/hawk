# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The role -> by-value ABI shape the entry emits.

Pins HAWK's role/mirror map table-for-table against eagle's
``classify_arg`` tags. The entry signature and the unpack are generated
from that map, so a role that packs the wrong mirror is a 32-byte handle
where the kernel expects a 40-byte mirror — deterministic garbage, no
crash. This file asserts the map emits from, and that the one shape it
cannot emit is refused loudly instead of packed wrongly.

The refusal is a real, measured limitation, not a defensive branch: a rank-1
``lookup`` plane (``Table[Vector[3]]``) — and the rank-1 ``accum_out`` every
VJP of such a gather mints — has no packing under, because that role is
pinned ``ScalarHandle``-shaped. Emitting a ``GRefMirror`` there would produce a
correct-looking artifact that decodes eagle's parameter block at the wrong
offsets, and closing that gap belongs to eagle, not here.

``hawk/_core``'s ArgBlock is told
each slot's by-value shape by a DESCRIPTOR, and if that descriptor were built
from a hand-written second table, the compiler and the marshal could disagree
about which mirror a role rides — the 32-byte-handle-for-a-40-byte-mirror
failure, with no crash. So ``hawk.runtime`` derives it from THIS map, and the
rows at the bottom pin the two derivations (from a ``Walk``, and from a deployed
artifact's SIDECAR, which is all a consumer holds) against each other on every
fixture kernel.
"""

from __future__ import annotations

import _deployable as D
import pytest
from conftest import sidecar_of

from hawk import runtime
from hawk.emit.aether import MIRROR_OF, mirror_of
from hawk.ir import HawkError
from hawk.types import TensorType

_V3 = TensorType((3,), "f64")
_M33 = TensorType((3, 3), "f64")
_S = TensorType((), "f64")


def test_the_map_is_h93s_table():
    assert MIRROR_OF == {
        "out": "GRefMirror", "vec_in": "GRefMirror", "mat_in": "GRefMirror",
        "per_sample": "ScalarHandle", "lookup": "ScalarHandle",
        "terminated": "ScalarHandle", "wide_in": "ScalarHandle",
        "wide_out": "ScalarHandle", "accum_out": "ScalarHandle",
        "uniform": "value", "nsamples": "value",
    }


def test_mutable_resolves_by_context():
    assert mirror_of("mutable", _M33) == "GRefMirror"
    assert mirror_of("mutable", _V3) == "GRefMirror"
    assert mirror_of("mutable", _S) == "ScalarHandle"


@pytest.mark.parametrize("role", ["per_sample", "lookup", "terminated", "wide_in",
                                  "wide_out", "accum_out"])
def test_a_rank1_plane_in_a_handle_only_role_is_refused(role):
    with pytest.raises(HawkError) as excinfo:
        mirror_of(role, _V3)
    message = str(excinfo.value)
    assert "the map" in message and "ScalarHandle" in message and "GRef accum/wide role" in message, (
        "the refusal must name the pinned map and the role that closes it, or a "
        f"reader cannot act on it: {message}"
    )


# --------------------------------------------------------------------------- #
# The descriptor `hawk._core` is built from is DERIVED from the map above.
# --------------------------------------------------------------------------- #
def test_the_core_descriptor_vocabulary_covers_the_whole_map():
    """Every by-value shape resolves to must have a ``_core`` kind. A role
    with no kind would raise at bind time — after the artifact was already
    built, loaded and declared good."""
    kinds = {runtime.kind_for(role, _S if MIRROR_OF[role] != "GRefMirror" else _V3)
             for role in MIRROR_OF if role != "nsamples"}
    assert kinds <= {"gref", "handle", "int_handle", "f64", "f32", "int"}, kinds
    assert runtime.kind_for("nsamples", TensorType((), "i32")) == "nsamples"


#: ``ScalarHandle`` and ``IntHandle`` are DISTINCT types that are
#: byte-identical in this build. A sidecar declares the dtype of every writable
#: slot (its ``mutables`` block) but NOT of an input plane, so a consumer
#: reading only the sidecar cannot tell an integer ``lookup``/``per_sample``
#: plane from a Real one — it resolves both to ``handle``. That is a real gap in
#: the field set, and it is harmless ONLY while the two PODs have the same
#: layout, which is a fact of the build and not an assumption: the row below
#: reads it off ``hawk._core.layout_sizes()`` and goes red the day it stops
#: holding, at which point the sidecar needs an input-dtype declaration.
_HANDLE_ALIASES = {"int_handle": "handle"}


def _by_bytes(descriptor):
    return [_HANDLE_ALIASES.get(kind, kind) for kind in descriptor]


def test_the_two_handle_pods_are_byte_identical_in_this_build():
    """The premise the row below rests on, read off the BINDING rather than
    assumed (``layout_sizes()`` fields 1 and 2, ``gref_abi.h``'s own order)."""
    from hawk import _core

    sizes = _core.layout_sizes()
    assert sizes[1] == sizes[2], (
        f"ScalarHandle is {sizes[1]} bytes and IntHandle {sizes[2]}: they are no "
        "longer interchangeable, so a sidecar that cannot declare an input "
        "plane's dtype can no longer describe one (needs the field)"
    )


@pytest.mark.parametrize("name,kernel", sorted({(c[0], c[1]) for c in D.cases(8)}),
                         ids=lambda p: p)
def test_the_walk_and_sidecar_derivations_of_the_descriptor_agree(built, name,
                                                                  kernel):
    """The two doors onto the same map. ``descriptor_for_walk`` reads the traced
    kernel; ``descriptor_for_sidecar`` reads what a CONSUMER actually holds — the
    JSON on disk, whose ``mutables`` dtype declarations are the context that
    resolves a ``mutable`` role's shape. If they disagreed about a slot's
    BYTE SHAPE, an artifact would marshal one way in the emitting process and
    another way in the consuming one.

    Compared by byte shape, not by kind NAME, and the difference is named rather
    than hidden: the only slots the two can spell differently are integer INPUT
    planes, which the sidecar carries no dtype for (see ``_HANDLE_ALIASES``).
    That is asserted too, so the allowance cannot silently widen."""
    sidecar = sidecar_of(built[name], kernel)
    from_sidecar = runtime.descriptor_for_sidecar(sidecar)
    from_walk = runtime.descriptor_for_walk(getattr(D, kernel).walk,
                                            scalar_type=sidecar["scalar_type"])
    assert _by_bytes(from_sidecar) == _by_bytes(from_walk), (
        f"{kernel}: the sidecar and the walk disagree about the by-value shapes "
        f"{sidecar['arg_spec']}\n  sidecar: {from_sidecar}\n  walk:    {from_walk}"
    )
    spec = [tuple(pair) for pair in sidecar["arg_spec"]]
    for (role, slot_name), a, b in zip(spec, from_sidecar, from_walk):
        assert a == b or (role not in ("mutable", "out")
                          and {a, b} == {"handle", "int_handle"}), (
            f"{kernel}: slot ({role}, {slot_name}) is {a!r} by the sidecar and "
            f"{b!r} by the walk — the only allowance is an INPUT plane whose "
            "integer dtype the sidecar does not declare"
        )
    assert len(from_walk) == len(spec), (
        "the descriptor must carry exactly one entry per arg_spec slot, in that "
        "ORDER — it is what indexes params[k]"
    )


def test_a_vector_mutable_is_a_mirror_and_a_scalar_one_a_handle(built):
    """HAWK's own worked example, on a REAL artifact: the 40-byte mirror and the
    32-byte handle, told apart by the sidecar's dtype declaration alone."""
    sidecar = sidecar_of(built["two_wire"], "two_wire")
    descriptor = runtime.descriptor_for_sidecar(sidecar)
    by_name = {name: descriptor[i]
               for i, (role, name) in enumerate(tuple(tuple(p) for p in
                                                      sidecar["arg_spec"]))}
    assert by_name["out_q"] == "gref" and by_name["out_w"] == "handle", by_name
