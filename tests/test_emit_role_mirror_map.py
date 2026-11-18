# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The by-value shape each argument role travels in is stated by the lowering answer and carried in the sidecar.

The entry signature is generated from that shape, so a slot the runner packs another way is a 32-byte handle where the
kernel reads a 40-byte mirror: deterministic garbage, no crash. The sidecar's ``arg_mirrors`` states the shape of every
slot, and ``hawk.runtime`` binds from it. The table below is the independent oracle: the pinned map of eagle's argument
classes (``eagle/roles.py``).
"""

from __future__ import annotations

import _deployable as D
import pytest
from conftest import sidecar_of

from hawk import HawkError, _ir, runtime

TensorType = _ir.TensorType

_V3 = TensorType((3,), "f64")
_S = TensorType((), "f64")

#: The mirror each role rides; ``mutable`` and ``out`` resolve by rank (see ``_expected``).
_MIRROR_OF = {
    "out": "GRefMirror", "vec_in": "GRefMirror", "mat_in": "GRefMirror",
    "per_sample": "ScalarHandle", "lookup": "ScalarHandle",
    "terminated": "ScalarHandle", "wide_in": "ScalarHandle",
    "wide_out": "ScalarHandle", "accum_out": "ScalarHandle",
    "uniform": "value", "nsamples": "value",
}


def _expected(role, name, sidecar):
    """The mirror of slot ``name`` under the pinned map; a writable slot is a mirror when it is a vector or a matrix."""
    if role in ("mutable", "out"):
        shape = next((m["dtype"] for m in sidecar["mutables"] if m["name"] == name), "scalar")
        return "GRefMirror" if shape in ("vector", "matrix") else "ScalarHandle"
    return _MIRROR_OF[role]


@pytest.mark.parametrize("name,kernel", sorted({(c[0], c[1]) for c in D.cases(8)}), ids=lambda p: p)
def test_the_sidecar_states_the_pinned_mirror_of_every_slot(built, name, kernel):
    sidecar = sidecar_of(built[name], kernel)
    got = {slot: sidecar["arg_mirrors"][slot] for _role, slot in sidecar["arg_spec"]}
    assert got == {slot: _expected(role, slot, sidecar) for role, slot in sidecar["arg_spec"]}


def test_every_mirror_has_a_core_descriptor_kind():
    """A mirror with no ``_core`` kind would fail at bind time, after the artifact was built, loaded and declared good."""
    kinds = {runtime.kind_for(role, _V3 if mirror == "GRefMirror" else _S, mirror)
             for role, mirror in _MIRROR_OF.items() if role != "nsamples"}
    assert kinds <= {"gref", "handle", "int_handle", "f64", "f32", "int"}, kinds
    assert runtime.kind_for("nsamples", TensorType((), "i32"), "value") == "nsamples"


def test_a_sidecar_without_the_mirror_of_a_slot_is_refused_by_name():
    sidecar = {"arg_spec": [["out", "y"]], "arg_mirrors": {}, "scalar_type": "float64",
               "mutables": [{"name": "y", "dtype": "scalar", "default": None}]}
    with pytest.raises(HawkError, match="arg_mirrors"):
        runtime.descriptor_for_sidecar(sidecar)


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


def test_a_vector_mutable_is_a_mirror_and_a_scalar_one_a_handle(built):
    """HAWK's own worked example, on a REAL artifact: the 40-byte mirror and the
    32-byte handle, told apart by the sidecar's dtype declaration alone."""
    sidecar = sidecar_of(built["two_wire"], "two_wire")
    descriptor = runtime.descriptor_for_sidecar(sidecar)
    by_name = {name: descriptor[i]
               for i, (role, name) in enumerate(tuple(tuple(p) for p in
                                                      sidecar["arg_spec"]))}
    assert by_name["out_q"] == "gref" and by_name["out_w"] == "handle", by_name
