# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""HAWK's role -> mirror map equals eagle's, and the
sidecar carries the ``mutables`` declarations that map depends on.

Two halves, both load-bearing:

* **the map, table for table.** Each of the 12 canonical roles resolves to
  exactly one by-value ABI shape, and HAWK's answer must equal the tag
  ``eagle.roles.classify_arg`` returns — the SAME classifier all three of
  eagle's own consumers dispatch on, so a HAWK disagreement is a disagreement
  with every launch path at once.
* **the declarations.** ``mutable`` resolves BY CONTEXT: ``GREF_MAT`` if the
  name is in the sidecar's matrix-mutable declarations, ``GREF_VEC`` if in the
  vector ones, else ``HANDLE``. Without them a ``Mutable[Vector[3]]`` packs as a
  32-byte handle where the kernel expects a 40-byte mirror — so the row asserts
  the block, the ``plan_view`` projection of it, the tag it produces, the BYTE
  size of the box eagle then packs, and finally that the kernel computes the
  right answer through it.

This test previously failed. Two plants: ``MIRROR_OF["out"] = "ScalarHandle"`` --
``AssertionError: : HAWK packs 'out' as 'ScalarHandle' but eagle's
classify_arg resolves it to GREF_VEC (GRefMirror)`` -- and
``_mutable_dtype`` collapsed to a constant ``"scalar"`` --
``At index 0 diff: {'name': 'y', 'dtype': 'scalar', ...} != {'name': 'y',
'dtype': 'vector', ...}``. Both plants were then removed.
"""

from __future__ import annotations

import _deploy as L
import _deployable as D
import numpy as np
import pytest
from conftest import sidecar_of

from hawk.artifact import plan_view
from hawk.emit.aether import MIRROR_OF, mirror_of
from hawk.types import TensorType

_V3 = TensorType((3,), "f64")
_M33 = TensorType((3, 3), "f64")
_S = TensorType((), "f64")

#: HAWK's shape name -> the ``eagle.roles`` tag(s) that shape corresponds to.
_SHAPE_OF_TAG = {"GREF_VEC": "GRefMirror", "GREF_MAT": "GRefMirror",
                 "HANDLE": "ScalarHandle", "WIDE_IN": "ScalarHandle",
                 "WIDE_OUT": "ScalarHandle", "ACCUM_OUT": "ScalarHandle",
                 "UNIFORM": "value", "NSAMPLES": "value"}


def test_the_map_equals_eagles_classify_arg_for_all_twelve_roles():
    from eagle.roles import ROLES, classify_arg

    assert set(ROLES) == set(MIRROR_OF) | {"mutable"}, (
        "the map must cover the canonical role vocabulary exactly"
    )
    for role in sorted(ROLES):
        if role == "mutable":
            continue
        tag = classify_arg(role, role)
        assert MIRROR_OF[role] == _SHAPE_OF_TAG[tag], (
            f"HAWK packs {role!r} as {MIRROR_OF[role]!r} but eagle's "
            f"classify_arg resolves it to {tag} ({_SHAPE_OF_TAG[tag]})"
        )


def test_mutable_resolves_by_context_exactly_as_classify_arg_does():
    from eagle.roles import classify_arg

    assert mirror_of("mutable", _V3) == "GRefMirror"
    assert mirror_of("mutable", _M33) == "GRefMirror"
    assert mirror_of("mutable", _S) == "ScalarHandle"
    assert classify_arg("mutable", "y", vec_mutables=frozenset({"y"})) == "GREF_VEC"
    assert classify_arg("mutable", "y", mat_mutables=frozenset({"y"})) == "GREF_MAT"
    assert classify_arg("mutable", "y") == "HANDLE"


def test_a_vector_mutable_carries_its_dtype_declaration(built):
    sc = sidecar_of(built["vec3"], "vec3_scale")
    assert sc["mutables"] == [
        {"name": "y", "dtype": "vector", "width": 3, "default": None}]
    view = plan_view(sc)
    assert view["vec_mutables"] == frozenset({"y"})
    # `arg_widths` is COMPLETE over every plane-bound slot, so the
    # vec_in input `x` carries its own width beside the mutable output `y`'s.
    assert view["arg_widths"] == {"x": 3, "y": 3}


def test_the_declaration_is_what_makes_the_box_forty_bytes(built):
    """The consequence names, measured: with the declaration the packed box
    is the 40-byte mirror; without it, the 32-byte handle the kernel would
    decode at the wrong offsets."""
    import ctypes

    from eagle.plan import _pack_args

    from hawk import _contracts

    sc = sidecar_of(built["vec3"], "vec3_scale")
    view = plan_view(sc)
    arrays = {"y": np.zeros((3, 8)), "x": np.ones((3, 8))}
    boxes, _addrs = _pack_args(view["arg_spec"], arrays, {"a": 1.0}, 8, 1,
                               vec_mutables=view["vec_mutables"],
                               mat_mutables=view["mat_mutables"], dtype=np.float64)
    assert ctypes.sizeof(boxes[0]) == _contracts.GREF_MIRROR_SIZE == 40

    undeclared, _ = _pack_args(view["arg_spec"], arrays, {"a": 1.0}, 8, 1,
                               dtype=np.float64)
    assert ctypes.sizeof(undeclared[0]) == _contracts.SCALAR_HANDLE_SIZE == 32


def test_a_scalar_and_a_matrix_mutable_declare_their_own_dtypes(built):
    multi = sidecar_of(built["multi"], "two_outputs")
    by_name = {m["name"]: m for m in multi["mutables"]}
    assert by_name["s"]["dtype"] == "scalar" and by_name["s"]["width"] == 1
    assert by_name["y"]["dtype"] == "vector" and by_name["y"]["width"] == 3
    assert plan_view(multi)["vec_mutables"] == frozenset({"y"})


def test_a_handle_only_role_refuses_a_rank1_plane():
    """The one shape the map CANNOT express, refused loudly rather than packed
    at the wrong size."""
    from hawk.ir import HawkError

    for role in ("per_sample", "lookup", "terminated", "wide_in", "wide_out",
                 "accum_out"):
        with pytest.raises(HawkError, match="ScalarHandle"):
            mirror_of(role, _V3)


def test_the_declared_kernel_computes_the_right_answer_through_the_mirror(built):
    import eagle.exec as eexec
    from eagle import plan as eplan

    bundle = built["vec3"]
    plugin = L.host_plugin(bundle.directory, "vec3_scale",
                           sidecar_of(bundle, "vec3_scale"))
    x = np.arange(24, dtype=float).reshape(3, 8)
    np.testing.assert_allclose(
        eplan.plan(plugin, structure=eexec.HostTeam).run(x=x, a=3.0),
        D.ref_vec3_scale(x, 3.0))
