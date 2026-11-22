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

This test previously failed under a plant: ``_mutable_dtype`` collapsed to a constant ``"scalar"`` --
``At index 0 diff: {'name': 'y', 'dtype': 'scalar', ...} != {'name': 'y',
'dtype': 'vector', ...}``. The plant was then removed.
"""

from __future__ import annotations

import _deployable as D
import numpy as np
import pytest
from conftest import sidecar_of

import hawk
from hawk.artifact import plan_view
from hawk import _ir

TensorType = _ir.TensorType

_V3 = TensorType((3,), "f64")
_M33 = TensorType((3, 3), "f64")
_S = TensorType((), "f64")

#: HAWK's shape name -> the ``eagle.roles`` tag(s) that shape corresponds to.
_SHAPE_OF_TAG = {"GREF_VEC": "GRefMirror", "GREF_MAT": "GRefMirror",
                 "HANDLE": "ScalarHandle", "WIDE_IN": "ScalarHandle",
                 "WIDE_OUT": "ScalarHandle", "ACCUM_OUT": "ScalarHandle",
                 "UNIFORM": "value", "NSAMPLES": "value"}


@pytest.mark.parametrize("name,kernel", sorted({(c[0], c[1]) for c in D.cases(8)}), ids=lambda p: p)
def test_the_stated_mirror_of_every_slot_is_what_eagles_classify_arg_resolves(built, name, kernel):
    """Each slot's ``arg_mirrors`` entry equals the shape eagle's argument class gives that role; a writable slot is
    classified by the sidecar's vector and matrix declarations, as eagle does."""
    from eagle.roles import classify_arg
    from eagle.sidecar import internal_roles

    sidecar = sidecar_of(built[name], kernel)
    mutables = {m["name"]: m["dtype"] for m in sidecar["mutables"]}
    vectors = frozenset(n for n, d in mutables.items() if d == "vector")
    matrices = frozenset(n for n, d in mutables.items() if d == "matrix")
    # eagle classifies the roles it names itself: the sidecar's one read role is read by the package's one rule first
    for role, slot in internal_roles(sidecar, name=name)["arg_spec"]:
        tag = classify_arg(role, slot, vec_mutables=vectors, mat_mutables=matrices)
        assert sidecar["arg_mirrors"][slot] == _SHAPE_OF_TAG[tag], (role, slot, tag)


def test_a_vector_mutable_carries_its_dtype_declaration(built):
    sc = sidecar_of(built["vec3"], "vec3_scale")
    assert sc["mutables"] == [
        {"name": "y", "dtype": "vector", "width": 3, "default": None}]
    view = plan_view(sc)
    assert view["vec_mutables"] == frozenset({"y"})
    # `arg_widths` is COMPLETE over every plane-bound slot, so the
    # plane_in input `x` carries its own width beside the mutable output `y`'s.
    assert view["arg_widths"] == {"x": 3, "y": 3}


def test_the_declaration_is_what_makes_the_box_forty_bytes(built):
    """The consequence names, measured: with the declaration the packed box
    is the 40-byte mirror; without it, the 32-byte handle the kernel would
    decode at the wrong offsets."""
    import ctypes

    from eagle.plan import _pack_args
    from eagle.sidecar import internal_roles

    from hawk import _contracts

    sc = sidecar_of(built["vec3"], "vec3_scale")
    view = plan_view(sc)
    arg_spec = internal_roles(view, name="vec3_scale")["arg_spec"]
    arrays = {"y": np.zeros((3, 8)), "x": np.ones((3, 8))}
    boxes, _addrs = _pack_args(arg_spec, arrays, {"a": 1.0}, 8, 1,
                               vec_mutables=view["vec_mutables"],
                               mat_mutables=view["mat_mutables"], dtype=np.float64)
    assert ctypes.sizeof(boxes[0]) == _contracts.GREF_MIRROR_SIZE == 40

    undeclared, _ = _pack_args(arg_spec, arrays, {"a": 1.0}, 8, 1,
                               dtype=np.float64)
    assert ctypes.sizeof(undeclared[0]) == _contracts.SCALAR_HANDLE_SIZE == 32


def test_a_scalar_and_a_matrix_mutable_declare_their_own_dtypes(built):
    multi = sidecar_of(built["multi"], "two_outputs")
    by_name = {m["name"]: m for m in multi["mutables"]}
    assert by_name["s"]["dtype"] == "scalar" and by_name["s"]["width"] == 1
    assert by_name["y"]["dtype"] == "vector" and by_name["y"]["width"] == 3
    assert plan_view(multi)["vec_mutables"] == frozenset({"y"})


def test_the_declared_kernel_computes_the_right_answer_through_the_mirror(built):
    x = np.arange(24, dtype=float).reshape(3, 8)
    y = np.zeros_like(x)
    hawk.run(hawk.load(built["vec3"].directory, "vec3_scale"), x=x, a=3.0, y=y)
    np.testing.assert_allclose(y, D.ref_vec3_scale(x, 3.0))


@pytest.mark.parametrize("cxx", ["g++", "clang++"])
def test_a_512_wide_vector_assign_compiles_and_runs_under_each_host_compiler(
        cxx, tmp_path, cache_dir, monkeypatch):
    """clang caps a fold expression at 256 operands; the whole-value assign
    must not grow a fold with the extent."""
    import shutil

    from hawk import Mutable, Param, Vector, kernel
    from hawk.artifact import build_bundle

    if shutil.which(cxx) is None:
        pytest.fail(f"no {cxx} on PATH; a skipped compile row is not a pass")
    monkeypatch.setenv("HAWK_CXX", cxx)

    @kernel
    def scale512(x: Vector[512], a: Param, y: Mutable[Vector[512]]):
        y = a * x

    bundle = build_bundle([scale512], tmp_path / cxx, targets=("host",),
                          cache_dir=str(tmp_path / ("cache_" + cxx)), host_profile="native")
    x = np.arange(512 * 8, dtype=float).reshape(512, 8)
    y = np.zeros_like(x)
    hawk.run(hawk.load(bundle.directory, "scale512"), x=x, a=3.0, y=y)
    np.testing.assert_array_equal(y, 3.0 * x)
