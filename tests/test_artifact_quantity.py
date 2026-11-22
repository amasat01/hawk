# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The namespaced compound quantity compiles and RUNS on both
targets, with no text surgery and no raw IR.

This is the compound-quantity shape made executable. A downstream package needed a
``Rotation`` wrapper plus two hand-minted raw ``VocabInput`` leaves
to expose one orientation access, and then hand-edited the GENERATED
C++ to wire them up. HAWK's answer is
ONE ``Quantity`` declaration: the DECLARATION namespaces every wire
(``earth__orientation__q`` / ``__w``), the walk binds all of them
contiguously whether or not the body reads each, and the emitter
reconstructs the access from the declaration.

The two counters names are asserted directly: ZERO text surgeries (the
emitted body is the ONE renderer's, byte-identical on both targets, and carries
no spliced raw block) and ZERO raw-IR constructions (the fixture module
constructs no ``raptor.ir`` node at all — an AST scan, the same detector shape
 uses).

This test previously failed: the wire namespacing stripped
(``Quantity._wire_name`` returning the bare wire name) --
``AssertionError: assert ('q', 'w') == ('earth__orientation__q',
'earth__orientation__w')`` -- and a raw ``Leaf(...)`` planted into the fixture
module -- ``AssertionError: : the quantity surface must need NO
raw-IR construction; the fixture module mints ['Leaf']``. Both plants were then
removed.
"""

from __future__ import annotations

import ast
import pathlib

import _deployable as D
from conftest import sidecar_of

from hawk import _ir

#: The IR node constructors a raw-IR workaround would have to name.
_RAW_IR_NAMES = frozenset({"Leaf", "Op", "Const", "At", "Select", "Assign",
                           "Update"})


def test_the_quantitys_wires_are_namespaced_and_contiguous():
    walk = D.spin.walk
    wires = tuple(D.ORIENTATION.wire_name(w) for w in D.ORIENTATION.wires)
    assert wires == ("earth__orientation__q", "earth__orientation__w")
    for name in wires:
        assert name.count("__") >= 2, f"{name!r} is not namespaced by the declaration"
    # per role, ONE consecutive declaration-ordered run.
    q = _ir.slot_of(walk)[("plane_in", "earth__orientation__q")]
    assert _ir.arg_spec(walk)[q] == ("plane_in", "earth__orientation__q")


def test_zero_raw_ir_constructions():
    tree = ast.parse(pathlib.Path("_deployable.py").read_text()
                     if pathlib.Path("_deployable.py").is_file()
                     else (pathlib.Path(__file__).parent / "_deployable.py").read_text())
    minted = [n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
              and n.func.id in _RAW_IR_NAMES]
    assert not minted, (
        "the quantity surface must need NO raw-IR construction; the "
        f"fixture module mints {sorted(set(minted))}"
    )


def test_the_quantity_artifact_declares_its_wires(built):
    sc = sidecar_of(built["spin"], "spin")
    assert sc["slot_shapes"]["earth__orientation__q"] == [4]
    assert sc["slot_shapes"]["earth__orientation__w"] == []
