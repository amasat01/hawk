# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""What the wing owns, tested without its front end: kinds, constructors, spellings, the entry list."""
import importlib
import sys

import pytest
from raptor.ir.kinds import is_a, kind_name
from raptor.ir.nodes import Leaf
from raptor.ir.types import TensorType

import raptor.wings.simt

F64 = TensorType((), "f64")


def test_hawk_is_absent():
    with pytest.raises(ImportError):
        importlib.import_module("hawk")
    assert not [m for m in sys.modules if m == "hawk" or m.startswith("hawk.")]


def test_the_two_warp_kinds_are_declared_and_built_by_the_constructors():
    assert {k["name"] for k in raptor.wings.simt.MANIFEST["kinds"]} == {"warp_lane", "warp_store"}
    x = Leaf("vocab_read", "per_sample", "x", F64)
    assert kind_name(raptor.wings.simt.WarpLane()) == "warp_lane" and is_a(raptor.wings.simt.WarpStore("y", x), "warp_store")


def test_every_declared_entry_resolves_and_every_module_imports_without_hawk():
    for part, names in raptor.wings.simt.MANIFEST["entries"].items():
        mod = importlib.import_module(f"raptor.wings.simt.{part}")
        assert all(hasattr(mod, n) for n in names), part
    for part in raptor.wings.simt.MANIFEST["emission_modules"]:
        importlib.import_module(part)


def test_the_wing_names_the_one_it_builds_on():
    assert list(raptor.wings.simt.MANIFEST)[:3] == ["family", "name", "parent"] and raptor.wings.simt.MANIFEST["parent"] == "base"
