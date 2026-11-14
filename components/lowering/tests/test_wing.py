# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""What the wing owns, tested without its front end: kinds, constructors, spellings, the log lowering, the entry list."""
import importlib
import sys

import pytest
from raptor.ir.kinds import is_a, kind_name
from raptor.ir.nodes import AccumWrite, Const, Leaf
from raptor.ir.types import TensorType
from raptor.ir import make

import lowering

F64, I32 = TensorType((), "f64"), TensorType((), "i32")


def test_hawk_is_absent():
    with pytest.raises(ImportError):
        importlib.import_module("hawk")
    assert not [m for m in sys.modules if m == "hawk" or m.startswith("hawk.")]


def test_the_twelve_kinds_are_declared_and_built_by_the_constructors():
    assert {k["name"] for k in lowering.MANIFEST["kinds"]} == {"team_store", "team_copy", "log_store", "scatter_log", "team_read", "box_store", "block_sum", "block_sum_write", "ordered_accum_write", "predicated_dispatch", "segmented_dispatch", "auto_dispatch"}
    x = Leaf("vocab_read", "per_sample", "x", F64)
    assert is_a(lowering.TeamStore("s", x), "team_store") and is_a(lowering.TeamCopy("s", x), "team_copy")


def test_every_declared_entry_resolves_and_every_module_imports_without_hawk():
    for part, names in lowering.MANIFEST["entries"].items():
        mod = importlib.import_module(f"lowering.{part}")
        assert all(hasattr(mod, n) for n in names), part
    for part in ("block", "box", "kinds", "lanes", "logs", "split", "team", "wide", "wrapper_text"):
        importlib.import_module(f"lowering.{part}")


def test_an_ordered_accumulate_lowers_to_the_pair_of_log_stores():
    from lowering.logs import log_ordered

    x, cell, key = (Leaf("vocab_read", "per_sample", n, t) for n, t in (("x", F64), ("cell", I32), ("key", I32)))
    plain = AccumWrite("g", make("mul", (x, x)), cell, F64)
    from lowering.logs import OrderedAccumWrite

    stated = OrderedAccumWrite("g", plain.value, plain.index, key, ((("key", None),)))
    out = log_ordered((stated, plain))
    assert [kind_name(s) for s in out] == ["log_store", "log_store", None] and out[2] is plain
    assert out[0].target == "g" and out[0].width == 1


def test_the_team_size_is_forced_by_an_argument_not_the_environment(monkeypatch):
    from lowering.team import threads

    monkeypatch.setenv("HAWK_TEAM_THREADS", "64")
    assert threads(10) == 128 and threads(10, 64) == 64 and threads(200) == 256
