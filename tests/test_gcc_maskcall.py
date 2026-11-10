# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Which GCC releases miscompile a conditional call of a function with masked
vector variants (aether-dsc drops those variants there), and what the
vectorisation predicate reports on them. No compiler is run: the macro dicts
are injected."""

from __future__ import annotations

import platform

import pytest
from test_host_loop_interchange import arith, sin_chain

from hawk.compile import toolchain as tc
from hawk.ir.cost import kernel_cost
from hawk.ir.vectorise import vectorises

VM = "native-vector-math"
X86 = platform.machine().lower() in ("x86_64", "amd64")


def gcc(major, minor=0, patch=0):
    return {"__GNUC__": str(major), "__GNUC_MINOR__": str(minor),
            "__GNUC_PATCHLEVEL__": str(patch)}


TRUTH = ([(gcc(13, m, p), m < 4) for m in range(0, 6) for p in (0, 1)]
         + [(gcc(14, m, p), m < 3) for m in range(0, 5) for p in (0, 1)]
         + [(gcc(12, 3, 0), False), (gcc(12, 5, 0), False),
            (gcc(15, 0, 0), False), (gcc(15, 1, 1), False)])


@pytest.mark.parametrize("macros,affected", TRUTH)
def test_truth_table(macros, affected):
    assert tc.gcc_maskcall_affected(macros) is affected


def test_clang_and_unprobed_are_never_affected():
    assert tc.gcc_maskcall_affected({**gcc(13, 2, 0), "__clang__": "1"}) is False
    assert tc.gcc_maskcall_affected({}) is False


def _inject(monkeypatch, macros):
    monkeypatch.setattr(tc, "compiler_macros", lambda _compiler: macros)
    # the GCC-family answer is memoised per compiler: keep the faked one out of later tests
    monkeypatch.setattr(tc, "_GCC_FAMILY_MEMO", {})


def test_no_compile_flag_is_added(monkeypatch):
    _inject(monkeypatch, gcc(13, 2, 0))
    assert "-fno-tree-loop-if-convert" not in tc.host_flags(profile=VM)


def test_predicate_does_not_depend_on_the_compiler_release(monkeypatch):
    # the host blend leaves no conditional math call, so no release needs a masked variant
    for macros in (gcc(12, 3, 0), gcc(13, 2, 0), gcc(14, 2, 0), gcc(14, 4, 0)):
        _inject(monkeypatch, macros)
        assert vectorises(sin_chain.walk, VM) is X86     # the vector math is x86-64 only
        assert kernel_cost(sin_chain.walk, VM).vectorised is X86
        assert vectorises(arith.walk, VM) is True
    # the exact profiles run library math scalar, whatever the compiler
    assert vectorises(sin_chain.walk, "native") is False
