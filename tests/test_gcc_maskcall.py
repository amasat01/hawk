# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Which GCC releases miscompile a conditional call of a function with masked
vector variants (aether-dsc drops those variants there), and what the
vectorisation predicate reports on them. No compiler is run: the macro dicts
are injected."""

from __future__ import annotations

import pytest
from test_host_loop_interchange import arith, sin_chain

from hawk.compile import toolchain as tc
from hawk.ir.cost import kernel_cost
from hawk.ir.vectorise import interchanged_loops, vectorises

VM = "native-vector-math"


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


@pytest.mark.parametrize("macros,dropped", [
    (gcc(13, 2, 0), True), (gcc(14, 2, 0), True), (gcc(12, 3, 0), True),
    (gcc(13, 4, 0), False), (gcc(13, 3, 0), True), (gcc(14, 4, 0), False), (gcc(15, 1, 0), False),
    ({**gcc(4, 2, 1), "__clang__": "1"}, False)])
def test_conditional_math_scalar_only_under_native_vector_math(macros, dropped):
    assert tc.conditional_math_scalar(VM, macros=macros) is dropped
    for profile in ("native", "portable"):
        assert tc.conditional_math_scalar(profile, macros=macros) is False


def _inject(monkeypatch, macros):
    monkeypatch.setattr(tc, "compiler_macros", lambda _compiler: macros)


def test_no_compile_flag_is_added(monkeypatch):
    _inject(monkeypatch, gcc(13, 2, 0))
    assert "-fno-tree-loop-if-convert" not in tc.host_flags(profile=VM)


def test_predicate_turns_off_only_conditional_math_trip_loops(monkeypatch):
    assert interchanged_loops(sin_chain.walk) and interchanged_loops(arith.walk)
    _inject(monkeypatch, gcc(13, 2, 0))
    # an interchanged loop with a math call: scalar, under native-vector-math only
    assert vectorises(sin_chain.walk, VM) is False
    assert kernel_cost(sin_chain.walk, VM).vectorised is False
    # other profiles: the exact ones already run library math scalar, so nothing changes
    assert vectorises(sin_chain.walk, "native") is False
    # a loop with no math call is unchanged
    assert vectorises(arith.walk, VM) is True
    assert kernel_cost(arith.walk, VM).vectorised is True
    # an unaffected compiler leaves the math loop vectorised
    _inject(monkeypatch, gcc(14, 4, 0))
    assert vectorises(sin_chain.walk, VM) is True
