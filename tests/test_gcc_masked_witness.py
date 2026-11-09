# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The masked-call witness (hawk/compile/witness/gcc_masked_call.cpp) and what
the host flags do with its verdict: a flagged GCC that runs it correctly gets
-DAETHER_HVM_MASKED=1; wrong code, an unrunnable witness, an explicit macro, an
unflagged compiler and clang change nothing; the verdict is probed once and
kept. Fakes stand in for the compiler version and the witness run; one test
runs the real local compilers when present."""

from __future__ import annotations

import os
import platform
import shutil

import pytest

from hawk.compile import toolchain as tc

DEFINE = "-DAETHER_HVM_MASKED=1"
FLAGS = ["-O3", "-march=native"]


@pytest.fixture
def probe(monkeypatch, tmp_path):
    """A fake host compiler (GCC 13.3 by default) and a counted witness run."""
    monkeypatch.setenv("HAWK_CACHE_DIR", str(tmp_path))
    tc.reset_identity_memo()
    state = {"macros": {"__GNUC__": "13", "__GNUC_MINOR__": "3"}, "verdict": "fixed", "runs": 0}
    monkeypatch.setattr(tc, "host_compiler", lambda: "/fake/g++")
    monkeypatch.setattr(tc, "compiler_macros", lambda c: state["macros"])
    monkeypatch.setattr(tc, "compiler_identity", lambda c: "fake-gcc-identity")

    def run(compiler, flags):
        state["runs"] += 1
        return state["verdict"], "fake"

    monkeypatch.setattr(tc, "_witness_run", run)
    yield state
    tc.reset_identity_memo()


def test_fixed_verdict_adds_the_define(probe):
    assert tc.masked_call_flag(FLAGS) == [DEFINE]


@pytest.mark.parametrize("verdict", ["miscompile", "unrun"])
def test_other_verdicts_add_nothing(probe, verdict):
    probe["verdict"] = verdict
    assert tc.masked_call_flag(FLAGS) == []


@pytest.mark.parametrize("macros", [
    {"__GNUC__": "13", "__GNUC_MINOR__": "4"}, {"__GNUC__": "14", "__GNUC_MINOR__": "3"},
    {"__GNUC__": "12", "__GNUC_MINOR__": "3"}, {"__GNUC__": "14", "__GNUC_MINOR__": "2", "__clang__": "1"}, {}])
def test_unflagged_compilers_are_not_probed(probe, macros):
    probe["macros"] = macros
    assert tc.masked_call_flag(FLAGS) == [] and probe["runs"] == 0


@pytest.mark.parametrize("explicit", ["-DAETHER_HVM_MASKED=0", "-DAETHER_HVM_MASKED=1", "-DAETHER_HVM_MASKED"])
def test_an_explicit_define_skips_the_probe(probe, explicit):
    assert tc.masked_call_flag(FLAGS, extra=("-DFOO", explicit)) == [] and probe["runs"] == 0


def test_the_verdict_is_probed_once_and_kept_on_disk(probe):
    assert tc.masked_call_flag(FLAGS) == [DEFINE] and tc.masked_call_flag(FLAGS) == [DEFINE]
    assert probe["runs"] == 1
    tc.reset_identity_memo()                      # a new process: the disk entry answers
    assert tc.masked_call_flag(FLAGS) == [DEFINE] and probe["runs"] == 1
    assert tc.masked_call_flag([*FLAGS, "-O2"]) == [DEFINE] and probe["runs"] == 2   # other flags, other verdict


def test_an_unrun_witness_is_not_stored(probe):
    probe["verdict"] = "unrun"
    tc.masked_call_flag(FLAGS)
    tc.reset_identity_memo()
    probe["verdict"] = "fixed"
    assert tc.masked_call_flag(FLAGS) == [DEFINE] and probe["runs"] == 2


def test_host_codegen_flags_carry_the_define_on_the_vector_math_profile(probe):
    if platform.machine().lower() in ("x86_64", "amd64"):   # the vector math is x86-64 only
        assert DEFINE in tc.host_codegen_flags("native-vector-math")
    else:
        assert DEFINE not in tc.host_codegen_flags("native-vector-math")
    assert DEFINE not in tc.host_codegen_flags("native")
    assert DEFINE not in tc.host_codegen_flags("native-vector-math", extra=("-DAETHER_HVM_MASKED=0",))


REAL = [("/home/mas/mambaforge/bin/x86_64-conda-linux-gnu-g++", "miscompile"),   # GCC 13.3.0
        ("/opt/rh/gcc-toolset-14/root/usr/bin/g++", "fixed")]                    # RH 14.2.1, carries the backport


@pytest.mark.parametrize("compiler,expected", REAL)
def test_real_compilers(compiler, expected, monkeypatch, tmp_path):
    if not os.path.exists(compiler) or shutil.which("taskset") is None:
        pytest.skip("this local compiler is not installed")
    monkeypatch.setenv("HAWK_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tc, "host_compiler", lambda: compiler)
    tc.reset_identity_memo()
    out = tc.masked_call_flag(tc.host_codegen_flags("native-vector-math", extra=("-DAETHER_HVM_MASKED=0",)))
    assert out == ([DEFINE] if expected == "fixed" else [])
    tc.reset_identity_memo()
