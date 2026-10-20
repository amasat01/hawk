# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The embedded C toolchain: discovery, strict flags, checks, shared build."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys

import pytest

from hawk.embedded import toolchain as tc
from hawk.ir import HawkError

VALID_C = """\
#include "t.h"

int t_add(int a, int b)
{
    return a + b;
}
"""
VALID_H = """\
#ifndef T_H
#define T_H

int t_add(int a, int b);

#endif
"""
# cppcheck sees one translation unit, so MISRA 8.7 flags any external function
# (nothing else references it); the clean file defines only main and a static function.
CLEAN_C = """\
#include "t.h"

static int t_square(int a)
{
    return a * a;
}

int main(void)
{
    return t_square(0);
}
"""
CLEAN_H = "#ifndef T_H\n#define T_H\n\n#endif\n"
PROMOTE_C = """\
#include <math.h>
#include "t.h"

float t_root(float x)
{
    return (float)sqrt((float)x);
}
"""
PROMOTE_H = "#ifndef T_H\n#define T_H\n\nfloat t_root(float x);\n\n#endif\n"


def test_hawk_cc_must_resolve(monkeypatch):
    monkeypatch.setenv("HAWK_CC", "/no/such/compiler-xyz")
    with pytest.raises(HawkError, match="/no/such/compiler-xyz"):
        tc.c_compiler()


def test_c_sibling_of_hawk_cxx(monkeypatch):
    if shutil.which("clang") is None:
        pytest.skip("clang not on PATH")
    monkeypatch.delenv("HAWK_CC", raising=False)
    monkeypatch.setenv("HAWK_CXX", "clang++")
    assert tc.c_compiler() == shutil.which("clang")


def test_parse_cppcheck_lines():
    text = (
        "t.c:12:5: style: Variable 'x' is assigned a value. [unreadVariable]\n"
        "noise line\n"
        "/tmp/a b/t.c:7:1: style: Rule 10.4 [misra-c2012-10.4]\n"
    )
    assert tc.parse_cppcheck(text) == [
        {"file": "t.c", "line": 12, "id": "unreadVariable",
         "text": "Variable 'x' is assigned a value."},
        {"file": "/tmp/a b/t.c", "line": 7, "id": "misra-c2012-10.4",
         "text": "Rule 10.4"},
    ]


def test_c_check_flags_rejects_other_std():
    assert tc.c_check_flags("c99")[0] == "-std=c99"
    with pytest.raises(HawkError):
        tc.c_check_flags("c17")


def test_run_checks_on_a_hand_written_c_file(monkeypatch):
    monkeypatch.delenv("HAWK_CXX", raising=False)
    result = tc.run_checks(CLEAN_C, CLEAN_H, "t")
    assert [c["std"] for c in result["cc"]] == ["c99", "c11"]
    assert all(c["rc"] == 0 for c in result["cc"])
    assert result["cxx_identity"] is None
    assert "||" in result["cc_identity"]
    cpp = result["cppcheck"]
    assert "skipped" in cpp or (cpp["rc"] == 0 and cpp["findings"] == [])
    assert result["seconds"] >= 0.0


def test_run_checks_requires_cppcheck_when_profile_says_so(monkeypatch):
    class Profile:
        require_cppcheck = True

    monkeypatch.setattr(tc, "cppcheck", lambda: None)
    with pytest.raises(HawkError, match="cppcheck"):
        tc.run_checks(VALID_C, VALID_H, "t", Profile())


def test_run_checks_refuses_a_double_promotion(tmp_path):
    with pytest.raises(HawkError) as err:
        tc.run_checks(PROMOTE_C, PROMOTE_H, "t")
    text = str(err.value)
    argv_line, _, stderr = text.partition("\n$ ")[2].partition("\n")
    assert "-Wdouble-promotion" in argv_line
    assert "double-promotion" in stderr and "precision" in stderr
    # The flag is what refuses it: the same file compiles without it.
    (tmp_path / "t.c").write_text(PROMOTE_C)
    (tmp_path / "t.h").write_text(PROMOTE_H)
    flags = [f for f in tc.c_check_flags("c99") if f != "-Wdouble-promotion"]
    done = subprocess.run([tc.c_compiler(), *flags, "t.c"], cwd=tmp_path,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_a_timeout_is_a_hawk_error(monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired(a[0], 600)

    monkeypatch.setattr(tc.subprocess, "run", boom)
    with pytest.raises(HawkError, match="timed out after 600s"):
        tc._run(["cc", "x.c"], ".")


def test_compile_shared_builds_a_loadable_library(tmp_path):
    c_path = tmp_path / "t.c"
    c_path.write_text(VALID_C)
    (tmp_path / "t.h").write_text(VALID_H)
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    out = tc.compile_shared(c_path, tmp_path / f"libt{suffix}")
    lib = ctypes.CDLL(str(out))
    lib.t_add.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.t_add.restype = ctypes.c_int
    assert lib.t_add(2, 3) == 5


def test_run_checks_passes_an_exported_function():
    if tc.cppcheck() is None:
        pytest.skip("cppcheck not installed")
    result = tc.run_checks(VALID_C, VALID_H, "t")
    assert result["cppcheck"]["findings"] == []
    assert result["cppcheck"]["suppressed"] == [d["rule"] for d in tc.DEVIATIONS]


def test_cppcheck_argv_carries_the_deviation_suppression():
    argv = tc.cppcheck_argv("cppcheck", "t.c", "/inc")
    assert "--suppress=misra-c2012-8.7" in argv
    assert argv.index("--suppress=missingIncludeSystem") < argv.index(
        "--suppress=misra-c2012-8.7")
