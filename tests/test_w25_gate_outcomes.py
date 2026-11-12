# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The gate's `toolchain-absent` outcome, checked in a subprocess pytest over a tiny temp suite that loads this conftest.

Decision: a session with only passed and toolchain-absent rows exits 0 (pytest counts failures from the `failed` outcome only),
and "N toolchain-absent" is always printed. The manifest judges read the outcome word from the recorder; a comparison treats
passed -> toolchain-absent as a regression (tools/w25_manifest.py in the wave workspace).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_SUITE = '''
import pytest
from hawk_cuda.nvrtc import NvrtcUnavailable
from raptor.protocols.refusal import ToolchainAbsent
from raptor.errors import RaptorError

def test_ok(): pass
def test_typed_refusal(): raise NvrtcUnavailable("NVRTC could not be loaded")
def test_assertion(): assert 1 == 2
def test_unrelated_hawk_error(): raise RaptorError("a different refusal")
def test_typed_refusal_nvcc(): raise ToolchainAbsent("no nvcc found", toolchain="nvcc")
def test_the_same_text_without_the_type_is_a_failure(): raise RaptorError("no nvcc found; set $HAWK_NVCC.")

@pytest.mark.needs_toolchain("cppcheck")
def test_declared_need(): pass

@pytest.fixture
def needs_device():
    raise NvrtcUnavailable("NVRTC could not be loaded")

def test_setup_refusal(needs_device): pass

@pytest.fixture
def broken():
    raise ValueError("unrelated setup failure")

def test_setup_unrelated(broken): pass
'''


def _env(**extra) -> dict:
    path = [str(_REPO / "tests"), str(_REPO), str(_REPO / "extensions" / "hawk_cuda"), os.environ.get("PYTHONPATH", "")]
    return dict(os.environ, PYTHONPATH=os.pathsep.join(path), PYTHONDONTWRITEBYTECODE="1", **extra)


def _run(tmp_path: Path, suite: str, *, absent: bool = True) -> tuple[subprocess.CompletedProcess, dict]:
    (tmp_path / "test_suite.py").write_text(textwrap.dedent(suite))
    env = _env()
    if absent:      # nvcc and cppcheck forced absent for the probes (the env override is checked first by the product too)
        env["HAWK_NVCC"] = str(tmp_path / "no-such-nvcc")
        env["HAWK_CPPCHECK"] = str(tmp_path / "no-such-cppcheck")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "conftest", "-rA", "--rootdir", str(tmp_path),
                        str(tmp_path / "test_suite.py")], capture_output=True, text=True, env=env, cwd=tmp_path)
    states = {m[1]: m[0] for m in re.findall(r"^(PASSED|FAILED|ERROR|TOOLCHAIN-ABSENT) \S*::(\w+)", r.stdout, re.M)}
    return r, states


def test_the_outcomes_are_told_apart(tmp_path):
    r, s = _run(tmp_path, _SUITE)
    assert s["test_ok"] == "PASSED", r.stdout[-1500:]
    assert s["test_typed_refusal"] == "TOOLCHAIN-ABSENT"
    assert s["test_setup_refusal"] == "TOOLCHAIN-ABSENT"                       # a setup-phase refusal, not an error
    assert s["test_typed_refusal_nvcc"] == "TOOLCHAIN-ABSENT"
    assert s["test_the_same_text_without_the_type_is_a_failure"] == "FAILED"     # never by message text
    assert s["test_assertion"] == "FAILED"
    assert s["test_unrelated_hawk_error"] == "FAILED"
    assert s["test_setup_unrelated"] == "ERROR"
    assert s["test_declared_need"] == "TOOLCHAIN-ABSENT"                       # the marker raises the typed refusal from setup
    assert re.search(r"^4 toolchain-absent", r.stdout, re.M), r.stdout[-800:]
    assert r.returncode == 1                                                      # the real failures decide the exit code


def test_a_session_of_passed_and_toolchain_absent_exits_zero_and_counts_them(tmp_path):
    r, s = _run(tmp_path, '''
        from hawk_cuda.nvrtc import NvrtcUnavailable
        def test_a(): pass
        def test_b(): raise NvrtcUnavailable("NVRTC could not be loaded")
    ''')
    assert r.returncode == 0, r.stdout[-800:]
    assert re.search(r"^1 toolchain-absent", r.stdout, re.M)
    assert re.search(r"1 passed.*1 toolchain-absent", r.stdout), r.stdout[-300:]
    assert s == {"test_a": "PASSED", "test_b": "TOOLCHAIN-ABSENT"}
