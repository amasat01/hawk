# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The `toolchain-absent` outcome of the suite, checked in a subprocess pytest over a tiny temp suite that loads this conftest.

A session with only passed and toolchain-absent rows exits 0 (pytest counts failures from the `failed` outcome only), and
"N toolchain-absent" is always printed. A row is toolchain-absent by the type of its refusal, never by its message text.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUITE = """
import pytest
from hawk.errors import HawkError, ToolchainAbsent

def test_ok(): pass
def test_typed_refusal(): raise ToolchainAbsent("no compiler found", toolchain="compiler")
def test_assertion(): assert 1 == 2
def test_unrelated_hawk_error(): raise HawkError("a different refusal")
def test_the_same_text_without_the_type_is_a_failure(): raise HawkError("no compiler found; set HAWK_CXX.")

@pytest.mark.needs_toolchain("linux")
def test_declared_need(): pass

@pytest.fixture
def needs_tool():
    raise ToolchainAbsent("no compiler found", toolchain="compiler")

def test_setup_refusal(needs_tool): pass

@pytest.fixture
def broken():
    raise ValueError("unrelated setup failure")

def test_setup_unrelated(broken): pass
"""


def run_suite(tmp_path: Path, suite: str) -> tuple[subprocess.CompletedProcess, dict]:
    """Run ``suite`` as a pytest session over this conftest; return the process and ``{test name: outcome word}``."""
    (tmp_path / "test_suite.py").write_text(textwrap.dedent(suite))
    path = os.pathsep.join(
        [str(REPO / "tests"), str(REPO), os.environ.get("PYTHONPATH", "")]
    )
    env = dict(os.environ, PYTHONPATH=path, PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "conftest", "-rA", "--rootdir", str(tmp_path), str(tmp_path / "test_suite.py")],
        capture_output=True, text=True, env=env, cwd=tmp_path, check=False,
    )  # fmt: skip
    found = re.findall(
        r"^(PASSED|FAILED|ERROR|TOOLCHAIN-ABSENT) \S*::(\w+)", done.stdout, re.M
    )
    return done, {name: word for word, name in found}


def test_the_outcomes_are_told_apart_by_the_type_of_the_refusal(tmp_path):
    """A typed toolchain refusal, in a test or in a fixture, and a declared need the host lacks are toolchain-absent; the rest keep their words."""
    done, outcome = run_suite(tmp_path, SUITE)
    assert outcome["test_ok"] == "PASSED", done.stdout[-1500:]
    assert outcome["test_typed_refusal"] == "TOOLCHAIN-ABSENT"
    assert outcome["test_setup_refusal"] == "TOOLCHAIN-ABSENT"
    assert outcome["test_declared_need"] == (
        "TOOLCHAIN-ABSENT" if not sys.platform.startswith("linux") else "PASSED"
    )
    assert outcome["test_the_same_text_without_the_type_is_a_failure"] == "FAILED"
    assert outcome["test_assertion"] == "FAILED"
    assert outcome["test_unrelated_hawk_error"] == "FAILED"
    assert outcome["test_setup_unrelated"] == "ERROR"
    assert done.returncode == 1  # the real failures decide the exit code


def test_a_session_of_passed_and_toolchain_absent_exits_zero_and_counts_them(tmp_path):
    """One passed and one toolchain-absent row: exit 0, the count line, and the passed row is not hidden."""
    done, outcome = run_suite(
        tmp_path,
        """
        from hawk.errors import ToolchainAbsent
        def test_a(): pass
        def test_b(): raise ToolchainAbsent("no compiler found", toolchain="compiler")
        """,
    )
    assert done.returncode == 0, done.stdout[-800:]
    assert re.search(r"^1 toolchain-absent", done.stdout, re.M)
    assert outcome == {"test_a": "PASSED", "test_b": "TOOLCHAIN-ABSENT"}
