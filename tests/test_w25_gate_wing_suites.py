# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A wing's own tests run with hawk blocked (their conftest), so they cannot share a session with hawk's tests: each one runs
here in a subprocess, one test id per wing. A vanished suite, a suite that ran nothing, or a red one is red in hawk's gate."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_WINGS = sorted(p.parent.name for p in (_REPO / "components").glob("*/pyproject.toml"))


def test_there_is_a_wing_to_run():
    assert _WINGS, "no wing under components/: the gate would wire nothing"


@pytest.mark.parametrize("wing", _WINGS)
def test_the_wings_own_suite_is_green_and_not_empty(wing):
    suite = _REPO / "components" / wing / "tests"
    assert suite.is_dir() and any(suite.glob("test_*.py")), f"components/{wing} has no tests: {suite}"
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=_REPO / "components" / wing,
                       capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    summary = next((l for l in reversed(r.stdout.strip().splitlines()) if re.search(r"\b(passed|failed|error|no tests ran)\b", l)), "(no summary line)")
    ran = re.search(r"(\d+) passed", summary)
    assert r.returncode == 0 and ran and int(ran.group(1)) > 0, f"components/{wing}/tests: rc={r.returncode}: {summary}"
