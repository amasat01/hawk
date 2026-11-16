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
#: every folder of a wing's own tests under staging/*/tests (a folder that holds test files), relative to the repository root
_SUITES = sorted({p.parent.relative_to(_REPO).as_posix() for p in (_REPO / "staging").glob("*/tests/**/test_*.py")})


def test_there_is_a_wing_to_run():
    assert {"staging/aether/tests", "staging/raptor/tests", "staging/raptor/tests/base", "staging/raptor/tests/simt"} <= set(_SUITES), _SUITES


@pytest.mark.parametrize("suite", _SUITES)
def test_the_wings_own_suite_is_green_and_not_empty(suite):
    folder = _REPO / suite
    own = [f"--ignore={s}" for s in _SUITES if s != suite and s.startswith(suite + "/")]     # a nested folder is a suite of its own
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", suite, *own], cwd=_REPO,
                       capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    summary = next((l for l in reversed(r.stdout.strip().splitlines()) if re.search(r"\b(passed|failed|error|no tests ran)\b", l)), "(no summary line)")
    ran = re.search(r"(\d+) passed", summary)
    assert folder.is_dir() and r.returncode == 0 and ran and int(ran.group(1)) > 0, f"{suite}: rc={r.returncode}: {summary}"
