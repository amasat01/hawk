# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Discovery from INSTALLED distributions only (no source tree on the path). The install environment is built by hand from wheels (the
hawk wheel, raptor-x86, raptor-arm, raptor-cuda, raptor-base, raptor-simt, raptor, and the family's other pieces); name its python in
``HAWK_INSTALL_PYTHON`` and these rows run the install smoke and the third-party leg in it. Without it they are skipped."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PYTHON = os.environ.get("HAWK_INSTALL_PYTHON")
TOOLS = Path(__file__).resolve().parent.parent / "tools"
pytestmark = pytest.mark.skipif(not PYTHON, reason="HAWK_INSTALL_PYTHON names no install environment")


def test_an_installed_hawk_builds_and_runs_a_host_kernel(tmp_path):
    done = subprocess.run([PYTHON, "-I", str(TOOLS / "w25_install_smoke.py")], capture_output=True, text=True, cwd=tmp_path,
                          env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    assert done.returncode == 0 and "INSTALL SMOKE OK host" in done.stdout, done.stdout[-1500:] + done.stderr[-1500:]
    assert "site-packages" in done.stdout.splitlines()[0]


def test_a_third_party_distribution_adds_a_wing_and_a_backend(tmp_path):
    done = subprocess.run([sys.executable, str(TOOLS / "w25_third_party_leg.py"), PYTHON, str(tmp_path)], capture_output=True, text=True)
    assert done.returncode == 0 and "THIRD-PARTY LEG OK" in done.stdout, done.stdout[-1500:] + done.stderr[-1500:]
