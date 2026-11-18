# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Discovery from INSTALLED distributions only (no source tree on the path).

Name the python of a clean environment holding hawk and the three OSPREY wheels (``raptor-osprey``, ``raptor-osprey-cpu``,
``raptor-osprey-cuda``) and nothing else of ours in ``HAWK_INSTALL_PYTHON``: this row runs the smoke below in it. Without
it the row is skipped.
"""

from __future__ import annotations

import os
import subprocess

import pytest

PYTHON = os.environ.get("HAWK_INSTALL_PYTHON")
pytestmark = pytest.mark.skipif(not PYTHON, reason="HAWK_INSTALL_PYTHON names no install environment")

#: Run with ``python -I`` from a folder that holds no source tree: where hawk came from, the OSPREYs the entry points
#: name, and one host kernel built and run.
SMOKE = """
import os, sys, tempfile
import hawk
from hawk import ospreys

print(hawk.__file__)
work = tempfile.mkdtemp()
os.environ["HAWK_CACHE_DIR"] = os.path.join(work, "cache")
found = ospreys.found()
assert {"cpu", "cuda"} <= set(found), sorted(found)
assert all(o.folder.is_dir() for o in found.values())

@hawk.kernel
def k(x: hawk.Scalar, a: hawk.Param) -> hawk.Scalar:
    return a * x + 1.0

hawk.build([k], os.path.join(work, "out"), targets=("host",))
print("INSTALL SMOKE OK host")
"""


def test_an_installed_hawk_finds_the_osprey_wheels_and_builds_a_host_kernel(tmp_path):
    done = subprocess.run(
        [PYTHON, "-I", "-c", SMOKE],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
    )
    assert done.returncode == 0 and "INSTALL SMOKE OK host" in done.stdout, done.stdout[-1500:] + done.stderr[-1500:]
    assert "site-packages" in done.stdout.splitlines()[0]
