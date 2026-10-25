# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The IR names no target: importing every ``hawk.ir`` module imports nothing
from ``hawk.emit`` or ``hawk.targets``."""

from __future__ import annotations

import json
import pkgutil
import subprocess
import sys

import hawk.ir


def test_importing_the_ir_pulls_in_no_emitter_and_no_target():
    names = sorted(m.name for m in pkgutil.walk_packages(hawk.ir.__path__, "hawk.ir."))
    probe = ("import importlib, json, sys\n"
             f"for name in {names!r}:\n    importlib.import_module(name)\n"
             "print(json.dumps(sorted(m for m in sys.modules "
             "if m.startswith(('hawk.emit', 'hawk.targets')))))")
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                          text=True, check=True)
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []
