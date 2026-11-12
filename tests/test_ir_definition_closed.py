# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The IR definition is raptor's (``raptor.ir``, closed there: its own guard). Here: ``hawk.ir`` and ``hawk.types``
define nothing and are the same objects as raptor's, loading them loads no hawk module beyond themselves, and the
passes read no machine global."""

from __future__ import annotations

import ast
import importlib
import json
import pathlib
import pkgutil
import re
import subprocess
import sys

import raptor.errors
import raptor.ir

import hawk.ir

_ROOT = pathlib.Path(hawk.ir.__file__).parent


def test_old_paths_are_raptors_objects():
    for m in pkgutil.iter_modules(raptor.ir.__path__):
        if m.name != "segment":           # split: the names are raptor's, the rewrite is a pass
            assert importlib.import_module(f"hawk.ir.{m.name}") is importlib.import_module(f"raptor.ir.{m.name}")
    assert importlib.import_module("hawk.types") is raptor.ir.types
    assert hawk.ir.HawkError is raptor.errors.RaptorError and hawk.ir.Walk is raptor.ir.Walk


def test_hawk_ir_files_define_nothing():
    stray = []
    for path in sorted(_ROOT.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef) or (isinstance(node, ast.FunctionDef) and node.name != "__getattr__"):
                stray.append(f"{path.name}:{node.lineno} defines {node.name}")
    assert not stray, stray


def test_runtime_imports_stay_inside_the_alias():
    names = sorted(m.name for m in pkgutil.walk_packages(hawk.ir.__path__, "hawk.ir.")
                   if m.name not in ("hawk.ir.chain", "hawk.ir.slotwise", "hawk.ir.contraction",
                                     "hawk.ir.segment"))
    probe = ("import importlib, json, sys\n"
             "import hawk\n"
             "before = set(sys.modules)\n"
             f"for name in ['hawk.ir', 'hawk.types'] + {names!r}:\n    importlib.import_module(name)\n"
             "print(json.dumps(sorted(m for m in set(sys.modules) - before if m.startswith('hawk'))))")
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    loaded = json.loads(done.stdout.strip().splitlines()[-1])
    assert [m for m in loaded if not (m == "hawk.ir" or m == "hawk.types" or m.startswith("hawk.ir."))] == []
def test_passes_read_no_machine_global():
    """No pass imports ``hawk.targets`` (top level, nested, or by name through importlib) or rebinds a module
    global: the machine descriptor comes from the build entry (``policy.using(machine=...)``)."""
    stray = []
    for path in sorted((_ROOT.parent / "passes").glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                stray.append(f"{path.name}:{node.lineno} global {', '.join(node.names)}")
            elif isinstance(node, ast.Import):
                stray += [f"{path.name}:{node.lineno} imports {a.name}" for a in node.names if a.name.startswith("hawk.targets")]
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if (node.level == 2 and mod.split(".")[0] == "targets") or mod.startswith("hawk.targets") or (
                        node.level == 2 and not mod and any(a.name == "targets" for a in node.names)):
                    stray.append(f"{path.name}:{node.lineno} imports targets")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and re.fullmatch(r"hawk\.targets(\.\w+)*", node.value):
                stray.append(f"{path.name}:{node.lineno} names targets by string")
            elif isinstance(node, ast.Attribute) and node.attr == "import_module":
                stray.append(f"{path.name}:{node.lineno} importlib.import_module")
    assert not stray, stray
