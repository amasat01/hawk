# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A node class that can appear in a unit lives in the IR definition (raptor.ir) or in a registered dialect, never in a
pass: no subclass of the IR ``Node`` is defined under ``hawk/passes``."""

from __future__ import annotations

import ast
import importlib
import pathlib
import pkgutil

import raptor.ir
from raptor.ir.nodes import Node

import hawk.backends
import hawk.passes

_PASSES = pathlib.Path(hawk.passes.__file__).parent


def _all_subclasses(cls):
    seen, todo = set(), [cls]
    while todo:
        for sub in todo.pop().__subclasses__():
            if sub not in seen:
                seen.add(sub)
                todo.append(sub)
    return seen


def test_no_node_class_is_defined_in_a_pass_statically():
    node_names = {c.__name__ for c in _all_subclasses(Node)} | {"Node"}
    stray = []
    for path in sorted(_PASSES.glob("*.py")):
        classes = [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.ClassDef)]
        grown = True
        while grown:
            grown = False
            for c in classes:
                bases = {b.id if isinstance(b, ast.Name) else getattr(b, "attr", "") for b in c.bases}
                if c.name not in node_names and bases & node_names:
                    node_names.add(c.name)
                    grown = True
        stray += [f"{path.name}:{c.lineno} {c.name}" for c in classes if c.name in node_names]
    assert not stray, stray


def test_no_node_class_is_defined_in_a_pass_at_run_time():
    for pkg in (hawk.passes, hawk.backends):
        for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            try:
                importlib.import_module(m.name)
            except ImportError:      # a backend whose toolchain is absent
                pass
    assert any(c.__name__ == "BlockSum" for c in _all_subclasses(Node))      # a kind of the lowering wing, declared when it loads
    stray = sorted(f"{c.__module__}.{c.__qualname__}" for c in _all_subclasses(Node) if c.__module__.startswith("hawk.passes"))
    assert not stray, stray
