# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""One door to the canonical form: hawk, the wings and the backends obtain the walk of a set of sinks through
``raptor.wings.base.cells.walk``, which declares the compensation planes of the summing sample sums. A direct call of raptor's
``canonical`` there would silently lose a plane and move a digest."""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOTS = ("hawk", "components", "staging", "extensions")
#: the door itself; every other module is refused
ALLOWED = {"staging/raptor/wings/base/cells.py": "the door"}


def _direct_calls(path: Path) -> list:
    tree = ast.parse(path.read_text())
    names = {"canonical"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("raptor"):
            names |= {a.asname for a in node.names if a.name == "canonical" and a.asname}
    return [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
            and ((isinstance(n.func, ast.Name) and n.func.id in names)
                 or (isinstance(n.func, ast.Attribute) and n.func.attr == "canonical"))]


def test_no_module_takes_the_canonical_form_but_through_the_door():
    bad = []
    for root in ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            if "build" in path.relative_to(REPO).parts:     # a copy an install leaves in the tree is not the source
                continue
            if rel not in ALLOWED:
                bad += [f"{rel}:{line}" for line in _direct_calls(path)]
    assert not bad, "raptor's canonical called directly, use raptor.wings.base.cells.walk:\n  " + "\n  ".join(bad)
