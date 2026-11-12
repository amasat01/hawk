# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""No module outside ``hawk/targets`` names a target: the build, the package
step and the compile drivers read a row of ``TARGETS``, so a new target is a
new row and nothing else."""

from __future__ import annotations

import ast
from pathlib import Path

import hawk

#: ``(file relative to the scanned root, literal)``: a site that cannot read a row, and why. Empty at v0.7.6 and empty by rule: a
#: new entry is a design change, not a test edit.
ALLOWED: dict[tuple[str, str], str] = {}

#: the target ids and the variant names: a string that selects a target or a variant is read from its row, never spelled.
NAMES = ("cuda", "host", "x86", "arm", "cpu", "simt")


def _roots():
    hawk_root = Path(hawk.__file__).parent
    yield "hawk", hawk_root
    for pkg in sorted((hawk_root.parent / "components").glob("*/*/__init__.py")):
        yield f"components/{pkg.parent.parent.name}", pkg.parent


def _hits():
    for label, root in _roots():
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(root)
            if label == "hawk" and rel.parts[0] == "targets":
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Constant) and node.value in NAMES:
                    yield f"{label}/{rel}:{node.lineno}", (f"{label}/{rel}", node.value)


def test_no_target_or_variant_name_is_spelled_outside_the_targets_package():
    hits = list(_hits())
    stray = [where for where, key in hits if key not in ALLOWED]
    assert not stray, f"{len(stray)} target or variant name(s) spelled here; read the row instead: {stray}"
    stale = set(ALLOWED) - {key for _where, key in hits}
    assert not stale, f"stale allowance: {sorted(stale)}"
