# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Runtime severance: no module under
``hawk/hawk/`` (the installable package) imports ``raptor`` or
``eagle`` at runtime.

This is a pure intra-repo, regex-over-source check: HAWK never imports the
packages it is severed from merely to prove it doesn't import them. Mirrors
the shape of ``eagle/python/tests/test_roles_raptor_repoint.py``'s structural
AST scan, but a plain source grep is enough here: ``grep`` finds no
``raptor``/``eagle`` import under ``hawk/hawk/``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

#: matches a top-level or indented ``import X`` / ``from X import ...`` for
#: any of the three severed packages, as either the leaf module or a dotted
#: submodule (``import raptor.schema`` counts; ``import raptorish`` must not).
_SEVERED_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+(raptor|eagle)(?:\.\S*)?\b", re.MULTILINE
)

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "hawk"


def _severed_imports_under(root: Path) -> list[str]:
    hits: list[str] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text()
        for m in _SEVERED_IMPORT_RE.finditer(text):
            hits.append(f"{path}:{text.count(chr(10), 0, m.start()) + 1}: {m.group(0).strip()}")
    return hits


@pytest.mark.repo_local
def test_package_root_exists():
    assert _PACKAGE_ROOT.is_dir(), f"expected the installable package at {_PACKAGE_ROOT}"


def test_no_raptor_eagle_import_under_hawk_hawk():
    hits = _severed_imports_under(_PACKAGE_ROOT)
    assert not hits, (
        "runtime severance broken — hawk/hawk/ must import zero "
        "raptor/eagle at runtime, found:\n" + "\n".join(hits)
    )
