# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The shared scanner for hardware facts held as code: an UPPER_CASE integer constant, at module level or in a class body, whose name
says a machine thing. Same rule as raptor's tests/test_hardware_free.py (``_MACHINE`` there plus the three words marked below): raptor's
copy and this one are to be unified at landing. Function-local arithmetic such as ``32 * 1024`` cannot be judged statically: NOT CHECKABLE."""

from __future__ import annotations

import ast
from pathlib import Path

MACHINE = {"warp", "bank", "banks", "lane", "lanes", "l1", "l2", "l3", "tile", "tiles", "width", "chunk", "chunks", "cache",
           "smem", "avx", "simd", "sse", "neon",
           "threads", "sm", "vector"}          # the three words raptor's list lacks


def machine_name(name: str) -> bool:
    return any(w in MACHINE or w.rstrip("0123456789") in MACHINE for w in name.lower().split("_"))


def _body_constants(body, rel):
    out = []
    for node in body:
        if isinstance(node, ast.ClassDef):
            out += _body_constants(node.body, rel)
            continue
        out += [f"{rel}:{node.lineno}: {t.id} = {v.value}" for t, v in _pairs(node)
                if isinstance(t, ast.Name) and t.id.isupper() and machine_name(t.id)
                and isinstance(v, ast.Constant) and isinstance(v.value, int) and not isinstance(v.value, bool)]
    return out


def _pairs(node):
    """``(target, value)`` of one statement; an unpacked ``A, B = 1, 2`` gives a pair per name."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
    value = getattr(node, "value", None)
    for t in targets:
        if isinstance(t, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)) and len(t.elts) == len(value.elts):
            yield from zip(t.elts, value.elts)
        else:
            yield t, value


def machine_constants(path: Path, rel: str) -> list:
    """``file:line: NAME = n`` for each machine-named integer constant at module or class level of one Python file."""
    return _body_constants(ast.parse(path.read_text()).body, rel)


def piece_constants(root: Path, base: Path) -> list:
    return [h for p in sorted(root.rglob("*.py")) if "tests" not in p.parts for h in machine_constants(p, p.relative_to(base).as_posix())]
