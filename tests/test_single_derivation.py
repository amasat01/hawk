# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Nothing outside ``hawk/ir/`` enumerates the walk.

``Walk`` owns the ONE traversal: every consumer derives from
``Walk.arg_spec`` / ``Walk.slot_of``, and a module that re-enumerates
``Walk.leaves`` or ``Walk.order`` is a second, redundant derivation. The
detector is an AST scan over the installable package
``hawk/hawk/`` excluding ``hawk/hawk/ir/``; the test suite itself is out of
scope, since a test asserting on ``walk.order`` is an observer, not a consumer
that ships.

Two assertions, deliberately: the first is as worded (an ITERATION over
``.leaves``/``.order`` — ``for``/comprehension iterables, iterating-builtin
arguments, ``*``-unpacking), the second is strictly stronger and closes the
aliasing hole the first has by construction (``x = w.order`` then ``for n in x``)
by refusing the attribute ACCESS itself.

This test previously failed when run against a planted real module
``hawk/hawk/_r4_probe.py`` carrying ``for leaf in walk.leaves:`` — both
assertions fired, naming the file and line; the probe was then removed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: The fields no consumer may enumerate.
_WALK_FIELDS = {"leaves", "order"}

#: Calls that consume an iterable whole.
_ITERATING_BUILTINS = {
    "iter", "list", "tuple", "set", "sorted", "enumerate", "reversed", "zip",
    "map", "filter", "any", "all", "sum", "len", "next", "frozenset",
}

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "hawk"
_IR_ROOT = _PACKAGE_ROOT / "ir"


def _scanned_files() -> list[Path]:
    return [p for p in sorted(_PACKAGE_ROOT.rglob("*.py")) if _IR_ROOT not in p.parents]


def _is_walk_field(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr in _WALK_FIELDS


def _hits(iteration_only: bool) -> list[str]:
    hits: list[str] = []
    for path in _scanned_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            targets: list[ast.AST] = []
            if isinstance(node, (ast.For, ast.AsyncFor)):
                targets = [node.iter]
            elif isinstance(node, ast.comprehension):
                targets = [node.iter]
            elif isinstance(node, ast.Starred):
                targets = [node.value]
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name)\
                    and node.func.id in _ITERATING_BUILTINS:
                targets = list(node.args)
            elif not iteration_only and isinstance(node, ast.Attribute):
                targets = [node]
            for t in targets:
                if _is_walk_field(t):
                    hits.append(f"{path}:{getattr(t, 'lineno', '?')}: .{t.attr}")  # type: ignore[union-attr]
    return sorted(set(hits))


@pytest.mark.repo_local
def test_scan_covers_the_package_and_excludes_ir():
    scanned = _scanned_files()
    assert scanned, f"the AST scan found no modules under {_PACKAGE_ROOT}"
    assert not any(_IR_ROOT in p.parents for p in scanned)


def test_no_module_outside_hawk_ir_iterates_walk_leaves_or_order():
    hits = _hits(iteration_only=True)
    assert not hits, (
        "only hawk/ir/ may enumerate Walk.leaves / Walk.order; every other "
        "consumer derives from Walk.arg_spec / Walk.slot_of. Found:\n" + "\n".join(hits)
    )


def test_no_module_outside_hawk_ir_even_names_walk_leaves_or_order():
    hits = _hits(iteration_only=False)
    assert not hits, (
        "(strict form, closing the aliasing hole): outside hawk/ir/ nothing may "
        "read Walk.leaves / Walk.order at all. Found:\n" + "\n".join(hits)
    )
