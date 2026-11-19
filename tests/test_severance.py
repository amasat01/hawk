# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The import directions of the ``hawk`` package, each as strong as its line, with no exemption list.

One scanner reads every module of the package: ``import`` and ``from`` statements at module level and inside functions,
classes and ``try`` blocks, and the string literals given to ``importlib.import_module``, ``__import__`` and
``importlib.util.find_spec``. Relative imports stay inside the package and are not edges.

1. hawk imports no ``raptor``, ``aether`` or ``ospreys`` package, at any level;
2. hawk imports no runtime (``eagle``) at import time;
3. at most one module of hawk names the runtime at all, and it does so lazily.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import NamedTuple

HAWK = Path(importlib.util.find_spec("hawk").origin).resolve().parent
IMPORT_CALLS = {"import_module", "__import__", "find_spec"}


class Edge(NamedTuple):
    """One import found in ``file``: the top-level package it names, and whether it runs when the module is imported."""

    file: str
    line: int
    top: str
    at_import: bool


def edges_of(file: str, tree: ast.Module) -> list:
    """Every absolute import of ``tree``, with whether it sits at module level outside any function."""
    found = []

    def visit(node, inside_function):
        for child in ast.iter_child_nodes(node):
            nested = inside_function or isinstance(
                child, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
            )
            if isinstance(child, ast.Import):
                found.extend(
                    Edge(file, child.lineno, a.name.split(".")[0], not inside_function)
                    for a in child.names
                )
            elif isinstance(child, ast.ImportFrom) and child.level == 0:
                found.append(
                    Edge(
                        file,
                        child.lineno,
                        (child.module or "").split(".")[0],
                        not inside_function,
                    )
                )
            elif (
                isinstance(child, ast.Call)
                and child.args
                and isinstance(child.args[0], ast.Constant)
            ):
                func = child.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr
                    if isinstance(func, ast.Attribute)
                    else ""
                )
                if name in IMPORT_CALLS and isinstance(child.args[0].value, str):
                    found.append(
                        Edge(
                            file,
                            child.lineno,
                            child.args[0].value.split(".")[0],
                            not inside_function,
                        )
                    )
            visit(child, nested)

    visit(tree, False)
    return found


def hawk_edges() -> list:
    """The imports of every module of the installed ``hawk`` package."""
    paths = sorted(HAWK.rglob("*.py"))
    assert len(paths) > 20, f"the scan found {len(paths)} modules under {HAWK}"
    return [
        e
        for p in paths
        for e in edges_of(p.relative_to(HAWK).as_posix(), ast.parse(p.read_text()))
    ]


def test_hawk_imports_no_raptor_aether_or_ospreys_package_at_any_level():
    """The family's libraries are reached through the compiled OSPREYs, so no Python package of theirs is named."""
    bad = [e for e in hawk_edges() if e.top in {"raptor", "aether", "ospreys"}]
    assert not bad, "\n".join(f"  hawk/{e.file}:{e.line} imports {e.top}" for e in bad)


def test_hawk_imports_no_runtime_at_import_time():
    """Importing ``hawk`` does not load ``eagle``: a machine without it can still author and build."""
    bad = [e for e in hawk_edges() if e.top == "eagle" and e.at_import]
    assert not bad, "\n".join(
        f"  hawk/{e.file}:{e.line} imports eagle at import time" for e in bad
    )


def test_hawk_has_one_lazy_importer_of_the_runtime():
    """At most one module of hawk names ``eagle`` at any level, so the runtime has one door."""
    files = sorted({e.file for e in hawk_edges() if e.top == "eagle"})
    assert len(files) <= 1, f"hawk names eagle from several modules: {files}"


def test_the_scan_tells_an_import_at_module_level_from_one_inside_a_function_and_reads_a_string_given_to_import_module():
    """Planted source: a top-level import and a literal-named import are at import time, the one in a function is not."""
    tree = ast.parse(
        "import eagle\nimport importlib\nimportlib.import_module('raptor.x')\n"
        "def door():\n    import aether\nfrom . import sibling\n"
    )
    found = {(e.top, e.at_import) for e in edges_of("planted.py", tree)}
    assert found == {
        ("eagle", True),
        ("importlib", True),
        ("raptor", True),
        ("aether", False),
    }
