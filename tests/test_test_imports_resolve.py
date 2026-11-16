# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Every import in tests/ resolves: the module imports and the imported name exists in it.

A test file that imports a deleted function inside a body only fails when that body runs, and a body that needs
a device compiler does not run on a machine without one, so the breakage hides until a device machine runs it.
This scan reads EVERY ``.py`` under ``tests/`` (helpers and conftest included), takes every ``import`` and
``from X import Y`` at any depth, and resolves each against the interpreter the suite runs in.

WHAT IS COVERED: module-level imports, imports inside functions, classes and ``if``/``try`` blocks, and imports
inside a string constant that parses as a Python module (the literal source a test hands a child interpreter).
WHAT IS NOT: f-strings and strings built by concatenation or formatting (their source is not literal), and
``importlib.import_module("x")`` calls with a computed name.

An import that is optional on purpose (a third-party package a test guards with ``importorskip``, a marker or
a ``try ... except ImportError``) is not a failure when absent, and the guard is checked to exist: such a module
must be named in :data:`OPTIONAL` with the reason."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import textwrap
from pathlib import Path

TESTS = Path(__file__).parent
#: the tools folder the w25 tests put on ``sys.path`` themselves before importing from it
TOOLS = TESTS.parent / "tools"

#: ``(file, module)`` -> why an unresolvable import is the test's DATA, not a leftover: a literal source the test
#: writes to a temporary package to prove a guard sees an undeclared name.
BY_DESIGN = {
    ("test_w25_wing_entries.py", "raptor.wings.base.box"): "the guard must see an undeclared name (not_an_entry)",
    ("test_w25_wing_entries.py", "raptor.wings.base.nosuch"): "the guard must see an undeclared module",
    ("test_w25_wing_entries.py", "ospreys.cuda.emit"): "the guard must see a leaf importing another leaf's module",
}

#: third-party modules a test may import without their being installed: module -> how the test is guarded.
OPTIONAL = {
    "cupy": "pytest.mark.gpu / importorskip: a device library",
    "cuda": "pytest.mark.gpu / needs_toolchain: the NVRTC python bindings",
    "torch": "importorskip: the autodiff example compares against torch",
    "jax": "importorskip",
    "scipy": "importorskip",
    "numba": "importorskip",
}


def _local_names(path: Path) -> set:
    """The names a test-folder module binds at module level (read from its source: importing it would run it)."""
    names: set = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
    return names


def _imports(tree: ast.AST, offset: int = 0):
    """``(line, module, names, guarded)`` for every import in ``tree``, plus those in literal source strings."""
    guarded: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(
                h.type is not None and any(n in ast.unparse(h.type) for n in ("ImportError", "ModuleNotFoundError"))
                for h in node.handlers):
            guarded.update(id(n) for stmt in node.body for n in ast.walk(stmt))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno + offset, alias.name, (), id(node) in guarded
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.lineno + offset, node.module, tuple(a.name for a in node.names), id(node) in guarded
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and "import " in node.value \
                and "\n" in node.value:
            try:
                inner = ast.parse(textwrap.dedent(node.value))
            except SyntaxError:
                continue
            yield from _imports(inner, node.lineno - 1 + offset)


def _resolve(module: str, names: tuple) -> str | None:
    """``None`` when ``module`` imports and carries every name, else why not."""
    top = module.split(".")[0]
    local = next((d / f"{top}.py" for d in (TESTS, TOOLS) if (d / f"{top}.py").is_file()), None)
    if local is not None and "." not in module:
        have = _local_names(local)
        missing = [n for n in names if n != "*" and n not in have and not (TESTS / f"{n}.py").is_file()]
        return f"{module} has no {missing}" if missing else None
    try:
        mod = importlib.import_module(module)
    except Exception as err:                                  # noqa: BLE001  (the reason is the report)
        return f"{module} does not import: {type(err).__name__}: {str(err)[:120]}"
    missing = [n for n in names if n != "*" and not hasattr(mod, n)
               and not (hasattr(mod, "__path__") and importlib.util.find_spec(f"{module}.{n}") is not None)]
    return f"{module} has no {missing}" if missing else None


def scan() -> tuple[int, list, list]:
    """``(imports scanned, unresolved, optional absent)``, the last two as ``file:line  import  reason``."""
    count, bad, optional = 0, [], []
    for path in sorted(TESTS.glob("*.py")):
        tree = ast.parse(path.read_text())
        for line, module, names, guarded in _imports(tree):
            count += 1
            why = _resolve(module, names)
            if why is None:
                continue
            tag = f"{path.name}:{line}  {'from ' if names else 'import '}{module}{' import ' + ', '.join(names) if names else ''}  -- {why}"
            if (path.name, module) in BY_DESIGN:
                continue
            if module.split(".")[0] in OPTIONAL or guarded:
                optional.append(tag)
            else:
                bad.append(tag)
    return count, bad, optional


def test_every_import_under_tests_resolves():
    count, bad, _optional = scan()
    assert count > 500, f"the scan saw {count} imports: it is not reading the folder"
    assert not bad, f"{len(bad)} of {count} imports do not resolve:\n" + "\n".join(bad)
