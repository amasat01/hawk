# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""One ast-based import scanner for the direction tests (design section 2a).

It reports every import of a source tree: ``import x`` and ``from x import y`` at module level and inside functions, classes
and try blocks (relative imports resolved to the package they name), and string literals handed to ``importlib.import_module``,
``__import__``, ``importlib.util.find_spec`` and ``importorskip``. A import_module / __import__ call whose argument is not a literal cannot be resolved:
it is reported with the target ``<non-literal>``, which belongs to no package.
"""

from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

_DYNAMIC = {"import_module", "__import__", "find_spec", "importorskip"}


@dataclass(frozen=True)
class Edge:
    path: Path
    line: int
    target: str      # dotted module name as written (relative imports resolved), or "<non-literal>"
    at_import: bool  # runs when the module is imported (module or class body, try blocks), not inside a function
    how: str         # "import" | "from" | "dynamic"

    @property
    def top(self) -> str:
        return self.target.split(".")[0]

    def where(self, base: Path | None = None) -> str:
        return f"{self.path.relative_to(base) if base else self.path}:{self.line}"


def _is_type_checking(test: ast.expr) -> bool:
    return "TYPE_CHECKING" in ast.unparse(test)


def _module_parts(path: Path, root: Path, pkg: str | None) -> list[str]:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return [*pkg.split('.'), *parts] if pkg else parts


def scan_file(path: Path, root: Path, pkg: str | None, *, include_type_checking: bool = True) -> list[Edge]:
    tree = ast.parse(path.read_text(), filename=str(path))
    package = _module_parts(path, root, pkg)
    if path.stem != "__init__":
        package = package[:-1]
    edges: list[Edge] = []

    def walk(node: ast.AST, in_func: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If) and _is_type_checking(child.test):
                if include_type_checking:
                    for sub in child.body:
                        walk_one(sub, in_func)
                for sub in child.orelse:
                    walk_one(sub, in_func)
                continue
            walk_one(child, in_func)

    def walk_one(n: ast.AST, in_func: bool) -> None:
        if isinstance(n, ast.Import):
            for a in n.names:
                edges.append(Edge(path, n.lineno, a.name, not in_func, "import"))
        elif isinstance(n, ast.ImportFrom):
            if n.level:
                if pkg is None:
                    return
                base = package[: len(package) - (n.level - 1)] if n.level > 1 else package
                target = ".".join([*base, *([n.module] if n.module else [])])
            else:
                target = n.module or ""
            edges.append(Edge(path, n.lineno, target, not in_func, "from"))
        elif isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
            if name in _DYNAMIC and n.args:
                a0 = n.args[0]
                if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
                    if not a0.value.startswith("."):
                        edges.append(Edge(path, n.lineno, a0.value, not in_func, "dynamic"))
                elif name != "find_spec":  # find_spec(variable) only probes, it imports nothing
                    edges.append(Edge(path, n.lineno, "<non-literal>", not in_func, "dynamic"))
        inner = in_func or isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        walk(n, inner)

    walk(tree, False)
    return edges


def scan_tree(root: Path, pkg: str | None, *, only: set[Path] | None = None, include_type_checking: bool = True) -> list[Edge]:
    out: list[Edge] = []
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts or (only is not None and p not in only):
            continue
        out.extend(scan_file(p, root, pkg, include_type_checking=include_type_checking))
    return out


def package_dir(name: str) -> Path:
    import importlib.util

    spec = importlib.util.find_spec(name)
    assert spec and spec.origin, f"piece {name!r} is not importable here: its direction test cannot run"
    return Path(spec.origin).parent


def is_stdlib(top: str) -> bool:
    return top in sys.stdlib_module_names


# ---- the pieces ----------------------------------------------------------------------------------------------------

def wing_homes() -> list[tuple[Path, str]]:
    """``[(folder, import name)]`` of every wing folder of the tree (``staging/<family>/wings/<name>``); the manifest beside it, not the folder, says the wing's family, name and parent."""
    out = [(d, f"{d.parent.parent.name}.wings.{d.name}") for d in sorted((REPO / "staging").glob("*/wings/*/")) if (d / "__init__.py").is_file()]
    return out


def wing_manifest(folder: Path) -> dict:
    f = folder / "wing.json"
    return json.loads(f.read_text()) if f.is_file() else {}


def osprey_homes() -> list[tuple[Path, str]]:
    """``[(folder, import name)]`` of every first-party OSPREY package of the tree."""
    out = [(d, f"ospreys.{d.name}") for d in sorted((REPO / "staging" / "ospreys").glob("*/")) if (d / "__init__.py").is_file()]
    return out


def _osprey_manifest(folder: Path) -> dict:
    f = folder / "osprey.json"
    return json.loads(f.read_text()) if f.is_file() else {}


def pieces() -> dict[str, tuple[Path, str | None]]:
    """piece id -> (source root, import name of the root or None): raptor, eagle, every wing (``wing:<family>.<name>``,
    from its manifest) and every OSPREY (``osprey:<name>``, from its manifest)."""
    out: dict[str, tuple[Path, str | None]] = {
        "raptor": (package_dir("raptor"), "raptor"),
        "eagle": (package_dir("eagle"), "eagle"),
    }
    for folder, module in wing_homes():
        m = wing_manifest(folder)
        out[f"wing:{m['family']}.{m['name']}"] = (folder, module)
    for folder, module in osprey_homes():
        out[f"osprey:{_osprey_manifest(folder)['name']}"] = (folder, module)
    return out


def piece_ids() -> list[str]:
    """The ids known without importing raptor or eagle (collection-time parametrisation)."""
    return ["raptor", "eagle",
            *(f"wing:{(m := wing_manifest(f))['family']}.{m['name']}" for f, _ in wing_homes()),
            *(f"osprey:{_osprey_manifest(f)['name']}" for f, _ in osprey_homes())]


@lru_cache(maxsize=None)
def piece_edges(piece: str) -> tuple[Edge, ...]:
    root, pkg = pieces()[piece]
    wings = root / "wings"      # a wing installed inside its family's package folder is a unit of its own, scanned under its wing id
    return tuple(e for e in scan_tree(root, pkg) if wings not in e.path.parents)


def manifest(piece: str) -> dict:
    """The wing.json of a wing or the osprey.json of an OSPREY (empty for a piece without one)."""
    root, _ = pieces()[piece]
    return wing_manifest(root) if piece.startswith("wing:") else _osprey_manifest(root)


def unit_module(family: str, name: str) -> str:
    """The import path of the wing ``name`` of ``family``, through the manifests' homes (never by spelling)."""
    for folder, module in wing_homes():
        m = wing_manifest(folder)
        if (m.get("family"), m.get("name")) == (family, name):
            return module
    raise KeyError(f"no wing {family}.{name} in this tree")


def wing_chain(family: str, name: str) -> list[str]:
    """``[name, parent, ...]`` of the wing and its parents, by the manifests."""
    out = []
    while name and name not in out:
        out.append(name)
        name = next((wing_manifest(f).get("parent") for f, _ in wing_homes()
                     if (wing_manifest(f).get("family"), wing_manifest(f).get("name")) == (family, name)), None)
    return out


def named_wings() -> dict[str, set[str]]:
    """OSPREY piece -> the import paths of the wings its rows name, through the chains of the leaves (the manifests are read as data)."""
    out = {}
    for pid in piece_ids():
        if pid.startswith("osprey:"):
            mods = set()
            for row in manifest(pid).get("backends", []):
                for family, leaf in (row.get("wings") or {}).items():
                    for leaf_name in ([] if leaf is None else [leaf] if isinstance(leaf, str) else list(leaf.values())):
                        mods |= {unit_module(family, n) for n in wing_chain(family, leaf_name)}
            out[pid] = mods
    return out


def writer_files(piece: str) -> set[Path] | None:
    """The writer side of a piece: every file of a wing; the ``emission_modules`` of an OSPREY."""
    root, pkg = pieces()[piece]
    if piece.startswith("wing:"):
        return None
    mods = manifest(piece).get("emission_modules", [])
    files = set()
    for m in mods:
        rel = Path(*m[len(pkg) + 1:].split(".")) if m.startswith(pkg + ".") else Path(*m.split("."))
        found = [cand for cand in (root / rel.with_suffix(".py"), root / rel / "__init__.py") if cand.is_file()]
        assert found, f"{piece}: the emission module {m!r} of its manifest is no file under {root}: the writer guard would see nothing"
        files.update(found)
    return files


def report(piece: str, rule: str, bad: list[Edge], all_counts: dict[str, int]) -> str:
    root, _ = pieces()[piece]
    base = root.parent
    head = f"{rule}: piece {piece}: {len(bad)} edge(s) NOT HELD"
    summary = "edges per piece: " + ", ".join(f"{k}={v}" for k, v in all_counts.items())
    first = "\n".join(f"  {e.where(base)}: {e.how} {e.target}" for e in bad[:5])
    by_file: dict[str, list[int]] = {}
    for e in bad:
        by_file.setdefault(str(e.path.relative_to(base)), []).append(e.line)
    full = "\n".join(f"  {f}: {','.join(map(str, ls))}" for f, ls in by_file.items())
    return f"{head}\n{summary}\nfirst:\n{first}\nall (file: lines):\n{full}"
