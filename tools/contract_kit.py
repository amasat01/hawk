# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The contribution kit: ``python tools/contract_kit.py <package dir> [--manifest path] [--allow NAME,file.py:NAME]``
(``--allow``: module-level tables the package states are read-only).

Runs, on a backend or component package that is not in this repository, the scans this repository holds its own pieces to, and prints
one line per rule of the contributor contract (DESIGN 2c): PASS, FAIL (file:line, ...) or NOT CHECKABLE HERE (why). The exit code is
non-zero on a FAIL. The package is read as source (nothing of it is imported); hawk must be importable (the names of the installed
variants and backends come from it, through the entry points, not from a list).

The module-level-mutable scan lives here and is imported by ``tests/test_module_mutable_state.py``: one scanner, two users."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests"))

MUTABLE_CALLS = {"dict", "list", "set", "bytearray", "defaultdict", "OrderedDict", "deque",
                 "Counter", "ChainMap", "WeakSet", "WeakKeyDictionary", "WeakValueDictionary",
                 "array", "local", "ContextVar"}

#: Bases that make a class immutable by construction (or a value type): its instances are not shared mutable state.
_VALUE_BASES = {"NamedTuple", "Enum", "IntEnum", "Flag", "IntFlag", "StrEnum", "TypedDict", "Protocol", "Exception", "BaseException"}


def nonfrozen_classes(trees) -> set:
    """Names of the classes defined in the scanned sources whose instances can change: not a frozen dataclass, not a value base."""
    out = set()
    for tree in trees:
        for n in ast.walk(tree):
            if not isinstance(n, ast.ClassDef):
                continue
            frozen = any("frozen=True" in ast.unparse(d) for d in n.decorator_list)
            bases = {(b.id if isinstance(b, ast.Name) else b.attr if isinstance(b, ast.Attribute) else "") for b in n.bases}
            if not frozen and not (bases & _VALUE_BASES) and not any(b.endswith(("Error", "Exception")) for b in bases):
                out.add(n.name)
    return out


def is_mutable(value, classes=frozenset()) -> bool:
    """A literal or comprehension container, a call of a mutable container / ``ContextVar`` / ``threading.local``, or an instance
    of a non-frozen class made by a call (``KINDS = Registry()``)."""
    if isinstance(value, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)):
        return True
    if isinstance(value, ast.Call):
        f = value.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        return name in MUTABLE_CALLS or name in classes
    return False


def module_level(body):
    for node in body:
        if isinstance(node, (ast.If, ast.Try)):
            for part in (node.body, node.orelse, *(h.body for h in getattr(node, "handlers", ()))):
                yield from module_level(part)
        elif isinstance(node, ast.With):
            yield from module_level(node.body)
        else:
            yield node


def mutables_of(trees: dict, classes=None) -> list:
    """``[(key, name, line)]`` of the module-level mutables of ``trees`` (``{key: ast module}``)."""
    classes = nonfrozen_classes(trees.values()) if classes is None else classes
    out = []
    for key, tree in trees.items():
        for node in module_level(tree.body):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            if is_mutable(value, classes):
                out += [(key, t.id, node.lineno) for t in targets if isinstance(t, ast.Name) and t.id != "__all__"]
    return out


# ---- the checks ----------------------------------------------------------------------------------------------------

def _files(root: Path) -> list:
    return [p for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts and "tests" not in p.relative_to(root).parts]


def _manifest(root: Path, given) -> tuple:
    for path in ([Path(given)] if given else [root / "extension.json", root / "component.json"]):
        if path.is_file():
            return path, json.loads(path.read_text())
    return None, {}


def component_entries() -> dict:
    """``{component: {module: set(names)}}`` of every installed component (its entry point, its ``component.json``), found without importing it."""
    from importlib import util

    from hawk import variants

    out = {}
    for ep in variants.entry_points("raptor.components"):
        spec = util.find_spec(ep.value.partition(":")[0].split(".")[0])
        home = Path(next(iter(spec.submodule_search_locations))) if spec and spec.submodule_search_locations else None
        if home and (home / "component.json").is_file():
            out[ep.name] = {m: set(n) for m, n in json.loads((home / "component.json").read_text()).get("entries", {}).items()}
    return out


def _names_at(path: Path) -> dict:
    """``{line: [names]}`` of the ``from x import a, b`` statements of one file."""
    return {n.lineno: [a.name for a in n.names] for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.ImportFrom)}


def entries_violations(edges, entries: dict, own: str = "") -> list:
    """The edges that reach a component's module or name that its manifest does not declare as an entry: ``[(edge, sentence)]``.
    The component's own files (``own``) reach their own modules freely."""
    bad = []
    for e in edges:
        parts = e.target.split(".")
        if parts[0] not in entries or parts[0] == own:
            continue
        declared = entries[parts[0]]
        if len(parts) == 1:
            continue
        if parts[1] not in declared:
            bad.append((e, f"{e.target}: module {parts[1]!r} is not an entry of the {parts[0]!r} component"))
        elif len(parts) == 2 and e.how == "from":
            for name in _names_at(e.path).get(e.line, ()):
                if name not in declared[parts[1]]:
                    bad.append((e, f"{e.target}: {name!r} is not an entry of {parts[0]}.{parts[1]}"))
    return bad


def run(root: Path, manifest_path=None, allow=()) -> list:
    """``[(rule number, title, status, detail)]``; status is PASS, FAIL or NOT CHECKABLE HERE."""
    import _w25_scan as scan
    from hawk import registry, variants

    root = root.resolve()
    pkg = root.name
    path, man = _manifest(root, manifest_path)
    edges = scan.scan_tree(root, pkg)
    entries = component_entries()
    named = {c for b in man.get("backends", []) for c in b.get("components", [])} | set(man.get("components", []))
    own_leaves = {ep.value.partition(":")[0].split(".")[0] for ep in variants.entry_points("raptor.extensions")} - {pkg}
    where = lambda e: f"{e.path.relative_to(root.parent)}:{e.line}"      # noqa: E731
    rows = []

    def row(n, title, bad, why=None):
        rows.append((n, title, "NOT CHECKABLE HERE" if why else "FAIL" if bad else "PASS", why or ", ".join(bad[:6]) + (f" (+{len(bad) - 6})" if len(bad) > 6 else "")))

    hawk_edges = [f"{where(e)} imports {e.target}" for e in edges if e.top == "hawk"]
    outside = [f"{where(e)} imports {e.target}" for e in edges if e.top not in ("hawk", pkg, "raptor", "<non-literal>") and e.top not in named
               and not scan.is_stdlib(e.top) and e.top not in own_leaves]
    sideways = [f"{where(e)} imports the leaf {e.target}" for e in edges if e.top in own_leaves]
    row(1, "leaf: 2a.3 no hawk; 2a.4 stdlib + raptor + named components; no other leaf",
        [f"[2a.3] {x}" for x in hawk_edges] + [f"[2a.4] {x}" for x in outside] + [f"[leaf] {x}" for x in sideways])
    trees = {p: ast.parse(p.read_text()) for p in _files(root)}
    exported = {b["ref"].partition(":")[2] for b in man.get("backends", []) if ":" in b.get("ref", "")}       # the object the manifest names
    row(2, "no side effects at import: no module-level mutable state",
        [f"{k.relative_to(root.parent)}:{ln} {name}" for k, name, ln in mutables_of(trees)
         if name not in exported and name not in allow and f"{k.name}:{name}" not in allow])
    rows.append((3, "sideways identity", "NOT CHECKABLE HERE", "needs two targets built and compared at run time; a source scan cannot see it"))
    names = {ep.name for ep in variants.entry_points("raptor.variants")} | set(registry.backend_ids()) | {"host", "cuda", "c"}
    names -= {b.get("name") for b in man.get("backends", [])}
    names -= {man.get("name"), man.get("dialect")}                # a piece names itself: its own dialect or component name is not a variant choice
    lit = [f"{p.relative_to(root.parent)}:{n.lineno} {n.value!r}" for p, t in trees.items() for n in ast.walk(t)
           if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in names]
    row(4, "the variant is an argument: no literal variant or target names", lit)
    rows.append((5, "declared override points only", "NOT CHECKABLE HERE", "the class a backend subclasses is hawk's; the check needs it imported, and a kit reads source only"))
    rows.append((6, "the frozen subset passes unmodified", "NOT CHECKABLE HERE", "it is a run of hawk's frozen test subset against the backend, not a scan"))
    bad = []
    if path is None:
        bad.append("no extension.json or component.json: pass --manifest")
    else:
        if man.get("contract") != registry.CONTRACT:
            bad.append(f"{path.name}: contract {man.get('contract')!r}, hawk is {registry.CONTRACT!r}")
        bad += [f"{where(e)} {why}" for e, why in entries_violations(edges, entries, pkg)]
        if path.name == "extension.json":
            from hawk.registry import _read

            got = _read(man.get("name", pkg), root)
            if got.get("refusal"):
                bad.append(got["refusal"])
    row(7, "one home per fact: the manifest is valid and every component name used is a declared entry", bad)
    row(8, "pinned and guarded: the manifest states the contract it was written for", [] if man.get("contract") == registry.CONTRACT else [f"contract {man.get('contract')!r}"])
    return rows


def main(argv: list) -> int:
    args = [a for a in argv if not a.startswith("--")]
    given = argv[argv.index("--manifest") + 1] if "--manifest" in argv else None
    allow = argv[argv.index("--allow") + 1].split(",") if "--allow" in argv else []
    for value in (given, ",".join(allow)):
        if value in args:
            args.remove(value)
    if len(args) != 1:
        print("usage: contract_kit.py <package dir> [--manifest path]", file=sys.stderr)
        return 2
    rows = run(Path(args[0]), given, allow)
    for n, title, status, detail in rows:
        print(f"rule {n} {title}: {status}" + (f" ({detail})" if detail else ""))
    return 1 if any(r[2] == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
