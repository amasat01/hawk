"""The ONE name check: every first-party name a set of sources imports (``from X import Y``, ``import X.Y``, attribute chains on such
aliases) resolves, each in a fresh subprocess.

usage: python tools/names_check.py --label NAME (--src DIR | --package NAME)... [--tree DIR] [--first-party a,b,...]
  --src DIR       a directory of sources to read (repeatable)
  --package NAME  an importable package whose directory is read (repeatable); it must resolve to a real package directory
  --tree DIR      the hawk tree whose packages resolve the names (default: the tree this file is in)
  --first-party   the packages whose names are resolved (default: hawk,raptor,lowering,simt,hawk_c,hawk_cuda)
Prints ``<label>: statements S; names N; unresolved U; toolchain-blocked T`` and every unresolved and blocked name.
Exit status: the number of unresolved names; 1000 when no statement was found, a package is not a real directory, or the resolver failed.
A name in a module that cannot be imported for want of a non-first-party library (a toolchain) is reported as blocked, never as resolved."""
import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
FIRST_PARTY = ("hawk", "raptor", "lowering", "simt", "hawk_c", "hawk_cuda")


def collect(sources, first_party=FIRST_PARTY):
    """``(statements, names)``: the count of first-party import statements and the list of ``(dotted module, attribute or None, where)``."""
    stmts, names = 0, []
    fp = set(first_party)
    for root in sources:
        for path in sorted(Path(root).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
            alias = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module.split(".")[0] in fp:
                    stmts += 1
                    for a in node.names:
                        names.append((node.module, None if a.name == "*" else a.name, f"{path}:{node.lineno}"))
                        alias[a.asname or a.name] = f"{node.module}.{a.name}"
                elif isinstance(node, ast.Import):
                    hit = [a for a in node.names if a.name.split(".")[0] in fp]
                    stmts += bool(hit)
                    for a in hit:
                        names.append((a.name, None, f"{path}:{node.lineno}"))
                        alias[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
            # a name the file also assigns or takes as an argument is a local variable, not the import
            local = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            local |= {a.arg for n in ast.walk(tree) if isinstance(n, ast.arguments)
                      for a in [*n.args, *n.kwonlyargs, *n.posonlyargs]}
            # a name used inside ``with pytest.raises(...)`` is expected not to resolve (a retired name that must refuse)
            expected = {id(n) for w in ast.walk(tree) if isinstance(w, ast.With)
                        and any(isinstance(i.context_expr, ast.Call) and getattr(i.context_expr.func, "attr", "") == "raises" for i in w.items)
                        for n in ast.walk(w)}
            for node in ast.walk(tree):     # attribute chains rooted at an imported first-party name
                if isinstance(node, ast.Attribute) and id(node) not in expected:
                    chain, cur = [], node
                    while isinstance(cur, ast.Attribute):
                        chain.append(cur.attr)
                        cur = cur.value
                    if isinstance(cur, ast.Name) and cur.id in alias and cur.id not in local:
                        full = ".".join([alias[cur.id], *reversed(chain)])
                        mod, _, attr = full.rpartition(".")
                        names.append((mod, attr, f"{path}:{node.lineno}"))
    return stmts, names


def package_dir(name, path=None):
    """The real directory of package ``name`` as ``path`` (a search path) resolves it; refused by name when it is not a directory
    holding ``__init__.py`` (an editable install's shim, a namespace portion)."""
    saved = sys.path[:]
    try:
        sys.path[:0] = list(path or ())
        spec = importlib.util.find_spec(name)
    finally:
        sys.path[:] = saved
    if spec is None:
        raise SystemExit(f"{name} is not importable here: install it or pass --src DIR")
    where = [Path(p) for p in (spec.submodule_search_locations or ())]
    real = [p for p in where if (p / "__init__.py").is_file()]
    if not real:
        raise SystemExit(f"{name} resolves to {[str(p) for p in where] or spec.origin}, which is not a package directory "
                         "(an editable install's shim?): pass --src DIR")
    return real[0]


PROBE = """
import importlib, json, sys
names, first_party = json.load(sys.stdin)        # on stdin: the list outgrows one command line
fp = set(first_party)
out = []


def _has(m, attr):
    try:
        getattr(m, attr)
        return True
    except Exception:       # a retired name raises its own error: it does not resolve either
        return False


def _reach(dotted):
    # the object ``dotted`` names: the longest importable module prefix, then attributes
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ModuleNotFoundError as exc:
            if exc.name and exc.name != ".".join(parts[:i]) and exc.name.split(".")[0] not in fp:
                raise
            continue
        for p in parts[i:]:
            obj = getattr(obj, p)
        return obj
    raise ModuleNotFoundError(f"No module named {parts[0]!r}", name=parts[0])


for mod, attr, where in names:
    kind = "unresolved"
    try:
        m = _reach(mod)
    except Exception as exc:
        if isinstance(exc, ImportError) and getattr(exc, "name", None) and exc.name.split(".")[0] not in fp:
            kind = "blocked"
        out.append([mod, attr, where, f"{type(exc).__name__}: {exc}", kind])
        continue
    if attr is not None and not _has(m, attr):
        try:
            importlib.import_module(mod + "." + attr)
        except Exception as exc:
            if isinstance(exc, ImportError) and getattr(exc, "name", None) and exc.name.split(".")[0] not in fp and exc.name != mod + "." + attr:
                kind = "blocked"
            out.append([mod, attr, where, f"no attribute {attr!r}" if kind == "unresolved" else f"{type(exc).__name__}: {exc}", kind])
print(json.dumps(out))
"""


def check(label, sources, tree=HERE, first_party=FIRST_PARTY, out=print):
    """Run the check over ``sources``; ``(status, unresolved, blocked)``."""
    stmts, names = collect(sources, first_party)
    if not stmts:
        out(f"{label}: FAILED: no first-party import statement found in {[str(s) for s in sources]}: the check read nothing")
        return 1000, [], []
    uniq = sorted({tuple(n) for n in names}, key=lambda t: (t[0], t[1] or "", t[2]))
    tree = Path(tree)
    paths = [str(tree), *(str(p) for p in sorted((tree / "extensions").glob("*"))), *(str(p) for p in sorted((tree / "components").glob("*")))]
    paths += [os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join(paths))
    flags = ["-P"] if sys.version_info >= (3, 11) else []
    with tempfile.TemporaryDirectory() as neutral:        # a working directory no package can come from
        res = subprocess.run([sys.executable, *flags, "-c", PROBE], input=json.dumps([uniq, list(first_party)]), env=env,
                             capture_output=True, text=True, cwd=neutral)
    if res.returncode:
        out(f"{label}: FAILED: the resolver died: {res.stderr}")
        return 1000, [], []
    rows = json.loads(res.stdout)
    bad = [r for r in rows if r[4] == "unresolved"]
    blocked = [r for r in rows if r[4] == "blocked"]
    out(f"{label}: statements {stmts}; names {len(uniq)}; unresolved {len(bad)}; toolchain-blocked {len(blocked)}")
    for mod, attr, where, why, _ in bad:
        out(f"UNRESOLVED {mod}.{attr or ''} at {where}: {why}")
    for mod, attr, where, why, _ in blocked:
        out(f"BLOCKED {mod}.{attr or ''} at {where}: {why}")
    return len(bad), bad, blocked


def main(argv):
    label, srcs, pkgs, tree, fp = "sources", [], [], HERE, FIRST_PARTY
    it = iter(argv)
    for a in it:
        if a == "--label":
            label = next(it)
        elif a == "--src":
            srcs.append(Path(next(it)))
        elif a == "--package":
            pkgs.append(next(it))
        elif a == "--tree":
            tree = Path(next(it))
        elif a == "--first-party":
            fp = tuple(next(it).split(","))
        else:
            print(__doc__)
            return 1000
    search = [str(tree), *os.environ.get("PYTHONPATH", "").split(os.pathsep)]
    srcs += [package_dir(p, [x for x in search if x]) for p in pkgs]
    if not srcs:
        print(f"{label}: FAILED: no source given")
        return 1000
    return check(label, srcs, tree, fp)[0]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
