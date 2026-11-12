"""List every hawk name eagle's Python package imports or reaches by attribute chain, and resolve each against the tree.

usage: python tools/eagle_hawk_names.py [--eagle DIR] [--tree DIR]
Defaults: the importable ``eagle`` package and the tree this file is in.
Prints the totals and every unresolved name; exit status is the number of unresolved names (0 = none)."""
import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def collect(eagle: Path):
    """``(statements, names)``: the count of hawk import statements and the set of ``(dotted module, attribute or None, where)``."""
    stmts, names = 0, []
    for path in sorted(eagle.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        alias = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module.split(".")[0] == "hawk":
                stmts += 1
                for a in node.names:
                    names.append((node.module, None if a.name == "*" else a.name, f"{path.name}:{node.lineno}"))
                    alias[a.asname or a.name] = f"{node.module}.{a.name}"
            elif isinstance(node, ast.Import):
                hit = [a for a in node.names if a.name.split(".")[0] == "hawk"]
                stmts += bool(hit)
                for a in hit:
                    names.append((a.name, None, f"{path.name}:{node.lineno}"))
                    alias[a.asname or "hawk"] = a.name if a.asname else "hawk"
        # a name the file also assigns or takes as an argument is a local variable, not the import
        local = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        local |= {a.arg for n in ast.walk(tree) if isinstance(n, ast.arguments)
                  for a in [*n.args, *n.kwonlyargs, *n.posonlyargs]}
        for node in ast.walk(tree):     # attribute chains rooted at an imported hawk name
            if isinstance(node, ast.Attribute):
                chain, cur = [], node
                while isinstance(cur, ast.Attribute):
                    chain.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name) and cur.id in alias and cur.id not in local and alias[cur.id].split(".")[0] == "hawk":
                    full = ".".join([alias[cur.id], *reversed(chain)])
                    mod, _, attr = full.rpartition(".")
                    names.append((mod, attr, f"{path.name}:{node.lineno}"))
    return stmts, names


PROBE = """
import importlib, json, sys
out = []
for mod, attr, where in json.loads(sys.argv[1]):
    try:
        m = importlib.import_module(mod)
    except Exception as exc:
        parent, _, last = mod.rpartition(".")
        try:    # the dotted path may be an attribute chain: a module, then names
            m = importlib.import_module(parent)
            m = getattr(m, last)
        except Exception:
            out.append([mod, attr, where, f"{type(exc).__name__}: {exc}"])
            continue
    if attr is not None and not hasattr(m, attr):
        try:
            importlib.import_module(mod + "." + attr)
        except Exception:
            out.append([mod, attr, where, f"no attribute {attr!r}"])
print(json.dumps(out))
"""


def main(argv):
    opts = dict(zip(argv[::2], argv[1::2]))
    tree = Path(opts.get("--tree", HERE))
    if "--eagle" in opts:
        eagle = Path(opts["--eagle"])
    else:
        spec = importlib.util.find_spec("eagle")
        if spec is None or not spec.submodule_search_locations:
            print("eagle is not importable here: install it or pass --eagle DIR")
            return 1000
        eagle = Path(list(spec.submodule_search_locations)[0])
    stmts, names = collect(eagle)
    uniq = sorted({tuple(n) for n in names}, key=lambda t: (t[0], t[1] or "", t[2]))
    paths = [str(tree / "extensions" / "hawk_c"), str(tree / "extensions" / "hawk_cuda"), str(tree)]
    paths += [os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join(paths))
    res = subprocess.run([sys.executable, "-c", PROBE, json.dumps(uniq)], env=env, capture_output=True, text=True, cwd=tree)
    if res.returncode:
        print(res.stderr)
        return 1000
    bad = json.loads(res.stdout)
    print(f"hawk import statements: {stmts}; distinct names checked: {len(uniq)}; unresolved: {len(bad)}")
    for mod, attr, where, why in bad:
        print(f"UNRESOLVED {mod}.{attr or ''} at {where}: {why}")
    return len(bad)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
