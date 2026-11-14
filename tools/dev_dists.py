# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Source-tree mode: make the in-tree distributions visible to ``importlib.metadata`` without installing anything.

``python tools/dev_dists.py <directory>`` writes, into ``<directory>`` (put it on ``PYTHONPATH``), for every in-tree distribution
(hawk's own pyproject, ``variants/``, ``components/*``, ``staging/packaging/*``): a ``<name>-<version>.dist-info`` folder (METADATA, entry_points.txt, RECORD)
generated FROM the distribution's pyproject, so an entry point is declared once, there; and a link per package that makes it
importable from that directory. hawk finds variants, wings and OSPREYs through these dist-info folders exactly as it
does through an installed wheel: there is no second discovery path. Standard library only (Python 3.11+ for ``tomllib``)."""

from __future__ import annotations

import os
import re
import sys
import tomllib
from pathlib import Path

TREE = Path(__file__).resolve().parent.parent


def pyprojects(tree: Path = TREE) -> list:
    return [p for p in [tree / "pyproject.toml", tree / "variants" / "pyproject.toml", *sorted((tree / "components").glob("*/pyproject.toml")),
                                   *sorted((tree / "staging" / "packaging").glob("*/pyproject.toml"))]
            if p.is_file()]


def _packages(doc: dict, home: Path) -> dict:
    """``{import name: directory}`` of the packages one pyproject ships."""
    tool = doc.get("tool", {})
    if "setuptools" in tool:
        mapping = tool["setuptools"].get("package-dir", {})
        names = tool["setuptools"].get("packages", [])
        return {n: (home / mapping[n] if n in mapping else home / n.replace(".", "/")) for n in names}
    listed = tool.get("scikit-build", {}).get("wheel", {}).get("packages", [])
    return {Path(p).name: home / p for p in listed}


def _link(target: Path, where: Path) -> None:
    where.parent.mkdir(parents=True, exist_ok=True)
    if where.is_symlink() or where.is_file():
        if where.is_symlink() and Path(os.readlink(where)) == target:
            return
        tmp = where.with_name(where.name + ".new")
        tmp.unlink(missing_ok=True)
        os.symlink(target, tmp)
        os.replace(tmp, where)
    elif not where.exists():
        os.symlink(target, where)


def write(pyproject: Path, out: Path) -> str:
    """The dist-info and links of one distribution; returns its name."""
    doc = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = doc["project"]
    name = project["name"]
    version = project.get("version", "0.0.0.dev0")
    folder = out / f"{re.sub(r'[-_.]+', '_', name)}-{version}.dist-info"
    folder.mkdir(parents=True, exist_ok=True)
    meta = [f"Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
    meta += [f"Requires-Dist: {d}" for d in project.get("dependencies", [])]
    (folder / "METADATA").write_text("\n".join(meta) + "\n", encoding="utf-8")
    lines = []
    for group, table in sorted(project.get("entry-points", {}).items()):
        lines += [f"[{group}]", *(f"{k} = {v}" for k, v in sorted(table.items())), ""]
    (folder / "entry_points.txt").write_text("\n".join(lines), encoding="utf-8")
    (folder / "INSTALLER").write_text("dev_dists\n", encoding="utf-8")
    (folder / "RECORD").write_text("", encoding="utf-8")
    for package, directory in sorted(_packages(doc, pyproject.parent).items()):
        if package == "hawk" or "." in package and (out / package.split(".")[0]).is_symlink():
            continue                                  # hawk itself is the caller's own path; a sub-package of a linked package is inside it
        place = out / Path(*package.split("."))
        if "." not in package and directory.resolve() == pyproject.parent.resolve():      # a package whose root is the distribution's folder
            place.mkdir(exist_ok=True)
            (place / "__init__.py").write_text("")
        else:
            _link(directory.resolve(), place)
    return name


def main(argv: list) -> int:
    if len(argv) != 1:
        print("usage: dev_dists.py <directory>", file=sys.stderr)
        return 2
    out = Path(argv[0]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for path in pyprojects():
        print(write(path, out), "<-", path.relative_to(TREE))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
