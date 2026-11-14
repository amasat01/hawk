# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The words by which hawk finds its pieces, strict, no exemption list: four entry-point groups, two manifest file names, and the
keys a manifest starts with. Each test prints its count; a count only falls."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GROUPS = {"raptor.wings", "aether.wings", "eagle.wings", "ospreys"}
#: the facts data keeps its own group until its wings carry it (an exemption of one name; it leaves with the facts step)
PENDING = {"raptor.variants"}


def _files(name_or_glob: str):
    return sorted(p for p in REPO.rglob(name_or_glob) if ".git" not in p.parts and "__pycache__" not in p.parts
                  and "node_modules" not in p.parts)


def _groups():
    out = []
    for p in _files("pyproject.toml"):
        for g in tomllib.loads(p.read_text()).get("project", {}).get("entry-points", {}):
            out.append((p, g))
    for p in _files("entry_points.txt"):
        for line in p.read_text(errors="replace").splitlines():
            if line.startswith("[") and line.rstrip().endswith("]"):
                out.append((p, line.strip()[1:-1]))
    return out


def _fail(rule, bad):
    assert not bad, f"{rule}: {len(bad)} NOT HELD\n" + "\n".join(f"  {b}" for b in bad)


def test_only_the_final_entry_point_groups_are_declared():
    _fail("entry-point groups other than raptor.wings, aether.wings, eagle.wings, ospreys",
          [f"{p.relative_to(REPO)}: {g}" for p, g in _groups() if g not in GROUPS | PENDING])


def test_no_manifest_has_a_retired_file_name():
    _fail("a manifest named component.json or extension.json",
          [str(p.relative_to(REPO)) for n in ("component.json", "extension.json") for p in _files(n)])


def test_every_wing_manifest_starts_with_family_name_parent_and_has_no_components():
    bad = []
    for p in _files("wing.json"):
        m = json.loads(p.read_text())
        if list(m)[:3] != ["family", "name", "parent"]:
            bad.append(f"{p.relative_to(REPO)}: starts with {list(m)[:3]}")
        if "components" in m:
            bad.append(f"{p.relative_to(REPO)}: has the key 'components'")
    # every wing folder has one
    for d in [*REPO.glob("components/*/*/"), *REPO.glob("staging/*/wings/*/")]:
        if (d / "__init__.py").is_file() and not (d / "wing.json").is_file():
            bad.append(f"{d.relative_to(REPO)}: no wing.json")
    _fail("wing manifests", bad)


def test_every_osprey_row_names_its_wings_and_no_variant_or_components():
    bad = []
    for p in _files("osprey.json"):
        for row in json.loads(p.read_text()).get("backends", []):
            where = f"{p.relative_to(REPO)}: backend {row.get('name')!r}"
            if not isinstance(row.get("wings"), dict) or not row["wings"] or set(row["wings"]) - {"raptor", "aether", "eagle"}:
                bad.append(f"{where}: no 'wings' with keys within raptor, aether, eagle")
            bad += [f"{where}: has the key {k!r}" for k in ("variant", "components") if k in row]
    for d in [*REPO.glob("extensions/*/*/"), *REPO.glob("staging/osprey_*/"), *REPO.glob("tests/plugins/*/*/")]:
        if (d / "__init__.py").is_file() and not (d / "osprey.json").is_file():
            bad.append(f"{d.relative_to(REPO)}: no osprey.json")
    _fail("OSPREY manifests", bad)


def test_no_python_file_spells_a_retired_group():
    words = ("raptor." + "components", "raptor." + "extensions")
    _fail("a string literal naming a retired entry-point group",
          [f"{p.relative_to(REPO)}:{n}" for p in _files("*.py") for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1)
           if any(w in line for w in words)])
