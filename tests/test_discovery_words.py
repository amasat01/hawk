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
          [f"{p.relative_to(REPO)}: {g}" for p, g in _groups() if g not in GROUPS])


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
    for d in [*REPO.glob("staging/*/wings/*/")]:
        if (d / "__init__.py").is_file() and not (d / "wing.json").is_file():
            bad.append(f"{d.relative_to(REPO)}: no wing.json")
    _fail("wing manifests", bad)


def test_every_osprey_row_names_its_wings_and_no_retired_key():
    bad = []
    for p in _files("osprey.json"):
        for row in json.loads(p.read_text()).get("backends", []):
            where = f"{p.relative_to(REPO)}: backend {row.get('name')!r}"
            if not isinstance(row.get("wings"), dict) or not row["wings"] or set(row["wings"]) - {"raptor", "aether", "eagle"}:
                bad.append(f"{where}: no 'wings' with keys within raptor, aether, eagle")
            bad += [f"{where}: has the key {k!r}" for k in ("vari" + "ant", "components") if k in row]
    for d in [*REPO.glob("staging/ospreys/*/"), *REPO.glob("tests/plugins/*/*/")]:
        if (d / "__init__.py").is_file() and not (d / "osprey.json").is_file():
            bad.append(f"{d.relative_to(REPO)}: no osprey.json")
    _fail("OSPREY manifests", bad)


def test_no_python_file_spells_a_retired_group():
    words = ("raptor." + "components", "raptor." + "extensions")
    _fail("a string literal naming a retired entry-point group",
          [f"{p.relative_to(REPO)}:{n}" for p in _files("*.py") for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1)
           if any(w in line for w in words)])


# ---- the four renamed packages: no old import name, no old folder, no old dialect name -----------------------------------------

OLD = ("low" + "ering", "simt", "hawk_c", "osprey_cpu", "hawk_cuda")
TEXT = (".py", ".json", ".toml", ".md", ".txt", ".yml", ".yaml", ".cfg", ".in", ".cmake", ".sh", ".ipynb", ".rst")


def _tracked():
    import subprocess

    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
    return [p for p in out if (REPO / p).is_file()]


def _text_files():
    return [p for p in _tracked() if p.endswith(TEXT) or p.rsplit("/", 1)[-1] in ("Makefile", "CMakeLists.txt")]


def test_no_import_statement_names_an_old_top_level_package():
    import ast

    bad = []
    for p in (p for p in _tracked() if p.endswith(".py")):
        try:
            tree = ast.parse((REPO / p).read_text())
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""] if isinstance(n, ast.ImportFrom) and not n.level else []
            bad += [f"{p}:{n.lineno}: {x}" for x in names if x.split(".")[0] in OLD]
    _fail("an import statement of lowering, simt, hawk_c, osprey_cpu or hawk_cuda", bad)


def test_no_dotted_module_string_names_an_old_top_level_package():
    import re

    pat = re.compile(r"(?<![\w.])(" + "|".join(OLD) + r")(?=\.[A-Za-z_])")
    _fail("a dotted module name starting with lowering, simt, hawk_c, osprey_cpu or hawk_cuda",
          [f"{p}:{n}" for p in _text_files() if p != "tests/test_discovery_words.py"
           for n, line in enumerate((REPO / p).read_text(errors="replace").splitlines(), 1) if pat.search(line)])


def test_the_old_folders_do_not_exist():
    prefixes = ("components/", "extensions/", "staging/osprey_cpu/")
    _fail("a tracked file under components/, extensions/ or staging/osprey_cpu", [p for p in _tracked() if p.startswith(prefixes)])


def test_no_manifest_or_spelling_row_names_the_dialect_of_the_old_base_wing():
    import re

    pat = re.compile(r'"dialect":\s*"' + OLD[0] + '"')
    _fail("a manifest row that names the dialect lowering",
          [f"{p}:{n}" for p in _tracked() if p.endswith(("wing.json", "osprey.json"))
           for n, line in enumerate((REPO / p).read_text().splitlines(), 1) if pat.search(line)])


# ---- the facts: a raptor wing carries its machine numbers; nothing is named by the old unit word -------------------------------

OLD_UNIT = "vari" + "ant"
OLD_HEADER = "raptor_" + OLD_UNIT + ".h"


def _unit_word_hits():
    import re

    pat = re.compile(r"(?i)(?<!in)" + OLD_UNIT)       # "invariant" is another word: the lookbehind excludes it
    hits = {}
    for p in _tracked():
        try:
            text = (REPO / p).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        lines = text.splitlines()
        if p == "CHANGELOG.md":         # the notes of a RELEASED version are history and stay as written: only the unreleased section is read
            lines = lines[:next((i for i, line in enumerate(lines) if re.match(r"## \d", line)), len(lines))]
        found = [n for n, line in enumerate(lines, 1) if pat.search(line)]
        if found:
            hits[p] = found
    return hits


def test_the_old_unit_word_occurs_in_no_tracked_text_file():
    hits = _unit_word_hits()
    n = sum(map(len, hits.values()))
    print(f"unit word: {n} lines in {len(hits)} files")
    _fail(f"the old unit word, any case, also inside identifiers: {n} lines in {len(hits)} files",
          [f"{p}: {len(v)} lines, first {v[0]}" for p, v in sorted(hits.items())])


def _wing_folders():
    return sorted(d for d in REPO.glob("staging/raptor/wings/*/") if (d / "wing.json").is_file())


def test_the_facts_live_in_the_wings_under_their_final_names():
    bad = [p for p in _tracked() if p.startswith(OLD_UNIT + "s/") or p.startswith(f"staging/raptor/wings/") and p.rsplit("/", 1)[-1] in ("data.py", OLD_HEADER)]
    bad += [f"{p}: group raptor.{OLD_UNIT}s" for p, g in _groups() if g == f"raptor.{OLD_UNIT}s"]
    for d in _wing_folders():
        facts = d / "facts.py"
        if facts.is_file():
            import ast

            literal = ast.literal_eval(facts.read_text())
            bad += [f"{facts.relative_to(REPO)}: the key {k!r} is stated by wing.json" for k in ("name", "extends") if k in literal]
    for p in _tracked():
        if p.endswith(("wing.json", "facts.py")) and '"extends"' in (REPO / p).read_text():
            bad.append(f"{p}: the key 'extends'")
    _fail("the facts of a raptor wing", bad)


def test_every_raptor_wing_with_facts_holds_the_header_its_resolved_facts_generate():
    import importlib.util

    tool = REPO / "tools" / "gen_facts_header.py"
    wings = [d for d in _wing_folders() if d.name != "base"]
    if not tool.is_file():
        _fail("no tools/gen_facts_header.py", [f"{d.relative_to(REPO)}: no raptor_facts.h" for d in wings])
    spec = importlib.util.spec_from_file_location("_gen_facts_header", tool)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    bad = [f"{d.relative_to(REPO)}: raptor_facts.h absent or not what the facts generate" for d in wings if d.name in gen.drift()]
    bad += [f"{d.relative_to(REPO)}: no raptor_facts.h" for d in wings if not (d / "raptor_facts.h").is_file()]
    _fail("a wing header that is not generated from its resolved facts", sorted(set(bad)))


def test_no_writer_includes_the_old_header_name():
    import re

    pat = re.compile(r"#\s*include\s*[\"<]" + re.escape(OLD_HEADER))
    _fail("a source line that emits an include of the old header",
          [f"{p}:{n}" for p in _tracked() if p.endswith(".py") and not p.startswith("tests/")
           for n, line in enumerate((REPO / p).read_text(errors="replace").splitlines(), 1) if pat.search(line)])
