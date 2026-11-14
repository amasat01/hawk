# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The direction family, for what a leaf may reach: an OSPREY that imports a wing's module reaches only what the wing's
manifest declares as an ENTRY (module and names); a leaf never imports another leaf; and the contribution kit runs on a package that is
not hawk's own. No list of names: the wings and the leaves are the ones the installed distributions declare."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
REPO = TESTS.parent
sys.path.insert(0, str(REPO / "tools"))

import _w25_scan as scan  # noqa: E402
import contract_kit as kit  # noqa: E402

EXTENSIONS = [(REPO / "staging" / "ospreys" / "c", "ospreys.c"), (REPO / "staging" / "ospreys" / "cuda", "ospreys.cuda")]


@pytest.mark.parametrize("root, pkg", EXTENSIONS, ids=[n for _, n in EXTENSIONS])
def test_an_osprey_reaches_only_the_declared_entries_of_a_wing(root, pkg):
    entries = kit.wing_entries()
    assert {"raptor.wings.base", "raptor.wings.simt"} <= set(entries), f"the wings are found through their entry points: {sorted(entries)}"
    bad = [f"{e.where(root.parent)}: {why}" for e, why in kit.entries_violations(scan.scan_tree(root, pkg), entries, pkg)]
    assert not bad, "\n".join(bad)


def test_the_entries_scan_sees_an_undeclared_name_and_module(tmp_path):
    (tmp_path / "leaf").mkdir()
    (tmp_path / "leaf" / "__init__.py").write_text("from raptor.wings.base.box import boxed_sinks, not_an_entry\nfrom raptor.wings.base.nosuch import x\n")
    bad = kit.entries_violations(scan.scan_tree(tmp_path / "leaf", "leaf"), kit.wing_entries(), "leaf")
    assert [why for _, why in bad] == ["raptor.wings.base.box: 'not_an_entry' is not an entry of raptor.wings.base.box",
                                       "raptor.wings.base.nosuch: module 'nosuch' is not an entry of the 'raptor.wings.base' wing"]


def test_no_leaf_imports_another_leaf():
    """The leaves are the packages that distributions declare in the ``ospreys`` group (hawk's own wheel declares the first-party ones)."""
    from hawk import wings

    leaves = {ep.value.partition(":")[0] for ep in wings.entry_points("ospreys")}
    assert {"ospreys.c", "ospreys.cuda"} <= leaves
    bad = [f"{e.where(root.parent)}: imports {e.target}" for root, pkg in EXTENSIONS for e in scan.scan_tree(root, pkg)
           if any(e.target == leaf or e.target.startswith(leaf + ".") for leaf in leaves if leaf != pkg)]
    assert not bad, "\n".join(bad)


def test_a_leaf_that_imports_another_leaf_is_seen(tmp_path):
    (tmp_path / "third").mkdir()
    (tmp_path / "third" / "__init__.py").write_text("import ospreys.cuda.emit\n")
    out = _kit(tmp_path / "third")
    assert out.returncode == 1 and "[leaf]" in out.stdout and "ospreys.cuda.emit" in out.stdout


def _kit(path, *more):
    return subprocess.run([sys.executable, str(REPO / "tools" / "contract_kit.py"), str(path), *more], capture_output=True, text=True,
                          env={"PYTHONPATH": ":".join(sys.path), "PATH": "/usr/bin"})


def _rules(out):
    return {int(line.split()[1]): line.split(": ")[-1].split(" (")[0] for line in out.stdout.splitlines() if line.startswith("rule ")}


def test_the_kit_fails_the_plugin_fixture_on_the_leaf_rule_only():
    """tests/plugins/backend_host2 subclasses the cpu OSPREY's Host: rule 1 (another leaf) fails, honestly; every other checked rule passes."""
    out = _kit(TESTS / "plugins" / "backend_host2" / "host2_plugin")
    rules = _rules(out)
    assert out.returncode == 1 and [n for n, st in rules.items() if st == "FAIL"] == [1], out.stdout
    assert "host2_plugin/__init__.py:8 imports the leaf ospreys.cpu.backend" in out.stdout and "[leaf]" in out.stdout
    assert [n for n, st in rules.items() if st == "NOT CHECKABLE HERE"] == [3, 5, 6]


#: every raptor wing that holds code beside its manifest and facts (a facts-only wing has nothing for the kit to read)
WINGS = [p for p in sorted((REPO / "staging" / "raptor" / "wings").glob("*"))
         if (p / "__init__.py").is_file() and any(f.name not in ("__init__.py", "facts.py", "cards.py") for f in p.glob("*.py"))]
assert WINGS, "no raptor wing with code found: the kit test would have no case"


@pytest.mark.parametrize("comp", WINGS, ids=[p.name for p in WINGS])
def test_the_kit_passes_a_wing(comp):
    """Every raptor wing with code found under staging/raptor/wings, not a named one."""
    from test_module_mutable_state import ALLOWED

    allow = ",".join(name for (file, name) in ALLOWED if file.startswith(comp.name + "/"))
    out = _kit(comp, "--allow", allow)
    rules = _rules(out)
    assert out.returncode == 0 and [n for n, st in rules.items() if st == "FAIL"] == [], out.stdout
    assert [n for n, st in rules.items() if st == "PASS"] == [1, 2, 4, 7, 8], out.stdout
