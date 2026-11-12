# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The direction family, for what a leaf may reach: an extension that imports a component's module reaches only what the component's
manifest declares as an ENTRY (module and names); a leaf never imports another leaf; and the contribution kit runs on a package that is
not hawk's own. No list of names: the components and the leaves are the ones the installed distributions declare."""

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

EXTENSIONS = [(p, p.name) for d in sorted((REPO / "extensions").glob("*/")) for p in sorted(d.iterdir()) if (p / "__init__.py").is_file()]


@pytest.mark.parametrize("root, pkg", EXTENSIONS, ids=[n for _, n in EXTENSIONS])
def test_an_extension_reaches_only_the_declared_entries_of_a_component(root, pkg):
    entries = kit.component_entries()
    assert {"lowering", "simt"} <= set(entries), f"the components are found through their entry points: {sorted(entries)}"
    bad = [f"{e.where(root.parent)}: {why}" for e, why in kit.entries_violations(scan.scan_tree(root, pkg), entries, pkg)]
    assert not bad, "\n".join(bad)


def test_the_entries_scan_sees_an_undeclared_name_and_module(tmp_path):
    (tmp_path / "leaf").mkdir()
    (tmp_path / "leaf" / "__init__.py").write_text("from lowering.box import BOX_OWN, not_an_entry\nfrom lowering.nosuch import x\n")
    bad = kit.entries_violations(scan.scan_tree(tmp_path / "leaf", "leaf"), kit.component_entries(), "leaf")
    assert [why for _, why in bad] == ["lowering.box: 'not_an_entry' is not an entry of lowering.box",
                                       "lowering.nosuch: module 'nosuch' is not an entry of the 'lowering' component"]


def test_no_leaf_imports_another_leaf():
    """The leaves are the packages that distributions declare in the ``raptor.extensions`` group (hawk's own wheel declares the first-party ones)."""
    from hawk import variants

    leaves = {ep.value.partition(":")[0].split(".")[0] for ep in variants.entry_points("raptor.extensions")}
    assert {"hawk_c", "hawk_cuda"} <= leaves
    bad = [f"{e.where(root.parent)}: imports {e.target}" for root, pkg in EXTENSIONS for e in scan.scan_tree(root, pkg)
           if e.top in leaves and e.top != pkg]
    assert not bad, "\n".join(bad)


def test_a_leaf_that_imports_another_leaf_is_seen(tmp_path):
    (tmp_path / "third").mkdir()
    (tmp_path / "third" / "__init__.py").write_text("import hawk_cuda.emit\n")
    out = _kit(tmp_path / "third")
    assert out.returncode == 1 and "[leaf]" in out.stdout and "hawk_cuda.emit" in out.stdout


def _kit(path, *more):
    return subprocess.run([sys.executable, str(REPO / "tools" / "contract_kit.py"), str(path), *more], capture_output=True, text=True,
                          env={"PYTHONPATH": ":".join(sys.path), "PATH": "/usr/bin"})


def _rules(out):
    return {int(line.split()[1]): line.split(": ")[-1].split(" (")[0] for line in out.stdout.splitlines() if line.startswith("rule ")}


def test_the_kit_fails_the_plugin_fixture_on_the_leaf_rule_only():
    """tests/plugins/backend_host2 subclasses hawk's Host: rule 1 (2a.3) fails, honestly; every other checked rule passes."""
    out = _kit(TESTS / "plugins" / "backend_host2" / "host2_plugin")
    rules = _rules(out)
    assert out.returncode == 1 and [n for n, st in rules.items() if st == "FAIL"] == [1], out.stdout
    assert "host2_plugin/__init__.py:8 imports hawk.backends.host" in out.stdout and "[2a.3]" in out.stdout
    assert [n for n, st in rules.items() if st == "NOT CHECKABLE HERE"] == [3, 5, 6]


COMPONENTS = [p for p in sorted((REPO / "components").glob("*")) if (p / p.name / "__init__.py").is_file()]


@pytest.mark.parametrize("comp", COMPONENTS, ids=[p.name for p in COMPONENTS])
def test_the_kit_passes_a_component(comp):
    """Every component found under components/*, not a named one."""
    from test_module_mutable_state import ALLOWED

    allow = ",".join(name for (file, name) in ALLOWED if file.startswith(comp.name + "/"))
    out = _kit(comp / comp.name, "--allow", allow)
    rules = _rules(out)
    assert out.returncode == 0 and [n for n, st in rules.items() if st == "FAIL"] == [], out.stdout
    assert [n for n, st in rules.items() if st == "PASS"] == [1, 2, 4, 7, 8], out.stdout
