# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A piece states in its manifest the variant keys it reads (``reads``), and every ``need("section", "key")`` call site in the piece is
covered by that list: the manifest is the one home, the scan has no list of its own. The first-party backends' variants hold the keys their
manifest states (refused when the manifest is loaded, not where a build reads)."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

from hawk import registry

REPO = Path(__file__).resolve().parent.parent


def _piece_roots() -> dict:
    out = {"hawk": Path(importlib.util.find_spec("hawk").origin).resolve().parent}
    out |= {f"component:{c.name}": c / c.name for c in sorted((REPO / "components").glob("*")) if (c / c.name / "__init__.py").is_file()}
    out |= {f"extension:{e.name}": e for d in sorted((REPO / "extensions").glob("*")) for e in sorted(d.iterdir()) if (e / "__init__.py").is_file()}
    return out


PIECES = _piece_roots()


def _declared(piece: str, root: Path) -> set:
    if piece == "hawk":
        return {r for b in registry._builtin()["backends"] for r in b.get("reads", ())}
    if piece.startswith("component:"):
        return set(json.loads((root / "component.json").read_text()).get("reads", ()))
    return {r for b in json.loads((root / "extension.json").read_text()).get("backends", ()) for r in b.get("reads", ())}


def need_sites(root: Path) -> list:
    """``(file:line, dotted key)`` of each ``<x>.need("section", "key")`` / ``<x>.need("section")`` call with literal arguments."""
    out = []
    for path in sorted(root.rglob("*.py")):
        if "tests" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "need"
                    and node.args and all(isinstance(a, ast.Constant) and isinstance(a.value, str) for a in node.args)):
                out.append((f"{path.relative_to(root.parent)}:{node.lineno}", ".".join(a.value for a in node.args)))
    return out


@pytest.mark.parametrize("piece", sorted(PIECES))
def test_every_variant_read_of_a_piece_is_stated_in_its_manifest(piece):
    declared = _declared(piece, PIECES[piece])
    bad = [f"{where}: need {key!r} is not in the reads of {piece}" for where, key in need_sites(PIECES[piece]) if key not in declared]
    assert not bad, "\n".join(bad)


def test_the_scan_finds_the_reads_of_the_components():
    assert {k for _, k in need_sites(PIECES["component:lowering"])} >= {"lanes.run_max", "scatter.max_chunks"}
    assert {k for _, k in need_sites(PIECES["component:simt"])} == {"launch.warp_size"}


def test_the_scan_sees_an_uncovered_read(tmp_path):
    (tmp_path / "m.py").write_text("def f(spec):\n    return spec.need('lanes', 'block_run'), spec.need('host_wide'), need(x)\n")
    assert [k for _, k in need_sites(tmp_path)] == ["lanes.block_run", "host_wide"]


def test_the_builtin_host_variants_hold_what_its_manifest_reads():
    for item in registry._builtin()["backends"]:
        registry._check_reads(item, "builtin")


def test_a_first_party_variant_that_lacks_a_key_its_backend_reads_is_refused_at_load(monkeypatch):
    from hawk import variants

    monkeypatch.setattr(variants, "load", lambda name: {"lanes": {"block_run": 4}})
    item = {"name": "x", "variant": "v", "reads": ["lanes.block_run", "lanes.run_max"]}
    with pytest.raises(ValueError, match=r"reads 'lanes.run_max', which its variant 'v' does not state"):
        registry._check_reads(item, "m.json")


def _manifest_with(tmp_path, **item):
    (tmp_path / "extension.json").write_text(json.dumps({"name": "toy", "version": "0", "contract": "1",
                                                          "backends": [{"name": "toy", "ref": "toy:B", **item}]}))
    return registry._read("toy", tmp_path)


def test_a_manifest_that_declares_an_unknown_policy_key_or_a_malformed_option_is_refused_by_name(tmp_path):
    assert "no_such_policy" in _manifest_with(tmp_path, policy=["no_such_policy"])["refusal"]
    assert "'Bad-Option'" in _manifest_with(tmp_path, options=["Bad-Option"])["refusal"]
    ok = _manifest_with(tmp_path, policy=["emit_form", "fold_rows"], options=["arch"])
    assert "refusal" not in ok


def test_a_third_party_cannot_declare_kinds_under_a_first_party_name_or_another_namespace():
    from hawk import HawkError
    from hawk.dialects import declare_kinds

    row = {"name": "simt.mine", "family": "expr", "operands": {"min": 1, "max": 1}, "result": ["same_as", 0],
           "spelling": {"dialect": "simt", "name": "x", "args": ["ref0"]}}
    for name in ("simt", "lowering", "hawk-cuda"):
        with pytest.raises(HawkError, match="belongs to a first-party piece"):
            declare_kinds({"name": name, "dialect": "simt", "kinds": [row]})
    with pytest.raises(HawkError, match="with the name of the manifest that declares it"):
        declare_kinds({"name": "toy", "dialect": "toy", "kinds": [{**row, "name": "other.mine"}]})
