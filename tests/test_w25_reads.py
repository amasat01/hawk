# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A piece states in its manifest the fact keys it reads (``reads``), and every ``need("section", "key")`` call site in the piece is
covered by that list: the manifest is the one home, the scan has no list of its own. The first-party backends' wings hold the keys their
manifest states (refused when the manifest is loaded, not where a build reads)."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

import _w25_scan as S
from hawk import registry

REPO = Path(__file__).resolve().parent.parent


def _piece_roots() -> dict:
    out = {"hawk": Path(importlib.util.find_spec("hawk").origin).resolve().parent}
    out |= {pid: root for pid, (root, _) in S.pieces().items() if pid in ("wing:raptor.base", "wing:raptor.simt", "osprey:c", "osprey:cuda")}
    return out


PIECES = _piece_roots()


def _declared(piece: str, root: Path) -> set:
    if piece == "hawk":      # hawk's front end reads the wing of the host row it builds: that row's reads (the cpu OSPREY's manifest)
        return set(registry.traits("host").get("reads", ()))
    if piece.startswith("wing:"):
        return set(json.loads((root / "wing.json").read_text()).get("reads", ()))
    return {r for b in json.loads((root / "osprey.json").read_text()).get("backends", ()) for r in b.get("reads", ())}


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
def test_every_facts_read_of_a_piece_is_stated_in_its_manifest(piece):
    declared = _declared(piece, PIECES[piece])
    bad = [f"{where}: need {key!r} is not in the reads of {piece}" for where, key in need_sites(PIECES[piece]) if key not in declared]
    assert not bad, "\n".join(bad)


def test_the_scan_finds_the_reads_of_the_wings():
    assert {k for _, k in need_sites(PIECES["wing:raptor.base"])} >= {"lanes.run_max", "scatter.max_chunks"}
    assert {k for _, k in need_sites(PIECES["wing:raptor.simt"])} == {"launch.warp_size"}


def test_the_scan_sees_an_uncovered_read(tmp_path):
    (tmp_path / "m.py").write_text("def f(spec):\n    return spec.need('lanes', 'block_run'), spec.need('host_wide'), need(x)\n")
    assert [k for _, k in need_sites(tmp_path)] == ["lanes.block_run", "host_wide"]


def test_the_host_facts_hold_what_its_manifest_reads():
    registry._check_reads(registry.traits("host"), "osprey-cpu")


def test_a_first_party_facts_that_lacks_a_key_its_backend_reads_is_refused_at_load(monkeypatch):
    from hawk import facts

    monkeypatch.setattr(facts, "load", lambda name: {"lanes": {"block_run": 4}})
    monkeypatch.setattr(facts, "chain", lambda name: ["base", name])
    item = {"name": "x", "wings": {"raptor": "v"}, "reads": ["lanes.block_run", "lanes.run_max"]}
    with pytest.raises(ValueError, match=r"reads 'lanes.run_max', which no wing of the raptor chain base -> v states"):
        registry._check_reads(item, "m.json")


def _manifest_with(tmp_path, **item):
    (tmp_path / "osprey.json").write_text(json.dumps({"name": "toy", "version": "0", "contract": "1",
                                                          "backends": [{"name": "toy", "ref": "toy:B", "reads": [], **item}]}))
    return registry._read("toy", tmp_path)


def test_a_manifest_that_declares_an_unknown_settings_key_or_a_malformed_option_is_refused_by_name(tmp_path):
    assert "no_such_setting" in _manifest_with(tmp_path, settings=["no_such_setting"])["refusal"]
    assert "'Bad-Option'" in _manifest_with(tmp_path, options=["Bad-Option"])["refusal"]
    ok = _manifest_with(tmp_path, settings=["emit_form", "fold_rows"], options=["arch"])
    assert "refusal" not in ok


def test_a_third_party_cannot_declare_kinds_under_a_first_party_name_or_another_namespace():
    from hawk import HawkError
    from hawk.dialects import declare_kinds

    row = {"name": "raptor.wings.simt.mine", "family": "expr", "operands": {"min": 1, "max": 1}, "result": ["same_as", 0],
           "spelling": {"dialect": "simt", "name": "x", "args": ["ref0"]}}
    for name in ("simt", "base", "cuda"):
        with pytest.raises(HawkError, match="belongs to a first-party piece"):
            declare_kinds({"name": name, "dialect": "simt", "kinds": [row]})
    with pytest.raises(HawkError, match="with the name of the manifest that declares it"):
        declare_kinds({"name": "toy", "dialect": "toy", "kinds": [{**row, "name": "other.mine"}]})


def test_a_backend_row_that_states_no_reads_is_refused_by_name(tmp_path):
    (tmp_path / "osprey.json").write_text(json.dumps({"name": "toy", "version": "0", "contract": "1",
                                                          "backends": [{"name": "toy", "ref": "toy:B"}]}))
    refusal = registry._read("toy", tmp_path)["refusal"]
    assert "backend 'toy' does not state `reads`" in refusal and "osprey.json" in refusal
    (tmp_path / "osprey.json").write_text(json.dumps({"name": "toy", "version": "0", "contract": "1",
                                                          "backends": [{"name": "toy", "ref": "toy:B", "reads": []}]}))
    assert "refusal" not in registry._read("toy", tmp_path)


def test_the_first_party_rows_state_their_reads_and_load():
    assert not registry.refused()
    for row in ("host", "c", "cuda"):
        assert registry.traits(row).get("reads"), row


def test_a_read_that_no_row_of_the_build_states_is_refused_by_name():
    import dataclasses

    from hawk import HawkError
    from hawk.targets import build_spec

    spec = build_spec(("host",))
    assert spec.need("lanes", "run_max") == 32
    with pytest.raises(HawkError, match="fact key 'uniform_run', which no backend row of the build states under `reads`"):
        spec.need("uniform_run")
    with pytest.raises(HawkError, match="'lanes.run_max'"):
        dataclasses.replace(spec, reads=frozenset()).need("lanes", "run_max")
