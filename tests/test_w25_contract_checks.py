# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The contribution contract is checkable: a manifest key that is unknown, a malformed manifest behind an unknown target, an id
outside the grammar, facts that break their schema, and facts changed after load are each refused BY NAME, early."""

from __future__ import annotations

import importlib
import json
import sys

import pytest
from _ospreys import backend, copy_of, make

from hawk import facts, registry
from hawk.backends import need
from raptor.errors import RaptorError as HawkError


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    registry.reset()
    importlib.invalidate_caches()
    yield tmp_path
    registry.reset()
    for name, module in list(sys.modules.items()):
        if str(getattr(module, "__file__", "") or "").startswith(str(tmp_path)):
            del sys.modules[name]


def make_wing(root, name: str, text: str, *, parent: str | None = "base", entry: str | None = None, package: str | None = None) -> None:
    """A distribution ``wing-<name>`` whose package holds ``wing.json`` (parent ``parent``) and ``facts.py`` = ``text`` and declares the
    raptor wing entry point ``entry``."""
    package = package or f"wing_{name}"
    dist = root / f"wing_{name}-0.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: wing-{name}\nVersion: 0.0\n")
    (dist / "entry_points.txt").write_text(f"[raptor.wings]\n{entry or name} = {package}\n")
    (root / package).mkdir()
    (root / package / "__init__.py").write_text("")
    (root / package / "wing.json").write_text(json.dumps({"family": "raptor", "name": entry or name, "parent": parent}))
    (root / package / "facts.py").write_text(text)
    facts.reset()
    importlib.invalidate_caches()


def refusal_of(ident: str) -> str:
    return registry.refused()[ident]


def test_an_unknown_key_in_a_backend_item_is_refused_naming_the_key_and_the_manifest(site):
    item = backend("toy", "toy_mod:BACKEND", varaint="cpu")
    make(site, "toy-ext", {"backends": [item]}, modules={"toy_mod": copy_of("ospreys.cpu.backend", "toy")})
    registry.reset()
    why = refusal_of("toy-ext")
    assert "varaint" in why and "osprey.json" in why and "unknown key" in why


def test_an_unknown_top_level_key_is_refused_by_name(site):
    make(site, "toy-ext", {"overrides": ["x"], "backends": []})
    registry.reset()
    assert "'overrides'" in refusal_of("toy-ext")


def test_a_malformed_manifest_never_shows_only_as_an_unknown_target(site):
    make(site, "toy-ext", {"backends": [backend("toy", "toy_mod:BACKEND", wings={"raptor": 7})]})
    registry.reset()
    with pytest.raises(HawkError, match=r"toy' is refused: OSPREY 'toy-ext' is refused: .*malformed"):
        need("toy")
    make(site, "toy-raw", raw="{not json")
    registry.reset()
    with pytest.raises(HawkError, match="'other' is refused: .*toy-raw.*could not be read"):
        need("other")


@pytest.mark.parametrize("bad", ["Toy-X!", "", "Host", "x y", "9lives"])
def test_a_backend_id_outside_the_grammar_is_refused_by_name(site, bad):
    make(site, "toy-ext", {"backends": [backend(bad, "toy_mod:BACKEND")]})
    registry.reset()
    assert "lowercase identifier" in refusal_of("toy-ext") and repr(bad) in refusal_of("toy-ext")


def test_an_osprey_name_outside_the_grammar_is_refused_by_name(site):
    make(site, "toy-ext", {"name": "Toy X!", "backends": []})
    registry.reset()
    assert "OSPREY name" in refusal_of("toy-ext")


def test_a_manifest_that_reads_a_number_its_wing_lacks_is_refused_at_load(site):
    make_wing(site, "bare", '{"linalg": {"max_dim": 32}}\n')
    item = backend("toy", "toy_mod:BACKEND", wings={"raptor": "bare"}, reads=["form.rolled_fold_min"])
    make(site, "toy-ext", {"backends": [item]})
    registry.reset()
    assert "'form.rolled_fold_min'" in refusal_of("toy-ext") and "base -> bare" in refusal_of("toy-ext")


def test_a_number_that_no_wing_of_a_three_wing_chain_states_is_refused_naming_the_chain(site):
    make_wing(site, "ra", '{"lanes": {"block_run": 4}}\n', parent=None)
    make_wing(site, "rb", '{"form": {"rolled_fold_min": 2}}\n', parent="ra")
    make_wing(site, "rc", '{"lanes": {"copy_threads": 8}}\n', parent="rb")
    item = {"name": "x", "wings": {"raptor": "rc"}, "reads": ["lanes.copy_threads", "lanes.block_run", "form.rolled_fold_min"]}
    registry._check_reads(item, "m.json")                       # every key is stated somewhere along ra -> rb -> rc
    with pytest.raises(ValueError, match=r"reads 'lanes.run_max', which no wing of the raptor chain ra -> rb -> rc states"):
        registry._check_reads({**item, "reads": [*item["reads"], "lanes.run_max"]}, "m.json")


def test_first_party_manifests_state_the_numbers_they_read():
    for backend_id in ("host", "c", "cuda"):
        assert registry.traits(backend_id).get("reads"), backend_id


@pytest.mark.parametrize("name", ["../cpu", "/tmp/cpu", "cpu/arm", "CPU", "", ".."])
def test_a_wing_name_that_is_not_one_segment_is_refused_by_name(name):
    with pytest.raises(ValueError, match="ONE identifier segment"):
        facts.load(name)


@pytest.mark.parametrize("text, word", [
    ('def width():\n    return 8\n\n{"form": {"rolled_fold_min": width()}}\n', "no def, no expression"),
    ('{"form": {"rolled_fold_min": 4 * 2}}\n', "facts.py"),
    ('{"extends": "cpu"}\n', "unknown top-level key 'extends'"),
    ('{"name": "other"}\n', "unknown top-level key 'name'"),
    ('{"frm": {}}\n', "unknown top-level key 'frm'"),
    ('{"form": 3}\n', "'form' is a int")])
def test_facts_are_checked_once_at_load(site, text, word):
    make_wing(site, "v", text)
    with pytest.raises(ValueError, match=word) as e:
        facts.load("v")
    assert "facts.py" in str(e.value)


def test_a_third_party_wing_is_found_by_its_entry_point_and_builds_on_a_first_party_one(site):
    make_wing(site, "acme", '{"form": {"rolled_fold_min": 5}}\n', parent="cpu")
    data = facts.load("acme")
    assert data["form"]["rolled_fold_min"] == 5 and data["linalg"] == facts.load("cpu")["linalg"]
    assert facts.include_flags(data) == (f"-I{site / 'wing_acme'}",)


def test_loaded_facts_are_read_only_down_to_its_nested_values():
    data = facts.load("arm")
    for change in (lambda: data.update(x=1), lambda: data["form"].__setitem__("rolled_fold_min", 1),
                   lambda: data["fusion_host"]["neon"].__setitem__("x", 1), lambda: data.pop("form")):
        with pytest.raises(TypeError, match="read-only"):
            change()
    assert facts.derive(data, {"form": {"rolled_fold_min": 1}})["form"]["rolled_fold_min"] == 1 and data["form"]["rolled_fold_min"] != 1


def test_an_emission_module_that_does_not_exist_is_refused_by_name(site):
    make(site, "toy-ext", {"emission_modules": ["toy_ext.nope"], "backends": []})
    registry.reset()
    assert "'toy_ext.nope'" in refusal_of("toy-ext") and "does not exist" in refusal_of("toy-ext")


def test_the_include_flag_carries_a_key_term_that_does_not_name_the_install_path(tmp_path, monkeypatch):
    """The same wing installed at two absolute paths: the flags differ (the path is on the compile line), the key terms are equal;
    a changed header changes the key term."""
    terms = []
    for where in ("one", "two"):
        root = tmp_path / where
        root.mkdir()
        make_wing(root, "acme", "{}\n", parent="cpu")
        (root / "wing_acme" / "raptor_facts.h").write_text("#define RAPTOR_LINALG_MAX_DIM 32\n")
        monkeypatch.syspath_prepend(str(root))
        facts.reset()
        (flag,) = facts.include_flags({"name": "acme"})
        terms.append((str(flag), flag.key_term))
        monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(root)])
    assert terms[0][0] != terms[1][0] and terms[0][1] == terms[1][1] and "acme" in terms[0][1]
    (tmp_path / "two" / "wing_acme" / "raptor_facts.h").write_text("#define RAPTOR_LINALG_MAX_DIM 64\n")
    monkeypatch.syspath_prepend(str(tmp_path / "two"))
    facts.reset()
    assert facts.include_flags({"name": "acme"})[0].key_term != terms[0][1]
    facts.reset()
