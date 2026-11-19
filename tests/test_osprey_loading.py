# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A library folder is found, loaded or refused by name, and the libraries are isolated from one another.

The rows use the real library folders the search list names (``HAWK_OSPREY_PATH``) and folders written under ``tmp_path``: a
third-party folder, a folder with a malformed or incomplete manifest, a library of another door version, a file that is no
library. A refusal names the folder and the member. The libraries' own isolation (each one's facts, in a folder holding only
itself) is the binding's ``test_sideways.py``; these rows are the same property read through hawk's search list.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hawk import ospreys
from hawk.artifact import build_key
from hawk.errors import HawkError

SUFFIX = "dylib" if sys.platform == "darwin" else "so"
TPU_ROW = {
    "request": "tpu",
    "target": "Tpu",
    "about": "a third-party target",
    "device": False,
    "entry": "host",
    "format": "host",
}


def real_folders() -> dict:
    """``{name: folder}`` of the library folders the search list names, read from their manifests."""
    folders = [
        Path(p) for p in os.environ.get(ospreys.SEARCH_ENV, "").split(os.pathsep) if p
    ]
    return {json.loads((f / "osprey.json").read_text())["name"]: f for f in folders}


@pytest.fixture
def searched(monkeypatch):
    """Point the search list at the folders a test names; nothing installed is read."""
    monkeypatch.setattr(ospreys.metadata, "entry_points", lambda group: [])

    def point(*folders):
        monkeypatch.setenv(ospreys.SEARCH_ENV, os.pathsep.join(str(f) for f in folders))
        ospreys.forget()

    yield point
    monkeypatch.undo()
    ospreys.forget()


def ask(osprey, tmp_path) -> dict:
    """The machine answer of ``osprey`` for its first target."""
    target = next(iter(osprey.targets))
    return osprey.machine(
        {
            "target": target,
            "options": {},
            "opt_level": "",
            "root": str(tmp_path / "store"),
        }
    )


def copied(tmp_path, name, *, library=None):
    """A copy of the real cpu folder as the third-party folder ``name`` (library file renamed to match)."""
    source = real_folders()["cpu"]
    folder = tmp_path / name
    shutil.copytree(source, folder)
    manifest = json.loads((folder / "osprey.json").read_text())
    manifest["name"] = name
    (folder / "osprey.json").write_text(json.dumps(manifest))
    for stale in folder.glob("libosprey_*"):
        stale.rename(folder / f"libosprey_{name}.{SUFFIX}")
    if library is not None:
        shutil.copy(library, folder / f"libosprey_{name}.{SUFFIX}")
    return folder


def test_a_manifest_that_is_not_json_is_refused_naming_the_file(tmp_path, searched):
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "osprey.json").write_text("{not json")
    searched(tmp_path / "bad")
    assert re.search(
        r"bad/osprey\.json: no readable osprey\.json",
        next(iter(ospreys.refused().values()), ""),
    )


def test_a_folder_without_a_manifest_is_refused_naming_the_file(tmp_path, searched):
    (tmp_path / "empty").mkdir()
    searched(tmp_path / "empty")
    assert re.search(
        r"empty/osprey\.json: no readable osprey\.json",
        next(iter(ospreys.refused().values()), ""),
    )


@pytest.mark.parametrize(
    "drop", ["request", "target", "about", "device", "entry", "format"]
)
def test_a_target_row_missing_a_member_is_refused_naming_the_folder_and_the_member(
    tmp_path, searched, drop
):
    folder = copied(tmp_path, "rowless")
    manifest = json.loads((folder / "osprey.json").read_text())
    del manifest["targets"][0][drop]
    (folder / "osprey.json").write_text(json.dumps(manifest))
    searched(folder)
    assert re.search(
        rf"rowless/osprey\.json: target .* lacks {drop}",
        next(iter(ospreys.refused().values()), ""),
    )


def test_a_folder_without_its_library_is_refused_naming_the_osprey_the_folder_and_the_file(
    tmp_path, searched
):
    folder = copied(tmp_path, "hollow")
    next(folder.glob("libosprey_*")).unlink()
    searched(folder)
    with pytest.raises(
        HawkError,
        match=rf"the OSPREY 'hollow' at .*hollow holds no libosprey_hollow\.{SUFFIX}",
    ):
        ask(ospreys.serving("host"), tmp_path)


def test_a_third_party_folder_is_loaded_and_answers_as_the_library_it_copies(
    tmp_path, searched
):
    searched(real_folders()["cpu"])
    original = ask(ospreys.serving("host"), tmp_path)
    searched(copied(tmp_path, "third"))
    assert ospreys.serving("host").name == "third"
    assert ask(ospreys.serving("host"), tmp_path)["stated"] == original["stated"]


def test_a_library_of_another_door_version_is_refused_by_name_with_its_folder(
    tmp_path, searched
):
    import _osprey

    other = Path(_osprey.__file__).parent / f"libabi1.{SUFFIX}"
    if not other.is_file():
        pytest.skip("the binding build holds no library of another door version")
    searched(copied(tmp_path, "oldabi", library=other))
    with pytest.raises(HawkError, match=r"oldabi.*abi number 1"):
        ask(ospreys.serving("host"), tmp_path)


def test_a_library_of_another_contract_version_is_refused_by_name_with_its_folder(
    tmp_path, searched
):
    import _osprey

    other = Path(_osprey.__file__).parent / f"libcontract0.{SUFFIX}"
    if not other.is_file():
        pytest.skip("the binding build holds no library of another contract version")
    searched(copied(tmp_path, "oldcontract", library=other))
    with pytest.raises(HawkError, match=r"oldcontract.*contract version"):
        ask(ospreys.serving("host"), tmp_path)


def test_a_library_without_the_entry_symbol_is_refused_by_name_with_its_folder_and_the_symbol(
    tmp_path, searched
):
    import _osprey

    other = Path(_osprey.__file__).parent / f"libfixture_unit.{SUFFIX}"
    if not other.is_file():
        pytest.skip("the binding build holds no library without the entry symbol")
    searched(copied(tmp_path, "nosymbol", library=other))
    with pytest.raises(HawkError, match=r"nosymbol.*osprey_v1"):
        ask(ospreys.serving("host"), tmp_path)


def test_the_machine_of_a_library_is_the_same_with_the_other_library_in_the_search_list_or_not(
    tmp_path, searched
):
    folders = real_folders()
    searched(folders["cpu"])
    alone = ask(ospreys.serving("host"), tmp_path)
    searched(folders["cuda"], folders["cpu"])
    assert ask(ospreys.serving("host"), tmp_path) == alone
    assert (
        ospreys.serving("host").folder == folders["cpu"]
        and ospreys.serving("cuda").folder == folders["cuda"]
    )


def test_a_distribution_adding_a_target_changes_neither_the_other_targets_nor_the_key_of_a_build_for_them(
    tmp_path, searched, monkeypatch
):
    import _deployable as D

    folders = real_folders()
    searched(folders["cpu"])
    before = (list(ospreys.targets()), build_key([D.axpb], targets=("host",)))
    third = copied(tmp_path, "tpu_osprey")
    manifest = json.loads((third / "osprey.json").read_text())
    manifest["targets"] = [TPU_ROW]
    (third / "osprey.json").write_text(json.dumps(manifest))
    point = SimpleNamespace(
        value="x", dist=SimpleNamespace(locate_file=lambda value: third)
    )
    monkeypatch.setattr(
        ospreys.metadata,
        "entry_points",
        lambda group: [point] if group == ospreys.ENTRY_POINT_GROUP else [],
    )
    ospreys.forget()
    assert sorted(ospreys.targets()) == sorted([*before[0], "tpu"])
    assert build_key([D.axpb], targets=("host",)) == before[1]
    assert (
        ospreys.serving("tpu").folder == third
        and ospreys.serving("host").folder == folders["cpu"]
    )


def test_a_host_only_search_list_builds_and_refuses_the_device_target_by_name(
    tmp_path, searched
):
    import _deployable as D

    from hawk.artifact import build_bundle

    searched(real_folders()["cpu"])
    assert list(ospreys.targets()) == ["host"]
    bundle = build_bundle([D.axpb], tmp_path / "out", options=None)
    assert bundle.manifest is not None
    with pytest.raises(HawkError, match="target 'cuda' is served by no OSPREY found"):
        build_bundle([D.axpb], tmp_path / "device", targets=("cuda",))


def test_a_broken_folder_in_the_search_list_never_breaks_a_call_for_a_library_that_is_fine(
    tmp_path, searched
):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "osprey.json").write_text("{not json")
    searched(real_folders()["cpu"], tmp_path / "broken")
    assert ask(ospreys.serving("host"), tmp_path)["stated"]


def test_an_unknown_manifest_key_is_refused_naming_the_key_and_the_file(
    tmp_path, searched
):
    folder = copied(tmp_path, "extra")
    manifest = json.loads((folder / "osprey.json").read_text())
    manifest["overrides"] = ["x"]
    (folder / "osprey.json").write_text(json.dumps(manifest))
    searched(folder)
    assert re.search(
        r"'overrides'.*osprey\.json|osprey\.json.*'overrides'",
        next(iter(ospreys.refused().values()), ""),
    )


@pytest.mark.parametrize("bad", ["Toy-X!", "", "Host", "x y", "9lives"])
def test_a_target_id_outside_the_grammar_is_refused_naming_it(tmp_path, searched, bad):
    folder = copied(tmp_path, "oddid")
    manifest = json.loads((folder / "osprey.json").read_text())
    manifest["targets"][0]["request"] = bad
    (folder / "osprey.json").write_text(json.dumps(manifest))
    searched(folder)
    assert re.search("request", next(iter(ospreys.refused().values()), ""))


def test_a_folder_that_is_refused_is_listed_with_its_reason_and_named_when_a_target_it_would_serve_is_asked_for(
    tmp_path, searched
):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "osprey.json").write_text("{not json")
    searched(real_folders()["cpu"], tmp_path / "broken")
    assert list(ospreys.found()) == ["cpu"]
    assert list(ospreys.refused()) == [str(tmp_path / "broken")]
    with pytest.raises(
        HawkError,
        match=r"target 'tpu' is served by no OSPREY.*refused: .*broken/osprey\.json: no readable",
    ):
        ospreys.serving("tpu")
