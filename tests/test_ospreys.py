# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Finding the OSPREY libraries a build calls: the search list, the manifest, the refusals by name."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
from _unit_sidecar import sidecar_of_kernel

from hawk import ospreys
from hawk.errors import HawkError, ToolchainAbsent
from hawk.targets import TARGETS, host_target

HOST_ROW = {"request": "host", "target": "Host", "about": "for the CPU", "device": False, "entry": "host", "format": "host"}
DEVICE_ROW = {"request": "cuda", "target": "Device", "about": "builds for the GPU", "device": True, "entry": "device", "format": "ptx"}


def write_osprey(folder, name, targets, **extra):
    """Write an OSPREY folder holding a manifest (the library is not needed to be found); ``targets`` maps id to row."""
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {"name": name, "version": "0", "door": "1", "targets": list(targets.values()), **extra}
    (folder / "osprey.json").write_text(json.dumps(manifest))
    return folder


@pytest.fixture
def searched(monkeypatch):
    """Point the search list at the folders a test names; nothing installed is read."""
    monkeypatch.setattr(ospreys.metadata, "entry_points", lambda group: [])

    def point(*folders):
        monkeypatch.setenv(ospreys.SEARCH_ENV, os.pathsep.join(str(f) for f in folders))
        ospreys.forget()

    yield point
    ospreys.forget()


def test_an_osprey_folder_named_by_the_environment_is_found(tmp_path, searched):
    searched(write_osprey(tmp_path / "cpu", "cpu", {"host": HOST_ROW}))
    assert list(ospreys.found()) == ["cpu"]
    assert ospreys.serving("host").name == "cpu"


def test_a_distribution_declares_a_folder_by_a_data_path(tmp_path, monkeypatch, searched):
    folder = write_osprey(tmp_path / "dist" / "data" / "cuda", "cuda", {"cuda": DEVICE_ROW})
    point = SimpleNamespace(value="data/cuda", dist=SimpleNamespace(locate_file=lambda value: tmp_path / "dist" / value))
    monkeypatch.setattr(ospreys.metadata, "entry_points", lambda group: [point] if group == ospreys.ENTRY_POINT_GROUP else [])
    ospreys.forget()
    assert ospreys.found()["cuda"].folder == folder


def test_a_target_nobody_serves_is_refused_by_name(tmp_path, searched):
    searched(write_osprey(tmp_path / "cpu", "cpu", {"host": HOST_ROW}))
    with pytest.raises(HawkError, match=r"target 'tpu' is served by no OSPREY found.*host"):
        ospreys.serving("tpu")


def test_two_ospreys_serving_one_target_are_refused_by_name(tmp_path, searched):
    searched(write_osprey(tmp_path / "a", "a", {"host": HOST_ROW}), write_osprey(tmp_path / "b", "b", {"host": HOST_ROW}))
    with pytest.raises(HawkError, match=r"target 'host' is served by both 'a' and 'b'"):
        ospreys.found()


def test_a_manifest_that_names_python_wings_is_refused(tmp_path, searched):
    searched(write_osprey(tmp_path / "w", "w", {"host": HOST_ROW}, wings={"raptor": "x"}))
    assert "Python wing is refused" in next(iter(ospreys.refused().values()), "")


@pytest.mark.parametrize("drop", ["name", "version", "door", "targets"])
def test_a_manifest_missing_a_member_is_refused_naming_it(tmp_path, searched, drop):
    folder = write_osprey(tmp_path / "m", "m", {"host": HOST_ROW})
    manifest = json.loads((folder / "osprey.json").read_text())
    del manifest[drop]
    (folder / "osprey.json").write_text(json.dumps(manifest))
    searched(folder)
    assert f"lacks {drop}" in next(iter(ospreys.refused().values()), "")


def test_the_targets_table_is_what_the_found_ospreys_serve(tmp_path, searched):
    searched(write_osprey(tmp_path / "cuda", "cuda", {"cuda": DEVICE_ROW}), write_osprey(tmp_path / "cpu", "cpu", {"host": HOST_ROW}))
    assert list(TARGETS) == ["cuda", "host"]
    assert TARGETS["cuda"].device and TARGETS["cuda"].target == "Device" and TARGETS["host"].host_entry
    assert host_target().id == "host"
    assert "tpu" not in TARGETS


def test_a_machine_with_no_tools_is_stated_and_only_a_build_raises_toolchain_absent(tmp_path, searched, monkeypatch):
    searched(write_osprey(tmp_path / "osprey_cpu", "cpu", {"host": HOST_ROW}))
    given = {"key": "k", "stated": {"isa": "x"}, "tools_absent": "no compiler is installed"}
    monkeypatch.setattr(ospreys.Osprey, "_call", lambda self, door, *args: given)
    osprey = ospreys.serving("host")
    assert osprey.machine({"target": "host"}) is given
    with pytest.raises(ToolchainAbsent, match="cpu: no compiler is installed") as caught:
        osprey.require_tools(given)
    assert caught.value.toolchain == "cpu"
    osprey.require_tools({**given, "tools_absent": None})


def test_a_refusal_with_the_code_tool_absent_is_toolchain_absent(tmp_path, searched):
    searched(write_osprey(tmp_path / "osprey_cpu", "cpu", {"host": HOST_ROW}))
    refusal = ValueError("no compiler")
    refusal.code = "tool_absent"
    assert isinstance(ospreys.serving("host")._refused("build", refusal), ToolchainAbsent)


def test_a_machine_answer_is_kept_as_given(tmp_path, searched, monkeypatch):
    searched(write_osprey(tmp_path / "cpu", "cpu", {"host": HOST_ROW}))
    given = {"key": "opaque", "stated": {"anything": [1, 2]}, "tools_absent": None}
    monkeypatch.setattr(ospreys.Osprey, "_call", lambda self, door, *args: given)
    assert ospreys.serving("host").machine({"target": "host"}) is given


def test_a_build_for_an_unserved_target_is_refused_by_name_before_any_lowering(tmp_path, searched):
    import _deployable as D

    from hawk.artifact import build_bundle

    searched(write_osprey(tmp_path / "cpu", "cpu", {"host": HOST_ROW}))
    with pytest.raises(HawkError, match="target 'cuda' is served by no OSPREY found"):
        build_bundle([D.axpb], tmp_path / "out", targets=("cuda",))


def test_the_sidecar_stores_what_the_osprey_stated_and_built_as_given():
    import _deployable as D

    tool, stated = {"cpu": {"figures": [1, 2, 3]}}, {"cpu": {"isa": "x"}}
    sidecar = sidecar_of_kernel(D.axpb, tool=tool, stated=stated, measured={"local_bytes": 64})
    assert sidecar["tool"] == tool and sidecar["stated"] == stated
    assert sidecar["launch"]["local_bytes"] == 64
