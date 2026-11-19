# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The defines a unit needs are the library's to state per target, and the build hands each target's request its own.

The libraries lower for real and no compiler runs: the build call of every OSPREY is replaced by one that records its request.
"""

from __future__ import annotations

import pytest
from test_host_interchange import oscillator

from hawk import HawkError, ospreys
from hawk.artifact import bundle
from hawk.build_settings import BuildSettings

pytest.importorskip("_osprey")
try:
    ospreys.serving("host")
    ospreys.serving("cuda")
except Exception as why:  # a library of the build is not at hand
    pytest.skip(
        f"the host and device OSPREYs are not available: {why}", allow_module_level=True
    )

#: The define the device library states for an until-done kernel built with fast entries.
FAST_ENTRIES = "HAWK_FAST_ENTRIES=1"


def requests_of_a_build(monkeypatch, tmp_path, targets):
    """``{request target: the build request of the OSPREY that serves it}`` for ``oscillator`` built with fast entries and the
    user's define ``USER=1``."""
    seen = {}

    def record(self, unit, root):
        seen[unit["target"]] = unit
        return {
            "key": "k",
            "hit": True,
            "seconds": 0.0,
            "artifact": "a",
            "source": "s",
            "format": "f",
        }

    monkeypatch.setattr(ospreys.Osprey, "build", record)
    settings = bundle._resolved_settings(BuildSettings(fast_entries=True), targets)
    kinds, machines, answers = bundle._lowered(
        [oscillator],
        [oscillator.name],
        None,
        mode="float64",
        targets=targets,
        cfg=settings,
        options={},
        opt_level="",
        root=tmp_path,
        sizes=None,
        tools_needed=False,
    )
    bundle._build_units(
        answers,
        targets=targets,
        machines=machines,
        options={},
        defines=("USER=1",),
        opt_level="",
        block=None,
        kinds=kinds,
        root=tmp_path,
    )
    return seen


@pytest.mark.parametrize("targets", [("host", "cuda"), ("cuda",)])
def test_the_device_build_request_carries_the_define_the_lowering_states_after_the_users(
    monkeypatch, tmp_path, targets
):
    seen = requests_of_a_build(monkeypatch, tmp_path, targets)
    assert seen[bundle.TARGETS["cuda"].target]["defines"] == ["USER=1", FAST_ENTRIES]


def test_the_host_build_request_of_a_host_and_device_build_carries_only_the_users_defines(
    monkeypatch, tmp_path
):
    seen = requests_of_a_build(monkeypatch, tmp_path, ("host", "cuda"))
    assert seen[bundle.TARGETS["host"].target]["defines"] == ["USER=1"]


def test_a_define_stated_for_a_target_the_build_does_not_have_is_refused_by_name():
    unit = {"defines": {"cuda": [FAST_ENTRIES]}, "texts": [], "meta": {"name": "k"}}
    with pytest.raises(HawkError, match="cuda"):
        bundle._texts(unit, ("host",))
