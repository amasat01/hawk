# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The builds of a unit's targets overlap, and a failure is raised in target order after every started build has finished.

The libraries lower for real and no compiler runs: the build call of every OSPREY is replaced by one that waits or fails.
"""

from __future__ import annotations

import threading

import pytest
from test_host_interchange import oscillator

from hawk import HawkError, ospreys
from hawk.artifact import bundle

pytest.importorskip("_osprey")
try:
    ospreys.serving("host")
    ospreys.serving("cuda")
except Exception as why:  # a library of the build is not at hand
    pytest.skip(
        f"the host and device OSPREYs are not available: {why}", allow_module_level=True
    )

TARGETS = ("host", "cuda")
ANSWER = {
    "key": "k",
    "hit": True,
    "seconds": 0.0,
    "artifact": "a",
    "source": "s",
    "format": "f",
}


def build_all(tmp_path):
    """Lower ``oscillator`` for both targets and run the builds of its units."""
    settings = bundle._resolved_settings(None, TARGETS)
    kinds, machines, answers = bundle._lowered(
        [oscillator], [oscillator.name], None, mode="float64", targets=TARGETS, cfg=settings,
        options={}, opt_level="", root=tmp_path, sizes=None, tools_needed=False)  # fmt: skip
    return bundle._build_units(
        answers, targets=TARGETS, machines=machines, options={}, defines=(), opt_level="",
        block=None, kinds=kinds, root=tmp_path)  # fmt: skip


def test_the_builds_of_the_two_targets_overlap(monkeypatch, tmp_path):
    meeting = threading.Barrier(2, timeout=10)

    def wait_for_the_other(self, unit, root):
        meeting.wait()  # a sequential build never lets the second start: the wait breaks
        return ANSWER

    monkeypatch.setattr(ospreys.Osprey, "build", wait_for_the_other)
    assert len(build_all(tmp_path)[0]) == 1


def test_the_first_failing_target_is_raised_after_both_builds_finished(
    monkeypatch, tmp_path
):
    finished = []

    def fail_on_the_host(self, unit, root):
        finished.append(unit["target"])
        if self.name == "cpu":
            raise HawkError("cpu: the compile failed")
        return ANSWER

    monkeypatch.setattr(ospreys.Osprey, "build", fail_on_the_host)
    with pytest.raises(HawkError, match="^cpu: the compile failed"):
        build_all(tmp_path)
    assert len(finished) == 2
