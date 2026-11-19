# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The form a per-lane eigh takes: a team stage only when the form or the size asks for one.

A matrix of at most 32 rows is worked by one thread (the per-lane form) under ``linalg_form="auto"`` and ``"thread"``; the
form ``"team"`` and a matrix above 32 rows make a team stage. The sidecar of the build states each stage's ``team`` record,
so the rows read the stages from the unit the library built, on the host and, where a device toolchain exists, on the device.
"""

from __future__ import annotations

import pytest
from _linalg_kernels import eigh_kernel, eigh_values_kernel
from conftest import sidecar_of

from hawk import _ir
from hawk.artifact import build_bundle, describe

#: The context of the shared kernel builders.
CTX = _ir.new_context()

DEVICE = pytest.mark.gpu

TARGETS = [pytest.param("host", id="host"), pytest.param("cuda", marks=DEVICE, id="cuda")]


def _team_stages(kernel, tmp_path, target, *, mode="float32", form="auto"):
    """The ``team`` records of the stages the sidecar of the ``target`` build of ``kernel`` states, in launch order."""
    bundle = build_bundle([kernel], tmp_path, targets=(target,), mode=mode, host_profile="native", linalg_form=form)
    return [stage["team"] for stage in sidecar_of(bundle, kernel.name).get("stages", []) if "team" in stage]


PER_LANE = [pytest.param("host", c, id=f"host-{c}") for c in ("float32", "float64", "thread", "vectors", "n31")]
PER_LANE += [pytest.param("cuda", c, marks=DEVICE, id=f"cuda-{c}") for c in ("float64", "thread", "vectors", "n31")]


@pytest.mark.parametrize("target,case", PER_LANE)
def test_a_matrix_of_at_most_32_rows_keeps_the_per_lane_form(case, target, tmp_path):
    kernel = {"vectors": eigh_kernel(CTX, 32), "n31": eigh_values_kernel(CTX, 31)}.get(case, eigh_values_kernel(CTX, 32))
    mode = "float64" if case == "float64" else "float32"
    form = "thread" if case == "thread" else "auto"
    assert _team_stages(kernel, tmp_path, target, mode=mode, form=form) == []


def test_a_float32_values_only_eigen_solve_of_at_most_32_rows_on_a_device_is_a_copy_team_then_a_solve_team_of_32_threads():
    """The sidecar a device build would write, read before any build, so it holds on every machine. The copy stage's team is wider
    than the solve's: its thread count is the device card's, which hawk does not expose, so the row pins its shape and not its value."""
    kernel = eigh_values_kernel(CTX, 32)
    stages = describe([kernel], targets=("cuda",), mode="float32")[kernel.name]["stages"]
    teams = [stage["team"] for stage in stages if "team" in stage]
    assert len(teams) == 2 and teams[1] == {"threads": 32}, teams
    assert set(teams[0]) == {"threads"} and teams[0]["threads"] > 32, teams


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("case", ["team", "n33"])
def test_the_team_form_and_a_matrix_above_32_rows_make_a_team_stage(case, target, tmp_path):
    kernel = eigh_values_kernel(CTX, 33 if case == "n33" else 32)
    assert _team_stages(kernel, tmp_path, target, form="team" if case == "team" else "auto")
