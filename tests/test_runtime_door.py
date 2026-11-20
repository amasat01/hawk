# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.run`` runs through the runtime door of the host library: one door per library, kept with it, and a unit's
slots named by the sidecar in the order the door binds by."""

from __future__ import annotations

import _oracle as O
import numpy as np
from conftest import sidecar_of

from hawk import ospreys, runtime


def test_one_door_is_made_for_a_library_and_every_kernel_runs_through_it(built):
    """The door belongs to the opened library: asked twice it is the same object, and two loaded kernels hold it."""
    door = ospreys.serving("host").runtime_door()
    assert ospreys.serving("host").runtime_door() is door
    axpb = O.load(built["axpb"].directory, "axpb", sidecar_of(built["axpb"], "axpb"))
    vec3 = O.load(built["vec3"].directory, "vec3_scale", sidecar_of(built["vec3"], "vec3_scale"))
    assert axpb._door is door and vec3._door is door


def test_the_sidecar_states_the_slot_table_a_run_binds_by(built):
    """Every plane of ``arg_spec`` is a slot of the unit, named once in the sidecar's ``slots``."""
    for bundle_name, kernel_name in (("axpb", "axpb"), ("vec3", "vec3_scale"), ("gather", "gather")):
        sidecar = sidecar_of(built[bundle_name], kernel_name)
        assert sorted(sidecar["slots"]) == sorted(name for _role, name in sidecar["arg_spec"])
        assert len(set(sidecar["slots"])) == len(sidecar["slots"])


def test_a_run_writes_the_callers_array_in_place_and_leaves_no_state_behind(built):
    """Repeated runs on one loaded kernel give the one answer: the planes and plans a run makes do not outlive it."""
    kernel = O.load(built["axpb"].directory, "axpb", sidecar_of(built["axpb"], "axpb"))
    x = np.arange(16, dtype=float)
    for _ in range(50):
        y = np.zeros(16)
        runtime.run(kernel, x=x, a=2.0, b=1.0, y=y)
        np.testing.assert_array_equal(y, 2.0 * x + 1.0)
