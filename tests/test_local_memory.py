# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The local memory one thread of a device unit holds is recorded in the sidecar as the device library states it.

The library refuses a unit past the device's per-thread limit by name (its own tests, with fake tools); here the sidecar of a
real device build carries the record the runtime reads at bind.
"""

from __future__ import annotations

from _linalg_kernels import lu_kernel
from conftest import sidecar_of

from hawk import _ir
from hawk.artifact import build_bundle

#: The device's per-thread local memory limit the card states (512 KB).
LOCAL_LIMIT_B = 512 * 1024


def test_a_team_form_matrix_holds_no_per_thread_frame_and_the_sidecar_states_the_limit(tmp_path, cache_dir):
    bundle = build_bundle([lu_kernel(_ir.new_context(), 71)], tmp_path, targets=("cuda",), cache_dir=cache_dir)
    launch = sidecar_of(bundle, "la_lu_71")["launch"]
    assert launch["local_limit"] == LOCAL_LIMIT_B and launch["local_source"] in ("ptx", "ptxas")
    assert 0 <= launch["local_bytes"] < LOCAL_LIMIT_B // 8       # the matrix lives in team scratch
