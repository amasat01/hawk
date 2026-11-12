# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""An eigenvalues-only f32 ``n = 32`` eigh is a one-warp team stage (``aether::linalg::warp::eighValues``).

The rule is a build-time one: a float32 build, ``linalg_form="auto"``, ``n == 32``, no eigenvector read, and a stage
of its own possible. The warp form is bit-identical to the thread form and was faster at every lane count measured
(Quadro P2000, f32, 256 to 131072 matrices: 24.5 vs 4.3, 52.8 vs 27.3, 176 vs 107, 659 vs 423, 2578 vs 1689 ms
thread vs warp), so no run-time fill test decides it.
"""

from __future__ import annotations
from _specs import DEVICE, HOST, device

import pytest
from _linalg_kernels import eigh_kernel, eigh_values_kernel

from hawk.passes import policy
from lowering.split import stages


def _ops(kernel, *, scalar="float32", form="auto", device_only=True, spec=DEVICE):
    found = stages("k", kernel.sinks, spec=spec.with_settings(linalg_form=form, scalar=scalar, device_only=device_only))
    return [] if found is None else [(st.team.threads, s.op) for st in found if st.team for s in st.team.steps]


def test_float32_values_only_n32_is_one_warp_stage():
    assert _ops(eigh_values_kernel(32)) == [(32, "eigh_warp")]


@pytest.mark.parametrize("arch,warp", [("sm_61", False), ("sm_89", False), ("", False), ("sm_80", True), ("sm_90", True)])
def test_float64_takes_the_warp_form_only_where_fp64_is_fast(arch, warp):
    from hawk.targets import cards

    spec = device(arch)
    got = _ops(eigh_values_kernel(32), scalar="float64", spec=spec)
    assert (got == [(32, "eigh_warp")]) is warp


@pytest.mark.parametrize("case", ["float64", "no_scalar", "thread", "team", "vectors", "n31", "n33"])
def test_every_other_build_keeps_its_form(case):
    kernel = {"vectors": eigh_kernel(32), "n31": eigh_values_kernel(31), "n33": eigh_values_kernel(33)}.get(case, eigh_values_kernel(32))
    scalar = {"float64": "float64", "no_scalar": ""}.get(case, "float32")
    form = case if case in ("thread", "team") else "auto"
    ops = _ops(kernel, scalar=scalar, form=form)
    assert all(op != "eigh_warp" for _, op in ops)
    if case in ("team", "n33"):
        assert ops and ops[0][1] == "eigh"          # a team stage, never the warp one
    else:
        assert ops == []                            # the per-lane (thread) form: no team stage at all


def test_a_build_with_a_host_target_keeps_the_thread_form():
    """The host has no warp: its team stage runs the 32 lanes in turn (R3: 5x slower on the CPU card)."""
    assert _ops(eigh_values_kernel(32), device_only=False) == []
