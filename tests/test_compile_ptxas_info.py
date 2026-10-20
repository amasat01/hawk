# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The per-kernel ptxas figures (registers, spills, stack frame) a device
compile reports are parsed off nvcc's stderr and recorded in the artifact's
sidecar. Text-level: a captured stderr fixture, no compiler run."""

from __future__ import annotations

import hawk
import hawk.math as M
from hawk.artifact.sidecar import sidecar_meta
from hawk.compile import toolchain as tc
from hawk.compile.drivers import parse_ptxas_info

_STDERR = """\
ptxas info    : 213 bytes gmem, 24 bytes cmem[4]
ptxas info    : Compiling entry function 'kan_routed' for 'sm_61'
ptxas info    : Function properties for kan_routed
    208 bytes stack frame, 212 bytes spill stores, 216 bytes spill loads
ptxas info    : Used 255 registers, used 0 barriers, 496 bytes cmem[0], 972 bytes cmem[2]
ptxas info    : Compiling entry function 'other' for 'sm_61'
ptxas info    : Function properties for other
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 96 registers, used 0 barriers, 400 bytes cmem[0]
"""

_WANT = {"kan_routed": {"regs": 255, "stack_B": 208, "spill_store_B": 212,
                        "spill_load_B": 216},
         "other": {"regs": 96, "stack_B": 0, "spill_store_B": 0, "spill_load_B": 0}}


@hawk.kernel
def _k(a: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    y = M.exp(a)


def test_ptxas_info_is_parsed_per_kernel():
    assert parse_ptxas_info(_STDERR) == _WANT
    assert parse_ptxas_info("nothing to see") == {}


def test_ptxas_is_asked_to_report_only_when_it_runs():
    assert "-v" in tc.device_flags(arch="sm_61", target="cubin")
    assert "-v" not in tc.device_flags(arch="compute_61", target="ptx")


def test_sidecar_carries_the_ptxas_figures_when_given():
    kw = dict(fmt="cubin", scalar_type="float64")
    assert sidecar_meta("k", _k.walk, ptxas=parse_ptxas_info(_STDERR), **kw)["ptxas"] == _WANT
    assert "ptxas" not in sidecar_meta("k", _k.walk, **kw)
