# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A gather leaves a guard when its index is a constant or the warp lane; a runtime index stays."""

from __future__ import annotations

import hawk
from simt import WarpLane
from hawk.ir.loops import boundary
from hawk.ir.nodes import At, Const, Leaf, LoopIndex, Select, TensorType
from hawk.ir.ops import make

_I32, _F64 = TensorType((), "i32"), TensorType((), "f64")
_TABLE = Leaf("table_read", "lookup", "w", TensorType((64,), "f64"))
_INDEX = LoopIndex(0, "r")


def _arm_boundary(index):
    """The boundary of a loop body whose ``select`` arm reads the table at ``index``; returns the arm and the boundary."""
    arm = make("add", (At(_TABLE, index, _F64), Const(1.0, _F64)))
    body = Select(make("lt", (_INDEX, Const(3, _I32))), arm, Const(2.0, _F64), _F64)
    return arm, boundary((body,), frozenset({id(_INDEX)}))


def _leaves_the_guard(index):
    arm, edge = _arm_boundary(index)
    return any(node is arm for node in edge)


def test_a_compile_time_or_warp_lane_index_is_not_a_guarded_gather():
    assert _leaves_the_guard(Const(5, _I32))
    assert _leaves_the_guard(make("add", (Const(8, _I32), WarpLane())))


def test_a_runtime_index_stays_in_the_guard():
    assert not _leaves_the_guard(_INDEX)
    runtime = Leaf("uniform", "uniform", "row", _I32)
    assert not _leaves_the_guard(make("add", (WarpLane(), runtime)))


def test_the_jvp_of_a_fused_matvec_reads_its_table_before_the_step_loop(tmp_path):
    """The loop-weighted FP64 count, as the lowering decides it: every table gather of ``w`` and ``dot_w`` sits before
    the K-step loop (executed once), none inside (executed K times). It was 48 gathers x 8 steps inside."""
    import re

    import _wide_kernels as wk
    import pytest
    from hawk.diff import step_jvp

    pytest.importorskip("cuda.bindings.nvrtc")
    kernel = hawk.steps(step_jvp(wk.steps_matvec_k1), 8)
    hawk.build([kernel], str(tmp_path), targets=("cuda",), device_arch="sm_89")
    text = (tmp_path / f"{kernel.name}.cu").read_text()
    head, body = text.split("_step = 0;", 1)
    gather = r"lut_(?:dot_)?w\["
    assert len(re.findall(gather, head)) >= 48 and not re.findall(gather, body)
