# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A gather leaves a guard when its index is a constant or the warp lane; a runtime index stays."""

from __future__ import annotations

import hawk
from hawk import _ir

_CONTEXT = _ir.new_context()
_I32, _F64 = ((), "i32", None), ((), "f64", None)
_TABLE = _ir.leaf(_CONTEXT, "table_read", "lookup", "w", ((64,), "f64", None))
_INDEX = _ir.make_node("LoopIndex", _I32, context=_CONTEXT, depth=0, name="r", bounds=None)


def _warp_lane():
    return _ir.make_node("WarpLane", _I32, context=_CONTEXT)


def _arm_boundary(index):
    """The boundary of a loop body whose ``select`` arm reads the table at ``index``; returns the arm and the boundary."""
    read = _ir.make_node("At", _F64, [_TABLE, index])
    arm = _ir.make("add", (read, _ir.const(_CONTEXT, 1.0, _F64)))
    cond = _ir.make("lt", (_INDEX, _ir.const(_CONTEXT, 3, _I32)))
    body = _ir.make_node("Select", _F64, [cond, arm, _ir.const(_CONTEXT, 2.0, _F64)])
    return arm, _ir.loop_boundary((body,), {_INDEX})


def _leaves_the_guard(index):
    arm, edge = _arm_boundary(index)
    return arm in edge


def test_a_compile_time_or_warp_lane_index_is_not_a_guarded_gather():
    assert _leaves_the_guard(_ir.const(_CONTEXT, 5, _I32))
    assert _leaves_the_guard(_ir.make("add", (_ir.const(_CONTEXT, 8, _I32), _warp_lane())))


def test_a_runtime_index_stays_in_the_guard():
    assert not _leaves_the_guard(_INDEX)
    runtime = _ir.leaf(_CONTEXT, "uniform", "uniform", "row", _I32)
    assert not _leaves_the_guard(_ir.make("add", (_warp_lane(), runtime)))


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
