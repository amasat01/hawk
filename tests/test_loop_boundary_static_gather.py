# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A gather leaves a guard when its index is a constant or the warp lane; a runtime index stays."""

from __future__ import annotations

import hawk
from hawk.ir.lane import WarpLane
from hawk.ir.loop_nodes import LoopIndex
from hawk.ir.loops import _gathers
from hawk.ir.nodes import At, Const, Leaf, TensorType
from hawk.ir.ops import make

_I32, _F64 = TensorType((), "i32"), TensorType((), "f64")
_TABLE = Leaf("table_read", "lookup", "w", TensorType((64,), "f64"))


def _read(index):
    return At(_TABLE, index, _F64)


def test_a_compile_time_or_warp_lane_index_is_not_a_guarded_gather():
    assert not _gathers(_read(Const(5, _I32)), {})
    lane = make("add", (Const(8, _I32), WarpLane()))
    assert not _gathers(_read(lane), {})
    assert not _gathers(make("add", (_read(lane), Const(1.0, _F64))), {})


def test_a_runtime_index_stays_in_the_guard():
    loop_index = LoopIndex(0, "r")
    assert _gathers(_read(loop_index), {})
    runtime = Leaf("uniform", "uniform", "row", _I32)
    assert _gathers(_read(make("add", (WarpLane(), runtime))), {})


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
