# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The aether writer's spellings of the simt nodes (registered; the writer module names none of them)."""

from __future__ import annotations

from types import MappingProxyType

from raptor.wings.simt.lane import slice_min


def warp_lane_value(w, r) -> str:
    """A ``WarpLane`` node: the lane's place in its warp (the team prologue binds ``hawk_wl``)."""
    return "hawk_wl"


def warp_store(w, r, sink, target: str, value: str) -> None:
    """The lane's slice of a wide value into elements ``j * 32 + lane`` of the Mutable's plane."""
    rows = r.spec.settings.staged_rows
    if rows > 1 and sink.ttype.shape and sink.ttype.shape[0] >= slice_min(sink.ttype.dtype, r.spec):        # the tile of the block: the copy-out after the body stores it coalesced
        n, width = sink.value.ttype.shape[0], sink.ttype.shape[0]
        tile = f"hawk_stg_{target}"
        r._emit("{")
        r._emit(f"const aether::Item<Real, {n}> hawk_ws = {value};")
        if n == width:       # a value every lane holds: lane 0 puts it in the tile
            r._emit(f"if (hawk_wl == 0) {{ for (Int hawk_lk = 0; hawk_lk < {n}; ++hawk_lk) {tile}[hawk_lk * {rows + 1} + hawk_wb] = hawk_ws.data()[hawk_lk]; }}")
        else:
            r._emit("#pragma unroll")
            r._emit(f"for (Int hawk_lk = 0; hawk_lk < {n}; ++hawk_lk) {tile}[(hawk_lk * 32 + hawk_wl) * {rows + 1} + hawk_wb] = hawk_ws.data()[hawk_lk];")
        r._emit("if (hawk_wl == 0) hawk_wrote[hawk_wb] = 1;")
        r._emit("}")
        return
    if sink.value.ttype.shape == sink.ttype.shape:      # the same in every lane
        r._emit(f"if (hawk_wl == 0) {target}[i] = {value};")
        return
    n = sink.value.ttype.shape[0]
    r._emit("{")
    r._emit(f"const aether::Item<Real, {n}> hawk_ws = {value};")
    r._emit("#pragma unroll")
    r._emit(f"for (Int hawk_lk = 0; hawk_lk < {n}; ++hawk_lk) "
            f"{target}[i].at(hawk_lk * 32 + hawk_wl) = hawk_ws.data()[hawk_lk];")
    r._emit("}")


#: The writer's spellings by (dialect, name): ``fn(w, r, *args)``, a read-only table written after the functions it names. A build reads
#: them through the manifest's spellings arm.
SPELLINGS = MappingProxyType({
    ("simt", "warp_lane_value"): warp_lane_value,
    ("simt", "warp_store"): warp_store,
})
