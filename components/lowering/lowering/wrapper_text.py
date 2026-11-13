# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The refusals of an entry wrapper that every lowering form carries: a banded build off the device, a banded team
reduction, and the sample sums an entry cannot serve. The helper texts are on the emit side (:mod:`aether.wings.base.wrapper_text`)."""

from __future__ import annotations

from raptor.errors import RaptorError


REDUCTION_REMEDY = "until the banded reduction lands; build this kernel in float64 or float32 meanwhile"


def refuse_cells_off_device(name: str, mode, target_id: str, target) -> None:
    """Refuse a ``cells`` (banded) build for a target that is not a device: banded is a device feature."""
    if mode.cells and not target.device:
        raise RaptorError(f"{name!r}: banded runs on device execution paths; on a CPU use float64 "
                        f"(target {target_id!r} is not a device)")


def refuse_team_reductions(team, mode) -> None:
    """Refuse a ``cells`` build of a block sum that is not the ordered plain sum (``team`` is its
    :class:`~simt.team.TeamSpec`): the general block sum is an aether team call that adds cells with a plain ``+=``,
    which would be the plain fold under the sum's name. The ordered stages fold ``BandAccum`` partials."""
    for step in () if team is None or not mode.cells else team.steps:
        if step.op == "block_sum" and "gp" not in dict(step.planes):
            raise RaptorError(f"a block sum over matrices ({step.op}) is not built in scalar mode {mode.id!r} yet: "
                            f"{REDUCTION_REMEDY}")

def refuse_sample_sums(sums, *, segment, active, one_step) -> None:
    """A row sums 32 consecutive samples of ONE warp; the segmented,
    active-set and ``_persist`` entries map a warp onto other samples."""
    entry = ("segmented" if segment is not None
             else "active-set" if active is not None and active.map_binding is not None
             else "_persist" if one_step is not None else None)
    if len({s.op for s in sums}) > 1:
        raise RaptorError(
            f"sample_sum {[s.name for s in sums]}: one kernel reduces over samples "
            "with one op (sum, max or min) in this release; split the kernel")
    if sums and entry is not None:
        raise RaptorError(
            f"sample_sum {[s.name for s in sums]}: a row is the sum over one "
            f"warp of 32 consecutive samples, and the {entry} entry's warp does not "
            "hold 32 consecutive samples; launch this kernel on its plain entry")
