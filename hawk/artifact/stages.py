# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The kernels a build makes of a split kernel: the team stage and the warp-form kernel."""

from __future__ import annotations

from typing import Any

from ..trace.kernel import Kernel, _declared
from raptor.ir.types import Slot
from .. import wings as _wings


class TeamStage(Kernel):
    """A team stage of a split kernel (:mod:`raptor.wings.simt.team`): the launch that works one matrix per
    lane with a team of threads. Its sinks only name the scratch planes it works on; its body is
    :func:`aether.wings.base.team_emit.team_body`, handed to the entry wrappers like a fused group's."""

    __slots__ = ()

    #: Its body is rendered once per target that wraps it (:meth:`hawk.backends.Backend.render`): a device row's text holds the device arms.
    per_target_body = True

    def __init__(self, name: str, sinks: tuple, team: Any, planes: dict[str, Any] | None = None) -> None:
        super().__init__(name, sinks, planes)
        self.team = team
        if team.reads:          # input planes the stage reads in place are slots of its own
            self.walk = _wings.view("raptor", "base", "cells").walk(self.sinks, (*_declared(self.planes), *(Slot(leaf.role, leaf.name, leaf.ttype) for leaf in team.reads)))

    def fused_body(self, kind: Any = None, cells: bool = False, *, spec, writer, device_text: Any = None) -> Any:
        from aether.wings.base.team_emit import stage_body

        from ..backends import need_device
        from .bundle import _active_spec

        return stage_body(self.team, _active_spec(kind), need_device().wrapper.device_text.team_cells if cells else None,
                          device_text)


class WarpStage(TeamStage):
    """A kernel in the warp form (:mod:`raptor.wings.simt.lane`): one block of a warp per lane, its sinks the sliced ones. Its body is
    the ordinary render of those sinks; the launch prologue binds the lane's place in its warp."""

    __slots__ = ()

    per_target_body = False

    def fused_body(self, kind: Any = None, cells: bool = False, *, spec, writer) -> Any:
        from aether.wings.base.render import render_body

        if self.one_step is not None:       # an automatic kernel renders its two bodies itself
            return None
        return render_body(self.sinks, self.walk, indent="        ", kind=kind, writer=writer, spec=spec)
