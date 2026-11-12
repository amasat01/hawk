# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The kernels a build makes of a split kernel: the team stage and the warp-form kernel."""

from __future__ import annotations

from typing import Any

from ..ir import canonical
from ..trace.kernel import Kernel, _declared
from ..types import Slot


class TeamStage(Kernel):
    """A team stage of a split kernel (:mod:`simt.team`): the launch that works one matrix per
    lane with a team of threads. Its sinks only name the scratch planes it works on; its body is
    :func:`simt.team_emit.team_body`, handed to the entry wrappers like a fused group's."""

    __slots__ = ()

    def __init__(self, name: str, sinks: tuple, team: Any, planes: dict[str, Any] | None = None) -> None:
        super().__init__(name, sinks, planes)
        self.team = team
        if team.reads:          # input planes the stage reads in place are slots of its own
            self.walk = canonical(self.sinks, (*_declared(self.planes), *(Slot(leaf.role, leaf.name, leaf.ttype) for leaf in team.reads)))

    def fused_body(self, kind: Any = None, cells: bool = False, *, spec) -> Any:
        from .bundle import _active_spec
        from ..emit.render import Body
        from ..emit.wrapper import active_reads
        from .. import components as _components

        return _components.view("lowering", "team_emit").team_body(self.team, _active_spec(kind), cells, body=Body, active_reads=active_reads)


class WarpStage(TeamStage):
    """A kernel in the warp form (:mod:`simt.lane`): one block of a warp per lane, its sinks the sliced ones. Its body is
    the ordinary render of those sinks; the launch prologue binds the lane's place in its warp."""

    __slots__ = ()

    def fused_body(self, kind: Any = None, cells: bool = False, *, spec) -> Any:
        from ..emit.render import render_body

        if self.one_step is not None:       # an automatic kernel renders its two bodies itself
            return None
        return render_body(self.sinks, self.walk, indent="        ", kind=kind, spec=spec)
