# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The fused-lane emitter's BODY path.

Lane fusion is a HAWK codegen surface because a fused emitter must see
``Walk``: lane boundaries, shared leaves and the merged ``arg_spec`` are
walk products, and an out-of-tree fuser would be a second enumeration.
This module delivers the BODY half — one body string per fused lane
group, each lane rendered by the same renderer every backend consumes
into its own scope, opened by a lane-local prologue hook the composing
consumer supplies. :func:`compose` closes the other half: it takes a
group whose membership the consumer decided and turns it into a
deployable kernel — the merged ``arg_spec``, one index prologue, one
body, and a :class:`~hawk.types.LaneMeta` record per lane in the sidecar.

Composition is still the consumer's; :func:`compose` only refuses, naming
both lanes, the disagreements a fused entry cannot survive: one NAME
meaning two different things across lanes (a binding hazard invisible at
the call site, since a consumer binds by name), and two lanes committing
to the same output plane (which would write it twice in one launch). A
mapreduce sink beside anything else is refused by ``canonical()`` itself,
the same way it refuses a single kernel's, since ``exec_access`` is one
scalar per manifest.

Everything else is the walk's own doing: the merged ``arg_spec`` IS
``canonical(every lane's sinks).arg_spec``, so a leaf two lanes share is
bound once by the same merge that dedups a leaf inside one kernel. A
lane scope is a real C++ block, so a lane's ``const auto`` names cannot
leak into the next lane's — two lanes reading the same leaf each bind
their own register copy, and the fused entry's one index prologue binds
the ``i`` they all read.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..ir import HawkError, Walk, canonical
from ..ir.nodes import LEAF_ROLES, SINK_ROLES
from ..types import LaneMeta, Slot
from .aether import Body, render_body


@dataclass(frozen=True)
class Lane:
    """One lane of a fused group: its sinks, its walk and its prologue
    hook — text the composing consumer splices at the top of this lane's
    scope; HAWK never invents one."""

    name: str
    sinks: tuple
    walk: Walk
    prologue: str = ""


@dataclass(frozen=True)
class FusedBody:
    """ONE body string for a fused lane group, plus its per-lane metadata."""

    text: str
    lanes: tuple = field(default=())
    reduce_op: str | None = None
    #: True the instant any lane's render used a random op — the OR of
    #: each lane's :attr:`~hawk.emit.aether.Body.needs_random`.
    needs_random: bool = False

    def as_body(self) -> Body:
        """The :class:`~hawk.emit.aether.Body` a backend wrapper consumes.

        A fused group's body is a body: the entry wrapper, unpack and
        index prologue are all the backend's, same as for a solo kernel."""
        return Body(self.text, self.reduce_op, self.needs_random)


def render_lane_body(lanes: Sequence[Lane], *, indent: str = "        ",
                     kind=None) -> FusedBody:
    """Render ONE body for a fused lane group (the body path).

    Every lane goes through :func:`hawk.emit.aether.render_body` — the
    same renderer, so a fused lane's text is what it would emit alone.
    ``kind`` is the :class:`hawk.ext.Kind` every lane's commits render
    under: one kind for the group, since a fused group is one entry."""
    lanes = tuple(lanes)
    if not lanes:
        raise HawkError(
            "render_lane_body(): an empty lane group — a fused body is one or more "
            "lanes, and which lanes fuse is the composing consumer's decision"
        )
    seen = set()
    chunks, meta, reduce_ops = [], [], []
    needs_random = False
    for lane in lanes:
        if lane.name in seen:
            raise HawkError(
                f"fused lane group: two lanes are both named {lane.name!r}; a lane "
                "name identifies its scope and its LaneMeta record"
            )
        seen.add(lane.name)
        body = render_body(lane.sinks, lane.walk, indent=indent + "    ", kind=kind)
        if body.reduce_op is not None:
            reduce_ops.append(body.reduce_op)
        needs_random = needs_random or body.needs_random
        chunks.append(f"{indent}{{  // lane {lane.name}")
        if lane.prologue:
            chunks.extend(f"{indent}    {line}"
                          for line in lane.prologue.splitlines())
        chunks.append(body.text)
        chunks.append(f"{indent}}}  // lane {lane.name}")
        meta.append(LaneMeta(lane.name, lane.walk.arg_spec, lane.walk.digest))
    return FusedBody("\n".join(chunks), tuple(meta),
                     reduce_ops[0] if reduce_ops else None, needs_random)


@dataclass(frozen=True)
class FusedKernel:
    """A composed lane group, shaped like a kernel so it deploys like one.

    ``name``, ``sinks`` and ``walk`` are what :mod:`hawk.artifact` reads
    off any kernel, so a fused group goes through the ordinary emit ->
    compile -> publish path. ``fused_body`` is the one hook that differs.
    ``lanes`` is the per-lane metadata that rides into the sidecar."""

    name: str
    members: tuple
    sinks: tuple
    walk: Walk
    lanes: tuple
    scalar_type: str = "float64"

    @property
    def arg_spec(self) -> tuple:
        """The MERGED slot order — one entry per distinct ``(role, name)``."""
        return self.walk.arg_spec

    def fused_body(self, kind=None) -> Body:
        """This group's body: one lane scope per member, one renderer."""
        return render_lane_body(self.members, kind=kind).as_body()


def _lane_of(member) -> Lane:
    """A group member as a :class:`Lane`; a traced kernel is adopted by
    its own name, sinks and walk."""
    if isinstance(member, Lane):
        return member
    for attribute in ("name", "sinks", "walk"):
        if not hasattr(member, attribute):
            raise HawkError(
                f"compose(): {member!r} is neither a Lane nor a traced kernel — a "
                f"group member must carry {attribute!r}")
    return Lane(member.name, tuple(member.sinks), member.walk)


def _check_slots(lanes: Sequence[Lane]) -> None:
    """Refuse the two cross-lane disagreements a merged ``arg_spec`` can't
    carry, naming both lanes."""
    seen: dict = {}
    committed: dict = {}
    for lane in lanes:
        for role, name in lane.walk.arg_spec:
            ttype = lane.walk.slot_types[(role, name)]
            held = seen.setdefault(name, (lane.name, role, ttype))
            if held[1:] != (role, ttype):
                raise HawkError(
                    f"fused group: lane {held[0]!r} binds {name!r} as "
                    f"(role={held[1]!r}, {held[2]!r}) but lane {lane.name!r} binds it "
                    f"as (role={role!r}, {ttype!r}). A fused entry's merged arg_spec "
                    "is bound BY NAME by whoever launches it, so one name may mean "
                    "exactly one thing across the group — split the lanes, or rename "
                    "one plane")
            if role not in SINK_ROLES:
                # A read-only role can never double-commit. Not `role in
                # LEAF_ROLES`: `mutable` is now BOTH, and this is about WRITES.
                continue
            owner = committed.setdefault((role, name), lane.name)
            if owner != lane.name:
                raise HawkError(
                    f"fused group: lanes {owner!r} and {lane.name!r} both commit to "
                    f"the output plane (role={role!r}, name={name!r}). One launch of "
                    "the fused entry would write it twice, and which write survived "
                    "would be the order the consumer happened to pass the lanes in")


def compose(group: Sequence, name: str, scalar_type: str = "float64",
            *, kind=None) -> FusedKernel:
    """Compose a lane GROUP into one deployable fused kernel (the open half).

    ``group`` is the membership the consumer decided, as :class:`Lane`
    records or traced kernels. ``scalar_type`` is asked for and refused
    here, so an unbuilt mode is refused outright rather than at compile
    time. Every slot each lane bound is carried into the merged walk as
    a declared slot, not only the ones its body reaches: a dropped
    ``terminated`` mask would give a fused entry a different signature."""
    from .backend import scalar_mode

    lanes = tuple(_lane_of(member) for member in group)
    if not lanes:
        raise HawkError(
            "compose(): an empty lane group — a fused entry is one or more lanes, "
            "and which lanes fuse is the composing consumer's decision")
    names = [lane.name for lane in lanes]
    if len(set(names)) != len(names):
        raise HawkError(
            f"compose(): two lanes share a name in {names}; a lane name identifies "
            "its scope and its LaneMeta record")
    mode = scalar_mode(scalar_type)
    _check_slots(lanes)

    sinks = tuple(sink for lane in lanes for sink in lane.sinks)
    declared = tuple(
        Slot(role, slot_name, lane.walk.slot_types[(role, slot_name)])
        for lane in lanes for role, slot_name in lane.walk.arg_spec
        if role in LEAF_ROLES
    )
    walk = canonical(sinks, declared)
    meta = tuple(LaneMeta(lane.name, lane.walk.arg_spec, lane.walk.digest)
                 for lane in lanes)
    fused = FusedKernel(name, lanes, sinks, walk, meta, mode.id)
    # Rendered once, here, so lanes that disagree on the sink/mask seams
    # fail at compose time rather than inside the artifact builder.
    fused.fused_body(kind)
    return fused
