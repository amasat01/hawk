# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The warp form of a wide kernel (``wide_form="team"``): one warp per sample, every wide value sliced across its lanes.

A value of ``W`` elements (``W`` a multiple of :data:`WARP`) is held as ``W / WARP`` elements per lane: element ``j`` of
lane ``l`` is element ``j * WARP + l`` of the value. An elementwise operation, a splat, a read of a plane at an
arithmetic run of indices and a loop carrying such values all act on the slice alone, so the 32 lanes share the work and
each keeps 1/32 of the value in registers. A narrow value (a scalar, a short vector) is computed by every lane alike.

:func:`slice_sinks` rewrites the sinks of a kernel into that form, or returns ``None`` when a node has no rule here
(then the kernel keeps its thread form). The stores come back as :class:`WarpStore`, which writes the slice back into the
Mutable's plane at the lane's own elements."""

from __future__ import annotations

from ..types import TensorType
from .loop_nodes import Loop, LoopCarry, LoopIndex, LoopValue
from .nodes import Assign, At, Const, Leaf, Node, Op, Select
from .ops import make

#: Lanes of the warp that shares one sample.
WARP = 32

#: The narrowest value worth slicing (a smaller one fits a thread's registers).
SLICE_MIN = 128

WARP_LANE_KIND = "warp_lane"


class WarpLane(Node):
    """The lane's place in its warp, ``0 .. WARP - 1`` (a terminal: it binds no slot)."""

    __slots__ = ()

    kind = WARP_LANE_KIND

    def __init__(self) -> None:
        super().__init__(TensorType((), "i32"))


class WarpStore(Assign):
    """The commit of a sliced value into the Mutable ``name``: element ``j`` of the slice goes to element
    ``j * WARP + lane`` of the plane. The sink keeps the plane's own type; its value is the slice."""

    __slots__ = ()


def sliced(ttype: TensorType) -> bool:
    """Whether a value of ``ttype`` is held sliced: a vector of a whole number of warps, wide enough to pay."""
    return len(ttype.shape) == 1 and ttype.shape[0] % WARP == 0 and ttype.shape[0] >= SLICE_MIN


def _slice_type(ttype: TensorType) -> TensorType:
    return TensorType((ttype.shape[0] // WARP,), ttype.dtype, ttype.tag)


class _Refused(Exception):
    """A node with no rule in the warp form."""


class _Slicer:
    def __init__(self) -> None:
        self.memo: dict = {}                       # id(node) -> (node, is_sliced)
        self.lane = WarpLane()

    def of(self, node: Node) -> tuple:
        got = self.memo.get(id(node))
        if got is None:
            got = self.memo[id(node)] = self._one(node)
        return got

    def _one(self, node: Node) -> tuple:
        if isinstance(node, (Const, LoopIndex, WarpLane)):
            return node, False
        if isinstance(node, LoopCarry):
            if sliced(node.ttype):
                raise _Refused("a sliced carry outside its loop")
            return node, False
        if isinstance(node, Leaf):
            if sliced(node.ttype):
                raise _Refused(f"a wide plane {node.name!r} read whole")
            return node, False
        if isinstance(node, LoopValue):
            loop, slot_sliced = self._loop(node.loop)
            return LoopValue(loop, node.slot), slot_sliced[node.slot]
        if isinstance(node, Loop):
            raise _Refused("a loop used as a value")
        if isinstance(node, At):
            return self._rep(node)
        if isinstance(node, (Op, Select)):
            return self._op(node)
        return self._rep(node)

    def _rep(self, node: Node) -> tuple:
        """A node every lane computes alike: none of its operands may be a slice."""
        ops = [self.of(o) for o in node.operands]
        if any(s for _, s in ops):
            raise _Refused(f"{type(node).__name__} {getattr(node, 'kind', '')} reads a slice")
        if all(a is b for (a, _), b in zip(ops, node.operands)):
            return node, False
        from .walk import _rebuild

        return _rebuild(node, [a for a, _ in ops]), False

    def _op(self, node: Node) -> tuple:
        if not sliced(node.ttype):
            return self._rep(node)
        kind = node.kind
        out = _slice_type(node.ttype)
        if kind == "splat":
            (a, s), = [self.of(o) for o in node.operands]
            if s:
                raise _Refused("a splat of a slice")
            return Op("splat", (a,), out, node.literal), True
        if kind == "vec":
            return self._gather(node, out), True
        ops = [self.of(o) for o in node.operands]
        if isinstance(node, Select) and node.operands[0].ttype.shape:
            raise _Refused("a select by a vector of conditions")
        from .wide import ELEMENTWISE

        if kind not in ELEMENTWISE:
            raise _Refused(f"{kind} of a wide value")
        for (_, s), o in zip(ops, node.operands):
            if not s and o.ttype.shape:
                raise _Refused("a narrow vector beside a wide one")
        new = [a for a, _ in ops]
        if isinstance(node, Select):
            return Select(new[0], new[1], new[2], out), True
        return Op(kind, tuple(new), out, node.literal), True

    def _gather(self, node: Node, out: TensorType) -> Op:
        """A ``vec`` of reads of one plane at an arithmetic run of indices: lane ``l`` reads the run's elements
        ``j * WARP + l``."""
        from .wide import _split_index, gather

        g = gather(node)
        if g is None:
            raise _Refused("a vector of parts that is not a run of reads")
        first = node.operands[0]
        index_type = first.index.ttype
        base = None if g.base is None else self.of(g.base)
        if base is not None and base[1]:
            raise _Refused("a gather based on a slice")
        parts = []
        for j in range(out.shape[0]):
            offset = g.start + j * WARP * g.step
            at = make("add", (make("mul", (self.lane, Const(g.step, index_type))), Const(offset, index_type)))
            if base is not None:
                at = make("add", (base[0], at))
            parts.append(At(g.plane, at, first.ttype))
        return Op("vec", tuple(parts), out)

    def _loop(self, loop: Loop) -> tuple:
        from .loops import build_loop

        got = self.memo.get(id(loop))
        if got is not None:
            return got
        if loop.exit_cond is not None or loop.count is not None or loop.body_sinks:
            raise _Refused("a loop with an exit, a run-time count or commits")
        slot_sliced = [sliced(c.ttype) for c in loop.carries]
        carries = [LoopCarry(c.depth, c.slot, c.name, _slice_type(c.ttype)) if s else c
                   for c, s in zip(loop.carries, slot_sliced)]
        for old, new in zip(loop.carries, carries):
            self.memo[id(old)] = (new, new is not old)
        inits, nexts = [], []
        for j, (init, nxt) in enumerate(zip(loop.inits, loop.nexts)):
            (i, si), (n, sn) = self.of(init), self.of(nxt)
            if si != slot_sliced[j] or sn != slot_sliced[j]:
                raise _Refused("a carry whose initial or next value is not held like the carry")
            inits.append(i)
            nexts.append(n)
        new_loop = build_loop(loop.index, carries, inits, nexts, start=loop.start, stop=loop.stop, step=loop.step,
                              dynamic=loop.dynamic)
        got = self.memo[id(loop)] = (new_loop, slot_sliced)
        return got


def slice_sinks(sinks) -> tuple | None:
    """``sinks`` in the warp form, or ``None`` when some node has no rule there. Every sink must commit a wide
    value to a Mutable."""
    sinks = tuple(sinks)
    if not sinks or not all(type(s) is Assign and sliced(s.ttype) for s in sinks):
        return None
    slicer = _Slicer()
    out = []
    try:
        for sink in sinks:
            value, is_sliced = slicer.of(sink.value)
            if not is_sliced:
                return None
            out.append(WarpStore(sink.name, value, sink.ttype))
    except _Refused:
        return None
    return tuple(out)
