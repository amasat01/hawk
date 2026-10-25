# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The thread box: one thread computes ``T`` lanes along a non-fastest axis.

Lanes that differ only in that axis read the same values from every plane read
the lane facts show invariant along it (:func:`hawk.ir.lanes.facts`, its box
candidates). Boxed, a thread reads those once and keeps the ``T`` lanes' own
values in registers: a loop's carried values become one ``aether::Item<Real, T>``
each. The fastest axis stays consecutive threads, and every lane is written to
its own column, so memory is unchanged.

Thread ``t`` owns lanes ``own + k * stride`` for ``k < T`` that are below
``nSamples``, with ``own = t + (t / stride) * (T - 1) * stride``: a launch runs
``ceil(n / (T * stride)) * stride`` threads (:func:`threads`). A box that
would run past ``nSamples`` is computed shifted back onto the last ``T`` rows;
its lanes below ``own`` are computed and not stored (another thread owns
them), so every lane index stays ``first + k * stride``. A launch with fewer
than ``T`` rows runs the box-free body lane by lane (:func:`frame`).

Lowered as an IR copy: every node that may move along the axis is copied per
lane of the box, every other node is shared, and the one renderer writes the
result. Each lane's terms keep their order, so a lane's value is the box-free
one, bit for bit, on the same target.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..ir.loop_nodes import Loop, LoopCarry, LoopCount, LoopIndex, LoopValue, TapeRead
from ..ir.loops import build_loop
from ..ir.nodes import (
    RAW_KIND,
    Assign,
    At,
    Const,
    Dispatch,
    Leaf,
    Node,
    Op,
    SampleIndex,
    Sink,
    WideWrite,
)
from ..ir.ops import RANDOM_OPS
from ..ir.walk import _rebuild
from ..types import TensorType

#: ``HAWK_BOX=0`` builds every kernel without a box (an A/B switch, not an API).
BOX_ENV = "HAWK_BOX"

#: The prologue's name for the first lane a thread owns.
BOX_OWN = "hawk_box_own"


@dataclass(frozen=True)
class Box:
    """A thread box: ``T`` lanes along lane axis ``axis`` (by position), whose
    one step moves the flat lane by ``stride``. What the sidecar's
    ``launch.box`` carries."""

    axis: int
    T: int
    stride: int

    def sidecar(self) -> dict:
        return {"axis": self.axis, "T": self.T, "stride": self.stride}


def threads(box: dict | None, n: int) -> int:
    """How many threads cover ``n`` lanes under a sidecar ``launch.box`` (``n`` without one)."""
    if not box:
        return n
    span = box["T"] * box["stride"]
    return -(-n // span) * box["stride"]


def owned_lane(ctype: str, thread: str, box: Box) -> str:
    """The prologue line binding thread ``thread``'s first owned lane."""
    return (f"        const {ctype} {BOX_OWN} = {thread} + {thread} / {box.stride} * "
            f"{(box.T - 1) * box.stride};")


def frame(box: Box, plain: str, boxed: str) -> str:
    """What follows :func:`owned_lane` in the entry: the ``boxed`` body over the
    thread's box, or, when the launch has fewer than ``T`` rows, the box-free
    ``plain`` body over each owned lane in turn. Both bodies bind ``hawk_i`` and
    ``i``, so both read exactly as the box-free entry's."""
    T, S = box.T, box.stride
    bind = ("const aether::SampleIndex i =\n"
            "{pad}    aether::SampleIndex::make(static_cast<std::size_t>(hawk_i));\n"
            "{pad}(void)hawk_i;\n{pad}(void)nSamples;")
    pad8, pad12, pad16 = " " * 8, " " * 12, " " * 16
    return "\n".join([
        f"{pad8}if (nSamples < {T * S}) {{",
        f"{pad12}for (long long hawk_k = 0; hawk_k < {T}; ++hawk_k) {{",
        f"{pad16}const long long hawk_i = {BOX_OWN} + hawk_k * {S};",
        f"{pad16}if (hawk_i >= nSamples) break;",
        pad16 + bind.format(pad=pad16),
        plain,
        f"{pad12}}}",
        f"{pad8}}} else {{",
        f"{pad12}const long long hawk_i = {BOX_OWN} + {(T - 1) * S} < nSamples",
        f"{pad12}    ? {BOX_OWN} : nSamples - {T * S} + {BOX_OWN} % {S};",
        pad12 + bind.format(pad=pad12),
        boxed,
        f"{pad8}}}",
    ])


class BoxStore(Sink):
    """The own-column store of one lane of the box, at ``lane``, skipped for a
    lane the thread does not own."""

    __slots__ = ()

    def __init__(self, sink: Sink, value: Node, lane: Node) -> None:
        super().__init__("box_store", sink.role, sink.name, value, lane, sink.ttype)


def thread_box(kernel, scalar_type: str, *, kind=None, segment=None, active=None,
               one_step=None, body=None) -> Box | None:
    """The box ``kernel``'s plain entry is emitted with, or ``None``.

    Read off the lane facts' mapping with the device card's ``T``
    (:func:`hawk.targets.cards.thread_box`); declined for anything but a plain
    elementwise entry (an active set, a segmented unit, a two-path or fused
    body, a compensated kind), for a kernel whose control or commits are not
    lane-local (:func:`_boxable`), when the box would straddle a compaction
    group, and under ``HAWK_BOX=0``."""
    if (os.environ.get(BOX_ENV) == "0" or segment is not None or active is not None
            or one_step is not None or body is not None
            or getattr(getattr(kind, "sink", None), "id", None) == "compensated"):
        return None
    sinks = tuple(kernel.sinks)
    if not _boxable(sinks) or any(role == "terminated" for role, _ in kernel.walk.arg_spec):
        return None
    from ..ir import lanes
    from ..targets.cards import thread_box as card

    f = lanes.facts(kernel, group=getattr(kernel, "group", 1))
    if not f.uniform_control:
        return None
    p1 = lanes.choose_mapping(f, card=card(scalar_type))["P1"]
    if p1 is None:
        return None
    T, axis = p1["T"], f.axes[p1["index"]]
    if T < 2 or (axis.extent is not None and axis.extent % T) \
            or (f.group > 1 and f.group % (T * axis.stride)):
        return None
    return Box(p1["index"], T, axis.stride)


def _boxable(sinks: tuple) -> bool:
    """Every commit is a scalar own-column store and every node is lane-local:
    no per-lane input read but a scalar one, no loop that exits, counts, commits
    or nests, a carried value only a scalar real, no dispatch, random draw or
    raw block (each reads the lane by its own name)."""
    from ..ir.lanes import _nodes_of

    if not all(isinstance(s, (Assign, WideWrite)) and s.index is None and not s.ttype.shape
               for s in sinks):
        return False
    for n in _nodes_of(sinks):
        if isinstance(n, (Dispatch, TapeRead, LoopCount)) or (
                isinstance(n, Op) and (n.kind in RANDOM_OPS or n.kind == RAW_KIND)):
            return False
        if isinstance(n, Leaf) and (n.role in ("vec_in", "mat_in", "mutable", "terminated")
                                    or n.role == "per_sample" and n.ttype.shape):
            return False
        if isinstance(n, Loop) and (n.exit_cond is not None or n.count is not None
                                    or n.body_sinks
                                    or any(isinstance(b, Loop) for b in n.body_nodes())
                                    or any(c.ttype.shape or c.ttype.dtype not in ("f64", "f32")
                                           for c in n.carries)):
            return False
    return True


def boxed_sinks(sinks, box: Box) -> tuple:
    """The sink set of ``box``'s entry: each sink once per lane of the box."""
    from ..ir.lanes import varies_along

    sinks = tuple(sinks)
    copy = _Copy(varies_along(sinks, box.axis), box)
    return tuple(BoxStore(s, copy(s.value, k), copy.lanes[k])
                 for s in sinks for k in range(box.T))


class _Copy:
    """Per-lane copies of a node graph. A node that may move along the box axis
    is copied for lane ``k`` (the lane ``i + k * stride``, a per-lane input read
    at that lane, a loop fused); any other node is lane 0's, shared by all. A
    lane coordinate ``i / D`` (``D`` dividing the stride) is lane 0's plus
    ``k * stride / D``, so lane ``k``'s addresses stay lane 0's plus a constant."""

    def __init__(self, varies, box: Box) -> None:
        self.varies, self.T, self.stride = varies, box.T, box.stride
        lane = SampleIndex()
        self.lanes = [lane] + [Op("add", (lane, Const(k * box.stride, lane.ttype)), lane.ttype)
                               for k in range(1, box.T)]
        self.memo: dict = {}
        self.fused: dict = {}

    def __call__(self, root: Node, k: int) -> Node:
        memo = self.memo
        stack = [(root, k)]
        while stack:
            node, kk = stack[-1]
            if (id(node), kk) in memo:
                stack.pop()
                continue
            if kk and not self.varies(node):
                if (id(node), 0) in memo:
                    memo[(id(node), kk)] = memo[(id(node), 0)]
                    stack.pop()
                else:
                    stack.append((node, 0))
                continue
            todo = [(c, kk) for c in self._operands(node) if (id(c), kk) not in memo]
            if todo:
                stack.extend(todo)
                continue
            stack.pop()
            memo[(id(node), kk)] = self._build(node, kk)
        return memo[(id(root), k)]

    def _fusing(self, node: Node) -> bool:
        """A loop's result that moves along the axis: read off the fused loop."""
        return isinstance(node, LoopValue) and self.varies(node.loop)

    def _operands(self, node: Node) -> tuple:
        if isinstance(node, At):
            return (node.index,)        # the plane is read, never copied
        if isinstance(node, (Leaf, Const, SampleIndex, LoopIndex, LoopCarry)) \
                or self._fusing(node):
            return ()
        return node.operands

    def _build(self, node: Node, k: int) -> Node:
        if isinstance(node, SampleIndex):
            return node if k == 0 else self.lanes[k]
        if k and _divides(node, self.stride):
            step = k * self.stride // int(node.operands[1].literal)
            return Op("add", (self(node, 0), Const(step, node.ttype)), node.ttype)
        if isinstance(node, Leaf):
            # reached only as an own-column read: lane k's own value
            if k == 0 or node.role != "per_sample":
                return node
            return At(node, self.lanes[k], node.ttype)
        if isinstance(node, (Const, LoopIndex)):
            return node
        if self._fusing(node):
            return _component(LoopValue(self._fuse(node.loop), node.slot), k)
        kids = [self.memo[(id(c), k)] for c in self._operands(node)]
        if isinstance(node, At):
            kids = [node.plane, *kids]
        if all(a is b for a, b in zip(kids, node.operands)):
            return node
        return _rebuild(node, kids)

    def _fuse(self, loop: Loop) -> Loop:
        """``loop`` once for the whole box: carried slot ``j`` becomes an
        ``Item`` of the ``T`` lanes' values, read back per lane by component."""
        held = self.fused.get(id(loop))
        if held is not None:
            return held
        T = self.T
        carries = [LoopCarry(c.depth, c.slot, c.name, TensorType((T,), c.ttype.dtype))
                   for c in loop.carries]
        for old, new in zip(loop.carries, carries):
            for k in range(T):
                self.memo[(id(old), k)] = _component(new, k)
        for k in range(T):
            self.memo[(id(loop.index), k)] = loop.index
        inits = [_vec([self(x, k) for k in range(T)], c.ttype)
                 for x, c in zip(loop.inits, carries)]
        nexts = [_vec([self(x, k) for k in range(T)], c.ttype)
                 for x, c in zip(loop.nexts, carries)]
        fused = build_loop(loop.index, carries, inits, nexts,
                           start=loop.start, stop=loop.stop, step=loop.step)
        self.fused[id(loop)] = fused
        return fused


def _divides(node: Node, stride: int) -> bool:
    """``node`` is ``lane / D`` with ``D`` a positive integer dividing ``stride``."""
    if not (isinstance(node, Op) and node.kind == "div"
            and isinstance(node.operands[0], SampleIndex)
            and isinstance(node.operands[1], Const)
            and node.ttype.dtype in ("i32", "i64")):
        return False
    d = int(node.operands[1].literal)
    return d > 0 and stride % d == 0


def _component(node: Node, k: int) -> Op:
    return Op("component", (node,), TensorType((), node.ttype.dtype), k)


def _vec(parts: list, ttype: TensorType) -> Op:
    return Op("vec", tuple(parts), ttype)
