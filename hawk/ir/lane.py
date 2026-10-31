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
from .nodes import Assign, At, Const, Finish, Leaf, Node, Op, RoleConst, Select
from .ops import make

#: Lanes of the warp that shares one sample.
WARP = 32

#: The narrowest value worth slicing, and so the shortest fold ``wide_form="auto"`` builds as a warp (provisional: a smaller
#: one fits a thread's registers).
SLICE_MIN = 128

WARP_LANE_KIND = "warp_lane"

#: The longest short vector a fill loop unrolls into registers (the elements are then computed in place of a run-time indexed frame).
NARROW_FILL_MAX = 32

#: The literal of a ``component_at`` that reads a plane element in place: each lane reads only the elements it commits.
WARP_READ = "warp_read"

#: The sum of a sliced value over its whole width: each lane sums its slice, then a fixed tree over the lanes.
WARP_SUM_KIND = "warp_sum"


class WarpLane(Node):
    """The lane's place in its warp, ``0 .. WARP - 1`` (a terminal: it binds no slot)."""

    __slots__ = ()

    kind = WARP_LANE_KIND

    def __init__(self) -> None:
        super().__init__(TensorType((), "i32"))


class WarpStore(Assign):
    """The commit of a value into the Mutable ``name``. A sliced one (the value is narrower than the plane): element ``j``
    of the slice goes to element ``j * WARP + lane``. Any other is the same in every lane: lane 0 stores it. The sink
    keeps the plane's own type."""

    __slots__ = ()


def sliced(ttype: TensorType) -> bool:
    """Whether a value of ``ttype`` is held sliced: a vector of a whole number of warps, wide enough to pay."""
    return len(ttype.shape) == 1 and ttype.shape[0] % WARP == 0 and ttype.shape[0] >= SLICE_MIN


def _slice_type(ttype: TensorType) -> TensorType:
    return TensorType((ttype.shape[0] // WARP,), ttype.dtype, ttype.tag)


def _reach(root: Node):
    """Every node under ``root``."""
    seen, stack = set(), [root]
    while stack:
        node = stack.pop()
        if id(node) not in seen:
            seen.add(id(node))
            yield node
            stack.extend(node.operands)


class _Refused(Exception):
    """A node with no rule in the warp form."""


class _Slicer:
    def __init__(self) -> None:
        self.memo: dict = {}                       # id(node) -> (node, is_sliced)
        self.lane = WarpLane()
        self.used = False                          # whether any value ended up sliced
        self.writes = 0                            # run-time indexed writes into a slice
        self.folds = 0                             # sums over the lanes
        self.keep: list = []                       # every node memoised, alive as long as its id is a key

    def of(self, node: Node) -> tuple:
        got = self.memo.get(id(node))
        if got is None:
            self.keep.append(node)
            got = self.memo[id(node)] = self._one(node)
            self.used = self.used or got[1]
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
                return self._read(node), True
            return node, False
        if isinstance(node, LoopValue):
            filled = self._fill(node)
            if filled is not None:
                return filled
            loop, slot_sliced = self._loop(node.loop)
            return LoopValue(loop, node.slot), slot_sliced[node.slot]
        if isinstance(node, Loop):
            raise _Refused("a loop used as a value")
        if isinstance(node, At):
            return self._rep(node)
        if isinstance(node, (Op, Select)):
            return self._op(node)
        return self._rep(node)

    def _read(self, leaf: Leaf) -> Op:
        """A per-sample vector input read whole: lane ``l`` reads its elements ``j * WARP + l``."""
        if leaf.role != "vec_in" and not (leaf.role == "mutable" and leaf.kind == "prior_read"):
            raise _Refused(f"a wide plane {leaf.name!r} read whole")
        index = TensorType((), "i32")
        scalar = TensorType((), leaf.ttype.dtype)
        parts = [Op("component_at", (leaf, make("add", (self.lane, Const(j * WARP, index)))), scalar, WARP_READ)
                 for j in range(leaf.ttype.shape[0] // WARP)]
        return Op("vec", tuple(parts), _slice_type(leaf.ttype))

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
            if node.kind in ("sum", "dot", "norm") and len(node.operands) == (2 if node.kind == "dot" else 1):
                ops = [self.of(o) for o in node.operands]
                if all(s for _, s in ops):      # each lane reduces its slice, then the lanes add up in a fixed tree
                    self.folds += 1
                    a = ops[0][0]
                    if node.kind == "sum":
                        return Op(WARP_SUM_KIND, (a,), node.ttype), False
                    partial = Op("dot", (a, ops[-1][0]), node.ttype)
                    total = Op(WARP_SUM_KIND, (partial,), node.ttype)
                    return (total if node.kind == "dot" else Op("sqrt", (total,), node.ttype)), False
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
        if kind == "set_component_at":
            return self._write(node, out), True
        ops = [self.of(o) for o in node.operands]
        if isinstance(node, Select) and node.operands[0].ttype.shape:
            raise _Refused("a select by a vector of conditions")
        from .wide import ELEMENTWISE

        if kind not in ELEMENTWISE:
            raise _Refused(f"{kind} of a wide value")
        for (_, s), o in zip(ops, node.operands):
            if not s and o.ttype.shape:
                raise _Refused(f"a narrow vector beside a wide one: {kind} flags={[s for _, s in ops]} {[(type(q).__name__, getattr(q, 'kind', ''), q.ttype.shape) for q in node.operands]}")
        new = [a for a, _ in ops]
        if isinstance(node, Select):
            return Select(new[0], new[1], new[2], out), True
        return Op(kind, tuple(new), out, node.literal), True

    def _gather(self, node: Node, out: TensorType) -> Op:
        """A ``vec`` of reads of one plane at an arithmetic run of indices: lane ``l`` reads the run's elements
        ``j * WARP + l``."""
        from .wide import gather

        g = gather(node)
        if g is None:
            return self._affine(node, out)
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

    def _fill(self, value: LoopValue) -> tuple | None:
        """A loop that writes element ``j`` of a wide vector at trip ``j`` from a value of ``j`` alone (``r[j] = f(j)`` over
        the whole vector) is the vector of its values: lane ``l`` computes ``f`` at its own elements."""
        from .walk import substitute

        loop = value.loop
        carry = loop.carries[value.slot]
        if (loop.exit_cond is not None or loop.count is not None or loop.body_sinks
                or len(carry.ttype.shape) != 1 or (loop.start, loop.step) != (0, 1) or loop.stop != carry.ttype.shape[0]):
            return None
        narrow = not sliced(carry.ttype)
        if narrow and carry.ttype.shape[0] > NARROW_FILL_MAX:
            return None
        nxt = loop.nexts[value.slot]
        if not (isinstance(nxt, Op) and nxt.kind == "set_component_at" and nxt.operands[0] is carry
                and nxt.operands[1] is loop.index):
            return None
        element = nxt.operands[2]
        if any(isinstance(n, LoopCarry) and any(n is c for c in loop.carries) for n in _reach(element)):
            return None             # the element is a function of the index alone: it reads no carry of the loop
        index_type = loop.index.ttype
        if narrow:      # a short vector every lane holds: its elements are unrolled, none is read or written at a run-time index
            return Op("vec", tuple(self._plain(substitute([element], {id(loop.index): Const(r, index_type)})[0])
                                   for r in range(loop.stop)), carry.ttype), False
        parts = []
        for j in range(loop.stop // WARP):
            at = make("add", (self.lane, Const(j * WARP, index_type)))
            part, = substitute([element], {id(loop.index): at})
            parts.append(self._plain(part))
        return Op("vec", tuple(parts), _slice_type(carry.ttype)), True

    def _write(self, node: Op, out: TensorType) -> Op:
        """``v[k] = x`` on a sliced vector: the lane whose element ``k`` is writes ``x``, the others keep theirs. A constant
        ``k`` touches one slot; a run-time one is a select in every slot, so no slot is chosen at run time."""
        (vec, was), index, value = self.of(node.operands[0]), self.of(node.operands[1]), self.of(node.operands[2])
        if not was or index[1] or value[1]:
            raise _Refused("a write of a slice or at a slice")
        k, x = index[0], value[0]
        scalar = TensorType((), node.ttype.dtype)
        index_type = k.ttype
        parts = []
        known = k.literal if isinstance(k, Const) else None
        self.writes += known is None
        for j in range(out.shape[0]):
            old = Op("component", (vec,), scalar, j)
            if known is not None and int(known) // WARP != j:
                parts.append(old)
                continue
            here = make("add", (self.lane, Const(j * WARP, index_type)))
            parts.append(Select(make("eq", (k, here)), x, old, scalar))
        return Op("vec", tuple(parts), out)

    def _affine(self, node: Node, out: TensorType) -> Op:
        """A ``vec`` whose parts are one expression of the element index: the parts agree node by node, and where they
        differ they are constants in an arithmetic run. Lane ``l`` computes the parts ``j * WARP + l`` from the run."""
        parts = node.operands
        built = [self._aligned(parts, j) for j in range(out.shape[0])]
        return Op("vec", tuple(built), out)

    def _aligned(self, parts, j: int) -> Node:
        """Part ``j * WARP + lane`` of the run ``parts`` (the same expression with its constants moved along)."""
        def walk(nodes: list) -> Node:
            first = nodes[0]
            if all(n is first for n in nodes):
                return self.of(first)[0] if not first.ttype.shape else self._plain(first)
            if all(isinstance(n, Const) and not isinstance(n, RoleConst) and n.ttype == first.ttype for n in nodes):
                lits = [n.literal for n in nodes]
                step = lits[1] - lits[0]
                if any(b - a != step for a, b in zip(lits, lits[1:])):
                    raise _Refused("constants that are not an arithmetic run")
                start = lits[0] + j * WARP * step
                lane = make("cast", (self.lane,), literal=first.ttype.dtype) if first.ttype.dtype != "i32" else self.lane
                return make("add", (Const(start, first.ttype), make("mul", (lane, Const(step, first.ttype)))))
            if not all(type(n) is type(first) and n.operands and len(n.operands) == len(first.operands)
                       and getattr(n, "kind", None) == getattr(first, "kind", None)
                       and getattr(n, "literal", None) == getattr(first, "literal", None)
                       and n.ttype == first.ttype for n in nodes):
                raise _Refused("parts that differ beyond their constants")
            from .walk import _rebuild

            return _rebuild(first, [walk([n.operands[k] for n in nodes]) for k in range(len(first.operands))])
        return walk(list(parts))

    def _plain(self, node: Node) -> Node:
        got, is_sliced = self.of(node)
        if is_sliced:
            raise _Refused("a slice inside a run of parts")
        return got

    def _loop(self, loop: Loop) -> tuple:
        from .loops import build_loop

        got = self.memo.get(id(loop))
        if got is not None:
            return got
        if loop.body_sinks:
            raise _Refused("a loop that commits once per iteration")
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
        exit_cond, exit_values, count = None, (), None
        if loop.exit_cond is not None:      # a flag every lane computes alike: the lanes leave together
            exit_cond, was = self.of(loop.exit_cond)
            if was:
                raise _Refused("an exit that reads a slice")
            exit_values = []
            for j, ev in enumerate(loop.exit_values):
                value, got = self.of(ev)
                if got != (slot_sliced[j] if j < len(slot_sliced) else False):
                    raise _Refused("an exit value not held like its carry")
                exit_values.append(value)
        if loop.count is not None:
            count, was = self.of(loop.count)
            if was:
                raise _Refused("a trip count that reads a slice")
        new_loop = build_loop(loop.index, carries, inits, nexts, start=loop.start, stop=loop.stop, step=loop.step,
                              exit_cond=exit_cond, exit_values=exit_values, count=count, count_tail=loop.count_tail,
                              dynamic=loop.dynamic)
        got = self.memo[id(loop)] = (new_loop, slot_sliced)
        return got


def slice_sinks(sinks, *, explicit: bool = True) -> tuple | None:
    """``sinks`` in the warp form, or ``None`` when some node has no rule there. ``explicit`` is whether the warp form was
    asked for: otherwise (``auto``) a kernel whose only wide work is writes at run-time indices keeps the placed form, which
    measured 4x faster there (each write is a select in every slot of the slice)."""
    sinks = tuple(sinks)
    if not sinks or not all(type(s) is Assign or isinstance(s, Finish) for s in sinks):
        return None
    slicer = _Slicer()
    out = []
    try:
        for sink in sinks:
            value, was = slicer.of(sink.value)
            if isinstance(sink, Finish):        # the flag is the same in every lane; each writes the one bit
                if was:
                    return None
                out.append(Finish(sink.name, value, sink.steps))
                continue
            out.append(WarpStore(sink.name, value, sink.ttype))
    except _Refused:
        return None
    if not slicer.used or (not explicit and slicer.writes and not slicer.folds):
        return None
    return tuple(out)
