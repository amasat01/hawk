# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The loop family of IR nodes: a lowered bounded ``for`` and the
placeholders and reads its body uses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..types import TensorType
from .nodes import HawkError, Node, Op, Sink

# --------------------------------------------------------------------------
# The LOOP family.
#
# A bounded ``for`` is lowered, not unrolled: unrolling blew ptxas's memory
# on a KAN-sized cell, and unrolling is a transpiler's answer, not a code
# generator's. The body is traced ONCE against a symbolic index with
# explicit loop-carried values, and the emitter writes a plain ``for`` —
# whether it unrolls is the C++/CUDA compiler's decision, not HAWK's.
#
# The body is NOT in ``operands``: a loop body is a SCOPE, evaluated once
# per iteration inside braces, so hoisting its nodes into the flat walk
# order would re-unroll it. A :class:`Loop` carries its body in fields the
# walk does not traverse; its ``operands`` are only the values crossing
# INTO the scope (carried inits and loop-invariant boundary subexpressions),
# which keeps every body leaf reachable through the ordinary walk.


#: The op KIND of a lowered bounded ``for``.
LOOP_KIND = "loop"
#: The KIND of a loop's symbolic index — the body's own ``k``.
LOOP_INDEX_KIND = "loop_index"
#: The KIND of a loop-carried value READ at the start of an iteration.
LOOP_CARRY_KIND = "loop_carry"
#: The KIND that projects one carried slot's value OUT of a finished loop.
LOOP_VALUE_KIND = "loop_value"
#: The KIND that reads a forward loop's per-iteration carry out of its tape
#: — the ONE form a reverse loop needs. The tape is a bounded local array a
#: derived kernel declares at a compiler-visible size, not a runtime tape.
TAPE_READ_KIND = "tape_read"
#: The KIND of a loop-with-a-``break``'s per-sample iteration count.
LOOP_COUNT_KIND = "loop_count"


class LoopIndex(Node):
    """A loop's symbolic index, as an IR VALUE (:data:`LOOP_INDEX_KIND`).

    Childless and binds no slot, like :class:`SampleIndex`, and a TERMINAL
    for both derivative directions for the same reason. Identified by the
    loop's NESTING DEPTH and the author's own variable name rather than an
    object counter, so the kernel's digest stays byte-identical across
    processes."""

    __slots__ = ("depth", "name")

    kind = LOOP_INDEX_KIND

    def __init__(self, depth: int, name: str, ttype: TensorType | None = None) -> None:
        super().__init__(ttype or TensorType((), "i32"))
        self.depth = int(depth)
        self.name = name

    @property
    def dedup_extra(self) -> Any:
        return (LOOP_INDEX_KIND, self.depth, self.name)


class LoopCarry(Node):
    """One loop-carried value, READ at the start of an iteration.

    ``slot`` is its position in the loop's carry tuple and ``name`` the author's
    own variable name (which is also what the emitted local is named after, so a
    generated body reads like the body that was written). Structurally
    identified the same way :class:`LoopIndex` is, and for the same reason."""

    __slots__ = ("depth", "slot", "name")

    kind = LOOP_CARRY_KIND

    def __init__(self, depth: int, slot: int, name: str, ttype: TensorType) -> None:
        super().__init__(ttype)
        self.depth = int(depth)
        self.slot = int(slot)
        self.name = name

    @property
    def dedup_extra(self) -> Any:
        return (LOOP_CARRY_KIND, self.depth, self.slot, self.name)


class Loop(Node):
    """A bounded ``for``, lowered rather than unrolled.

    ``start``/``stop``/``step`` are compile-time ints (the trip count must
    RESOLVE at trace time); ``trip`` is the iteration count they imply.
    ``index`` is the body's symbolic index; ``carries`` are the loop-carried
    reads, one per name; ``nexts[j]`` is carry ``j``'s end-of-iteration
    value. ``operands`` are the values crossing into the scope: the
    carries' initial values, then the body's loop-invariant boundary
    subexpressions (``free``) — what makes every leaf the body reads
    reachable from the ordinary walk.

    A carried slot may be rank>=1: the emitter declares it as
    ``aether::Item<T, Es...>`` (STORAGE, not an expression tree) rather than
    ``const auto``, since a rank>=1 aether expression stores operands BY
    REFERENCE and would dangle once the iteration that built it ends. A
    REVERSE loop needs no flag of its own — it is the same node with its
    header run backwards (a negative ``step``), one emitter arm.

    The node's own ``ttype`` is the first carry's; a loop is a STATEMENT,
    never rendered as a value (a :class:`LoopValue` projects one result
    out).

    A loop a sample can LEAVE EARLY carries ``exit_cond`` (a rank-0 bool)
    and ``exit_values`` (each carried slot's value at the break);
    :attr:`exit_form` reads where the break sits from those values. A
    DERIVED loop of such a loop instead carries ``count``, a runtime
    ``i32`` (:class:`LoopCount`): the first ``count`` iterations, or the
    last ``count`` with ``count_tail`` set (a reverse loop). Both stay
    ``None`` otherwise."""

    __slots__ = ("index", "carries", "nexts", "start", "stop", "step", "trip",
                 "carry_count", "body_sinks", "_body_key", "exit_cond",
                 "exit_values", "count", "count_tail", "_free_end")

    kind = LOOP_KIND

    def __init__(self, index: LoopIndex, carries: Sequence[LoopCarry],
                 inits: Sequence[Node], nexts: Sequence[Node], *,
                 start: int, stop: int, step: int,
                 free: Sequence[Node] = (),
                 body_sinks: Sequence[Sink] = (),
                 exit_cond: Node | None = None,
                 exit_values: Sequence[Node] = (),
                 count: Node | None = None,
                 count_tail: bool = False) -> None:
        carries, inits, nexts = tuple(carries), tuple(inits), tuple(nexts)
        body_sinks = tuple(body_sinks)
        exit_values = tuple(exit_values)
        if exit_cond is None and exit_values:
            raise HawkError("Loop: exit values without an exit condition")
        if exit_cond is not None:
            if exit_cond.ttype.shape != () or exit_cond.ttype.dtype != "bool":
                raise HawkError(
                    f"Loop: a `break` condition must be a rank-0 bool, got "
                    f"{exit_cond.ttype!r}")
            if len(exit_values) != len(carries):
                raise HawkError(
                    f"Loop: {len(carries)} carried slot(s) but {len(exit_values)} "
                    "value(s) at the `break`")
            for j, (carry, value) in enumerate(zip(carries, exit_values)):
                if value.ttype != carry.ttype:
                    raise HawkError(
                        f"Loop: carried slot {j} ({carry.name!r}) is typed "
                        f"{carry.ttype!r} but its value at the `break` is typed "
                        f"{value.ttype!r}")
        if count is not None and (count.ttype.shape != ()
                                  or count.ttype.dtype != "i32"):
            raise HawkError(f"Loop: a runtime iteration count is a rank-0 i32, "
                            f"got {count.ttype!r}")
        for sink in body_sinks:
            if not isinstance(sink, Sink):
                raise HawkError(f"Loop: {sink!r} is not a sink")
        if not carries:
            raise HawkError(
                "Loop: a lowered `for` must carry at least one value — a body that "
                "assigns nothing the loop's consumers read computes nothing, and an "
                "empty loop is not a HAWK form"
            )
        if not (len(carries) == len(inits) == len(nexts)):
            raise HawkError(
                f"Loop: {len(carries)} carried slot(s) but {len(inits)} initial "
                f"value(s) and {len(nexts)} body result(s) — the three are one "
                "tuple read three ways")
        if int(step) == 0:
            raise HawkError("Loop: a `for` step of zero never terminates")
        for j, (carry, init, nxt) in enumerate(zip(carries, inits, nexts)):
            for what, node in (("initial value", init), ("body result", nxt)):
                if node.ttype != carry.ttype:
                    raise HawkError(
                        f"Loop: carried slot {j} ({carry.name!r}) is typed "
                        f"{carry.ttype!r} but its {what} is typed {node.ttype!r}. A "
                        "carried value has ONE type for the whole loop — it is a "
                        "single C++ local the body reassigns, so a type that changed "
                        "between iterations could not be declared at all")
            if len(carry.ttype.shape) > 2:
                raise HawkError(   # pragma: no cover - no rank>2 type exists yet
                    f"Loop: carried slot {j} ({carry.name!r}) is typed "
                    f"{carry.ttype!r}; a carried slot is one aether value with "
                    "static extents and HAWK declares none above rank 2")
        free = tuple(free)
        tail = () if count is None else (count,)
        super().__init__(carries[0].ttype, (*inits, *free, *tail))
        self._free_end = len(inits) + len(free)
        self.exit_cond = exit_cond
        self.exit_values = exit_values
        self.count = count
        self.count_tail = bool(count_tail)
        self.index = index
        self.carries = carries
        self.nexts = nexts
        self.carry_count = len(carries)
        self.start, self.stop, self.step = int(start), int(stop), int(step)
        self.trip = len(range(self.start, self.stop, self.step))
        self.body_sinks = body_sinks
        self._body_key = None

    @property
    def inits(self) -> tuple:
        """The carried slots' initial values, in carry order."""
        return self.operands[:self.carry_count]

    @property
    def free(self) -> tuple:
        """The body's loop-invariant boundary values, computed ONCE outside."""
        return self.operands[self.carry_count:self._free_end]

    @property
    def exit_form(self) -> str | None:
        """Where the loop's ``break`` sits, from its VALUES: ``"head"`` if
        every carried value at the break is the iteration's start (it did
        nothing), ``"tail"`` if every one is the result (it ran in full),
        ``"middle"`` otherwise, ``None`` without a break."""
        if self.exit_cond is None:
            return None
        if all(v is c for v, c in zip(self.exit_values, self.carries)):
            return "head"
        if all(v is n for v, n in zip(self.exit_values, self.nexts)):
            return "tail"
        return "middle"

    def body_nodes(self) -> tuple:
        """Every body-subgraph node, post-order, boundary values and the
        loop's own placeholders excluded. Iterative on purpose — called by
        the emitter, access-class inference and the digest, so a recursive
        walk would stack three separate times."""
        boundary = {id(o) for o in self.operands}
        seen: set = set()
        order: list = []
        stack: list = [(n, False) for n in reversed(self.body_roots())]
        while stack:
            node, expanded = stack.pop()
            if id(node) in seen or id(node) in boundary:
                continue
            if isinstance(node, (LoopIndex, LoopCarry)):
                continue
            if not expanded:
                stack.append((node, True))
                for child in reversed(node.operands):
                    if id(child) not in seen:
                        stack.append((child, False))
                continue
            seen.add(id(node))
            order.append(node)
        return tuple(order)

    def body_roots(self) -> tuple:
        """What the body EVALUATES, in emission order: the carried results,
        then every sink the body commits once per iteration.

        A body sink is how a GATHER's reverse inside a loop is expressed
        (``bar_t[e(k)] += …`` once per iteration, at that iteration's row) —
        there is no way to say that with a top-level sink, since the scatter
        form transposes a gather and here the gather is inside a scope."""
        exits = (() if self.exit_cond is None
                 else (self.exit_cond, *self.exit_values))
        return self.nexts + tuple(
            operand for sink in self.body_sinks for operand in sink.operands
        ) + exits

    @property
    def dedup_extra(self) -> Any:
        """Loop's dedup key tail: the header plus a STRUCTURAL key of the
        body, computed once. Two loops with the same header, boundary
        operands and structurally equal bodies are ONE loop; the body is
        hashed, not compared, so the key stays small."""
        if self._body_key is None:
            self._body_key = _body_digest(self)
        key = (LOOP_KIND, self.start, self.stop, self.step,
               tuple(c.name for c in self.carries),
               tuple((s.kind, s.role, s.name) for s in self.body_sinks),
               self._body_key)
        if self.exit_cond is not None:
            key += ("exit",)
        if self.count is not None:
            key += ("count", self.count_tail)
        return key


#: How wide ONE body-node key is, in hex characters (:func:`_body_keys`).
#: 128 bits of blake2b: the collision exposure of a structural key is the same
#: exposure the walk's own sha256 digest already accepts one level up, and 16 bytes
#: is where the key stops being the thing that dominates a body's memory.
BODY_KEY_HEX = 32


def _body_keys(loop: Loop) -> tuple[dict, tuple]:
    """``(id(node) -> its structural key, the body's node list)`` — the ONE
    implementation of a loop body's structural keying.

    Boundary values are keyed by POSITION in ``loop.operands`` (never
    identity), placeholders by their slot, and every other node by the same
    ``(kind, dtype, shape, tag, child keys, literal)`` tuple
    :func:`hawk.ir.walk._key` uses — HASHED TO A FIXED WIDTH rather than
    embedded verbatim, so a key is O(fan-in) bytes and the whole body O(n)
    instead of O(2**depth): embedding children's key text cost gigabytes of
    RAM on a deep KAN-style chain.

    STILL STRUCTURAL (a Merkle key over the same tuple), POSITIONAL
    (swapping two children changes the key) and PROCESS-STABLE (blake2b of
    a ``repr``, never Python's salted ``hash()`` or an object address) —
    everything the walk digest requires."""
    import hashlib

    key_of: dict[int, str] = {}
    for k, operand in enumerate(loop.operands):
        key_of.setdefault(id(operand), f"free{k}")
    body = loop.body_nodes()
    for node in body:
        parts = []
        for child in node.operands:
            held = key_of.get(id(child))
            if held is None:
                if isinstance(child, LoopIndex):
                    held = "index"
                elif isinstance(child, LoopCarry):
                    held = f"carry{child.slot}"
                else:                                   # pragma: no cover - closed
                    raise HawkError(
                        f"Loop body: {child!r} is neither a boundary value, a "
                        "placeholder nor a body node — the boundary computation "
                        "and the body traversal disagree")
                key_of[id(child)] = held
            parts.append(held)
        t = node.ttype
        raw = repr((node.kind, t.dtype, t.shape, t.tag, tuple(parts),
                    node.dedup_extra))
        key_of[id(node)] = hashlib.blake2b(
            raw.encode(), digest_size=BODY_KEY_HEX // 2).hexdigest()
    return key_of, body


def body_key_widths(loop: Loop) -> tuple[int, ...]:
    """The BYTE width of every structural key :func:`_body_keys` builds for
    ``loop``'s body, in body order — the one diagnostic that can see the
    O(2**depth) defect from outside.

    A key embedding its children's text is indistinguishable from a
    fixed-width one by every other observable (same digest length, same
    walk nodes, same emitted body); only the key SIZE reveals it, computed
    by the SAME function the digest uses."""
    key_of, body = _body_keys(loop)
    return tuple(len(key_of[id(node)]) for node in body)


def _body_digest(loop: Loop) -> str:
    """A process-stable structural hash of ``loop``'s body (:func:`_body_keys`).

    Walked ONCE — the key map and node order come from the same call — since
    this runs inside ``dedup_extra``, asked of every loop by the canonical
    walk, and a second traversal would double the cost for nothing."""
    import hashlib

    key_of, body = _body_keys(loop)
    roots = []
    for nxt in loop.body_roots():
        held = key_of.get(id(nxt))
        if held is None:
            held = ("index" if isinstance(nxt, LoopIndex)
                    else f"carry{nxt.slot}" if isinstance(nxt, LoopCarry) else None)
        roots.append(held)
    h = hashlib.sha256()
    h.update(b"hawk.ir.loop/2\n")
    for node in body:
        h.update(f"{key_of[id(node)]}\n".encode())
    h.update(f"roots|{roots!r}\n".encode())
    return h.hexdigest()[:32]


class LoopValue(Op):
    """One carried slot's FINAL value, projected out of a finished loop.

    A loop produces as many values as it carries and a HAWK node carries one
    type, so the projection is its own node — the same shape ``component``
    already has for a vector's element. ``literal`` is the carried slot."""

    __slots__ = ()

    def __init__(self, loop: Loop, slot: int) -> None:
        if not isinstance(loop, Loop):
            raise HawkError(f"LoopValue: {loop!r} is not a Loop")
        if not 0 <= slot < loop.carry_count:
            raise HawkError(
                f"LoopValue: slot {slot} is outside the loop's {loop.carry_count} "
                "carried slot(s)")
        super().__init__(LOOP_VALUE_KIND, (loop,), loop.carries[slot].ttype, slot)

    @property
    def loop(self) -> Loop:
        return self.operands[0]          # type: ignore[return-value]

    @property
    def slot(self) -> int:
        return int(self.literal)


class LoopCount(Op):
    """How many iterations of a loop with a ``break`` applied their update
    — per sample, a runtime ``i32`` (:data:`LOOP_COUNT_KIND`).

    Reverse mode needs it: the derived loop runs over exactly the
    iterations that sample ran. A head-form ``break`` doesn't count the
    exiting iteration; a tail-form one does. An index-like value, so a
    TERMINAL for both directions."""

    __slots__ = ()

    def __init__(self, loop: Loop) -> None:
        if not isinstance(loop, Loop) or loop.exit_cond is None:
            raise HawkError(f"LoopCount: {loop!r} is not a loop with a `break`")
        super().__init__(LOOP_COUNT_KIND, (loop,), TensorType((), "i32"))

    @property
    def loop(self) -> Loop:
        return self.operands[0]          # type: ignore[return-value]


class TapeRead(Op):
    """The value carried slot ``slot`` held at iteration ``index`` of a
    FORWARD loop — what a reverse-mode loop needs that the forward loop
    doesn't keep on its own.

    An ACCUMULATOR loop never mints one (its adjoint doesn't depend on the
    carried value, so its reverse is a plain loop with nothing stored); a
    RECURRENCE's does, writing each iteration's carry into a fixed-size
    local array the reverse loop reads back. Operand 0 is the forward loop,
    emitted before this one by the ordinary walk order."""

    __slots__ = ("slot",)

    def __init__(self, loop: Loop, slot: int, index: Node) -> None:
        if not isinstance(loop, Loop):
            raise HawkError(f"TapeRead: {loop!r} is not a Loop")
        if not 0 <= slot < loop.carry_count:
            raise HawkError(
                f"TapeRead: slot {slot} is outside the loop's {loop.carry_count} "
                "carried slot(s)")
        super().__init__(TAPE_READ_KIND, (loop, index), loop.carries[slot].ttype,
                         slot)
        self.slot = int(slot)

    @property
    def loop(self) -> Loop:
        return self.operands[0]          # type: ignore[return-value]

    @property
    def index(self) -> Node:
        return self.operands[1]
