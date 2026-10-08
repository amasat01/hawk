# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Building and classifying a lowered ``for``: the boundary, the classes.

Two questions live here, belonging in neither :mod:`hawk.ir.nodes` (node
SHAPES) nor :mod:`hawk.ir.walk` (the ONE traversal).

WHICH VALUES CROSS INTO THE SCOPE: a :class:`~hawk.ir.loop_nodes.Loop`'s
``operands`` are the carried initial values plus the body's loop-INVARIANT
boundary subexpressions, which :func:`build_loop` computes as the MAXIMAL
subexpressions depending on neither the index nor any carry. This keeps
every leaf the body reads reachable from those operands, lets the emitter
name each boundary value once instead of per iteration, and lets the
digest key them by POSITION, not object identity.

WHAT KIND OF LOOP IT IS: :func:`classify_loop` answers, from the body's
STRUCTURE rather than an author-supplied name, the question reverse mode
must ask before differentiating a loop:

* ``accumulator`` — every carry updates as ``c = c + a`` with ``a`` free of
  EVERY carry, read nowhere else; the reverse pass is a plain forward loop
  storing NOTHING;
* ``recurrence`` — anything else the rule table covers: the forward loop
  keeps one bounded array per carry and the reverse loop runs backwards;
* refused, NAMED — a nested loop inside a differentiated body would need a
  tape of a tape, which reverse mode does not ask for.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .loop_nodes import Loop, LoopCarry, LoopIndex
from .nodes import RAW_KIND, HawkError, Node

#: The class whose adjoint stores nothing.
ACCUMULATOR = "accumulator"
#: The class whose adjoint runs in reverse over the forward loop's own
#: carried values.
RECURRENCE = "recurrence"
#: The two classes — a loop is one of these or REFUSED by name. No third
#: "assume it works" class.
LOOP_CLASSES = (ACCUMULATOR, RECURRENCE)


@dataclass(frozen=True)
class LoopClass:
    """What :func:`classify_loop` found: the class, plus an accumulator's
    per-carry ADDEND ``a`` in ``c = c + a`` (empty for a recurrence)."""

    name: str
    addends: tuple = ()


# --------------------------------------------------------------------------
# the boundary


def _depends(node: Node, own: frozenset, memo: dict) -> bool:
    """Whether ``node``'s subtree reads THIS loop's index or one of ITS
    carries.

    ``own`` is this loop's own placeholder ids: an inner loop legitimately
    reads the OUTER index, loop-INVARIANT to it, crossing its boundary as
    an operand rather than being trapped inside.

    Iterative and memoised on ``id`` — a recursive form would stack the
    body's depth once per query."""
    hit = memo.get(id(node))
    if hit is not None:
        return hit
    stack = [(node, False)]
    while stack:
        current, expanded = stack.pop()
        if id(current) in memo and not expanded:
            continue
        if isinstance(current, (LoopIndex, LoopCarry)):
            memo[id(current)] = id(current) in own
            continue
        if not expanded:
            stack.append((current, True))
            for child in current.operands:
                if id(child) not in memo:
                    stack.append((child, False))
            continue
        memo[id(current)] = any(memo.get(id(c), False) for c in current.operands)
    return memo[id(node)]


def boundary(nexts: Sequence[Node], own: frozenset) -> tuple:
    """The body's loop-invariant BOUNDARY values, in deterministic order.

    Maximal: the walk stops at the highest invariant node on each path, so
    its children ride along inside that one definition. A ``next`` that is
    invariant in its entirety is its own boundary value — a loop whose body
    ignores the index, odd but well-formed."""
    memo: dict = {}
    out: list = []
    seen: set = set()
    stack = [n for n in reversed(tuple(nexts))]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        if not _depends(node, own, memo):
            seen.add(id(node))
            out.append(node)
            continue
        seen.add(id(node))
        for child in reversed(node.operands):
            stack.append(child)
    # de-dup, first-seen order: two paths may reach one invariant subexpression.
    unique: list = []
    held: set = set()
    for node in out:
        if id(node) not in held:
            held.add(id(node))
            unique.append(node)
    return tuple(unique)


def build_loop(index: LoopIndex, carries: Sequence[LoopCarry],
               inits: Sequence[Node], nexts: Sequence[Node], *, start: int,
               stop: int, step: int, body_sinks: Sequence[Node] = (),
               exit_cond: Node | None = None, exit_values: Sequence[Node] = (),
               count: Node | None = None, count_tail: bool = False) -> Loop:
    """Mint a :class:`~hawk.ir.loop_nodes.Loop`, computing its boundary for it.

    THE constructor every producer uses (the tracer, :mod:`hawk.diff.loops`
    for a derived loop), so a derived loop's operands are computed by
    exactly the code that computed the primal's."""
    nexts, body_sinks = tuple(nexts), tuple(body_sinks)
    exit_values = tuple(exit_values)
    roots = nexts + tuple(o for sink in body_sinks for o in sink.operands)
    if exit_cond is not None:
        roots += (exit_cond, *exit_values)
    own = frozenset({id(index), *(id(c) for c in carries)})
    held = {id(n) for n in inits}
    if count is not None:
        held.add(id(count))
    free = tuple(n for n in boundary(roots, own) if id(n) not in held)
    loop = Loop(index, carries, inits, nexts, start=start, stop=stop, step=step,
                free=free, body_sinks=body_sinks, exit_cond=exit_cond,
                exit_values=exit_values, count=count, count_tail=count_tail)
    _check_placeholders(loop)
    return loop


def exit_kwargs(loop: Loop) -> dict:
    """``loop``'s ``break``/count as :func:`build_loop` keywords, for a
    REBUILD that must leave early exactly where the original did."""
    return {"exit_cond": loop.exit_cond, "exit_values": loop.exit_values,
            "count": loop.count, "count_tail": loop.count_tail}


def _check_placeholders(loop: Loop) -> None:
    """Every index/carry placeholder inside the body must be THIS loop's.

    A derived loop substitutes the primal's carried reads for something
    else; a missed substitution would leave a foreign placeholder the
    digest keys as this loop's own carry — a silently wrong recurrence,
    checked once, here."""
    own = {id(loop.index)} | {id(c) for c in loop.carries}
    for node in loop.body_nodes():
        for child in node.operands:
            if isinstance(child, (LoopIndex, LoopCarry)) and id(child) not in own:
                raise HawkError(
                    f"the loop over {loop.index.name!r} reads {child.kind} "
                    f"{getattr(child, 'name', '?')!r} (depth {child.depth}), which "
                    "belongs to a DIFFERENT loop. A loop body may only read its own "
                    "index and its own carried slots; anything else must cross the "
                    "boundary as an operand")
    for root in loop.body_roots():
        if isinstance(root, (LoopIndex, LoopCarry)) and id(root) not in own:
            raise HawkError(                      # pragma: no cover - as above
                f"the loop over {loop.index.name!r} returns a placeholder of a "
                "different loop")


def addends(node: Node) -> list:
    """Flatten an ``add`` chain into its terms (``add`` only — a ``sub`` at the
    top alternates the sign of what follows and is not the additive shape)."""
    if getattr(node, "kind", None) == "add":
        return addends(node.operands[0]) + addends(node.operands[1])
    return [node]


# --------------------------------------------------------------------------
# the classifier


def classify_loop(loop: Loop) -> LoopClass:
    """Which of :data:`LOOP_CLASSES` ``loop`` is, or a refusal NAMING why.

    Narrow and structural: every carry's body result must be an ``add``
    chain containing that carry EXACTLY once as a bare term, read nowhere
    else. Anything weaker could differentiate a body whose adjoint
    genuinely depends on the carried value as if it didn't — a wrong
    gradient, not a refused one."""
    for inner in loop.body_nodes():
        if isinstance(inner, Loop):
            raise HawkError(
                f"loop over {loop.index.name!r} (bound {loop.trip}) cannot be "
                "differentiated: its body contains a nested `for`. Reverse mode "
                "would have to keep the inner loop's carried history for every "
                "outer iteration — a tape of a tape — which is beyond what "
                "lowers. Flatten the two loops into one, or take the derivative "
                "of the inner loop's body as its own kernel")
        if getattr(inner, "kind", None) == RAW_KIND:
            raise HawkError(
                f"loop over {loop.index.name!r} cannot be differentiated: its "
                "body splices a @raw_device block, whose text is opaque to the "
                "IR, so no rule can exist for it")

    for sink in loop.body_sinks:
        raise HawkError(
            f"loop over {loop.index.name!r} cannot be differentiated: its body "
            f"commits the sink {sink.name!r} (role {sink.role!r}) once per "
            "iteration. A body sink is what a DERIVED loop carries (the reverse "
            "of a gather is a scatter); a PRIMAL body may not commit one, "
            "so differentiating this loop would be differentiating a derivative")

    per_carry: list = []
    carry_ids = {id(c) for c in loop.carries}
    for carry, nxt in zip(loop.carries, loop.nexts):
        terms = addends(nxt)
        bare = [t for t in terms if t is carry]
        rest = [t for t in terms if t is not carry]
        if len(bare) != 1:
            return LoopClass(RECURRENCE)
        if any(_touches(t, carry_ids) for t in rest):
            return LoopClass(RECURRENCE)
        per_carry.append(_sum_of(rest))
    # no carry may be read anywhere except as those bare additive terms
    reads = 0
    for node in loop.body_nodes():
        reads += sum(1 for c in node.operands if id(c) in carry_ids)
    if reads != len(loop.carries):
        return LoopClass(RECURRENCE)
    if any(a is None for a in per_carry):
        return LoopClass(RECURRENCE)
    return LoopClass(ACCUMULATOR, tuple(per_carry))


def _touches(node: Node, carry_ids: set) -> bool:
    stack, seen = [node], set()
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if id(current) in carry_ids:
            return True
        stack.extend(current.operands)
        if isinstance(current, Loop):
            stack.extend(current.nexts)
    return False


def _sum_of(terms: list) -> Node | None:
    """``terms`` re-summed, or ``None`` when there are none (``c = c``, a carry
    the body never advances — not an accumulator, just a pass-through)."""
    from .ops import make as _make

    if not terms:
        return None
    out = terms[0]
    for term in terms[1:]:
        out = _make("add", (out, term))
    return out
