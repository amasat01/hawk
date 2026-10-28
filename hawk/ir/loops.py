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
* refused, NAMED — a nested loop that leaves early or commits per iteration
  (:func:`refuse_unsupported_nesting`); any other nested loop differentiates.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .loop_nodes import Loop, LoopCarry, LoopIndex
from .nodes import RAW_KIND, At, Const, Dispatch, HawkError, Node, Select

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


def _gathers(node: Node, memo: dict) -> bool:
    """Whether ``node``'s subtree contains an :class:`At` (memoised on ``id``)."""
    hit = memo.get(id(node))
    if hit is not None:
        return hit
    stack = [(node, False)]
    while stack:
        current, expanded = stack.pop()
        if not expanded:
            if id(current) in memo:
                continue
            if isinstance(current, At):
                memo[id(current)] = True
                continue
            stack.append((current, True))
            stack.extend((c, False) for c in current.operands)
            continue
        memo[id(current)] = any(memo.get(id(c), False) for c in current.operands)
    return memo[id(node)]


def _gated_reads(nexts: Sequence[Node], own: frozenset) -> tuple:
    """``nexts`` with each loop-invariant plane read of a ``dispatch`` arm gated by
    the arm's own condition, and the ids of the gates made.

    With a loop-invariant selector, a scalar invariant subtree of an arm that
    gathers is valid exactly where the selector picks that arm. Wrapped as
    ``select(arm taken, read, 0)`` it is valid everywhere, so :func:`boundary`
    hoists it out of the loop and the arm reads the hoisted value. A read under a
    nested arm keeps that arm's guard and is left alone."""
    from .ops import make
    from .walk import substitute

    memo: dict = {}
    gathers: dict = {}
    swaps: dict = {}
    gates: set = set()
    seen: set = set()
    stack = list(nexts)
    while stack:
        node = stack.pop()
        if id(node) in seen or not _depends(node, own, memo):
            continue
        seen.add(id(node))
        if not isinstance(node, Dispatch) or _depends(node.operands[0], own, memo):
            stack.extend(node.operands)
            continue
        stack.append(node.operands[0])      # the arms are handled here, not descended
        sel, arms = node.operands[0], list(node.operands[1:])
        for i, arm in enumerate(arms):
            found: dict = {}
            walk, visited = [arm], set()
            while walk:
                n = walk.pop()
                if id(n) in visited:
                    continue
                visited.add(id(n))
                if isinstance(n, Loop):
                    continue                        # a nested loop is never gated
                if not _depends(n, own, memo):
                    if _gathers(n, gathers) and not n.ttype.shape:
                        found[id(n)] = n
                        continue
                    walk.extend(n.operands)
                elif isinstance(n, (Dispatch, Select)):
                    walk.append(n.operands[0])      # a nested arm keeps its guard
                else:
                    walk.extend(n.operands)
            if not found:
                continue
            last = len(arms) - 1
            kind = "le" if i == 0 else "ge" if i == last else "eq"
            taken = make(kind, (sel, Const(i, sel.ttype)))
            gate = {}
            for key, g in found.items():
                zero = Const(0 if g.ttype.dtype in ("i32", "int32") else 0.0, g.ttype)
                gate[key] = Select(taken, g, zero, g.ttype)
            gates.update(id(v) for v in gate.values())
            arms[i] = substitute([arm], gate)[0]
        if any(a is not b for a, b in zip(arms, node.operands[1:])):
            swaps[id(node)] = Dispatch(sel, arms, node.policy, node.ttype)
    return (substitute(nexts, swaps) if swaps else tuple(nexts)), frozenset(gates)


def boundary(nexts: Sequence[Node], own: frozenset,
             gates: frozenset = frozenset()) -> tuple:
    """The body's loop-invariant BOUNDARY values, in deterministic order.

    Maximal: the walk stops at the highest invariant node on each path, so
    its children ride along inside that one definition. A ``next`` that is
    invariant in its entirety is its own boundary value — a loop whose body
    ignores the index, odd but well-formed."""
    memo: dict = {}
    gathers: dict = {}
    out: list = []
    seen: dict = {}                    # id -> was it last visited guarded
    # (node, guarded): a node under a `dispatch` / `select` ARM is guarded — the arm
    # runs only where its case holds, and a plane read there at a runtime row is only
    # valid there. Hoisting such a read out of the loop would evaluate it for every
    # lane and iteration, so a guarded subtree that gathers stays in the body (its
    # gather-free invariant parts still cross the boundary).
    stack = [(n, False) for n in reversed(tuple(nexts))]
    while stack:
        node, guarded = stack.pop()
        # an UNGUARDED visit overrides a guarded one that stayed in the body: the
        # node is read outside every arm too, so it is safe to hoist
        if id(node) in seen and (guarded or not seen[id(node)]):
            continue
        seen[id(node)] = guarded
        if not _depends(node, own, memo) and not (
                guarded and id(node) not in gates and _gathers(node, gathers)):
            out.append(node)
            continue
        arms = (frozenset(id(a) for a in node.operands[1:])
                if isinstance(node, (Dispatch, Select)) else frozenset())
        for child in reversed(node.operands):
            stack.append((child, guarded or id(child) in arms))
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
               count: Node | None = None, count_tail: bool = False,
               dynamic: str | None = None) -> Loop:
    """Mint a :class:`~hawk.ir.loop_nodes.Loop`, computing its boundary for it.

    THE constructor every producer uses (the tracer, :mod:`hawk.diff.loops`
    for a derived loop), so a derived loop's operands are computed by
    exactly the code that computed the primal's."""
    own = frozenset({id(index), *(id(c) for c in carries)})
    nexts, gates = _gated_reads(nexts, own)
    body_sinks = tuple(body_sinks)
    exit_values = tuple(exit_values)
    roots = nexts + tuple(o for sink in body_sinks for o in sink.operands)
    if exit_cond is not None:
        roots += (exit_cond, *exit_values)
    held = {id(n) for n in inits}
    if count is not None:
        held.add(id(count))
    free = tuple(n for n in boundary(roots, own, gates) if id(n) not in held)
    loop = Loop(index, carries, inits, nexts, start=start, stop=stop, step=step,
                free=free, body_sinks=body_sinks, exit_cond=exit_cond,
                exit_values=exit_values, count=count, count_tail=count_tail,
                dynamic=dynamic)
    _check_placeholders(loop)
    return loop


def exit_kwargs(loop: Loop) -> dict:
    """``loop``'s ``break``/count as :func:`build_loop` keywords, for a
    REBUILD that must leave early exactly where the original did."""
    return {"exit_cond": loop.exit_cond, "exit_values": loop.exit_values,
            "count": loop.count, "count_tail": loop.count_tail,
            "dynamic": loop.dynamic}


def _check_placeholders(loop: Loop) -> None:
    """Every index/carry placeholder inside the body must be THIS loop's.

    A derived loop substitutes the primal's carried reads for something
    else; a missed substitution would leave a foreign placeholder the
    digest keys as this loop's own carry — a silently wrong recurrence,
    checked once, here."""
    own = {id(loop.index)} | {id(c) for c in loop.carries}
    # an ENCLOSING loop's index or carry (shallower depth) legitimately crosses as
    # an operand; a placeholder of the SAME depth that is not ours is a missed
    # substitution and stays refused
    crossing = {id(o) for o in loop.operands
                if isinstance(o, (LoopIndex, LoopCarry)) and o.depth < loop.index.depth}
    for node in loop.body_nodes():
        for child in node.operands:
            if (isinstance(child, (LoopIndex, LoopCarry)) and id(child) not in own
                    and id(child) not in crossing):
                raise HawkError(
                    f"the loop over {loop.index.name!r} reads {child.kind} "
                    f"{getattr(child, 'name', '?')!r} (depth {child.depth}), which "
                    "belongs to a DIFFERENT loop. A loop body may only read its own "
                    "index and its own carried slots; anything else must cross the "
                    "boundary as an operand")
    for root in loop.body_roots():
        if (isinstance(root, (LoopIndex, LoopCarry)) and id(root) not in own
                and id(root) not in crossing):
            raise HawkError(                      # pragma: no cover - as above
                f"the loop over {loop.index.name!r} returns a placeholder of a "
                "different loop")


def addends(node: Node) -> list:
    """Flatten an ``add`` chain into its terms (``add`` only — a ``sub`` at the
    top alternates the sign of what follows and is not the additive shape)."""
    if getattr(node, "kind", None) == "add":
        return addends(node.operands[0]) + addends(node.operands[1])
    return [node]


def refuse_unsupported_nesting(loop: Loop) -> None:
    """Refuse, BY NAME, a body whose nested ``for`` neither direction can carry.

    A nested loop differentiates: reverse recomputes the inner forward loop inside
    each reversed outer iteration (its carried history lives in a LOCAL tape of
    the inner trip count, never a tape of a tape), forward carries the inner
    tangents in the inner loop. Three inner shapes are not built: one that leaves
    early (its per-sample trip count would have to be threaded through every outer
    iteration), one that commits a sink per iteration (a derived loop), and, in
    reverse, one that gathers a plane being differentiated (its scatter would sit
    inside a scope inside a scope: checked where the scatter is made)."""
    for inner in loop.body_nodes():
        if not isinstance(inner, Loop):
            continue
        if inner.exit_cond is not None:
            raise HawkError(
                f"loop over {loop.index.name!r} cannot be differentiated: its body "
                f"contains a nested `for` over {inner.index.name!r} with a `break`. "
                "A nested loop differentiates, but one that leaves early would "
                "need its per-sample trip count threaded through every outer "
                "iteration, which is not built. Move the `break` to the outer loop, "
                "or run the inner loop to its bound and mask")
        if inner.body_sinks or inner.count is not None:
            raise HawkError(
                f"loop over {loop.index.name!r} cannot be differentiated: its body "
                f"contains a nested loop over {inner.index.name!r} that commits a "
                "sink or runs a recorded trip count (a derived loop); "
                "differentiating a derivative's nested loop is not built")


# --------------------------------------------------------------------------
# the classifier


def classify_loop(loop: Loop) -> LoopClass:
    """Which of :data:`LOOP_CLASSES` ``loop`` is, or a refusal NAMING why.

    Narrow and structural: every carry's body result must be an ``add``
    chain containing that carry EXACTLY once as a bare term, read nowhere
    else. Anything weaker could differentiate a body whose adjoint
    genuinely depends on the carried value as if it didn't — a wrong
    gradient, not a refused one."""
    refuse_unsupported_nesting(loop)
    for inner in loop.body_nodes():
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
