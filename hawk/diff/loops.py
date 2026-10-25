# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Autodiff THROUGH a lowered ``for`` — the two directions.

The straight-line transform (:mod:`hawk.diff.transform`) folds a flat DAG
into one derivative expression per input. A body carrying a
:class:`~hawk.ir.Loop` cannot be folded that way: a loop is a scope
evaluated ``trip`` times, and unrolling it would reintroduce the emission
that was deleted. So a loop's derivative mirrors it — another loop, over
the same range.

:func:`hawk.ir.loops.classify_loop` picks between two classes, by whether a
carried value's adjoint depends on the carry itself:

* **accumulator** (``c = c + a``, ``a`` free of every carry): the carry's
  adjoint is the same number at every iteration, so the reverse pass is a
  plain forward loop accumulating ``x_bar += (d a / d x)(k) * c_bar``.
  Nothing is stored.
* **recurrence** (anything else, e.g. ``acc = tanh(g*acc + x)``): the
  adjoint at iteration ``k`` reads the primal carry at ``k``, which the
  forward loop overwrites, so the forward loop keeps each iteration's
  carried value in a bounded local array and the adjoint loop runs
  backwards over it — generated code, not a runtime tape.

Forward mode needs no such split: one loop carries both the primal slots
and their tangents side by side.

A loop neither class fits refuses by name — a :class:`~hawk.ir.HawkError`
naming the index variable and the reason — never a silent zero or a
derivative taken as if the carry did not exist.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..ir import (
    AccumWrite,
    Assign,
    At,
    Const,
    Dispatch,
    HawkError,
    Leaf,
    Loop,
    LoopCarry,
    LoopCount,
    LoopIndex,
    LoopValue,
    Node,
    Primitive,
    SampleIndex,
    Select,
    Sink,
    TapeRead,
)
from ..ir import make as _mk
from ..ir.loops import ACCUMULATOR, RECURRENCE, LoopClass, build_loop, classify_loop
from ..ir.slotwise import chain as _chain
from ..ir.slotwise import is_structural_zero, slotwise
from ..ir.walk import substitute as _substitute
from .rules import rule_for, zero_like

#: Node families neither direction pushes through inside a body (the
#: straight-line transform's terminals, plus the two loop placeholders).
_TERMINALS = (Sink, Leaf, Const, SampleIndex, LoopIndex, LoopCarry, Loop)

#: dtypes a derivative flows through (the straight-line transform's rule).
_DIFFERENTIABLE = ("f64", "f32")


@dataclass(frozen=True)
class LoopReverse:
    """What :func:`reverse_loop` produced for one primal loop.

    ``contributions`` are ``(primal operand, adjoint)`` pairs the caller
    accumulates into its own store. ``roots`` is the derived loop when it
    commits scatters (a scatter inside a scope has nowhere else to live),
    and ``scattered`` the plane names it scattered into, so the caller
    does not also emit a dense gradient plane for them."""

    contributions: tuple = ()
    roots: tuple = ()
    scattered: frozenset = frozenset()
    #: Planes the derived loop ACCUMULATES at runtime indices once per
    #: iteration (``bar[i].at(k) += e``): the caller zeroes them and emits no
    #: dense gradient plane for them either.
    accumulated: frozenset = frozenset()


# --------------------------------------------------------------------------
# a reverse pass over ONE body


def _post_order(roots: Sequence[Node], stop: set) -> tuple:
    """The body's nodes, post-order, stopping at ``stop`` (ids) and terminals."""
    order: list = []
    seen: set = set()
    stack = [(r, False) for r in reversed(tuple(roots))]
    while stack:
        node, expanded = stack.pop()
        if id(node) in seen:
            continue
        if not expanded:
            stack.append((node, True))
            if id(node) in stop or isinstance(node, _TERMINALS):
                continue
            for child in reversed(node.operands):
                if id(child) not in seen:
                    stack.append((child, False))
            continue
        seen.add(id(node))
        order.append(node)
    return tuple(order)


def _accumulate(store: dict, node: Node, contribution: Node) -> None:
    held = store.get(id(node))
    store[id(node)] = (contribution if held is None
                       else _mk("add", (held, contribution)))


def _reduce_to(contribution: Node, target) -> Node:
    if contribution.ttype.shape == target.shape:
        return contribution
    if target.shape == ():
        return _mk("sum", (contribution,))
    raise HawkError(                              # pragma: no cover - typed above
        f"an adjoint typed {contribution.ttype!r} cannot flow into an operand "
        f"typed {target!r} — only a rank-0 operand un-broadcasts")


def _pullback(roots: Sequence[Node], seeds: Sequence[tuple], stop: set,
              inputs, name, jump=None, local_pass: bool = False) -> tuple:
    """One reverse pass over a loop body. Returns
    ``(adjoints, scatters, scattered)``.

    ``adjoints`` is ``{id(node): adjoint}`` for every node reached, read by
    the caller at the stop nodes. ``scatters`` are the body sinks the
    transpose produces: the reverse of a gather ``t.at(e(k))`` is a
    scatter-add at the same row, once per iteration. ``jump`` is the body's
    :class:`~hawk.diff.jump.JumpPlan` of a ``jump=`` kernel, else ``None``."""
    store: dict = {}
    for node, seed in seeds:
        _accumulate(store, node, seed)
    scatters: list = []
    scattered: set = set()
    for node in reversed(_post_order(roots, stop)):
        g = store.get(id(node))
        if g is None or id(node) in stop or isinstance(node, _TERMINALS):
            continue
        if isinstance(node, At):
            if local_pass:
                raise HawkError(
                    "a plane gathered with at() inside a `dispatch` branch inside "
                    "a loop body being differentiated in REVERSE: the scatter "
                    "that transposes it would have to be committed under the "
                    "branch's case, which is not built. Read the plane outside "
                    "the dispatch (the branches may use the value)")
            if node.plane in inputs:
                scattered.add(node.plane.name)
                scatters.append(AccumWrite(name(node.plane.name), g, node.index,
                                           g.ttype))
            continue
        if isinstance(node, Dispatch):
            _dispatch_pullback(node, g, store, stop, inputs, name, jump)
            continue
        if isinstance(node, (LoopValue, TapeRead)):
            raise HawkError(
                "a lowered loop's TAPE was read inside a body being "
                "differentiated in REVERSE: reverse-over-reverse through a "
                "`for` is not built. The adjoint of a tape "
                "read is a scatter INTO the tape — a write to storage the "
                "forward loop owns — and HAWK carries no rule for that. "
                "FORWARD-over-reverse IS built and computes the same "
                "second-order object: write jvp(vjp(kernel)), whose tangent of "
                "a tape read is the same row of the AUGMENTED forward loop")
        if isinstance(node, Primitive):
            for child, contribution in node.rules.vjp_contributions(node, g):
                _accumulate(store, child, _reduce_to(contribution, child.ttype))
            continue
        if jump is not None and jump.apply_vjp(
                node, g, lambda child, c: _accumulate(store, child,
                                                      _reduce_to(c, child.ttype))):
            continue
        rule = rule_for(node.kind, len(node.ttype.shape))
        if rule.vjp is None:
            raise HawkError(
                f"node kind {node.kind!r} inside a loop body carries no reverse "
                f"rule ({rule.mode})")
        for child, contribution in rule.vjp(node, g):
            _accumulate(store, child, _reduce_to(contribution, child.ttype))
    return store, tuple(scatters), frozenset(scattered)


def _dispatch_pullback(node: Dispatch, g: Node, store: dict, stop: set,
                       inputs, name, jump) -> None:
    """The reverse of a ``dispatch`` inside a loop body — the loop twin of
    :func:`hawk.diff.transform._dispatch_vjp`: K isolated pullbacks, one per
    branch, each seeded at that branch's root with ``g`` and stopped at the
    body's own stops plus what every branch shares; what each reaches at a
    stop node (a boundary operand, the index, a tape row, a leaf) is merged
    into ONE ``dispatch(kind, [c_0 or 0, ...])`` term per target, accumulated
    into the enclosing pass. A shared node is seeded once by the merge and
    differentiated through once, by the enclosing pass. ``kind`` takes no
    adjoint."""
    if jump is not None:
        raise HawkError(
            "a `dispatch` inside a loop body of a `jump=` kernel differentiated "
            "in REVERSE: the switching comparisons of the plan are not applied "
            "under a branch; this is not built")
    from .transform import _shared_by_all

    shared = _shared_by_all(node, id)
    cut = set(stop) | set(shared)
    reached: dict[int, tuple] = {}
    for j, branch in enumerate(node.branches):
        if isinstance(branch, Const):
            continue
        local, _scatters, _scattered = _pullback(
            [branch], [(branch, g)], cut, inputs, name, None, local_pass=True)
        for target in _post_order([branch], cut):
            c = local.get(id(target))
            if c is None or isinstance(target, Const):
                continue
            if not (id(target) in cut or isinstance(target, _TERMINALS)):
                continue        # differentiated through within the local pass
            if target.ttype.dtype not in _DIFFERENTIABLE:
                continue
            held = reached.setdefault(id(target), (target, {}))
            held[1][j] = c
    for target, per_branch in reached.values():
        terms = [per_branch.get(j) if per_branch.get(j) is not None
                 else zero_like(target.ttype) for j in range(node.K)]
        _accumulate(store, target, Dispatch(node.selector, terms, node.policy,
                                            target.ttype))


# --------------------------------------------------------------------------
# reverse mode


def _readable(loop: Loop) -> tuple:
    """The loop's operands whose adjoint the reverse pass may accumulate.

    A plane read through ``at()`` is excluded: its adjoint is a scatter at
    the row the gather read, not a dense value to sum. A literal, a nested
    loop and a non-differentiable dtype are excluded too. This is only the
    pre-filter; whether an operand is actually reached is decided
    afterwards, by the pullback's own store."""
    from ..ir.nodes import GATHER_ROLES

    return tuple(
        operand for operand in loop.operands
        if not isinstance(operand, (Const, Loop))
        and not (isinstance(operand, Leaf) and operand.role in GATHER_ROLES)
        and operand.ttype.dtype in _DIFFERENTIABLE
    )


def reverse_loop(loop: Loop, seeds: dict, inputs, name, jump=None,
                 exclusive: frozenset = frozenset()) -> LoopReverse:
    """The reverse-mode derivative of one lowered ``for``.

    ``seeds`` maps a carried SLOT to the adjoint of that slot's final value.
    ``inputs`` are the ``wrt`` leaves (an ``at()`` read of one scatters), and
    ``name`` is the transform's own :class:`hawk.diff.transform._Namer`;
    ``jump`` the kernel's :class:`~hawk.diff.jump.JumpPlan`, else ``None``."""
    if not seeds:
        return LoopReverse()
    if loop.body_sinks:
        raise HawkError(
            f"the loop over {loop.index.name!r} commits a sink once per iteration, "
            "which only a DERIVED loop does: second-order reverse mode through a "
            "`for` is not built")
    if loop.count is not None:
        raise HawkError(
            f"the loop over {loop.index.name!r} is a DERIVED loop bounded by a "
            "forward loop's per-sample iteration count: reverse-over-reverse "
            "through a `for` with a `break` is not built")
    nexts = loop.nexts
    if loop.exit_form == "middle":
        # The exiting iteration runs only up to the break and its carries leave
        # as `exit_values`: the tail form of the same loop, whose body result
        # is `select(exit_cond, exit_values, nexts)`. The forward loop is
        # untouched; its tape row for the exiting iteration is written before
        # the test, and its count includes that iteration.
        nexts = tuple(Select(loop.exit_cond, ev, nx, nx.ttype)
                      for ev, nx in zip(loop.exit_values, loop.nexts))
        kind = LoopClass(RECURRENCE)
    else:
        kind = classify_loop(loop)
    targets = _readable(loop)
    count = None if loop.exit_cond is None else LoopCount(loop)
    if kind.name == ACCUMULATOR:
        return _reverse_accumulator(loop, kind, seeds, targets, inputs, name,
                                    count, jump, exclusive)
    return _reverse_recurrence(loop, seeds, targets, inputs, name, count, jump,
                               exclusive, nexts)


def _reverse_accumulator(loop, kind, seeds, targets, inputs, name,
                         count=None, jump=None,
                         exclusive: frozenset = frozenset()) -> LoopReverse:
    """``c = c + a(k)``: the adjoint of the carry never changes, so the reverse
    is a FORWARD loop accumulating ``d a / d x`` weighted by that one adjoint,
    and nothing is stored ((a))."""
    contributions = [(loop.inits[slot], g) for slot, g in seeds.items()]
    live = [(kind.addends[slot], g) for slot, g in seeds.items()
            if kind.addends[slot] is not None]
    if not live:
        return LoopReverse(tuple(contributions))
    stop = {id(o) for o in loop.operands} | {id(loop.index)}
    roots = [a for a, _g in live]
    store, scatters, scattered = _pullback(
        roots, live, stop, inputs, name,
        None if jump is None else jump.body(loop, roots))
    carried = [t for t in targets if store.get(id(t)) is not None]
    carried, runtime, accumulated = _runtime_sinks(carried, store, inputs, name,
                                                   exclusive)
    scatters = scatters + runtime
    if not carried and not scatters:
        return LoopReverse(tuple(contributions))
    if carried:
        accums = tuple(LoopCarry(loop.index.depth, j, _bar_name(t, j), t.ttype)
                       for j, t in enumerate(carried))
        inits = tuple(zero_like(t.ttype) for t in carried)
        nexts = tuple(_mk("add", (accums[j], store[id(t)]))
                      for j, t in enumerate(carried))
    else:
        # Scatters only: every input the body reads is a plane, so nothing
        # sums into a local. The loop still carries one inert slot (a HAWK
        # loop's output is its carries) purely to stay expressible; the
        # compiler drops it.
        ttype = loop.carries[0].ttype
        accums = (LoopCarry(loop.index.depth, 0, "bar_step", ttype),)
        inits = (zero_like(ttype),)
        nexts = (accums[0],)
    nexts, scatters = _slotwise_body(nexts, scatters)
    derived = build_loop(loop.index, accums, inits, nexts,
                         start=loop.start, stop=loop.stop, step=loop.step,
                         body_sinks=scatters, count=count)
    for j, target in enumerate(carried):
        contributions.append((target, LoopValue(derived, j)))
    roots = (derived,) if scatters else ()
    return LoopReverse(tuple(contributions), roots, scattered, accumulated)


def _reverse_recurrence(loop, seeds, targets, inputs, name,
                        count=None, jump=None,
                        exclusive: frozenset = frozenset(),
                        nexts=None) -> LoopReverse:
    """The general case: the carry's adjoint evolves, so the derived loop
    runs backwards over the forward loop's own per-iteration carried
    values, reached through :class:`~hawk.ir.loop_nodes.TapeRead` — why it
    declares one bounded local array per carried slot."""
    m = loop.carry_count
    tapes = tuple(TapeRead(loop, i, loop.index) for i in range(m))
    body = _substitute(loop.nexts if nexts is None else nexts,
                       {id(loop.carries[i]): tapes[i] for i in range(m)})
    bars = tuple(
        LoopCarry(loop.index.depth, j, f"bar_{_clean(loop.carries[j].name)}",
                  loop.carries[j].ttype)
        for j in range(m))
    stop = ({id(o) for o in loop.operands} | {id(loop.index)}
            | {id(t) for t in tapes})
    # Every carried slot is pulled back, seeded or not. A slot with no
    # incoming seed starts its adjoint at zero, but a recurrence is exactly
    # the shape where that adjoint stops being zero mid-way: `bar_j`
    # accumulates across iterations from every other slot whose body
    # result reads slot `j`. Restricting the pullback to only the seeded
    # slots dropped those terms silently: a zero adjoint costs a folded
    # `add` of zero, but a missing one is a plausible wrong number nothing
    # folds.
    live = [(body[j], bars[j]) for j in range(m)]
    roots = [b for b, _g in live]
    store, scatters, scattered = _pullback(
        roots, live, stop, inputs, name,
        None if jump is None else jump.body(loop, roots))
    carried = [t for t in targets if store.get(id(t)) is not None]
    carried, runtime, accumulated = _runtime_sinks(carried, store, inputs, name,
                                                   exclusive)
    scatters = scatters + runtime
    accums = tuple(
        LoopCarry(loop.index.depth, m + j, _bar_name(t, j), t.ttype)
        for j, t in enumerate(carried))
    inits = tuple(seeds.get(j) or zero_like(loop.carries[j].ttype)
                  for j in range(m)) + tuple(zero_like(t.ttype) for t in carried)
    nexts = tuple(store.get(id(tapes[i])) or zero_like(loop.carries[i].ttype)
                  for i in range(m)) + tuple(
        _mk("add", (accums[j], store[id(t)])) for j, t in enumerate(carried))
    nexts, scatters = _slotwise_body(nexts, scatters)
    derived = build_loop(
        loop.index, bars + accums, inits, nexts,
        start=loop.start + (loop.trip - 1) * loop.step,
        stop=loop.start - loop.step, step=-loop.step, body_sinks=scatters,
        count=count, count_tail=True)
    contributions = [(loop.inits[j], LoopValue(derived, j)) for j in range(m)]
    for j, target in enumerate(carried):
        contributions.append((target, LoopValue(derived, m + j)))
    roots = (derived,) if scatters else ()
    return LoopReverse(tuple(contributions), roots, scattered, accumulated)


def _runtime_sinks(carried: list, store: dict, inputs, name,
                   exclusive: frozenset) -> tuple:
    """Constraint 7 (the flat form): a carried target that is a ``wrt`` leaf
    read ONLY inside loop bodies (``exclusive``, the transform's pre-scan of
    the flat walk) and whose body cotangent is an ``add_at`` chain over a
    structural zero is NOT carried — its links commit as BODY SINKS,
    ``bar[i].at(k) += e`` once per iteration on the own column (no atomics:
    one sample, one column), so the reverse loop never holds the leaf's full
    width. Any other target is carried as before (a wide ``Item``). Returns
    ``(still carried, body sinks, accumulated plane names)``."""
    kept, sinks, names = [], [], set()
    for t in carried:
        if not (isinstance(t, Leaf) and id(t) in exclusive and t.ttype.shape):
            kept.append(t)
            continue
        (value,) = slotwise((store[id(t)],))
        base, links = _chain(value)
        if not links or not is_structural_zero(base):
            kept.append(t)
            continue
        plane = name(t.name)
        sinks.append(Assign(plane, value, t.ttype))
        names.add(plane)
    return kept, tuple(sinks), frozenset(names)


def _slotwise_body(nexts: Sequence, scatters: Sequence) -> tuple:
    """A derived loop's body roots in slotwise form (:mod:`hawk.ir.slotwise`)."""
    both = slotwise((*nexts, *scatters))
    return both[:len(nexts)], both[len(nexts):]


def _clean(text: str) -> str:
    return text or "c"


def _bar_name(target: Node, j: int) -> str:
    """The carried adjoint accumulator's own name — informational (it names the
    emitted local), so it says which value it accumulates for when it can."""
    slot = target.binding
    return f"bar_{_clean(slot.name)}" if slot is not None else f"bar_t{j}"


# --------------------------------------------------------------------------
# forward mode


def _tape_tangent(node: TapeRead, augmented_of) -> Node | None:
    """The tangent of one :class:`~hawk.ir.loop_nodes.TapeRead` — the rule
    that makes forward-over-reverse work through a recurrence.

    A reverse loop over a recurrence reads the forward loop's carried value
    at iteration ``k`` from that loop's tape; its tangent is the same row
    of the forward loop's augmented twin, whose slot ``m + j`` IS the
    tangent of slot ``j``: ``TapeRead(F, j, k)``'s tangent is
    ``TapeRead(F', m + j, k)``.

    ``augmented_of`` returns ``None`` when the forward loop carries no
    tangent at all, meaning the taped value's tangent is genuinely absent
    rather than zero."""
    if augmented_of is None:
        raise HawkError(
            "a lowered loop's TAPE was read inside a body being differentiated "
            "forwards, but this transform supplied no augmented-loop lookup: "
            "forward-over-reverse needs the AUGMENTED twin of the taped forward "
            "loop, whose slot m+j carries the tangent of slot j")
    augmented = augmented_of(node.loop)
    if augmented is None:
        return None
    return TapeRead(augmented, node.loop.carry_count + node.slot, node.index)


def forward_loop(loop: Loop, tangent_of, name, augmented_of=None,
                 jump=None) -> Node | None:
    """The forward-mode derivative of one lowered ``for``: one loop carrying
    the primal slots and their tangents side by side.

    ``tangent_of(node)`` returns the tangent of a value outside the loop,
    or ``None``; the result is the augmented loop, whose slot ``m + j`` is
    the tangent of the primal loop's slot ``j``. ``augmented_of(loop)``
    returns the augmented twin of another loop this one's body reads a
    tape of, or ``None`` — needed only for forward-over-reverse (see
    :func:`_tape_tangent`).

    Also called on a VJP-derived IR's loops (forward over reverse): those
    are ordinary accumulator/recurrence loops needing no new mode, except
    for one node with no tangent rule,
    :class:`~hawk.ir.loop_nodes.TapeRead` (:func:`_tape_tangent`). The
    remaining refusals are structural: a loop committing once per
    iteration is a derived loop's scatter, and reverse-over-reverse still
    has no tape rule. ``jump`` is the kernel's
    :class:`~hawk.diff.jump.JumpPlan` of a ``jump=`` kernel, else ``None``."""
    del name
    if loop.body_sinks:
        raise HawkError(
            f"the loop over {loop.index.name!r} commits a sink once per iteration, "
            "which only a DERIVED loop does: second-order forward mode through a "
            "`for` is not built")
    classify_loop(loop)          # refuses a body no direction can differentiate
    m = loop.carry_count
    # The primal carries are reused verbatim as the augmented loop's first m
    # slots, so the primal half of the body needs no substitution and stays
    # structurally shared with the loop it mirrors.
    carries = tuple(loop.carries)
    dots = tuple(
        LoopCarry(loop.index.depth, m + j, f"dot_{_clean(loop.carries[j].name)}",
                  loop.carries[j].ttype)
        for j in range(m))
    body = tuple(loop.nexts)
    exits = tuple(loop.exit_values)
    tangent: dict = {}
    for j in range(m):
        tangent[id(carries[j])] = dots[j]
    for operand in loop.operands:
        got = tangent_of(operand)
        if got is not None:
            tangent[id(operand)] = got
    stop = ({id(o) for o in loop.operands} | {id(loop.index)}
            | {id(c) for c in carries})
    plan = None if jump is None else jump.body(loop, body + exits)

    def t(c: Node) -> Node | None:
        return tangent.get(id(c))

    for node in _post_order(body + exits, stop):
        if isinstance(node, _TERMINALS) or id(node) in stop:
            continue
        if plan is not None:
            handled, got = plan.apply_jvp(
                node, t, lambda n: rule_for(n.kind, len(n.ttype.shape)).jvp(n, t))
            if got is not None:
                tangent[id(node)] = got
            if handled:
                continue
        if isinstance(node, At):
            plane = tangent.get(id(node.plane))
            if plane is not None:
                tangent[id(node)] = At(plane, node.index, node.ttype)
            continue
        if isinstance(node, TapeRead):
            got = _tape_tangent(node, augmented_of)
            if got is not None:
                tangent[id(node)] = got
            continue
        if isinstance(node, LoopValue):     # pragma: no cover - structurally dead
            # Loop-invariant projections are cut from the body by
            # `hawk.ir.loops.boundary`; reaching one here means that
            # computation and this traversal disagree.
            raise HawkError(
                f"the loop over {loop.index.name!r} reads another loop's result "
                "as a BODY node rather than as a boundary operand — the boundary "
                "computation and the forward traversal disagree")
        if isinstance(node, Primitive):
            supplied = [tangent.get(id(c)) for c in node.inputs]
            if any(t is not None for t in supplied):
                got = node.rules.jvp_tangent(node, supplied)
                if got is not None:
                    tangent[id(node)] = got
            continue
        if isinstance(node, Dispatch):
            # the straight-line rule (hawk.diff.transform.jvp): the same
            # selector picks among the branch tangents, a tangent-free
            # branch contributing the structural zero
            branch_tangents = [tangent.get(id(b)) for b in node.branches]
            if any(t is not None for t in branch_tangents):
                filled = [t if t is not None else zero_like(b.ttype)
                          for t, b in zip(branch_tangents, node.branches)]
                tangent[id(node)] = Dispatch(node.selector, filled, node.policy,
                                             node.ttype)
            continue
        if any(tangent.get(id(c)) is not None for c in node.operands):
            rule = rule_for(node.kind, len(node.ttype.shape))
            got = rule.jvp(node, lambda c: tangent.get(id(c)))
            if got is not None:
                tangent[id(node)] = got
    inits = tuple(loop.inits) + tuple(
        tangent_of(loop.inits[j]) or zero_like(loop.carries[j].ttype)
        for j in range(m))
    dot_nexts = tuple(
        tangent.get(id(body[j])) or zero_like(loop.carries[j].ttype)
        for j in range(m))
    nexts = tuple(body) + dot_nexts
    extra: dict = {"count": loop.count, "count_tail": loop.count_tail}
    if loop.exit_cond is not None:
        # The augmented loop leaves where the primal does, over the same
        # condition, so tangents are carried along exactly the iterations
        # the sample ran.
        dot_exits = tuple(
            dot_nexts[j] if exits[j] is body[j]
            else tangent.get(id(exits[j])) or zero_like(loop.carries[j].ttype)
            for j in range(m))
        extra.update(exit_cond=loop.exit_cond, exit_values=exits + dot_exits)
    n_next, n_exit = len(nexts), len(extra.get("exit_values", ()))
    exits_now = tuple(extra.get("exit_values", ()))
    rooted = slotwise((*nexts, *exits_now, *([loop.exit_cond] if n_exit else [])))
    nexts = rooted[:n_next]
    if n_exit:
        extra.update(exit_values=rooted[n_next:n_next + n_exit], exit_cond=rooted[-1])
    return build_loop(loop.index, carries + dots, inits, nexts,
                      start=loop.start, stop=loop.stop, step=loop.step, **extra)
