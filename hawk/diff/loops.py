# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Autodiff THROUGH a lowered ``for`` — the two directions.

The straight-line transform (:mod:`hawk.diff.transform`) folds a flat DAG
into one derivative expression per input. A body carrying a
``Loop`` node cannot be folded that way: a loop is a scope
evaluated ``trip`` times, and unrolling it would reintroduce the emission
that was deleted. So a loop's derivative mirrors it — another loop, over
the same range.

:func:`hawk._ir.classify_loop` picks between two classes, by whether a
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

A loop neither class fits refuses by name — a :class:`~hawk.HawkError`
naming the index variable and the reason — never a silent zero or a
derivative taken as if the carry did not exist.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .. import _ir
from ..errors import HawkError
from .functions import call_adjoints, call_tangent
from ..trace.value import dispatch_node, dispatch_strategy
from . import rules
from .tape import _carry

#: Kinds of node a loop BODY's walk does not enter: bound values, literals, sinks and the loop's
#: own index and carries (``_LEAF_KINDS``), and a nested loop (handled as one unit, ``_TERMINALS``).
#: The straight-line transform's own list differs by design (it has ``plane_extent``, no loop).
_LEAF_KINDS = ("sink", "leaf", "const", "sample_index", "loop_index", "loop_carry")
_TERMINALS = (*_LEAF_KINDS, "loop")

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
# reading and building the nodes of a loop


def _index_of(loop: Any) -> Any:
    """The loop's index variable."""
    return loop.field_nodes("index")[0]


def _index_name(loop: Any) -> str:
    """The name of the loop's index variable, for a refusal."""
    return _index_of(loop).field_text("name")


def _carries(loop: Any) -> tuple:
    """The loop's carried values (its ``LoopCarry`` placeholders), in slot order."""
    return loop.field_nodes("carries")


def _inits(loop: Any) -> tuple:
    """The values the carried slots start from: the first operands, one per carry."""
    return loop.operands[: loop.field_int("carry_count")]


def _bounds(loop: Any) -> dict:
    """The loop's ``start``, ``stop`` and ``step`` as :func:`hawk._ir.loop` keywords."""
    return {name: loop.field_int(name) for name in ("start", "stop", "step")}


def with_body_sinks(loop: Any, body_sinks: Sequence[Any]) -> Any:
    """``loop`` over the same range, carries and exit, committing ``body_sinks`` once per iteration."""
    return _ir.loop(
        _index_of(loop),
        _carries(loop),
        _inits(loop),
        loop.field_nodes("nexts"),
        **_bounds(loop),
        body_sinks=tuple(body_sinks),
        **_ir.exit_kwargs(loop),
    )


def _loop_carry(loop: Any, slot: int, name: str, ttype: Any) -> Any:
    """A carried placeholder at ``slot`` of a loop at the depth of ``loop``."""
    depth = _index_of(loop).field_int("depth")
    return _carry(loop.ctx, depth, slot, name, ttype)


def _loop_count(loop: Any) -> Any:
    """The number of iterations a loop with a ``break`` ran."""
    return _ir.make_node("LoopCount", ((), "i32", None), [loop], literal=None)


def _tape_read(loop: Any, slot: int, index: Any) -> Any:
    """The value carried slot ``slot`` of ``loop`` had at iteration ``index``."""
    ttype = _ir.type_of(_carries(loop)[slot])
    return _ir.make_node("TapeRead", ttype, [loop, index], literal=slot, slot=slot)


def _scatter_add(plane: str, value: Any, indices: Sequence[Any], edge: str, axes: Sequence[Any]) -> Any:
    """The scatter-add of ``value`` into the plane ``plane`` at ``indices``, under the edge rule of the read it transposes."""
    return _ir.make_node(
        "Update",
        _ir.type_of(value),
        [value, *indices],
        name=plane,
        op="add",
        reduced=None,
        edge=edge,
        axes=tuple(axes),
    )


def _select(cond: Any, on_true: Any, on_false: Any) -> Any:
    """Make ``cond ? on_true : on_false``, typed as ``on_true`` (both arms share one type)."""
    return _ir.make_node("Select", _ir.type_of(on_true), [cond, on_true, on_false])


# --------------------------------------------------------------------------
# a reverse pass over ONE body


def _post_order(roots: Sequence[Any], stop: set) -> tuple:
    """The body's nodes, post-order, stopping at ``stop`` and terminals.

    A NESTED ``Loop`` is descended into (its operands, the boundary values it
    reads) and appears in the order after them; the loop a tape read or a trip
    count names is NOT (that is a statement the reverse loop reads, not a node
    of this body)."""
    order: list = []
    seen: set = set()
    stack = [(r, False) for r in reversed(tuple(roots))]
    while stack:
        node, expanded = stack.pop()
        if node in seen:
            continue
        if not expanded:
            stack.append((node, True))
            if node in stop or _ir.is_a(node, *_LEAF_KINDS):
                continue
            if _ir.is_a(node, "tape_read", "loop_count"):
                if _ir.is_a(node, "tape_read"):
                    owner, index = node.operands
                    stack.append((index, False))
                    if owner not in stop:  # a NESTED forward loop's tape
                        stack.append((owner, False))
                continue
            for child in reversed(node.operands):
                if child not in seen:
                    stack.append((child, False))
            continue
        seen.add(node)
        order.append(node)
    return tuple(order)


def _accumulate(store: dict, node: Any, contribution: Any) -> None:
    """Add ``contribution`` to the adjoint ``store`` holds for ``node``."""
    held = store.get(node)
    store[node] = contribution if held is None else _ir.make("add", (held, contribution))


def _reduce_to(contribution: Any, operand: Any) -> Any:
    """Un-broadcast: a rank-0 ``operand``'s adjoint is the SUM of the contributions its broadcast
    produced (the transpose of a broadcast)."""
    flowing, target = _ir.type_of(contribution), _ir.type_of(operand)
    if flowing.shape == target.shape:
        return contribution
    if target.shape == ():
        return _ir.make("sum", (contribution,))
    raise HawkError(
        f"an adjoint typed {flowing!r} cannot flow into an operand typed "
        f"{target!r} — only a rank-0 operand un-broadcasts"
    )


def _pullback(
    roots: Sequence[Any],
    seeds: Sequence[tuple],
    stop: set,
    inputs,
    name,
    jump=None,
    local_pass: bool = False,
    inject: Sequence[tuple] = (),
) -> tuple:
    """One reverse pass over a loop body. Returns
    ``(adjoints, scatters, scattered)``.

    ``adjoints`` is ``{node: adjoint}`` for every node reached, read by
    the caller at the stop nodes. ``scatters`` are the body sinks the
    transpose produces: the reverse of a gather ``t.at(e(k))`` is a
    scatter-add at the same row, once per iteration. ``jump`` is the body's
    :class:`~hawk.diff.jump.JumpPlan` of a ``jump=`` kernel, else ``None``.
    ``inject`` holds the ``(comparison, weight)`` pairs of an engaged plan's
    alternate path: the weight is the pseudo-adjoint of ``l - r``."""
    store: dict = {}
    for node, seed in seeds:
        _accumulate(store, node, seed)
    for site, weight in inject:
        lhs, rhs = site.operands
        _accumulate(store, lhs, weight)
        _accumulate(store, rhs, _ir.make("neg", (weight,)))
    scatters: list = []
    scattered: set = set()
    loop_seeds: dict = {}
    for node in reversed(_post_order(roots, stop)):
        g = store.get(node)
        if _ir.is_a(node, "loop_value") and node not in stop:
            if g is not None:
                owner, slot = _ir.loop_value_of(node)
                loop_seeds.setdefault(owner, {})[slot] = g
            continue
        if _ir.is_a(node, "loop") and node not in stop:
            _nested_pullback(node, loop_seeds.get(node, {}), store, inputs, name, jump)
            continue
        if g is None or node in stop or _ir.is_a(node, *_TERMINALS):
            continue
        if _ir.is_a(node, "at"):
            if local_pass:
                # a branch-local pass: ``g`` stays in the local store and the
                # enclosing :func:`_dispatch_pullback` commits ONE scatter, at a
                # row only the taken branch computes
                continue
            plane, *indices = node.operands
            if plane in inputs:
                scattered.add(plane.field_text("name"))
                scatters.append(
                    _scatter_add(
                        name(plane.field_text("name")), g, indices, node.field_text("edge"),
                        _ir.field_pairs(plane, "axes"),
                    )
                )
            continue
        if _ir.is_a(node, "dispatch"):
            _dispatch_pullback(
                node, g, store, stop, inputs, name, jump, scatters, scattered, local_pass
            )
            continue
        if _ir.is_a(node, "loop_value", "tape_read"):
            raise HawkError(
                "a lowered loop's TAPE was read inside a body being "
                "differentiated in REVERSE: reverse-over-reverse through a "
                "`for` is not built. The adjoint of a tape "
                "read is a scatter INTO the tape — a write to storage the "
                "forward loop owns — and HAWK carries no rule for that. "
                "FORWARD-over-reverse IS built and computes the same "
                "second-order object: write jvp(vjp(kernel)), whose tangent of "
                "a tape read is the same row of the AUGMENTED forward loop"
            )
        if _ir.is_a(node, "call"):
            for child, contribution in call_adjoints(node, g, name.derived):
                _accumulate(store, child, _reduce_to(contribution, child))
            continue
        if jump is not None and jump.apply_vjp(
            node,
            g,
            lambda child, c: _accumulate(store, child, _reduce_to(c, child)),
        ):
            continue
        if jump is None:
            guarded = guarded_select(node)
            if guarded is not None:
                _dispatch_pullback(
                    guarded, g, store, stop, inputs, name, jump, scatters, scattered, local_pass
                )
                continue
        rule = rules.rule_of(node)
        if rule.vjp is None:
            raise HawkError(
                f"node kind {node.field_text('kind')!r} inside a loop body carries no reverse "
                f"rule ({rule.mode})"
            )
        for child, contribution in rule.vjp(node, g):
            _accumulate(store, child, _reduce_to(contribution, child))
    return store, tuple(scatters), frozenset(scattered)


def _nested_pullback(inner: Any, seeds: dict, store: dict, inputs, name, jump) -> None:
    """The reverse of a ``for`` nested in the body being reversed: the loop's own
    reverse (:func:`reverse_loop`), whose forward half is rebuilt INSIDE the
    reversed iteration with its carried history in a local tape of the inner trip
    count, folded into this body's adjoints like any other node's rule."""
    if not seeds:
        return
    if jump is not None:
        raise HawkError(
            f"the nested loop over {_index_name(inner)!r} of a `jump=` kernel "
            "differentiated in REVERSE: the switching comparisons of the plan are "
            "not applied through a nested loop; this is not built"
        )
    derived = reverse_loop(inner, seeds, inputs, name)
    if derived.roots or derived.accumulated:
        raise HawkError(
            f"the nested loop over {_index_name(inner)!r} gathers a plane being "
            "differentiated (`at()` of a `wrt` input) inside a loop inside a loop, "
            "differentiated in REVERSE: its scatter would be committed inside two "
            "scopes, which is not built. Read the plane in the OUTER loop (or "
            "before both) and let the inner loop use the value, or flatten the "
            "two loops into one"
        )
    for child, contribution in derived.contributions:
        _accumulate(store, child, _reduce_to(contribution, child))


def _dispatch_pullback(
    node: Any,
    g: Any,
    store: dict,
    stop: set,
    inputs,
    name,
    jump,
    scatters: list,
    scattered: set,
    local_pass: bool = False,
) -> None:
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
            "under a branch; this is not built"
        )
    from .transform import _shared_by_all

    selector, *branches = node.operands
    shared = _shared_by_all(node, lambda n: n)  # a body is keyed on node identity
    cut = set(stop) | set(shared)
    reached: dict = {}
    for j, branch in enumerate(branches):
        if _ir.is_a(branch, "const"):
            continue
        local, _scatters, _scattered = _pullback(
            [branch], [(branch, g)], cut, inputs, name, None, local_pass=True
        )
        for target in _post_order([branch], cut):
            c = local.get(target)
            if c is None or _ir.is_a(target, "const"):
                continue
            if not (target in cut or _ir.is_a(target, *_TERMINALS, "at")):
                continue  # differentiated through within the local pass
            if _ir.type_of(target).dtype not in _DIFFERENTIABLE:
                continue
            held = reached.setdefault(target, {})
            held[j] = c
    for target, per_branch in reached.items():
        ttype = _ir.type_of(target)
        terms = [
            per_branch[j] if per_branch.get(j) is not None else _ir.zero_like(target.ctx, ttype)
            for j in range(len(branches))
        ]
        combined = dispatch_node(selector, terms, dispatch_strategy(node), ttype)
        if _ir.is_a(target, "at"):
            if local_pass:
                raise HawkError(
                    "a plane gathered with at() inside a `dispatch` inside a "
                    "`dispatch` branch of a loop body differentiated in REVERSE: "
                    "the scatter's row is only valid under both selectors, which "
                    "is not built. Read the plane before the outer `dispatch`"
                )
            plane = target.operands[0]
            if plane in inputs:
                scattered.add(plane.field_text("name"))
                scatters.append(
                    _scatter_add(
                        name(plane.field_text("name")),
                        combined,
                        _taken_indices(node, target, per_branch),
                        target.field_text("edge"),
                        _ir.field_pairs(plane, "axes"),
                    )
                )
            continue
        _accumulate(store, target, combined)


def has_gather(node: Any, memo: dict | None = None) -> bool:
    """Whether ``node``'s subtree reads a plane at a runtime row (an ``At`` node)."""
    memo = {} if memo is None else memo
    stack = [(node, False)]
    while stack:
        current, expanded = stack.pop()
        if current in memo and not expanded:
            continue
        if not expanded:
            if _ir.is_a(current, "at"):
                memo[current] = True
                continue
            stack.append((current, True))
            stack.extend((c, False) for c in current.operands if c not in memo)
            continue
        memo[current] = any(memo.get(c, False) for c in current.operands)
    return memo[node]


def guarded_select(node: Any) -> Any:
    """A ``select`` with a gather in an arm, as the two-case ``switch`` it guards
    like: the arm's rows are valid only where its condition holds, so the reverse
    must read and scatter them only there (a ``select``'s pullback would evaluate
    both arms' gathers for every lane). ``None`` for any other node."""
    if not (_ir.is_a(node, "select") and any(has_gather(a) for a in node.operands[1:])):
        return None
    cond, on_true, on_false = node.operands
    one = _ir.const(node.ctx, 1, ((), "i32", None))
    zero = _ir.const(node.ctx, 0, ((), "i32", None))
    chosen = _select(cond, one, zero)
    return dispatch_node(chosen, [on_false, on_true], "switch", _ir.type_of(node))


def _taken_indices(node: Any, target: Any, per_branch: dict) -> list:
    """The indices a branch-gathered plane's scatter commits at.

    The gather's indices are computed only by the branches that read it; the scatter
    sits outside the ``dispatch`` and would evaluate them for every lane, so a lane
    whose branch did NOT read the plane could commit (a zero) at a position that is out
    of range for it. Indices every branch computes are kept as they are; otherwise each index
    is itself a ``dispatch`` of the same selector: the gather's index where the branch
    gathered, 0 (always in range, and a zero is added there) elsewhere."""
    selector, *branches = node.operands
    indices = target.operands[1:]
    if len(per_branch) == len(branches):
        return list(indices)
    taken = []
    for index in indices:
        index_type = _ir.type_of(index)
        zero = _ir.const(target.ctx, 0, index_type)
        arms = [index if j in per_branch else zero for j in range(len(branches))]
        taken.append(dispatch_node(selector, arms, dispatch_strategy(node), index_type))
    return taken


# --------------------------------------------------------------------------
# reverse mode


def _readable(loop: Any) -> tuple:
    """The loop's operands whose adjoint the reverse pass may accumulate.

    A plane read through ``at()`` is excluded: its adjoint is a scatter at
    the row the gather read, not a dense value to sum. A literal, a nested
    loop and a non-differentiable dtype are excluded too. This is only the
    pre-filter; whether an operand is actually reached is decided
    afterwards, by the pullback's own store."""
    return tuple(
        operand
        for operand in loop.operands
        if not _ir.is_a(operand, "const", "loop")
        and not (_ir.is_a(operand, "leaf") and operand.field_text("role") in _ir.GATHER_ROLES)
        and _ir.type_of(operand).dtype in _DIFFERENTIABLE
    )


def reverse_loop(
    loop: Any, seeds: dict, inputs, name, jump=None, exclusive: frozenset = frozenset(), alt=None
) -> LoopReverse:
    """The reverse-mode derivative of one lowered ``for``.

    ``seeds`` maps a carried SLOT to the adjoint of that slot's final value.
    ``inputs`` are the ``wrt`` leaves (an ``at()`` read of one scatters), and
    ``name`` is the transform's own :class:`hawk.diff.transform._Namer`;
    ``jump`` the kernel's :class:`~hawk.diff.jump.JumpPlan`, else ``None``;
    ``alt`` the :class:`~hawk.diff.altpath.AltPath` of an engaged one, whose
    drawn comparison injects its weight into the body at its iteration."""
    sites = () if alt is None else alt.sites_of(loop)
    if not seeds and not sites:
        return LoopReverse()
    _ir.refuse_unsupported_nesting(loop)
    exits = _ir.exit_kwargs(loop)
    if loop.field_nodes("body_sinks"):
        raise HawkError(
            f"the loop over {_index_name(loop)!r} commits a sink once per iteration, "
            "which only a DERIVED loop does: second-order reverse mode through a "
            "`for` is not built"
        )
    if exits["dynamic"] == "open":
        raise HawkError(
            f"the loop over {_index_name(loop)!r} has a run-time trip count and no "
            "`max=`, so reverse mode cannot size its tape: give `max=` so the tape "
            "has a size, `range(n, max=K)`"
        )
    if exits["count"] is not None and exits["dynamic"] is None:
        raise HawkError(
            f"the loop over {_index_name(loop)!r} is a DERIVED loop bounded by a "
            "forward loop's per-sample iteration count: reverse-over-reverse "
            "through a `for` with a `break` is not built"
        )
    nexts = loop.field_nodes("nexts")
    if _ir.exit_form(loop) == "middle":
        # The exiting iteration runs only up to the break and its carries leave
        # as `exit_values`: the tail form of the same loop, whose body result
        # is `select(exit_cond, exit_values, nexts)`. The forward loop is
        # untouched; its tape row for the exiting iteration is written before
        # the test, and its count includes that iteration.
        nexts = tuple(
            _select(exits["exit_cond"], ev, nx) for ev, nx in zip(exits["exit_values"], nexts)
        )
        kind = _ir.LoopClass(_ir.RECURRENCE)
    elif sites:
        kind = _ir.LoopClass(_ir.RECURRENCE)
    else:
        kind = _ir.classify_loop(loop)
    targets = _readable(loop)
    if exits["dynamic"]:
        count = exits["count"]
    else:
        count = None if exits["exit_cond"] is None else _loop_count(loop)
    if kind.name == _ir.ACCUMULATOR:
        return _reverse_accumulator(
            loop, kind, seeds, targets, inputs, name, count, jump, exclusive
        )
    return _reverse_recurrence(
        loop, seeds, targets, inputs, name, count, jump, exclusive, nexts, alt
    )


def _reverse_accumulator(
    loop,
    kind,
    seeds,
    targets,
    inputs,
    name,
    count=None,
    jump=None,
    exclusive: frozenset = frozenset(),
) -> LoopReverse:
    """``c = c + a(k)``: the adjoint of the carry never changes, so the reverse
    is a FORWARD loop accumulating ``d a / d x`` weighted by that one adjoint,
    and nothing is stored ((a))."""
    inits = _inits(loop)
    contributions = [(inits[slot], g) for slot, g in seeds.items()]
    live = [(kind.addends[slot], g) for slot, g in seeds.items() if kind.addends[slot] is not None]
    if not live:
        return LoopReverse(tuple(contributions))
    stop = set(loop.operands) | {_index_of(loop)}
    roots = [a for a, _g in live]
    store, scatters, scattered = _pullback(
        roots, live, stop, inputs, name, None if jump is None else jump.body(loop, roots)
    )
    carried = [t for t in targets if store.get(t) is not None]
    carried, runtime, accumulated = _runtime_sinks(carried, store, inputs, name, exclusive)
    scatters = scatters + runtime
    if not carried and not scatters:
        return LoopReverse(tuple(contributions))
    if carried:
        accums = tuple(
            _loop_carry(loop, j, _bar_name(t, j), _ir.type_of(t)) for j, t in enumerate(carried)
        )
        starts = tuple(_ir.zero_like(loop.ctx, _ir.type_of(t)) for t in carried)
        nexts = tuple(_ir.make("add", (accums[j], store[t])) for j, t in enumerate(carried))
    else:
        # Scatters only: every input the body reads is a plane, so nothing
        # sums into a local. The loop still carries one inert slot (a HAWK
        # loop's output is its carries) purely to stay expressible; the
        # compiler drops it.
        ttype = _ir.type_of(_carries(loop)[0])
        accums = (_loop_carry(loop, 0, "bar_step", ttype),)
        starts = (_ir.zero_like(loop.ctx, ttype),)
        nexts = (accums[0],)
    nexts, scatters = _slotwise_body(nexts, scatters)
    derived = _ir.loop(
        _index_of(loop), accums, starts, nexts, **_bounds(loop), body_sinks=scatters, count=count
    )
    for j, target in enumerate(carried):
        contributions.append((target, _ir.loop_value(derived, j)))
    roots = (derived,) if scatters else ()
    return LoopReverse(tuple(contributions), roots, scattered, accumulated)


def _reverse_recurrence(
    loop,
    seeds,
    targets,
    inputs,
    name,
    count=None,
    jump=None,
    exclusive: frozenset = frozenset(),
    nexts=None,
    alt=None,
) -> LoopReverse:
    """The general case: the carry's adjoint evolves, so the derived loop
    runs backwards over the forward loop's own per-iteration carried
    values, reached through ``TapeRead`` nodes — why it
    declares one bounded local array per carried slot."""
    carries = _carries(loop)
    m = len(carries)
    index = _index_of(loop)
    tapes = tuple(_tape_read(loop, i, index) for i in range(m))
    forward = tuple(loop.field_nodes("nexts") if nexts is None else nexts)
    sites = () if alt is None else alt.sites_of(loop)
    body = _ir.substitute(forward + sites, {carries[i]: tapes[i] for i in range(m)})
    body, rest = body[: len(forward)], body[len(forward) :]
    stop = set(loop.operands) | {index} | set(tapes)
    sites = alt.inject(loop, rest, body, stop) if sites else ()
    bars = tuple(
        _loop_carry(
            loop, j, f"bar_{_clean(carries[j].field_text('name'))}", _ir.type_of(carries[j])
        )
        for j in range(m)
    )
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
        roots,
        live,
        stop,
        inputs,
        name,
        None if jump is None else jump.body(loop, roots),
        inject=sites,
    )
    carried = [t for t in targets if store.get(t) is not None]
    carried, runtime, accumulated = _runtime_sinks(carried, store, inputs, name, exclusive)
    scatters = scatters + runtime
    accums = tuple(
        _loop_carry(loop, m + j, _bar_name(t, j), _ir.type_of(t)) for j, t in enumerate(carried)
    )
    starts = tuple(_seed_or_zero(seeds.get(j), carries[j]) for j in range(m)) + tuple(
        _ir.zero_like(loop.ctx, _ir.type_of(t)) for t in carried
    )
    nexts = tuple(_seed_or_zero(store.get(tapes[i]), carries[i]) for i in range(m)) + tuple(
        _ir.make("add", (accums[j], store[t])) for j, t in enumerate(carried)
    )
    nexts, scatters = _slotwise_body(nexts, scatters)
    start, step, trip = (loop.field_int(key) for key in ("start", "step", "trip"))
    derived = _ir.loop(
        index,
        bars + accums,
        starts,
        nexts,
        start=start + (trip - 1) * step,
        stop=start - step,
        step=-step,
        body_sinks=scatters,
        count=count,
        count_tail=True,
    )
    inits = _inits(loop)
    contributions = [(inits[j], _ir.loop_value(derived, j)) for j in range(m)]
    for j, target in enumerate(carried):
        contributions.append((target, _ir.loop_value(derived, m + j)))
    roots = (derived,) if scatters else ()
    return LoopReverse(tuple(contributions), roots, scattered, accumulated)


def _seed_or_zero(value: Any, like: Any) -> Any:
    """``value``, or the structural zero of ``like``'s type when there is none."""
    return value if value is not None else _ir.zero_like(like.ctx, _ir.type_of(like))


def _runtime_sinks(carried: list, store: dict, inputs, name, exclusive: frozenset) -> tuple:
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
        if not (_ir.is_a(t, "leaf") and t in exclusive and _ir.type_of(t).shape):
            kept.append(t)
            continue
        (value,) = _ir.slotwise((store[t],))
        base, links = _ir.slotwise_chain(value)
        if not links or not _ir.is_structural_zero(base):
            kept.append(t)
            continue
        plane = name(t.field_text("name"))
        sinks.append(_ir.make_node("Assign", _ir.type_of(t), [value], name=plane))
        names.add(plane)
    return kept, tuple(sinks), frozenset(names)


def _slotwise_body(nexts: Sequence, scatters: Sequence) -> tuple:
    """A derived loop's body roots in slotwise form (see :func:`hawk._ir.slotwise`)."""
    both = _ir.slotwise((*nexts, *scatters))
    return both[: len(nexts)], both[len(nexts) :]


def _clean(text: str) -> str:
    """``text``, or ``c`` for an unnamed carry."""
    return text or "c"


def _bar_name(target: Any, j: int) -> str:
    """The carried adjoint accumulator's own name — informational (it names the
    emitted local), so it says which value it accumulates for when it can."""
    if _ir.is_a(target, "leaf", "sink"):
        return f"bar_{_clean(target.field_text('name'))}"
    return f"bar_t{j}"


# --------------------------------------------------------------------------
# forward mode


def _tape_tangent(node: Any, augmented_of) -> Any:
    """The tangent of one ``TapeRead`` — the rule
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
            "loop, whose slot m+j carries the tangent of slot j"
        )
    owner, index = node.operands
    augmented = augmented_of(owner)
    if augmented is None:
        return None
    return _tape_read(augmented, owner.field_int("carry_count") + node.field_int("slot"), index)


def forward_loop(loop: Any, tangent_of, name, augmented_of=None, jump=None, alt=None) -> Any:
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
    ``TapeRead`` (:func:`_tape_tangent`). The
    remaining refusals are structural: a loop committing once per
    iteration is a derived loop's scatter, and reverse-over-reverse still
    has no tape rule. ``jump`` is the kernel's
    :class:`~hawk.diff.jump.JumpPlan` of a ``jump=`` kernel, else ``None``;
    ``alt`` the :class:`~hawk.diff.altpath.AltPath` of an engaged one, which
    adds the alternate path to the loop."""
    del name
    if loop.field_nodes("body_sinks"):
        raise HawkError(
            f"the loop over {_index_name(loop)!r} commits a sink once per iteration, "
            "which only a DERIVED loop does: second-order forward mode through a "
            "`for` is not built"
        )
    _ir.classify_loop(loop)  # refuses a body no direction can differentiate
    m = loop.field_int("carry_count")
    index = _index_of(loop)
    # The primal carries are reused verbatim as the augmented loop's first m
    # slots, so the primal half of the body needs no substitution and stays
    # structurally shared with the loop it mirrors.
    carries = _carries(loop)
    dots = tuple(
        _loop_carry(
            loop, m + j, f"dot_{_clean(carries[j].field_text('name'))}", _ir.type_of(carries[j])
        )
        for j in range(m)
    )
    body = tuple(loop.field_nodes("nexts"))
    extra = _ir.exit_kwargs(loop)
    exits = tuple(extra["exit_values"])
    tangent: dict = {}
    for j in range(m):
        tangent[carries[j]] = dots[j]
    for operand in loop.operands:
        got = tangent_of(operand)
        if got is not None:
            tangent[operand] = got
    stop = set(loop.operands) | {index} | set(carries)
    plan = None if jump is None else jump.body(loop, body + exits)

    def t(c: Any) -> Any:
        return tangent.get(c)

    nested: dict = {}
    derived: dict = {}  # the functions this body's calls were derived into, one each

    def twin(lp):
        """The augmented twin of a loop this body reads the tape of: a loop NESTED in
        the body is augmented here, any other one is the caller's."""
        if lp in nested:
            return nested[lp]
        return None if augmented_of is None else augmented_of(lp)

    for node in _post_order(body + exits, stop):
        if _ir.is_a(node, "loop") and node not in stop:
            # a NESTED loop: its augmented twin carries the inner tangents; this
            # body's tangent of one of its results is the twin's slot m + j
            if plan is not None:
                raise HawkError(
                    f"the nested loop over {_index_name(node)!r} of a `jump=` kernel "
                    "differentiated FORWARD: the switching comparisons of the plan "
                    "are not applied through a nested loop; this is not built"
                )
            nested[node] = forward_loop(node, t, None, twin)
            continue
        if _ir.is_a(node, "loop_value") and node.operands[0] in nested:
            owner, slot = _ir.loop_value_of(node)
            tangent[node] = _ir.loop_value(nested[owner], owner.field_int("carry_count") + slot)
            continue
        if _ir.is_a(node, *_TERMINALS) or node in stop:
            continue
        if plan is not None:
            handled, got = plan.apply_jvp(node, t, lambda n: rules.rule_of(n).jvp(n, t))
            if got is not None:
                tangent[node] = got
            if handled:
                continue
        if _ir.is_a(node, "at"):
            plane, *indices = node.operands
            plane_tangent = tangent.get(plane)
            if plane_tangent is not None:
                tangent[node] = _ir.make_node(
                    "At", _ir.type_of(node), [plane_tangent, *indices], edge=node.field_text("edge")
                )
            continue
        if _ir.is_a(node, "tape_read"):
            owner = node.operands[0]
            got = _tape_tangent(
                node, twin if (owner in nested or augmented_of is not None) else None
            )
            if got is not None:
                tangent[node] = got
            continue
        if _ir.is_a(node, "loop_value"):  # pragma: no cover - structurally dead
            # Loop-invariant projections are cut from the body by the loop's
            # boundary computation; reaching one here means that
            # computation and this traversal disagree.
            raise HawkError(
                f"the loop over {_index_name(loop)!r} reads another loop's result "
                "as a BODY node rather than as a boundary operand — the boundary "
                "computation and the forward traversal disagree"
            )
        if _ir.is_a(node, "call"):
            supplied = [tangent.get(c) for c in node.operands]
            if any(s is not None for s in supplied):
                tangent[node] = call_tangent(node, supplied, derived)
            continue
        if _ir.is_a(node, "dispatch"):
            # the straight-line rule (hawk.diff.transform.jvp): the same
            # selector picks among the branch tangents, a tangent-free
            # branch contributing the structural zero
            selector, *branches = node.operands
            branch_tangents = [tangent.get(b) for b in branches]
            if any(s is not None for s in branch_tangents):
                filled = [
                    s if s is not None else _ir.zero_like(b.ctx, _ir.type_of(b))
                    for s, b in zip(branch_tangents, branches)
                ]
                tangent[node] = dispatch_node(
                    selector, filled, dispatch_strategy(node), _ir.type_of(node)
                )
            continue
        if any(tangent.get(c) is not None for c in node.operands):
            got = rules.rule_of(node).jvp(node, lambda c: tangent.get(c))
            if got is not None:
                tangent[node] = got
    inits = tuple(_inits(loop)) + tuple(
        _seed_or_zero(tangent_of(_inits(loop)[j]), carries[j]) for j in range(m)
    )
    dot_nexts = tuple(_seed_or_zero(tangent.get(body[j]), carries[j]) for j in range(m))
    nexts = tuple(body) + dot_nexts
    if extra["exit_cond"] is not None:
        # The augmented loop leaves where the primal does, over the same
        # condition, so tangents are carried along exactly the iterations
        # the sample ran.
        dot_exits = tuple(
            dot_nexts[j]
            if exits[j] == body[j]
            else _seed_or_zero(tangent.get(exits[j]), carries[j])
            for j in range(m)
        )
        extra["exit_values"] = exits + dot_exits
    if alt is not None and alt.affects(loop):
        from .altpath import Parts

        return alt.joint(
            loop,
            Parts(
                carries,
                dots,
                tuple(inits),
                body,
                dot_nexts,
                exits,
                tuple(extra["exit_values"])[m:],
                t,
            ),
        )
    n_next, n_exit = len(nexts), len(extra["exit_values"])
    exits_now = tuple(extra["exit_values"])
    rooted = _ir.slotwise((*nexts, *exits_now, *([extra["exit_cond"]] if n_exit else [])))
    nexts = rooted[:n_next]
    if n_exit:
        extra.update(exit_values=rooted[n_next : n_next + n_exit], exit_cond=rooted[-1])
    return _ir.loop(index, carries + dots, inits, nexts, **_bounds(loop), **extra)
