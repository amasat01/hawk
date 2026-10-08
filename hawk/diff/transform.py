# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""IR -> IR autodiff: reverse ``vjp`` and forward ``jvp``.

A transform walks the primal DAG once and synthesises a new IR whose sinks are
the derivative's. HAWK keeps no runtime tape: the derived IR goes through the
same ``canonical()`` walk, emitter and cache as any primal, so a partially-read
compound quantity must still bind every wire (a gradient w.r.t. position never
reads ``vel``).

Reverse mode seeds one adjoint plane per primal sink and accumulates
contributions from the rule table, un-broadcasting a rank-0 operand's adjoint
with a ``sum``. Forward mode seeds one tangent plane per ``wrt`` input and
pushes tangents along the same traversal. A ``lookup`` gather ``at(e)`` and a
scatter-add ``AccumWrite`` are each other's reverse-mode transpose.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

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
    LoopIndex,
    LoopValue,
    MapreducePartial,
    Node,
    Primitive,
    SampleIndex,
    Select,
    Sink,
    WideWrite,
    build_loop,
    canonical_nodes,
)
from ..ir import make as _mk
from ..ir.loops import exit_kwargs
from ..ir.nodes import NAMESPACE_SEP, RAW_KIND, Finish
from ..types import TensorType
from .loops import forward_loop, reverse_loop
from .rules import rule_for, zero_like

#: Prefixes of the planes a derivative declares (never a wire namespace).
ADJOINT_PREFIX = "bar"
TANGENT_PREFIX = "dot"


class Derived(tuple):
    """The sink tuple :func:`vjp`/:func:`jvp` return, tagged with facts the
    transform already knows: which kernel it differentiated (:attr:`primal`,
    or ``None`` for an unnamed sink set), which direction (:attr:`kind`,
    ``"vjp"``/``"jvp"``), with respect to which leaves (:attr:`wrt`, resolved
    names in traversal order) and which already-published unit the primal
    lives in (:attr:`primal_unit`, ``None`` meaning "same bundle as this
    derivative").

    A plain ``tuple`` subclass: every consumer of ``vjp``/``jvp`` only
    iterates the result or hands it to ``canonical()``, so tagging is
    invisible to them. :func:`hawk.artifact.bundle.build_bundle` reads the tag
    to tell a derivative kernel from a primal one, and to let a derivative's
    primal live in a different unit named by that unit's digest rather than by
    bundle membership — needed because a VJP's scatter and its primal's
    cross-sample read are different execution axes and cannot share a bundle.
    """

    def __new__(cls, sinks, *, primal: str | None, kind: str, wrt: tuple[str, ...],
                primal_unit: str | None = None):
        self = super().__new__(cls, sinks)
        self.primal = primal
        self.kind = kind
        self.wrt = wrt
        self.primal_unit = primal_unit
        return self


def _termination_mask(primal: Any) -> Node | None:
    """The OR of every guard mask ``primal`` binds, as a fresh
    ``terminated``-kind leaf — ``None`` when ``primal`` carries none.

    Reads ``primal`` itself (not the walk) because a declared-but-unread
    ``terminated`` plane (the guard alone consumes it) never surfaces by
    walking the sinks. A duck-typed sink tuple or a second-order ``Derived``
    primal has no ``.walk``/``.kind`` and answers ``None`` safely: its own
    masking is already baked into its ``Select`` arithmetic, which the chain
    rule differentiates through like any other node."""
    walk = getattr(primal, "walk", None)
    if walk is None:
        return None
    # A kernel-like wrapper may carry `.walk` but no `.kind`; treated the same
    # as "no guard", never an error.
    kind = getattr(primal, "kind", None)
    guard = getattr(kind, "guard", None) if kind is not None else None
    if guard is None:
        return None
    names = [m for m in guard.names if ("terminated", m) in walk.slot_of]
    if not names:
        return None
    mask: Node = Leaf("terminated", "terminated", names[0],
                      walk.slot_types[("terminated", names[0])])
    for extra in names[1:]:
        other = Leaf("terminated", "terminated", extra,
                     walk.slot_types[("terminated", extra)])
        mask = _mk("lor", (mask, other))
    return mask


def _masked_sink(sink: Node, mask: Node) -> Node:
    """``sink``, its value replaced by ``terminated ? 0 : value`` — the
    structural zero :func:`~hawk.diff.rules.zero_like` gives any missing
    contribution, so every sink shape alike stores or accumulates nothing for
    a terminated sample.

    A ``sink`` may be a :class:`~hawk.ir.loop_nodes.Loop` (a reverse-mode
    loop's own scatter commit): only its per-iteration commits are masked,
    not its header/carries/``nexts``. Rebuilt through
    :func:`~hawk.ir.build_loop` rather than :class:`Loop` directly so the
    loop's boundary is recomputed by the same code every other loop producer
    uses."""
    if isinstance(sink, Loop):
        masked_body = tuple(_masked_sink(s, mask) for s in sink.body_sinks)
        return build_loop(sink.index, sink.carries, sink.inits, sink.nexts,
                          start=sink.start, stop=sink.stop, step=sink.step,
                          body_sinks=masked_body, **exit_kwargs(sink))
    value = Select(mask, zero_like(sink.value.ttype), sink.value, sink.value.ttype)
    if isinstance(sink, MapreducePartial):
        return MapreducePartial(sink.name, value, sink.op, sink.ttype)
    if isinstance(sink, AccumWrite):
        return AccumWrite(sink.name, value, sink.index, sink.ttype)
    if isinstance(sink, WideWrite):
        return WideWrite(sink.name, value, sink.index, sink.ttype)
    return Assign(sink.name, value, sink.ttype)


def _mask_termination(sinks: Sequence[Node], primal: Any) -> tuple[Node, ...]:
    """``sinks`` gated by :func:`_termination_mask` when ``primal`` declares
    one. The single call both :func:`vjp` and :func:`jvp` make before
    tagging their result, masking every sink shape uniformly."""
    mask = _termination_mask(primal)
    if mask is None:
        return tuple(sinks)
    return tuple(_masked_sink(s, mask) for s in sinks)


def _refuse_prior_read(seq: Sequence[Node]) -> None:
    """Refuses a primal containing a ``prior_read`` leaf (a Mutable read
    before its first store), naming the plane.

    The value it reads is a recurrence across launches; its derivative needs
    the value the primal launch overwrites, and HAWK keeps no tape. Left
    unguarded, a jvp would bind the primal's own prior plane read-only
    (correct only if the derived kernel launches before the primal) and a vjp
    would bind the same adjoint plane twice — plausible wrong answers, not
    crashes, so this must raise instead."""
    for node in seq:
        if isinstance(node, Leaf) and node.kind == "prior_read":
            raise HawkError(
                f"{node.name!r} is read before it is assigned: a plane's "
                "launch-start value is a recurrence across launches; its "
                "derivative needs the value the primal launch overwrites, "
                "and hawk keeps no tape — differentiate the per-launch term "
                "instead"
            )


def _single_step(primal: Any) -> Any:
    """The single step a default automatic kernel carries as ``.step``
    (same name, same planes), else ``primal``. An explicit ``steps="auto"``
    or ``steps=K`` kernel carries none and is named ``<name>_x...``, so a
    derivative tagged with the step's name would point at a kernel its
    bundle does not hold."""
    step = getattr(primal, "step", None)
    return primal if step is None else step


def _primal_name(primal: Any) -> str | None:
    """The primal's own ``.name``, or ``None`` for a duck-typed sink set
    that carries none."""
    name = getattr(primal, "name", None)
    return name if isinstance(name, str) else None

#: Node families a transform never pushes through: a bound plane, a literal,
#: a sink (seeded, not traversed), and the lane index (an address, not a
#: differentiable quantity).
_TERMINALS = (Sink, Leaf, Const, SampleIndex, LoopIndex, LoopCarry)

#: dtypes a derivative flows through; everything else is a structural constant.
_DIFFERENTIABLE = ("f64", "f32")
#: How a read plane's RANK picks its role when a derivative mints one.
_READ_ROLE = {0: "per_sample", 1: "vec_in", 2: "mat_in"}


def vjp(primal: Any, *, wrt: Sequence[str] | None = None,
        primal_unit: str | None = None) -> Derived:
    """The reverse-mode derivative IR of ``primal``: one adjoint plane per
    primal sink in, one gradient plane per ``wrt`` input out. Returned as
    :class:`Derived`, tagging ``primal``'s name and the resolved ``wrt``.

    ``primal_unit=`` names the digest of the unit ``primal`` was already
    published into (:attr:`hawk.artifact.bundle.Bundle.digest`), when the
    caller knows it. Left ``None``, it means either the primal will publish
    in the same bundle as this derivative, or the caller supplies the unit
    later through ``derivative={name: (primal_name, unit_digest)}``.

    A default automatic kernel is differentiated through its single step:
    the derivative of one step, as for the plain kernel."""
    primal = _single_step(primal)
    sinks = _sinks_of(primal)
    seq, position = canonical_nodes(sinks)
    _refuse_prior_read(seq)
    inputs = _wrt(seq, wrt)
    name = _Namer(ADJOINT_PREFIX, _taken(seq))
    key = _keyer(position)
    adjoint: dict[Any, Node] = {}
    for sink in sinks:
        if isinstance(sink, Loop):
            raise HawkError(
                "this IR's root set contains a lowered loop that COMMITS once per "
                "iteration, which only a DERIVED kernel does: second-order reverse "
                "mode through a `for` is not built. Publish the first "
                "derivative and differentiate that kernel")
        _accumulate(adjoint, key, sink.value, _seed(sink, name))
    scatters: list[Node] = []
    scattered: set[str] = set()
    _reverse_pass(seq, key, adjoint, inputs, name, scatters, scattered)
    gradients = [
        Assign(name(leaf.name), adjoint.get(key(leaf)) or zero_like(leaf.ttype),
               leaf.ttype)
        for leaf in inputs
        if leaf.role != "lookup" and leaf.name not in scattered
    ]
    out = _mask_termination(gradients + scatters, primal)
    return Derived(out, primal=_primal_name(primal), kind="vjp",
                   wrt=tuple(leaf.name for leaf in inputs), primal_unit=primal_unit)


def jvp(primal: Any, *, wrt: Sequence[str] | None = None,
        primal_unit: str | None = None) -> Derived:
    """The forward-mode derivative IR of ``primal``: one tangent plane
    per ``wrt`` input in, one directional-derivative sink per primal sink out.
    Returned as :class:`Derived`, tagging ``primal``'s name and the resolved
    ``wrt`` (see that class). ``primal_unit=`` is :func:`vjp`'s (see there).
    A default automatic kernel is differentiated through its single step, as
    in :func:`vjp`."""
    primal = _single_step(primal)
    sinks = _sinks_of(primal)
    for sink in sinks:
        if isinstance(sink, Loop):
            raise HawkError(
                "this IR's root set contains a lowered loop that COMMITS once per "
                "iteration, which only a DERIVED kernel does: second-order forward "
                "mode through a `for` is not built")
    seq, position = canonical_nodes(sinks)
    _refuse_prior_read(seq)
    inputs = _wrt(seq, wrt)
    name = _Namer(TANGENT_PREFIX, _taken(seq))
    key = _keyer(position)
    tangent: dict[Any, Node] = {
        key(leaf): Leaf(leaf.kind, leaf.role, name(leaf.name), leaf.ttype)
        for leaf in inputs
    }
    #: The AUGMENTED loop a lowered `for` becomes in forward mode — one loop
    #: carrying the primal slots and their tangents side by side; slot `m + j`
    #: is the tangent of the primal's slot `j` (hawk/diff/loops.py).
    loop_tangent: dict[Any, Node] = {}
    for node in seq:
        if isinstance(node, _TERMINALS):
            continue
        if isinstance(node, Loop):
            # `loop_tangent` is also consulted by the LoopValue arm below: a
            # reverse loop's body reads the forward loop's tape, and the
            # tangent of that read is the same row of this augmented twin
            # (hawk/diff/loops._tape_tangent). The walk visits the taped
            # forward loop before the reverse loop that reads it, so the
            # twin is always already in this table when needed.
            got = forward_loop(node, lambda c: tangent.get(key(c)), name,
                               lambda lp: loop_tangent.get(key(lp)))
            if got is not None:
                loop_tangent[key(node)] = got
            continue
        if isinstance(node, LoopValue):
            augmented = loop_tangent.get(key(node.loop))
            if augmented is not None:
                tangent[key(node)] = LoopValue(
                    augmented, node.loop.carry_count + node.slot)
            continue
        if isinstance(node, Primitive):
            supplied = [tangent.get(key(c)) for c in node.inputs]
            if any(t is not None for t in supplied):
                got = node.rules.jvp_tangent(node, supplied)
                if got is not None:
                    tangent[key(node)] = got
            continue
        if isinstance(node, At):
            plane = tangent.get(key(node.plane))
            if plane is not None:
                tangent[key(node)] = At(plane, node.index, node.ttype)
            continue
        if isinstance(node, Dispatch):
            # `t_out = dispatch(kind, [t_e0, ..., t_eK])`, same policy and
            # selector (`kind` carries no tangent of its own). A branch with
            # no tangent contributes the structural zero — select's own
            # missing-tangent rule — and branch tangents are used only
            # inside this node, so the emitter inlines each in its own
            # case/select arm.
            branch_tangents = [tangent.get(key(b)) for b in node.branches]
            if any(t is not None for t in branch_tangents):
                filled = [t if t is not None else zero_like(b.ttype)
                         for t, b in zip(branch_tangents, node.branches)]
                tangent[key(node)] = Dispatch(node.selector, filled, node.policy,
                                              node.ttype)
            continue
        if any(tangent.get(key(c)) is not None for c in node.operands):
            got = _rule(node).jvp(node, lambda c: tangent.get(key(c)))
            if got is not None:
                tangent[key(node)] = got
    out = _mask_termination(
        tuple(_tangent_sink(s, tangent.get(key(s.value)), name) for s in sinks),
        primal)
    return Derived(
        out, primal=_primal_name(primal), kind="jvp",
        wrt=tuple(leaf.name for leaf in inputs), primal_unit=primal_unit)


# --------------------------------------------------------------------------
# the reverse-mode dispatch rule


def _reverse_pass(seq: Sequence[Node], key, adjoint: dict[Any, Node],
                  inputs: Sequence[Leaf], name: _Namer, scatters: list[Node],
                  scattered: set[str], *, local_pass: bool = False) -> None:
    """One reverse-accumulation pass over ``seq``, given an already-seeded
    ``adjoint`` — the core step :func:`vjp` runs once over the whole walk,
    seeded at every primal sink.

    :func:`_dispatch_vjp` reuses this in isolation over one
    :class:`~hawk.ir.nodes.Dispatch` branch's exclusive sub-DAG, with
    ``local_pass=True``: a fresh ``adjoint`` seeded at only that branch's
    root. A node reachable from branch j alone keeps exactly that seed's
    contribution; one also reachable elsewhere accumulates a second
    contribution the ordinary way, via the same ``_accumulate`` fan-in any
    shared subexpression uses.

    ``local_pass`` changes one thing: an :class:`~hawk.ir.nodes.At`'s reverse
    (gather -> scatter) is a real sink, not a value, so it cannot be masked
    in place. A local pass records its contribution into ``adjoint`` instead
    and leaves the scatter itself to :func:`_dispatch_vjp`'s merge step,
    which has all K branches' contributions in hand and commits one masked
    value once — committing it here unconditionally would scatter branch j's
    contribution for every lane regardless of which branch that lane
    actually took."""
    loop_seeds: dict[Any, dict] = {}
    for node in reversed(seq):
        g = adjoint.get(key(node))
        if isinstance(node, LoopValue):
            if g is not None:
                loop_seeds.setdefault(key(node.loop), {})[node.slot] = g
            continue
        if isinstance(node, Loop):
            derived = reverse_loop(node, loop_seeds.get(key(node), {}), inputs,
                                   name)
            for child, contribution in derived.contributions:
                _accumulate(adjoint, key, child,
                            _reduce_to(contribution, child.ttype))
            scatters.extend(derived.roots)
            scattered |= set(derived.scattered)
            continue
        if g is None or isinstance(node, _TERMINALS):
            continue
        if isinstance(node, At):
            if local_pass:
                # `g` already holds this node's fully-accumulated local
                # contribution — from consumers reached within this pass, or
                # as the branch's own seed. Folding it in again here would
                # double every scatter this local pass reaches.
                continue
            if node.plane in inputs:
                scattered.add(node.plane.name)
                scatters.append(AccumWrite(name(node.plane.name), g, node.index,
                                           g.ttype))
            continue
        if isinstance(node, Primitive):
            # Operand 0 (the forward subgraph) is deliberately skipped: the
            # supplied rule already is this primitive's derivative.
            for child, contribution in node.rules.vjp_contributions(node, g):
                _accumulate(adjoint, key, child,
                            _reduce_to(contribution, child.ttype))
            continue
        if isinstance(node, Dispatch):
            _dispatch_vjp(node, g, seq, key, inputs, name, adjoint, scatters,
                         scattered)
            continue
        for child, contribution in _rule(node).vjp(node, g):
            _accumulate(adjoint, key, child, _reduce_to(contribution, child.ttype))


def _dispatch_vjp(node: Dispatch, g: Node, seq: Sequence[Node], key,
                  inputs: Sequence[Leaf], name: _Namer, adjoint: dict[Any, Node],
                  scatters: list[Node], scattered: set[str]) -> None:
    """K independent reverse passes, one per branch, each seeded only at that
    branch's own root with the dispatch's incoming adjoint ``g`` — never a
    per-node masked seed pushed through the ordinary chain rule, which would
    still run every branch's arithmetic under a zero factor.

    Every node any branch's pass reached gets one combined adjoint term,
    ``dispatch(kind, [c_0 or 0, ..., c_K or 0])`` (same policy as the primal,
    so the emitter elides branches that stayed structurally zero),
    accumulated into the enclosing pass like any other contribution, so a
    node reached through multiple paths sums correctly. A reached
    :class:`~hawk.ir.nodes.At` is the one exception: its combined value
    becomes one scattered ``AccumWrite`` here, since a scatter is a sink, not
    something a downstream node could read further. ``kind`` itself is never
    seeded and receives no adjoint."""
    reached: dict[int, dict[int, Node]] = {}
    for j, branch in enumerate(node.branches):
        seed_slot = key(branch)
        local: dict[Any, Node] = {seed_slot: g}
        _reverse_pass(seq, key, local, inputs, name, scatters, scattered,
                      local_pass=True)
        for slot, contribution in local.items():
            target = seq[slot]
            if not isinstance(target, (Leaf, At)):
                # Not a leaf of the sub-transform: every other node this
                # isolated pass reached was already differentiated through,
                # down to its own leaves, within this one pass. Reporting it
                # upward would make the enclosing pass push through it again
                # via the ordinary chain rule — a double count. Only a
                # genuine leaf or an At (whose reverse is a sink the merge
                # below commits) has nowhere further to propagate on its own.
                continue
            reached.setdefault(slot, {})[j] = contribution
    for slot, per_branch in reached.items():
        target = seq[slot]
        branch_terms = [per_branch.get(j) if per_branch.get(j) is not None
                        else zero_like(target.ttype) for j in range(node.K)]
        combined = Dispatch(node.selector, branch_terms, node.policy, target.ttype)
        if isinstance(target, At):
            if target.plane in inputs:
                scattered.add(target.plane.name)
                scatters.append(AccumWrite(name(target.plane.name), combined,
                                           target.index, combined.ttype))
            continue
        _accumulate(adjoint, key, target, combined)


# --------------------------------------------------------------------------
# the traversal and its keys


def _sinks_of(primal: Any) -> tuple[Sink, ...]:
    sinks = tuple(getattr(primal, "sinks", primal))
    if not sinks or not all(isinstance(s, (Sink, Loop)) for s in sinks):
        raise HawkError(
            f"{primal!r} is not a kernel or a sink set — a transform differentiates "
            "an IR whose roots are its sinks"
        )
    # Termination is DISCRETE: a primal's finish statement has no derivative,
    # so the derived kernel binds no counter, writes no mask and carries no
    # sidecar ``finish``; it keeps the primal's mask READ (``_mask_termination``).
    kept = tuple(s for s in sinks if not isinstance(s, Finish))
    if not kept:
        raise HawkError(
            f"{primal!r} only finishes its sample — it commits no plane to "
            "differentiate")
    return kept


def _keyer(position: Mapping[int, int]):
    """The walk's own structural identity, used as the derivative's memo key:
    two leaf OBJECTS sharing ``(role, name)`` never split a gradient, and
    a duplicated subexpression accumulates ONE adjoint."""
    def key(node: Node) -> int:
        return position[id(node)]
    return key


def _rule(node: Node):
    rule = rule_for(node.kind, len(node.ttype.shape))
    if rule.vjp is None and rule.jvp is None:
        raise HawkError(
            f"node kind {node.kind!r} carries no derivative rule ({rule.mode}); "
            "it is differentiated by the transform itself, not by the table"
        )
    return rule


# --------------------------------------------------------------------------
# inputs, seeds and derived names


def _wrt(seq: Iterable[Node], wrt: Sequence[str] | None) -> tuple[Leaf, ...]:
    """The leaves a derivative is taken with respect to, in traversal order."""
    for node in seq:
        if getattr(node, "kind", None) == RAW_KIND:
            raise HawkError(
                "a spliced @raw_device block cannot be differentiated: its "
                "text is opaque to the IR, so no rule can exist for it"
            )
    leaves = [n for n in seq
              if isinstance(n, Leaf) and n.ttype.dtype in _DIFFERENTIABLE]
    if wrt is None:
        return tuple(leaves)
    wanted = tuple(wrt)
    unknown = [w for w in wanted if not any(leaf.name == w for leaf in leaves)]
    if unknown:
        raise HawkError(
            f"wrt names {unknown} are not differentiable inputs of this kernel; its "
            f"inputs are {[leaf.name for leaf in leaves]}"
        )
    return tuple(leaf for leaf in leaves if leaf.name in wanted)


def _taken(seq: Iterable[Node]) -> set[str]:
    return {n.binding.name for n in seq if n.binding is not None}


class _Namer:
    """Derived plane names: ``<prefix>_<primal name>`` with the compound-wire
    namespace flattened, since only a Quantity declaration may mint a name in it. A
    flattening that would collide is refused, never shared."""

    def __init__(self, prefix: str, taken: set[str]) -> None:
        self.prefix = prefix
        self.taken = taken
        self.source: dict[str, str] = {}

    def __call__(self, name: str) -> str:
        """The derived plane name for one primal plane."""
        derived = f"{self.prefix}_{name.replace(NAMESPACE_SEP, '_')}"
        held = self.source.setdefault(derived, name)
        if held != name or derived in self.taken:
            raise HawkError(
                f"the derived name {derived!r} for {name!r} collides with "
                f"{held if held != name else 'a primal plane'!r} — flattening the "
                "compound-wire namespace made two planes one"
            )
        return derived


def _seed(sink: Sink, name: _Namer) -> Node:
    """The incoming adjoint plane of one primal sink (the four sink kinds)."""
    derived, t = name(sink.name), sink.ttype
    if isinstance(sink, MapreducePartial):
        if sink.op != "sum" or t.shape != ():
            raise HawkError(
                f"no reverse rule for a mapreduce_partial with op={sink.op!r} "
                f"type={t!r} (sink {sink.name!r}). The ONLY seeding rule HAWK "
                "defines for a mapreduce sink is the rank-0 'sum' one — the reverse "
                "of a sum-reduction is a broadcast of its adjoint — and a 'max' fold "
                "has no such rule at all: its subgradient is ambiguous at a tie, so "
                "differentiating it would return one of several answers with nothing "
                "recording which. A two-pass reduction whose first pass is a max "
                "(a softmax's row peak) declares that pass FORWARD-ONLY; the pass "
                "that carries the gradient is the sum"
            )
        return Leaf("uniform", "uniform", derived, t)
    if isinstance(sink, (AccumWrite, WideWrite)) and sink.index is not None:
        # the transpose of a scatter-add is a gather of the adjoint plane
        return At(Leaf("table_read", "lookup", derived, t), sink.index, t)
    return Leaf("vocab_read", _READ_ROLE[len(t.shape)], derived, t)


def _tangent_sink(sink: Sink, tangent: Node | None, name: _Namer) -> Node:
    """The forward-mode twin of one primal sink, same kind and same lane."""
    value = tangent if tangent is not None else zero_like(sink.ttype)
    derived = name(sink.name)
    if isinstance(sink, MapreducePartial):
        return MapreducePartial(derived, value, sink.op, sink.ttype)
    if isinstance(sink, AccumWrite):
        return AccumWrite(derived, value, sink.index, sink.ttype)
    if isinstance(sink, WideWrite):
        return WideWrite(derived, value, sink.index, sink.ttype)
    return Assign(derived, value, sink.ttype)


# --------------------------------------------------------------------------
# accumulation and un-broadcasting


def _accumulate(store: dict[Any, Node], key: Any, node: Node,
                contribution: Node) -> None:
    slot = key(node)
    held = store.get(slot)
    store[slot] = contribution if held is None else _mk("add", (held, contribution))


def _reduce_to(contribution: Node, target: TensorType) -> Node:
    """Un-broadcast: a rank-0 operand's adjoint is the SUM of the contributions
    its broadcast produced (the transpose of a broadcast)."""
    if contribution.ttype.shape == target.shape:
        return contribution
    if target.shape == ():
        return _mk("sum", (contribution,))
    raise HawkError(
        f"an adjoint typed {contribution.ttype!r} cannot flow into an operand typed "
        f"{target!r} — only a rank-0 operand un-broadcasts"
    )
