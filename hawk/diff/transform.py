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

import functools
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
    Node,
    Primitive,
    SampleIndex,
    Select,
    Sink,
    WideWrite,
    build_loop,
    canonical,
    canonical_nodes,
)
from ..ir import make as _mk
from ..ir.loops import exit_kwargs
from ..ir.nodes import (ADJOINT_PREFIX, COMPANION_SUFFIX, DISPATCH_POLICIES, NAMESPACE_SEP, RAW_KIND, Finish,
                        Op, ReduceSamples, SampleSum, param_cell_names)
from ..ir.slotwise import slotwise
from ..ir.walk import dispatch_groups
from ..types import INT_PLANE_DTYPES, TensorType
from .altpath import AltPath
from .jump import JumpPlan, drop_width, jump_of
from .loops import _taken_row, forward_loop, guarded_select, reverse_loop
from .rules import rule_for, zero_like

#: Prefixes of the planes a derivative declares (never a wire namespace).
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
                primal_unit: str | None = None, jump: dict | None = None):
        self = super().__new__(cls, sinks)
        #: The sidecar ``jump`` record of a derivative smoothed through its
        #: switches (``@kernel(jump=...)``), else ``None``.
        self.jump = jump
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
        masked_body = slotwise(tuple(_masked_sink(s, mask) for s in sink.body_sinks))
        return build_loop(sink.index, sink.carries, sink.inits, sink.nexts,
                          start=sink.start, stop=sink.stop, step=sink.step,
                          body_sinks=masked_body, **exit_kwargs(sink))
    value = Select(mask, zero_like(sink.value.ttype), sink.value, sink.value.ttype)
    if isinstance(sink, AccumWrite):
        return AccumWrite(sink.name, value, sink.index, sink.ttype, sink.policy)
    if isinstance(sink, WideWrite):
        return WideWrite(sink.name, value, sink.index, sink.ttype)
    if isinstance(sink, SampleSum):
        return sink.rebuilt(value)
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
                "and hawk keeps no tape inside a launch. Across launches, "
                "step_jvp (forward) or step_vjp with tape / eagle record=True "
                "(reverse) carry it; to differentiate this kernel as it stands, "
                "start the plane from zeros (`curve = hawk.math.vec(0.0, ...)`) "
                "or write it element by element over every index, and "
                "differentiate the per-launch term"
            )


def _single_step(primal: Any) -> Any:
    """The single step a default automatic kernel carries as ``.step``
    (same name, same planes), else ``primal``. An explicit ``steps="auto"``
    or ``steps=K`` kernel is named ``<name>_x...``, so a derivative tagged
    with its step's name would point at a kernel its bundle does not hold."""
    from ..trace.steps import default_step

    return default_step(primal)


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


def _rederivable(derive):
    """``derive`` with its result remembering how to be made again: the primal
    it differentiated, a ``rebuild(primal)`` that differentiates another one the
    same way (with the resolved ``wrt``), and the primal's launch shape and
    group, so a kernel built from it launches over the same lanes and the
    broadcast door can differentiate a broadcast variant of the primal instead
    (:mod:`hawk._broadcast`)."""
    @functools.wraps(derive)
    def wrapped(primal, **kw):
        out = derive(primal, **kw)
        out.source = primal
        out.rebuild = lambda other: wrapped(other, **{**kw, "wrt": out.wrt})
        out.lane_space = getattr(primal, "lane_space", ((None, None),))
        out.group = getattr(primal, "group", 1)
        return out

    return wrapped


@_rederivable
def vjp(primal: Any, *, wrt: Sequence[str] | None = None,
        primal_unit: str | None = None,
        policy: Mapping[str, str] | None = None) -> Derived:
    """The reverse-mode derivative IR of ``primal``: one adjoint plane per
    primal sink in, one gradient plane per ``wrt`` input out. Returned as
    :class:`Derived`, tagging ``primal``'s name and the resolved ``wrt``.

    ``primal_unit=`` names the digest of the unit ``primal`` was already
    published into (:attr:`hawk.artifact.bundle.Bundle.digest`), when the
    caller knows it. Left ``None``, it means either the primal will publish
    in the same bundle as this derivative, or the caller supplies the unit
    later through ``derivative={name: (primal_name, unit_digest)}``.

    A default automatic kernel is differentiated through its single step:
    the derivative of one step, as for the plain kernel.

    ``policy={group: policy}`` sets the lowering of the derived dispatches
    of the primal's dispatch GROUP ``group`` (the sidecar's positional
    ``dispatch<i>`` name), which otherwise inherit the primal's policy: the
    reverse takes its own structure. Any of
    :data:`~hawk.ir.nodes.DISPATCH_POLICIES` is accepted; an unknown group
    or policy refuses by name."""
    primal = _single_step(primal)
    sinks = _sinks_of(primal)
    cells = param_cell_names(canonical_nodes(sinks)[0])
    if cells:
        return _vjp_param_cells(primal, cells, wrt, primal_unit, policy)
    return _vjp(primal, sinks, wrt, primal_unit, policy, summed=())


def _vjp(primal: Any, sinks: tuple, wrt: Sequence[str] | None,
         primal_unit: str | None, policy: Mapping[str, str] | None,
         summed: tuple) -> Derived:
    """:func:`vjp` proper; the gradient of each input named in ``summed``
    is summed over samples (:class:`~hawk.ir.nodes.SampleSum`)."""
    overrides = _policy_overrides(primal, policy)
    jump = jump_of(primal)
    drop_width((), wrt, jump)
    seq, position = canonical_nodes(sinks)
    _refuse_prior_read(seq)
    inputs = drop_width(_wrt(seq, wrt), None, jump)
    name = _Namer(ADJOINT_PREFIX, _taken(seq),
                  _named(sinks, inputs))
    name.counts = _compact_counts(primal)
    key = _keyer(position)
    plan = _plan(seq, key, inputs, jump, primal)
    alt = AltPath.of(plan, seq, reverse=True)
    if alt is not None:
        for node in seq:
            if not isinstance(node, _TERMINALS):
                alt.visit(node, None)
    adjoint: dict[Any, Node] = {}
    for sink in sinks:
        if isinstance(sink, Loop):
            raise HawkError(
                "this IR's root set contains a lowered loop that COMMITS once per "
                "iteration, which only a DERIVED kernel does: second-order reverse "
                "mode through a `for` is not built. Publish the first "
                "derivative and differentiate that kernel")
        if sink.ttype.dtype in INT_PLANE_DTYPES:
            continue                # an integer plane takes no adjoint
        _accumulate(adjoint, key, sink.value, _seed(sink, name))
    if alt is not None:
        alt.g = alt.seed_gain(sinks, lambda s: _seed(s, name))
        for site, weight in alt.injections():
            lhs, rhs = site.operands
            _accumulate(adjoint, key, lhs, weight)
            _accumulate(adjoint, key, rhs, _mk("neg", (weight,)))
    scatters: list[Node] = []
    scattered: set[str] = set()
    accumulated: set[str] = set()
    _reverse_pass(seq, key, adjoint, inputs, name, scatters, scattered,
                  overrides=overrides, jump=None if alt is not None else plan,
                  accumulated=accumulated, exclusive=_loop_exclusive(seq, inputs),
                  alt=alt)
    for leaf in inputs:
        if name(leaf.name) not in accumulated:
            continue
        if adjoint.get(key(leaf)) is not None:   # pragma: no cover - by the pre-scan
            raise HawkError(
                f"the gradient of {leaf.name!r} is accumulated at runtime indices "
                "inside a loop AND reaches a contribution outside that loop")
    gradients = [
        _gradient_sink(leaf, name(leaf.name), summed,
                       adjoint.get(key(leaf)) or zero_like(leaf.ttype))
        for leaf in inputs
        if leaf.role != "lookup" and leaf.name not in scattered
        and name(leaf.name) not in accumulated
    ]
    held = getattr(primal, "broadcast", ())
    scatters = _compensated(scatters, {name(leaf.name) for leaf in inputs if leaf.name in held})
    out = slotwise(_mask_termination(gradients + list(scatters), primal))
    names = tuple(leaf.name for leaf in inputs)
    if plan is not None:
        out += plan.spread_cells()
    return Derived(out, primal=_primal_name(primal), kind="vjp", wrt=names,
                   primal_unit=primal_unit,
                   jump=None if plan is None else plan.record(names))



def _compact_counts(primal: Any) -> dict:
    """``{plane: the column count of its compact plane}`` (an ``i32`` node) for each
    plane a broadcast variant reads at the axes it varies along: the product of the
    kept axes' extents, a kept runtime first axis through the lane count."""
    held = getattr(primal, "broadcast", None) or {}
    axes = getattr(primal, "lane_space", None)
    out: dict = {}
    for plane, kept in held.items():
        if axes is None or kept is None:
            continue
        t = TensorType((), "i32")
        fixed = 1
        for (_label, extent), k in zip(axes, kept):
            fixed *= extent if k and extent is not None else 1
        if kept[0] and axes[0][1] is None:
            rest = 1
            for (_label, extent), k in zip(axes, kept):
                rest *= 1 if k else extent
            lanes = Leaf("nsamples", "nsamples", "n_samples", t)
            out[plane] = lanes if rest == 1 else _mk("floordiv", (lanes, Const(rest, t)))
        else:
            out[plane] = Const(fixed, t)
    return out


def _scatter_payload(name: _Namer, plane: str, g: Node, index: Node) -> list:
    """The scattered accumulate of ``g`` into the adjoint of ``plane`` at ``index``:
    a rank-0 value as it is; a vector or matrix one scalar accumulate per entry
    ``k`` (row-major) at ``index + k * count`` of ONE component-major plane, so the
    gradient has the layout of the compact plane it is the gradient of. Only a
    broadcast variant knows its column count (:func:`_compact_counts`); any other
    rank >= 1 scatter is left whole, which the emitter refuses by name."""
    shape = g.ttype.shape
    if not shape:
        return [AccumWrite(name(plane), g, index, g.ttype)]
    count = name.counts.get(plane)
    if count is None:       # not a broadcast variant: the emitter refuses a rank >= 1 scatter by name
        return [AccumWrite(name(plane), g, index, g.ttype)]
    t, scalar = index.ttype, TensorType((), g.ttype.dtype)
    out = []
    for k in range(g.ttype.shape[0] * (g.ttype.shape[1] if len(shape) == 2 else 1)):
        at = (k // shape[1], k % shape[1]) if len(shape) == 2 else k
        key = index if k == 0 else _mk("add", (index, _mk("mul", (count, Const(k, t)))))
        out.append(AccumWrite(name(plane), _mk("component", (g,), literal=at), key, scalar))
    return out

def _compensated(scatters: Sequence[Node], planes: set) -> list:
    """``scatters`` with each accumulate into a plane of ``planes`` compensated,
    its companion bound by a zero commit at the same key. A gradient plane summed
    over the lane axes a broadcast input does not vary along."""
    out: list = []
    for s in scatters:
        if isinstance(s, AccumWrite) and s.name in planes:
            s = AccumWrite(s.name, s.value, s.index, s.ttype, "compensated")
            out.append(s)
            out.append(AccumWrite(s.name + COMPANION_SUFFIX, Const(0.0, TensorType((), s.ttype.dtype)),
                                  s.index, s.ttype))
        else:
            out.append(s)
    return out


def _loop_exclusive(seq: Sequence[Node], inputs: Sequence[Leaf]) -> frozenset:
    """The ids of the ``wrt`` leaves read ONLY inside lowered-loop bodies: no
    top-level node (a sink included) reads one, and none is a carried slot's
    initial value. Such a leaf's gradient can be committed by the loops alone
    (:func:`hawk.diff.loops.reverse_loop`, the per-iteration accumulate)."""
    dense: set = set()
    for n in seq:
        held = n.inits if isinstance(n, Loop) else n.operands
        dense.update(id(c) for c in held if isinstance(c, Leaf))
    return frozenset(id(leaf) for leaf in inputs if id(leaf) not in dense)


def _plan(seq: Sequence[Node], key, inputs: Sequence[Leaf], jump: Any,
          primal: Any) -> JumpPlan | None:
    """The switching comparisons of a ``jump=`` primal, ``None`` when it
    declares none or none of its comparisons switches."""
    if jump is None:
        return None
    plan = JumpPlan(seq, key, inputs, jump, _primal_name(primal))
    return plan if plan.sites else None


def _gradient_sink(leaf: Node, name: str, summed: tuple, value: Node) -> Node:
    """The sink of one input's gradient: the sum over samples of the per-sample
    adjoint for a promoted parameter, else the adjoint plane itself."""
    if leaf.name in summed:
        if leaf.ttype.shape:
            raise HawkError(f"the gradient of {leaf.name!r} cannot be summed over samples: "
                            f"only a scalar leaf is, and it has shape {leaf.ttype.shape}")
        return ReduceSamples("sum", value).stored(name, leaf.ttype)[0]
    return Assign(name, value, leaf.ttype)


def _vjp_param_cells(primal: Any, cells: tuple, wrt: Sequence[str] | None,
                     primal_unit: str | None,
                     policy: Mapping[str, str] | None) -> Derived:
    """The reverse of a kernel reading parameter cells (see
    :func:`hawk.diff.param_planes`): differentiate the form that reads each
    cell as a per-sample plane, sum each cell's gradient over every 32
    samples (``bar_<name>`` rows, :class:`~hawk.ir.nodes.SampleSum`), then
    read the cells again in the derived kernel."""
    from .params import demote, promote

    per_sample = _Demoted(primal, demote(_sinks_of(primal)))
    d = _vjp(per_sample, per_sample.sinks, wrt, primal_unit, policy, summed=cells)
    return Derived(promote(d, cells), primal=_primal_name(primal), kind="vjp",
                   wrt=d.wrt, primal_unit=primal_unit, jump=d.jump)


class _Demoted:
    """A primal's name and guard bindings over replaced sinks."""

    def __init__(self, primal: Any, sinks: tuple) -> None:
        self.name, self.sinks, self.walk = primal.name, sinks, primal.walk
        self.planes = getattr(primal, "planes", {})
        self.kind = getattr(primal, "kind", None)
        self.jump = getattr(primal, "jump", None)


@_rederivable
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
    jump = jump_of(primal)
    drop_width((), wrt, jump)
    seq, position = canonical_nodes(sinks)
    _refuse_prior_read(seq)
    inputs = drop_width(_wrt(seq, wrt), None, jump)
    name = _Namer(TANGENT_PREFIX, _taken(seq),
                  _named(sinks, inputs))
    key = _keyer(position)
    plan = _plan(seq, key, inputs, jump, primal)
    alt = AltPath.of(plan, seq)
    tangent: dict[Any, Node] = {
        key(leaf): Leaf(leaf.kind, leaf.role, name(leaf.name), leaf.ttype,
                        sorted=leaf.sorted)
        for leaf in inputs
    }
    #: The AUGMENTED loop a lowered `for` becomes in forward mode — one loop
    #: carrying the primal slots and their tangents side by side; slot `m + j`
    #: is the tangent of the primal's slot `j` (hawk/diff/loops.py).
    loop_tangent: dict[Any, Node] = {}
    for node in seq:
        if isinstance(node, _TERMINALS):
            continue
        if alt is not None:
            alt.visit(node, lambda c: tangent.get(key(c)))
        elif plan is not None and _jump_jvp(node, plan, tangent, key):
            continue
        if isinstance(node, Loop):
            # `loop_tangent` is also consulted by the LoopValue arm below: a
            # reverse loop's body reads the forward loop's tape, and the
            # tangent of that read is the same row of this augmented twin
            # (hawk/diff/loops._tape_tangent). The walk visits the taped
            # forward loop before the reverse loop that reads it, so the
            # twin is always already in this table when needed.
            got = forward_loop(node, lambda c: tangent.get(key(c)), name,
                               lambda lp: loop_tangent.get(key(lp)),
                               jump=None if alt is not None else plan, alt=alt)
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
    if alt is not None:
        made = alt.sinks(sinks, lambda v: tangent.get(key(v)),
                         lambda s, v, index=None: _tangent_sink(s, v, name, index))
    else:
        made = tuple(_tangent_sink(s, tangent.get(key(s.value)), name) for s in sinks
                     if s.ttype.dtype not in INT_PLANE_DTYPES)
    out = slotwise(_mask_termination(made, primal))
    names = tuple(leaf.name for leaf in inputs)
    if plan is not None:
        out += plan.spread_cells()
    return Derived(
        out, primal=_primal_name(primal), kind="jvp", wrt=names,
        primal_unit=primal_unit,
        jump=None if plan is None else plan.record(names))


def _jump_jvp(node: Node, plan: JumpPlan, tangent: dict, key) -> bool:
    """Forward mode's boolean-op rules of a ``jump=`` primal: whether
    ``node`` was one (its tangent, if any, recorded)."""
    def t(c: Node) -> Node | None:
        return tangent.get(key(c))

    handled, got = plan.apply_jvp(node, t, lambda n: _rule(n).jvp(n, t))
    if got is not None:
        tangent[key(node)] = got
    return handled


# --------------------------------------------------------------------------
# the reverse-mode dispatch rule


def _reverse_pass(seq: Sequence[Node], key, adjoint: dict[Any, Node],
                  inputs: Sequence[Leaf], name: _Namer, scatters: list[Node],
                  scattered: set[str], *, local_pass: bool = False,
                  overrides: Mapping[int, str] | None = None,
                  cut: frozenset = frozenset(),
                  jump: JumpPlan | None = None,
                  accumulated: set | None = None,
                  exclusive: frozenset = frozenset(), alt=None) -> None:
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
        if cut and key(node) in cut:
            continue        # a shared node: :func:`_dispatch_vjp` seeds it once
        if isinstance(node, LoopValue):
            if g is not None:
                loop_seeds.setdefault(key(node.loop), {})[node.slot] = g
            continue
        if isinstance(node, Loop):
            derived = reverse_loop(node, loop_seeds.get(key(node), {}), inputs,
                                   name, jump=jump, exclusive=exclusive, alt=alt)
            for child, contribution in derived.contributions:
                _accumulate(adjoint, key, child,
                            _reduce_to(contribution, child.ttype))
            scatters.extend(derived.roots)
            scattered |= set(derived.scattered)
            if derived.accumulated:
                if accumulated is None:
                    raise HawkError(
                        "a loop that accumulates a gradient plane per iteration "
                        "reached a reverse pass that cannot own it (a dispatch "
                        "branch): not built")
                accumulated |= set(derived.accumulated)
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
                scatters.extend(_scatter_payload(name, node.plane.name, g, node.index))
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
                         scattered, overrides, jump)
            continue
        if jump is not None and jump.apply_vjp(
                node, g, lambda child, c: _accumulate(adjoint, key, child,
                                                      _reduce_to(c, child.ttype))):
            continue
        if jump is None:
            guarded = guarded_select(node)
            if guarded is not None:
                _dispatch_vjp(guarded, g, seq, key, inputs, name, adjoint, scatters,
                              scattered, overrides, jump)
                continue
        for child, contribution in _rule(node).vjp(node, g):
            _accumulate(adjoint, key, child, _reduce_to(contribution, child.ttype))


def _dispatch_vjp(node: Dispatch, g: Node, seq: Sequence[Node], key,
                  inputs: Sequence[Leaf], name: _Namer, adjoint: dict[Any, Node],
                  scatters: list[Node], scattered: set[str],
                  overrides: Mapping[int, str] | None = None,
                  jump: JumpPlan | None = None) -> None:
    """K independent reverse passes, one per branch, each seeded only at that
    branch's own root with the dispatch's incoming adjoint ``g`` — never a
    per-node masked seed pushed through the ordinary chain rule, which would
    still run every branch's arithmetic under a zero factor.

    Every node any branch's pass reached gets one combined adjoint term,
    ``dispatch(kind, [c_0 or 0, ..., c_K or 0])`` (same policy as the primal,
    so the emitter elides branches that stayed structurally zero),
    accumulated into the enclosing pass like any other contribution (under
    ``overrides``' policy for this node when :func:`vjp` was given one), so a
    node reached through multiple paths sums correctly. A reached
    :class:`~hawk.ir.nodes.At` is the one exception: its combined value
    becomes one scattered ``AccumWrite`` here, since a scatter is a sink, not
    something a downstream node could read further. ``kind`` itself is never
    seeded and receives no adjoint.

    A compute node every non-zero branch reads (:func:`_shared_by_all`) is NOT
    differentiated through once per branch: each branch's pass stops at it and
    reports its contribution, and the combined ``dispatch(kind, [c_0, ...])``
    seeds its own reverse ONCE in the enclosing pass (the adjoint is linear in
    the seed). Every case contributes a real term, so no chain ever runs on a
    zero seed; a node only some branches read keeps the per-branch path."""
    policy = (overrides or {}).get(id(node), node.policy)
    shared = _shared_by_all(node, key)
    reached: dict[int, dict[int, Node]] = {}
    for j, branch in enumerate(node.branches):
        seed_slot = key(branch)
        local: dict[Any, Node] = {seed_slot: g}
        _reverse_pass(seq, key, local, inputs, name, scatters, scattered,
                      local_pass=True, overrides=overrides, cut=shared,
                      jump=jump)
        for slot, contribution in local.items():
            target = seq[slot]
            if not isinstance(target, (Leaf, At)) and slot not in shared:
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
        combined = Dispatch(node.selector, branch_terms, policy, target.ttype)
        if isinstance(target, At):
            if target.plane in inputs:
                scattered.add(target.plane.name)
                scatters.extend(_scatter_payload(name, target.plane.name, combined,
                                                 _taken_row(node, target, per_branch)))
            continue
        _accumulate(adjoint, key, target, combined)


def _shared_by_all(node: Dispatch, key) -> frozenset:
    """The keys of the compute nodes (operators, selects, dispatches) read, at any
    depth, by EVERY branch of ``node`` that is not a literal constant — empty
    unless two or more branches are live."""
    live = [b for b in node.branches if not isinstance(b, Const)]
    if len(live) < 2:
        return frozenset()
    common: set | None = None
    for branch in live:
        seen: dict = {}
        stack = [branch]
        while stack:
            n = stack.pop()
            if key(n) not in seen:
                seen[key(n)] = n
                stack.extend(n.operands)
        mine = {k for k, n in seen.items() if isinstance(n, (Op, Dispatch, Select))}
        common = mine if common is None else common & mine
    return frozenset(common)


def _policy_overrides(primal: Any, policy: Mapping[str, str] | None
                      ) -> dict[int, str]:
    """``{id(dispatch node): policy}`` for :func:`vjp`'s ``policy=``,
    resolving each group name through the primal's own walk (the numbering
    its sidecar publishes). Refuses an unknown group or policy by name."""
    if not policy:
        return {}
    walk = getattr(primal, "walk", None)
    if walk is None:
        walk = canonical(tuple(getattr(primal, "sinks", primal)))
    groups = dispatch_groups(walk)
    out: dict[int, str] = {}
    for group, chosen in policy.items():
        if group not in groups:
            raise HawkError(
                f"vjp(policy=...): {group!r} is not a dispatch group of "
                f"{_primal_name(primal) or 'this IR'}; its groups are "
                f"{sorted(groups)}")
        if chosen not in DISPATCH_POLICIES:
            raise HawkError(
                f"vjp(policy=...): unknown policy {chosen!r} for {group!r}; "
                f"declared are {DISPATCH_POLICIES}")
        out[id(groups[group])] = chosen
    return out


# --------------------------------------------------------------------------
# the traversal and its keys


def _sets_bits(node: Node) -> bool:
    """Whether ``node`` is an OR-scatter, or a loop whose body holds one."""
    return (isinstance(node, AccumWrite) and node.policy == "or") or any(
        _sets_bits(s) for s in getattr(node, "body_sinks", ()))


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
    if any(_sets_bits(s) for s in kept):
        raise HawkError(
            f"{primal!r} sets bits by key (hawk.scatter(y, policy=\"or\")): a bit mask is "
            "discrete and has no derivative; differentiate a kernel that does not "
            "write it")
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
    integers = sorted({n.name for n in seq if isinstance(n, Leaf) and n.name in wanted
                       and n.ttype.dtype in INT_PLANE_DTYPES})
    if integers:
        raise HawkError(
            f"wrt names {integers}: an integer plane (int32/uint32) carries no "
            "derivative; drop it from wrt= (a float computed from it through "
            "hawk.math.real is differentiated with respect to its float inputs)")
    unknown = [w for w in wanted if not any(leaf.name == w for leaf in leaves)]
    if unknown:
        raise HawkError(
            f"wrt names {unknown} are not differentiable inputs of this kernel; its "
            f"inputs are {[leaf.name for leaf in leaves]}"
        )
    return tuple(leaf for leaf in leaves if leaf.name in wanted)


def _taken(seq: Iterable[Node]) -> set[str]:
    return {n.binding.name for n in seq if n.binding is not None}


def _named(sinks: Iterable[Node], inputs: Iterable[Leaf]) -> list[str]:
    """The primal planes a derivative will name: its sinks (a lowered loop has none
    and is refused later) and its inputs."""
    return [n.name for n in (*sinks, *inputs) if hasattr(n, "name")]


class _Namer:
    """Derived plane names: ``<prefix>_<primal name>`` with the compound-wire
    namespace flattened, since only a Quantity declaration may mint a name in it.
    A flattening that would make two primal planes one is refused, never shared.

    A derivative of a derivative meets its own first-level names among the
    primal's planes (the reverse of a reverse kernel has ``bar_x`` as a sink and
    wants ``bar_x`` as the gradient of ``x``). Such a name is never reused: the
    whole derivative moves up one level together, ``bar2_x``, ``bar3_x``, taking
    the first level at which none of the planes it must name is already held, so
    the level counts the nesting depth. ``names`` are the primal planes the
    derivative will name; a kernel that never collides keeps ``<prefix>_<name>``
    exactly. Every name is a pure function of the primal's planes."""

    def __init__(self, prefix: str, taken: set[str],
                 names: Iterable[str] = ()) -> None:
        self.prefix = prefix
        self.taken = taken
        #: ``{plane: column count}`` of the compact planes of a broadcast kernel
        #: (:func:`_compact_counts`)
        self.counts: dict = {}
        self.source: dict[str, str] = {}
        flat = [n.replace(NAMESPACE_SEP, "_") for n in names]
        self.level = 1
        while any(self._spell(self.level, f) in taken for f in flat):
            self.level += 1

    def _spell(self, level: int, flat: str) -> str:
        return f"{self.prefix}{level if level > 1 else ''}_{flat}"

    def __call__(self, name: str) -> str:
        """The derived plane name for one primal plane."""
        flat = name.replace(NAMESPACE_SEP, "_")
        derived = self._spell(self.level, flat)
        while derived in self.taken:        # a plane this derivative was not told of
            self.level += 1
            derived = self._spell(self.level, flat)
        held = self.source.setdefault(derived, name)
        if held != name:
            raise HawkError(
                f"the derived name {derived!r} for {name!r} collides with "
                f"{held!r} — flattening the compound-wire namespace made two "
                "planes one"
            )
        return derived


_TIE_TEXT = (
    "no derivative for the {op} over samples stored to {name!r}: its subgradient "
    "is ambiguous at a tie, so differentiating it would return one of several "
    "answers with nothing recording which. Differentiate the per-sample value "
    "and take the {op} on the host, or reduce with a sum")


def _seed(sink: Sink, name: _Namer) -> Node:
    """The incoming adjoint plane of one primal sink (the four sink kinds)."""
    derived, t = name(sink.name), sink.ttype
    if isinstance(sink, SampleSum):
        if sink.op != "sum":
            raise HawkError(_TIE_TEXT.format(name=sink.name, op=sink.op))
        # the reverse of a sum over samples is a broadcast of its adjoint
        return Leaf("uniform", "uniform", derived, t)
    if isinstance(sink, (AccumWrite, WideWrite)) and sink.index is not None:
        # the transpose of a scatter-add is a gather of the adjoint plane
        return At(Leaf("table_read", "lookup", derived, t), sink.index, t)
    return Leaf("vocab_read", _READ_ROLE[len(t.shape)], derived, t)


def _tangent_sink(sink: Sink, tangent: Node | None, name: _Namer,
                  index: Node | None = None) -> Node:
    """The forward-mode twin of one primal sink, same kind and same lane
    (at ``index`` when the alternate path moved the key)."""
    value = tangent if tangent is not None else zero_like(sink.ttype)
    derived = name(sink.name)
    index = sink.index if index is None else index
    if isinstance(sink, AccumWrite):
        return AccumWrite(derived, value, index, sink.ttype, sink.policy)
    if isinstance(sink, WideWrite):
        return WideWrite(derived, value, index, sink.ttype)
    if isinstance(sink, SampleSum):
        if sink.op != "sum":
            raise HawkError(_TIE_TEXT.format(name=sink.name, op=sink.op))
        return SampleSum(derived, value, sink.ttype)
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
