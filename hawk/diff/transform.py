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
scatter-add ``Update`` are each other's reverse-mode transpose.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .. import _ir
from ..errors import HawkError
from ..ext.primitive import jvp_tangent, vjp_contributions
from ..trace.value import DISPATCH_STRATEGIES, dispatch_node, dispatch_strategy
from . import rules
from .altpath import AltPath
from .jump import JumpPlan, drop_width, jump_of
from .loops import (
    _reduce_to,
    _taken_indices,
    forward_loop,
    guarded_select,
    reverse_loop,
)
from .ordered import kept_of, order


#: Prefixes of the planes a derivative declares (never a wire namespace).
TANGENT_PREFIX = "dot"
ADJOINT_PREFIX = _ir.ADJOINT_PREFIX


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

    def __new__(
        cls,
        sinks,
        *,
        primal: str | None,
        kind: str,
        wrt: tuple[str, ...],
        primal_unit: str | None = None,
        jump: dict | None = None,
        gradients: dict | None = None,
        cotangents: dict | None = None,
        next_values: dict | None = None,
        tangents: dict | None = None,
    ):
        self = super().__new__(cls, sinks)
        #: A reverse derivative's planes as the transform named them: ``gradients`` maps each gradient plane to the
        #: input it is the gradient of, ``cotangents`` each cotangent (seed) plane to the forward output it seeds.
        #: Empty for a forward derivative.
        self.gradients = {} if gradients is None else gradients
        self.cotangents = {} if cotangents is None else cotangents
        #: A step derivative's next-value planes: each plane holding a carried plane's value after the step (the transform
        #: named it), mapped to the carried plane. Empty when the derivative is not of a step.
        self.next_values = {} if next_values is None else next_values
        #: A forward derivative's tangent planes as the transform named them, inputs and outputs alike: each tangent
        #: plane mapped to the primal plane it is the tangent of. Empty for a reverse derivative.
        self.tangents = {} if tangents is None else tangents
        #: The sidecar ``jump`` record of a derivative smoothed through its
        #: switches (``@kernel(jump=...)``), else ``None``.
        self.jump = jump
        #: ``K`` of a fused K-step reverse (``step_vjp(..., steps_reverse="fused")``), else ``None``.
        self._fused_steps = None
        #: The kind a kernel built from this derivative takes when it names none.
        self._kernel_kind = None
        self.primal = primal
        self.kind = kind
        self.wrt = wrt
        self.primal_unit = primal_unit
        return self


def _termination_mask(primal: Any, context: Any) -> Any:
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
    names = [m for m in guard.names if ("terminated", m) in _ir.slot_of(walk)]
    if not names:
        return None
    mask = _ir.leaf(
        context,
        "terminated",
        "terminated",
        names[0],
        _ir.slot_type(walk, "terminated", names[0]),
    )
    for extra in names[1:]:
        other = _ir.leaf(
            context,
            "terminated",
            "terminated",
            extra,
            _ir.slot_type(walk, "terminated", extra),
        )
        mask = _ir.make("lor", (mask, other))
    return mask


def _masked_sink(sink: Any, mask: Any) -> Any:
    """``sink``, its value replaced by ``terminated ? 0 : value`` — the
    structural zero :func:`~hawk._ir.zero_like` gives any missing
    contribution, so every sink shape alike stores or accumulates nothing for
    a terminated sample.

    A ``sink`` may be a lowered loop (a reverse-mode loop's own scatter
    commit): only its per-iteration commits are masked, not its
    header/carries/``nexts``. Rebuilt through :func:`~hawk._ir.loop` so the
    loop's boundary is recomputed by the same code every other loop producer
    uses."""
    if _ir.is_a(sink, "loop"):
        masked_body = _ir.slotwise(
            tuple(_masked_sink(s, mask) for s in sink.field_nodes("body_sinks"))
        )
        return _ir.loop(
            sink.field_nodes("index")[0],
            sink.field_nodes("carries"),
            sink.field_nodes("inits"),
            sink.field_nodes("nexts"),
            start=sink.field_int("start"),
            stop=sink.field_int("stop"),
            step=sink.field_int("step"),
            body_sinks=masked_body,
            **_ir.exit_kwargs(sink),
        )
    committed = sink.operands[0]
    value_type = _ir.type_of(committed)
    value = _ir.make_node(
        "Select", value_type, [mask, _ir.zero_like(mask.ctx, value_type), committed]
    )
    return _ir.unreduced(sink, (value, *sink.operands[1:]))


def _mask_termination(
    sinks: Sequence[Any], primal: Any, context: Any
) -> tuple[Any, ...]:
    """``sinks`` gated by :func:`_termination_mask` when ``primal`` declares
    one. The single call both :func:`vjp` and :func:`jvp` make before
    tagging their result, masking every sink shape uniformly. ``context`` is
    the one the primal's nodes live in."""
    mask = _termination_mask(primal, context)
    if mask is None:
        return tuple(sinks)
    return tuple(_masked_sink(s, mask) for s in sinks)


def _refuse_prior_read(seq: Sequence[Any]) -> None:
    """Refuses a primal containing a ``prior_read`` leaf (a Mutable read
    before its first store), naming the plane.

    The value it reads is a recurrence across launches; its derivative needs
    the value the primal launch overwrites, and HAWK keeps no tape. Left
    unguarded, a jvp would bind the primal's own prior plane read-only
    (correct only if the derived kernel launches before the primal) and a vjp
    would bind the same adjoint plane twice — plausible wrong answers, not
    crashes, so this must raise instead."""
    for node in seq:
        if _ir.is_a(node, "leaf") and node.field_text("kind") == "prior_read":
            raise HawkError(
                f"{node.field_text('name')!r} is read before it is assigned: a plane's "
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


#: Node kinds a transform never pushes through: a bound plane, a literal,
#: a sink (seeded, not traversed), and the lane index (an address, not a
#: differentiable quantity). The loop-body walk keeps its own list (``loops._TERMINALS``): it has no
#: ``plane_extent`` and it stops at a nested loop, which this transform handles itself.
_TERMINALS = (
    "sink",
    "leaf",
    "const",
    "sample_index",
    "plane_extent",
    "loop_index",
    "loop_carry",
)


def _is_terminal(node: Any) -> bool:
    """Whether ``node`` is of a kind in :data:`_TERMINALS`."""
    return _ir.is_a(node, *_TERMINALS)


def _indices_of(node: Any) -> tuple:
    """The indices a scattering sink commits at or a read gathers at, one per axis; empty for a commit at the lane's own cell."""
    return tuple(node.operands[1:])


def _scatter(
    name: str, value: Any, indices: Sequence[Any], ttype: Any, fields: dict | None = None,
    edge: str = "unchecked",
    axes: Sequence[Any] = (),
) -> Any:
    """An accumulating commit of ``value`` into plane ``name``, whose axes are ``axes``, at ``indices`` (none: the lane's own cell)."""
    return _ir.make_node(
        "Update", ttype, [value, *indices], name=name, **(fields or {"op": "add"}), reduced=None, edge=edge,
        axes=tuple(axes),
    )


#: dtypes a derivative flows through; everything else is a structural constant.
_DIFFERENTIABLE = ("f64", "f32")


def _rederivable(derive):
    """``derive`` with its result remembering how to be made again: the primal
    it differentiated, a ``rebuild(primal)`` that differentiates another one the
    same way (with the resolved ``wrt``), and the primal's launch shape and
    group, so a kernel built from it launches over the same lanes and the
    broadcast door can differentiate a broadcast twin of the primal instead
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
def vjp(
    primal: Any,
    *,
    wrt: Sequence[str] | None = None,
    primal_unit: str | None = None,
    strategy: Mapping[str, str] | None = None,
) -> Derived:
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

    ``strategy={group: strategy}`` sets the lowering of the derived dispatches
    of the primal's dispatch GROUP ``group`` (the sidecar's positional
    ``dispatch<i>`` name), which otherwise inherit the primal's strategy: the
    reverse takes its own structure. Any of
    the dispatch strategies of the library is accepted; an unknown group
    or strategy refuses by name."""
    if getattr(primal, "members", None):
        from .chain import vjp_chain

        return vjp_chain(primal, wrt, primal_unit, strategy)
    primal = _single_step(primal)
    sinks = _sinks_of(primal)
    cells = _ir.param_cell_names(_ir.canonical_nodes(sinks)[0])
    if cells:
        return _vjp_param_cells(primal, cells, wrt, primal_unit, strategy)
    return _vjp(
        primal,
        sinks,
        wrt,
        primal_unit,
        strategy,
        summed=(),
        kept=kept_of(tuple(getattr(primal, "sinks", primal))),
    )


def _vjp(
    primal: Any,
    sinks: tuple,
    wrt: Sequence[str] | None,
    primal_unit: str | None,
    strategy: Mapping[str, str] | None,
    summed: tuple,
    kept: tuple | None = None,
) -> Derived:
    """:func:`vjp` proper; the gradient of each input named in ``summed``
    is summed over samples (a ``SampleSum`` sink). ``kept`` is set when the primal
    stores an ordered keyed sum: its scatters become an ordered log (:mod:`hawk.diff.ordered`)."""
    overrides = _strategy_overrides(primal, strategy)
    jump = jump_of(primal)
    drop_width((), wrt, jump)
    seq, position = _ir.canonical_nodes(sinks)
    _refuse_prior_read(seq)
    context = sinks[0].ctx
    inputs = drop_width(_wrt(seq, wrt), None, jump)
    name = _Namer(ADJOINT_PREFIX, _taken(seq), _named(sinks, inputs))
    name.counts = _compact_counts(primal, context)
    key = _keyer(position)
    plan = _plan(seq, key, inputs, jump, primal)
    alt = AltPath.of(plan, seq, reverse=True)
    if alt is not None:
        for node in seq:
            if not _is_terminal(node):
                alt.visit(node, None)
    adjoint: dict[Any, Any] = {}
    for sink in sinks:
        if _ir.is_a(sink, "loop"):
            raise HawkError(
                "this IR's root set contains a lowered loop that COMMITS once per "
                "iteration, which only a DERIVED kernel does: second-order reverse "
                "mode through a `for` is not built. Publish the first "
                "derivative and differentiate that kernel"
            )
        if _ir.type_of(sink).dtype in _ir.INT_PLANE_DTYPES:
            continue  # an integer plane takes no adjoint
        _accumulate(adjoint, key, sink.operands[0], _seed(sink, name))
    if alt is not None:
        alt.g = alt.seed_gain(sinks, lambda s: _seed(s, name))
        for site, weight in alt.injections():
            lhs, rhs = site.operands
            _accumulate(adjoint, key, lhs, weight)
            _accumulate(adjoint, key, rhs, _ir.make("neg", (weight,)))
    scatters: list[Any] = []
    scattered: set[str] = set()
    accumulated: set[str] = set()
    _reverse_pass(
        seq,
        key,
        adjoint,
        inputs,
        name,
        scatters,
        scattered,
        overrides=overrides,
        jump=None if alt is not None else plan,
        accumulated=accumulated,
        exclusive=_loop_exclusive(seq, inputs),
        alt=alt,
    )
    for leaf in inputs:
        if name(leaf.field_text("name")) not in accumulated:
            continue
        if adjoint.get(key(leaf)) is not None:  # pragma: no cover - by the pre-scan
            raise HawkError(
                f"the gradient of {leaf.field_text('name')!r} is accumulated at runtime indices "
                "inside a loop AND reaches a contribution outside that loop"
            )
    gradients = [
        _gradient_sink(
            leaf,
            name(leaf.field_text("name")),
            summed,
            adjoint.get(key(leaf)) or _ir.zero_like(context, _ir.type_of(leaf)),
        )
        for leaf in inputs
        if leaf.field_text("role") != "lookup"
        and leaf.field_text("name") not in scattered
        and name(leaf.field_text("name")) not in accumulated
    ]
    held = getattr(primal, "broadcast", ())
    scatters = _compensated(
        scatters,
        {
            name(leaf.field_text("name"))
            for leaf in inputs
            if leaf.field_text("name") in held
        },
    )
    out = _ir.slotwise(_mask_termination(gradients + list(scatters), primal, context))
    if kept is not None:
        out = order(out, kept)
    names = tuple(leaf.field_text("name") for leaf in inputs)
    if plan is not None:
        out += plan.spread_cells()
    return Derived(
        out,
        primal=_primal_name(primal),
        kind="vjp",
        wrt=names,
        primal_unit=primal_unit,
        jump=None if plan is None else plan.record(names),
        gradients={name(n): n for n in names},
        cotangents={
            name(sink.field_text("name")): sink.field_text("name")
            for sink in sinks
            if _ir.type_of(sink).dtype not in _ir.INT_PLANE_DTYPES
        },
    )


def _compact_counts(primal: Any, context: Any) -> dict:
    """``{plane: the column count of its compact plane}`` (an ``i32`` node of ``context``) for each
    plane a broadcast twin reads at the axes it varies along: the product of the
    kept axes' extents, a kept runtime first axis through the lane count."""
    held = getattr(primal, "broadcast", None) or {}
    axes = getattr(primal, "lane_space", None)
    out: dict = {}
    for plane, kept in held.items():
        if axes is None or kept is None:
            continue
        t = _ir.TensorType((), "i32")
        fixed = 1
        for (_label, extent), k in zip(axes, kept):
            fixed *= extent if k and extent is not None else 1
        if kept[0] and axes[0][1] is None:
            rest = 1
            for (_label, extent), k in zip(axes, kept):
                rest *= 1 if k else extent
            lanes = _ir.leaf(context, "nsamples", "nsamples", "n_samples", t)
            out[plane] = (
                lanes
                if rest == 1
                else _ir.make("floordiv", (lanes, _ir.const(context, rest, t)))
            )
        else:
            out[plane] = _ir.const(context, fixed, t)
    return out


def _scatter_payload(
    name: _Namer, plane: str, g: Any, indices: Sequence[Any], edge: str, axes: Sequence[Any]
) -> list:
    """The scattered accumulate of ``g`` into the adjoint of ``plane`` at ``indices``, under the read's edge rule:
    a rank-0 value as it is; a vector or matrix one scalar accumulate per entry
    ``k`` (row-major) at ``index + k * count`` of ONE component-major plane, so the
    gradient has the layout of the compact plane it is the gradient of. Only a
    broadcast twin knows its column count (:func:`_compact_counts`); any other
    rank >= 1 scatter is left whole, which the emitter refuses by name."""
    g_type = _ir.type_of(g)
    shape = g_type.shape
    count = name.counts.get(plane)
    if (
        not shape or count is None or len(indices) != 1
    ):  # not a broadcast twin of one axis: the emitter refuses a rank >= 1 scatter by name
        return [_scatter(name(plane), g, indices, g_type, edge=edge, axes=axes)]
    (index,) = indices
    t, scalar = _ir.type_of(index), _ir.TensorType((), g_type.dtype)
    out = []
    for k in range(shape[0] * (shape[1] if len(shape) == 2 else 1)):
        at = (k // shape[1], k % shape[1]) if len(shape) == 2 else k
        key = (
            index
            if k == 0
            else _ir.make(
                "add", (index, _ir.make("mul", (count, _ir.const(index.ctx, k, t))))
            )
        )
        out.append(
            _scatter(name(plane), _ir.make("component", (g,), literal=at), (key,), scalar, edge=edge, axes=axes)
        )
    return out


def _compensated(scatters: Sequence[Any], planes: set) -> list:
    """``scatters`` with each accumulate into a plane of ``planes`` compensated,
    its companion bound by a zero commit at the same key. A gradient plane summed
    over the lane axes a broadcast input does not vary along."""
    out: list = []
    for s in scatters:
        if _ir.is_accumulate(s) and s.field_text("name") in planes:
            name, ttype, indices = s.field_text("name"), _ir.type_of(s), _indices_of(s)
            edge, axes = s.field_text("edge"), _ir.field_pairs(s, "axes")
            out.append(_scatter(name, s.operands[0], indices, ttype, _ir.contract_fields("compensated"), edge, axes))
            zero = _ir.const(s.ctx, 0.0, _ir.TensorType((), ttype.dtype))
            out.append(_scatter(name + _ir.COMPANION_SUFFIX, zero, indices, ttype, edge=edge, axes=axes))
        else:
            out.append(s)
    return out


def _loop_exclusive(seq: Sequence[Any], inputs: Sequence[Any]) -> frozenset:
    """The ``wrt`` leaves read ONLY inside lowered-loop bodies: no
    top-level node (a sink included) reads one, and none is a carried slot's
    initial value. Such a leaf's gradient can be committed by the loops alone
    (:func:`hawk.diff.loops.reverse_loop`, the per-iteration accumulate)."""
    dense: set = set()
    for n in seq:
        held = n.field_nodes("inits") if _ir.is_a(n, "loop") else n.operands
        dense.update(c for c in held if _ir.is_a(c, "leaf"))
    return frozenset(leaf for leaf in inputs if leaf not in dense)


def _plan(
    seq: Sequence[Any], key, inputs: Sequence[Any], jump: Any, primal: Any
) -> JumpPlan | None:
    """The switching comparisons of a ``jump=`` primal, ``None`` when it
    declares none or none of its comparisons switches."""
    if jump is None:
        return None
    plan = JumpPlan(seq, key, inputs, jump, _primal_name(primal))
    return plan if plan.sites else None


def _gradient_sink(leaf: Any, name: str, summed: tuple, value: Any) -> Any:
    """The sink of one input's gradient: the sum over samples of the per-sample
    adjoint for a promoted parameter, else the adjoint plane itself."""
    ttype = _ir.type_of(leaf)
    if leaf.field_text("name") in summed:
        if ttype.shape:
            raise HawkError(
                f"the gradient of {leaf.field_text('name')!r} cannot be summed over "
                f"samples: only a scalar leaf is, and it has shape {ttype.shape}"
            )
        reduction = _ir.make_node(
            "ReduceSamples",
            _ir.type_of(value),
            [value],
            op="sum",
            contract="auto",
            key=None,
            count=None,
            kept=None,
        )
        return _ir.cells_stored(reduction, name)[0]
    return _ir.make_node("Assign", ttype, [value], name=name)


def _vjp_param_cells(
    primal: Any,
    cells: tuple,
    wrt: Sequence[str] | None,
    primal_unit: str | None,
    strategy: Mapping[str, str] | None,
) -> Derived:
    """The reverse of a kernel reading parameter cells (see
    :func:`hawk.diff.param_planes`): differentiate the form that reads each
    cell as a per-sample plane, sum each cell's gradient over every 32
    samples (``bar_<name>`` rows, ``SampleSum`` sinks), then
    read the cells again in the derived kernel."""
    from .params import demote, promote

    per_sample = _Demoted(primal, demote(_sinks_of(primal)))
    d = _vjp(per_sample, per_sample.sinks, wrt, primal_unit, strategy, summed=cells)
    return Derived(
        promote(d, cells),
        primal=_primal_name(primal),
        kind="vjp",
        wrt=d.wrt,
        primal_unit=primal_unit,
        jump=d.jump,
        gradients=d.gradients,
        cotangents=d.cotangents,
    )


class _Demoted:
    """A primal's name and guard bindings over replaced sinks."""

    def __init__(self, primal: Any, sinks: tuple) -> None:
        self.name, self.sinks, self.walk = primal.name, sinks, primal.walk
        self.planes = getattr(primal, "planes", {})
        self.kind = getattr(primal, "kind", None)
        self.jump = getattr(primal, "jump", None)


def jvp(
    primal: Any, *, wrt: Sequence[str] | None = None, primal_unit: str | None = None
) -> Derived:
    """The forward-mode derivative IR of ``primal``: one tangent plane
    per ``wrt`` input in, one directional-derivative sink per primal sink out.
    Returned as :class:`Derived`, tagging ``primal``'s name and the resolved
    ``wrt`` (see that class). ``primal_unit=`` is :func:`vjp`'s (see there).
    A default automatic kernel is differentiated through its single step, as
    in :func:`vjp`. A ``hawk.steps(k, K)`` kernel is carried forward,
    ``hawk.steps(step_jvp(k), K)`` (``primal_unit`` does not apply to it)."""
    from ..trace.steps import _fused_steps_of

    fused = _fused_steps_of(primal)
    if fused is not None:
        from ..trace.steps import steps
        from .tape import step_jvp

        return steps(step_jvp(fused[0], wrt=wrt), fused[1])
    return _jvp(primal, wrt=wrt, primal_unit=primal_unit)


def with_derivative(
    primal: Any, *, wrt: Sequence[str] | None = None, name: str | None = None
) -> Any:
    """ONE kernel that computes ``primal``'s outputs and their forward-mode tangents along ``wrt``: the sinks of
    ``primal`` followed by the sinks of :func:`jvp`. The result is a plain :class:`hawk.Kernel` (not a derivative
    kernel), so it runs anywhere a kernel runs, as a step of an ``eagle.Graph`` builder included. Its planes are
    ``primal``'s, plus one tangent plane per ``wrt`` input and one per output, named with :data:`TANGENT_PREFIX`.
    Named ``<primal>_with_derivative`` unless ``name`` is given. A ``hawk.steps(k, K)`` kernel is refused: its tangent
    is carried across the steps by :func:`jvp`."""
    from ..trace.kernel import Kernel
    from ..trace.steps import _fused_steps_of

    if _fused_steps_of(primal) is not None:
        raise HawkError(
            "hawk.diff.with_derivative: this is a hawk.steps(k, K) kernel; its tangent is carried across the steps, "
            "so take jvp(k) (one kernel) or with_derivative of the single step"
        )
    one = _single_step(primal)
    tangent = _jvp(primal, wrt=wrt)
    sinks = (*_sinks_of(one), *tangent)
    label = name or f"{_primal_name(one) or 'kernel'}_with_derivative"
    declared = dict(getattr(one, "planes", None) or {})
    if declared:  # a tangent plane is declared like the plane it is the tangent of, so calls bind and size it
        names = [
            nm
            for _role, nm in Kernel(
                label, sinks, kind=getattr(one, "kind", None)
            ).arg_spec
        ]
        extra = [
            (nm, primal_plane)
            for nm, primal_plane in tangent.tangents.items()
            if nm in names and nm not in declared and primal_plane in declared
        ]
        # input tangents first, then output tangents, each in the primal's own order
        order = list(declared)
        extra.sort(
            key=lambda e: (
                getattr(declared[e[1]], "form", "") == "mutable",
                order.index(e[1]),
            )
        )
        declared.update((nm, declared[primal_plane]) for nm, primal_plane in extra)
    out = Kernel(label, sinks, declared or None, kind=getattr(one, "kind", None))
    out.tangents = tangent.tangents
    return out


@_rederivable
def _jvp(
    primal: Any, *, wrt: Sequence[str] | None = None, primal_unit: str | None = None
) -> Derived:
    if getattr(primal, "members", None):
        from .chain import jvp_chain

        return jvp_chain(primal, wrt, primal_unit)
    primal = _single_step(primal)
    sinks = _sinks_of(primal)
    for sink in sinks:
        if _ir.is_a(sink, "loop"):
            raise HawkError(
                "this IR's root set contains a lowered loop that COMMITS once per "
                "iteration, which only a DERIVED kernel does: second-order forward "
                "mode through a `for` is not built"
            )
    jump = jump_of(primal)
    drop_width((), wrt, jump)
    seq, position = _ir.canonical_nodes(sinks)
    _refuse_prior_read(seq)
    context = sinks[0].ctx
    inputs = drop_width(_wrt(seq, wrt), None, jump)
    name = _Namer(TANGENT_PREFIX, _taken(seq), _named(sinks, inputs))
    key = _keyer(position)
    plan = _plan(seq, key, inputs, jump, primal)
    alt = AltPath.of(plan, seq)
    tangent: dict[Any, Any] = {
        key(leaf): _ir.leaf(
            context,
            leaf.field_text("kind"),
            leaf.field_text("role"),
            name(leaf.field_text("name")),
            _ir.type_of(leaf),
            sorted=leaf.field_flag("sorted"),
            axes=_ir.field_pairs(leaf, "axes"),
        )
        for leaf in inputs
    }
    #: The AUGMENTED loop a lowered `for` becomes in forward mode — one loop
    #: carrying the primal slots and their tangents side by side; slot `m + j`
    #: is the tangent of the primal's slot `j` (hawk/diff/loops.py).
    loop_tangent: dict[Any, Any] = {}
    for node in seq:
        if _is_terminal(node):
            continue
        if alt is not None:
            alt.visit(node, lambda c: tangent.get(key(c)))
        elif plan is not None and _jump_jvp(node, plan, key, tangent):
            continue
        if _ir.is_a(node, "loop"):
            # `loop_tangent` is also consulted by the LoopValue arm below: a
            # reverse loop's body reads the forward loop's tape, and the
            # tangent of that read is the same row of this augmented twin
            # (hawk/diff/loops._tape_tangent). The walk visits the taped
            # forward loop before the reverse loop that reads it, so the
            # twin is always already in this table when needed.
            got = forward_loop(
                node,
                lambda c: tangent.get(key(c)),
                name,
                lambda lp: loop_tangent.get(key(lp)),
                jump=None if alt is not None else plan,
                alt=alt,
            )
            if got is not None:
                loop_tangent[key(node)] = got
            continue
        if _ir.is_a(node, "loop_value"):
            loop, slot = _ir.loop_value_of(node)
            augmented = loop_tangent.get(key(loop))
            if augmented is not None:
                tangent[key(node)] = _ir.loop_value(
                    augmented, len(loop.field_nodes("carries")) + slot
                )
            continue
        if _ir.is_a(node, "primitive"):
            supplied = [tangent.get(key(c)) for c in node.operands[1:]]
            if any(t is not None for t in supplied):
                got = jvp_tangent(node, supplied)
                if got is not None:
                    tangent[key(node)] = got
            continue
        if _ir.is_a(node, "at"):
            plane, *indices = node.operands
            plane = tangent.get(key(plane))
            if plane is not None:
                tangent[key(node)] = _ir.make_node(
                    "At", _ir.type_of(node), [plane, *indices], edge=node.field_text("edge")
                )
            continue
        if _ir.is_a(node, "dispatch"):
            # `t_out = dispatch(kind, [t_e0, ..., t_eK])`, same strategy and
            # selector (`kind` carries no tangent of its own). A branch with
            # no tangent contributes the structural zero — select's own
            # missing-tangent rule — and branch tangents are used only
            # inside this node, so the emitter inlines each in its own
            # case/select arm.
            selector, *branches = node.operands
            branch_tangents = [tangent.get(key(b)) for b in branches]
            if any(t is not None for t in branch_tangents):
                filled = [
                    t if t is not None else _ir.zero_like(context, _ir.type_of(b))
                    for t, b in zip(branch_tangents, branches)
                ]
                tangent[key(node)] = dispatch_node(
                    selector, filled, dispatch_strategy(node), _ir.type_of(node)
                )
            continue
        if any(tangent.get(key(c)) is not None for c in node.operands):
            got = _rule(node).jvp(node, lambda c: tangent.get(key(c)))
            if got is not None:
                tangent[key(node)] = got
    if alt is not None:
        made = alt.sinks(
            sinks,
            lambda v: tangent.get(key(v)),
            lambda s, v, indices=None: _tangent_sink(s, v, name, indices),
        )
    else:
        made = tuple(
            _tangent_sink(s, tangent.get(key(s.operands[0])), name)
            for s in sinks
            if _ir.type_of(s).dtype not in _ir.INT_PLANE_DTYPES
        )
    out = _ir.slotwise(_mask_termination(made, primal, context))
    names = tuple(leaf.field_text("name") for leaf in inputs)
    if plan is not None:
        out += plan.spread_cells()
    return Derived(
        out,
        primal=_primal_name(primal),
        kind="jvp",
        wrt=names,
        primal_unit=primal_unit,
        jump=None if plan is None else plan.record(names),
        tangents=dict(name.source),
    )


def _jump_jvp(node: Any, plan: JumpPlan, key, tangent: dict) -> bool:
    """Forward mode's boolean-op rules of a ``jump=`` primal: whether
    ``node`` was one (its tangent, if any, recorded)."""

    def t(c: Any) -> Any:
        return tangent.get(key(c))

    handled, got = plan.apply_jvp(node, t, lambda n: _rule(n).jvp(n, t))
    if got is not None:
        tangent[key(node)] = got
    return handled


# --------------------------------------------------------------------------
# the reverse-mode dispatch rule


def _reverse_pass(
    seq: Sequence[Any],
    key,
    adjoint: dict[Any, Any],
    inputs: Sequence[Any],
    name: _Namer,
    scatters: list[Any],
    scattered: set[str],
    *,
    local_pass: bool = False,
    overrides: Mapping[Any, str] | None = None,
    cut: frozenset = frozenset(),
    jump: JumpPlan | None = None,
    accumulated: set | None = None,
    exclusive: frozenset = frozenset(),
    alt=None,
) -> None:
    """One reverse-accumulation pass over ``seq``, given an already-seeded
    ``adjoint`` — the core step :func:`vjp` runs once over the whole walk,
    seeded at every primal sink.

    :func:`_dispatch_vjp` reuses this in isolation over one
    dispatch branch's exclusive sub-DAG, with
    ``local_pass=True``: a fresh ``adjoint`` seeded at only that branch's
    root. A node reachable from branch j alone keeps exactly that seed's
    contribution; one also reachable elsewhere accumulates a second
    contribution the ordinary way, via the same ``_accumulate`` fan-in any
    shared subexpression uses.

    ``local_pass`` changes one thing: a gather's (``at``) reverse
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
            continue  # a shared node: :func:`_dispatch_vjp` seeds it once
        if _ir.is_a(node, "loop_value"):
            if g is not None:
                owner, slot = _ir.loop_value_of(node)
                loop_seeds.setdefault(key(owner), {})[slot] = g
            continue
        if _ir.is_a(node, "loop"):
            derived = reverse_loop(
                node,
                loop_seeds.get(key(node), {}),
                inputs,
                name,
                jump=jump,
                exclusive=exclusive,
                alt=alt,
            )
            for child, contribution in derived.contributions:
                _accumulate(adjoint, key, child, _reduce_to(contribution, child))
            scatters.extend(derived.roots)
            scattered |= set(derived.scattered)
            if derived.accumulated:
                if accumulated is None:
                    raise HawkError(
                        "a loop that accumulates a gradient plane per iteration "
                        "reached a reverse pass that cannot own it (a dispatch "
                        "branch): not built"
                    )
                accumulated |= set(derived.accumulated)
            continue
        if g is None or _is_terminal(node):
            continue
        if _ir.is_a(node, "at"):
            plane, *indices = node.operands
            if local_pass:
                # `g` already holds this node's fully-accumulated local
                # contribution — from consumers reached within this pass, or
                # as the branch's own seed. Folding it in again here would
                # double every scatter this local pass reaches.
                continue
            if plane in inputs:
                scattered.add(plane.field_text("name"))
                scatters.extend(
                    _scatter_payload(
                        name, plane.field_text("name"), g, indices, node.field_text("edge"),
                        _ir.field_pairs(plane, "axes"),
                    )
                )
            continue
        if _ir.is_a(node, "primitive"):
            # Operand 0 (the forward subgraph) is deliberately skipped: the
            # supplied rule already is this primitive's derivative.
            for child, contribution in vjp_contributions(node, g):
                _accumulate(adjoint, key, child, _reduce_to(contribution, child))
            continue
        if _ir.is_a(node, "dispatch"):
            _dispatch_vjp(
                node,
                g,
                seq,
                key,
                inputs,
                name,
                adjoint,
                scatters,
                scattered,
                overrides,
                jump,
            )
            continue
        if jump is not None and jump.apply_vjp(
            node, g, lambda child, c: _accumulate(adjoint, key, child, _reduce_to(c, child))
        ):
            continue
        if jump is None:
            guarded = guarded_select(node)
            if guarded is not None:
                _dispatch_vjp(
                    guarded,
                    g,
                    seq,
                    key,
                    inputs,
                    name,
                    adjoint,
                    scatters,
                    scattered,
                    overrides,
                    jump,
                )
                continue
        for child, contribution in _rule(node).vjp(node, g):
            _accumulate(adjoint, key, child, _reduce_to(contribution, child))


def _dispatch_vjp(
    node: Any,
    g: Any,
    seq: Sequence[Any],
    key,
    inputs: Sequence[Any],
    name: _Namer,
    adjoint: dict[Any, Any],
    scatters: list[Any],
    scattered: set[str],
    overrides: Mapping[Any, str] | None = None,
    jump: JumpPlan | None = None,
) -> None:
    """K independent reverse passes, one per branch, each seeded only at that
    branch's own root with the dispatch's incoming adjoint ``g`` — never a
    per-node masked seed pushed through the ordinary chain rule, which would
    still run every branch's arithmetic under a zero factor.

    Every node any branch's pass reached gets one combined adjoint term,
    ``dispatch(kind, [c_0 or 0, ..., c_K or 0])`` (same strategy as the primal,
    so the emitter elides branches that stayed structurally zero),
    accumulated into the enclosing pass like any other contribution (under
    ``overrides``' strategy for this node when :func:`vjp` was given one), so a
    node reached through multiple paths sums correctly. A reached
    gather (``at``) is the one exception: its combined value
    becomes one scattered ``Update`` that adds here, since a scatter is a sink, not
    something a downstream node could read further. ``kind`` itself is never
    seeded and receives no adjoint.

    A compute node every non-zero branch reads (:func:`_shared_by_all`) is NOT
    differentiated through once per branch: each branch's pass stops at it and
    reports its contribution, and the combined ``dispatch(kind, [c_0, ...])``
    seeds its own reverse ONCE in the enclosing pass (the adjoint is linear in
    the seed). Every case contributes a real term, so no chain ever runs on a
    zero seed; a node only some branches read keeps the per-branch path."""
    strategy = (overrides or {}).get(node, dispatch_strategy(node))
    selector, *branches = node.operands
    shared = _shared_by_all(node, key)
    reached: dict[int, dict[int, Any]] = {}
    for j, branch in enumerate(branches):
        local: dict[int, Any] = {key(branch): g}
        _reverse_pass(
            seq,
            key,
            local,
            inputs,
            name,
            scatters,
            scattered,
            local_pass=True,
            overrides=overrides,
            cut=shared,
            jump=jump,
        )
        for slot, contribution in local.items():
            target = seq[slot]
            if not _ir.is_a(target, "leaf", "at") and slot not in shared:
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
        target_type = _ir.type_of(target)
        branch_terms = [
            per_branch.get(j)
            if per_branch.get(j) is not None
            else _ir.zero_like(node.ctx, target_type)
            for j in range(len(branches))
        ]
        combined = dispatch_node(selector, branch_terms, strategy, target_type)
        if _ir.is_a(target, "at"):
            plane = target.operands[0]
            if plane in inputs:
                scattered.add(plane.field_text("name"))
                scatters.extend(
                    _scatter_payload(
                        name,
                        plane.field_text("name"),
                        combined,
                        _taken_indices(node, target, per_branch),
                        target.field_text("edge"),
                        _ir.field_pairs(plane, "axes"),
                    )
                )
            continue
        _accumulate(adjoint, key, target, combined)


def _shared_by_all(node: Any, key) -> frozenset:
    """The keys of the compute nodes (operators, selects, dispatches) read, at any
    depth, by EVERY branch of ``node`` that is not a literal constant — empty
    unless two or more branches are live."""
    live = [b for b in node.operands[1:] if not _ir.is_a(b, "const")]
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
        mine = {k for k, n in seen.items() if _ir.is_a(n, "op", "dispatch", "select")}
        common = mine if common is None else common & mine
    return frozenset(common)


def _strategy_overrides(
    primal: Any, strategy: Mapping[str, str] | None
) -> dict[Any, str]:
    """``{dispatch node: strategy}`` for :func:`vjp`'s ``strategy=``,
    resolving each group name through the primal's own walk (the numbering
    its sidecar publishes). Refuses an unknown group or strategy by name."""
    if not strategy:
        return {}
    walk = getattr(primal, "walk", None)
    if walk is None:
        walk = _ir.cells_walk(tuple(getattr(primal, "sinks", primal)))
    groups = {found.name: found.node for found in _ir.dispatch_groups(walk)}
    out: dict[Any, str] = {}
    for group, chosen in strategy.items():
        if group not in groups:
            raise HawkError(
                f"vjp(strategy=...): {group!r} is not a dispatch group of "
                f"{_primal_name(primal) or 'this IR'}; its groups are "
                f"{sorted(groups)}"
            )
        if chosen not in DISPATCH_STRATEGIES:
            raise HawkError(
                f"vjp(strategy=...): unknown strategy {chosen!r} for {group!r}; "
                f"declared are {DISPATCH_STRATEGIES}"
            )
        out[groups[group]] = chosen
    return out


# --------------------------------------------------------------------------
# the traversal and its keys


def _keyer(position: Mapping[Any, int]):
    """The walk's own structural identity, used as the derivative's table key: two equal
    nodes made apart share one position, so a duplicated subexpression accumulates ONE
    adjoint (or tangent) like a shared one. A node outside the walk has no key."""

    def key(node: Any) -> int:
        return position[node]

    return key


def _accumulate(store: dict, key: Any, node: Any, contribution: Any) -> None:
    """Add ``contribution`` to the adjoint ``store`` holds for ``node``'s slot."""
    slot = key(node)
    held = store.get(slot)
    store[slot] = contribution if held is None else _ir.make("add", (held, contribution))


def _sets_bits(node: Any) -> bool:
    """Whether ``node`` is an OR-scatter, or a loop whose body holds one."""
    if _ir.is_a(node, "loop"):
        return any(_sets_bits(s) for s in node.field_nodes("body_sinks"))
    return _ir.sets_bits(node)


def _sinks_of(primal: Any) -> tuple[Any, ...]:
    sinks = tuple(getattr(primal, "sinks", primal))
    if not sinks or not all(_ir.is_a(s, "sink", "loop") for s in sinks):
        raise HawkError(
            f"{primal!r} is not a kernel or a sink set — a transform differentiates "
            "an IR whose roots are its sinks"
        )
    # Termination is DISCRETE: a primal's finish statement has no derivative,
    # so the derived kernel binds no counter, writes no mask and carries no
    # sidecar ``finish``; it keeps the primal's mask READ (``_mask_termination``).
    kept = _ir.expand_stores([s for s in sinks if not _ir.is_a(s, "finish")])
    if any(_sets_bits(s) for s in kept):
        raise HawkError(
            f"{primal!r} sets bits by key (`|=`): a bit mask is "
            "discrete and has no derivative; differentiate a kernel that does not "
            "write it"
        )
    if not kept:
        raise HawkError(
            f"{primal!r} only finishes its sample — it commits no plane to differentiate"
        )
    return kept


def _rule(node: Any):
    kind = node.field_text("kind")
    rule = rules.rule_for(kind, len(_ir.type_of(node).shape))
    if rule.vjp is None and rule.jvp is None:
        raise HawkError(
            f"node kind {kind!r} carries no derivative rule ({rule.mode}); "
            "it is differentiated by the transform itself, not by the table"
        )
    return rule


# --------------------------------------------------------------------------
# inputs, seeds and derived names


def _wrt(seq: Iterable[Any], wrt: Sequence[str] | None) -> tuple[Any, ...]:
    """The leaves a derivative is taken with respect to, in traversal order."""
    for node in seq:
        if node.field_text("kind") == _ir.RAW_KIND:
            raise HawkError(
                "a spliced @raw_device block cannot be differentiated: its "
                "text is opaque to the IR, so no rule can exist for it"
            )
    leaves = [
        n
        for n in seq
        if _ir.is_a(n, "leaf") and _ir.type_of(n).dtype in _DIFFERENTIABLE
    ]
    if wrt is None:
        return tuple(leaves)
    wanted = tuple(wrt)
    integers = sorted(
        {
            n.field_text("name")
            for n in seq
            if _ir.is_a(n, "leaf")
            and n.field_text("name") in wanted
            and _ir.type_of(n).dtype in _ir.INT_PLANE_DTYPES
        }
    )
    if integers:
        raise HawkError(
            f"wrt names {integers}: an integer plane (int32/uint32) carries no "
            "derivative; drop it from wrt= (a float computed from it through "
            "hawk.math.real is differentiated with respect to its float inputs)"
        )
    unknown = [
        w for w in wanted if not any(leaf.field_text("name") == w for leaf in leaves)
    ]
    if unknown:
        raise HawkError(
            f"wrt names {unknown} are not differentiable inputs of this kernel; its "
            f"inputs are {[leaf.field_text('name') for leaf in leaves]}"
        )
    return tuple(leaf for leaf in leaves if leaf.field_text("name") in wanted)


def _taken(seq: Iterable[Any]) -> set[str]:
    """The names of the planes ``seq`` binds: its leaves and its sinks."""
    return {n.field_text("name") for n in seq if _ir.is_a(n, "leaf", "sink")}


def _named(sinks: Iterable[Any], inputs: Iterable[Any]) -> list[str]:
    """The primal planes a derivative will name: its sinks (a lowered loop has none
    and is refused later) and its inputs."""
    return [
        n.field_text("name") for n in (*sinks, *inputs) if _ir.is_a(n, "leaf", "sink")
    ]


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

    def __init__(self, prefix: str, taken: set[str], names: Iterable[str] = ()) -> None:
        self.prefix = prefix
        self.taken = taken
        #: ``{plane: column count}`` of the compact planes of a broadcast kernel
        #: (:func:`_compact_counts`)
        self.counts: dict = {}
        self.source: dict[str, str] = {}
        flat = [n.replace(_ir.NAMESPACE_SEP, "_") for n in names]
        self.level = 1
        while any(self._spell(self.level, f) in taken for f in flat):
            self.level += 1

    def _spell(self, level: int, flat: str) -> str:
        return f"{self.prefix}{level if level > 1 else ''}_{flat}"

    def __call__(self, name: str) -> str:
        """The derived plane name for one primal plane."""
        flat = name.replace(_ir.NAMESPACE_SEP, "_")
        derived = self._spell(self.level, flat)
        while derived in self.taken:  # a plane this derivative was not told of
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
    "and take the {op} on the host, or reduce with a sum"
)


def _seed(sink: Any, name: _Namer) -> Any:
    """The incoming adjoint plane of one primal sink (the four sink kinds)."""
    derived, t = name(sink.field_text("name")), _ir.type_of(sink)
    context = sink.ctx
    if _ir.is_a(sink, "sample_sum"):
        if sink.field_text("op") != "sum":
            raise HawkError(
                _TIE_TEXT.format(name=sink.field_text("name"), op=sink.field_text("op"))
            )
        # the reverse of a sum over samples is a broadcast of its adjoint
        return _ir.leaf(context, "uniform", "uniform", derived, t)
    indices = _indices_of(sink)
    if _ir.is_a(sink, "update") and indices:
        # the transpose of a scatter-add is a gather of the adjoint plane, at the same indices under the same edge rule
        plane = _ir.leaf(context, "table_read", "lookup", derived, t, axes=_ir.field_pairs(sink, "axes"))
        return _ir.make_node("At", t, [plane, *indices], edge=sink.field_text("edge"))
    return _ir.leaf(context, "vocab_read", "plane_in", derived, t)


def _tangent_sink(sink: Any, tangent: Any, name: _Namer, indices: Any = None) -> Any:
    """The forward-mode twin of one primal sink, same kind and same lane
    (at ``indices`` when the alternate path moved the key)."""
    ttype = _ir.type_of(sink)
    value = tangent if tangent is not None else _ir.zero_like(sink.ctx, ttype)
    derived = name(sink.field_text("name"))
    indices = _indices_of(sink) if indices is None else indices
    if _ir.is_accumulate(sink):
        return _scatter(
            derived, value, indices, ttype, _ir.operation_fields(sink), sink.field_text("edge"),
            _ir.field_pairs(sink, "axes"),
        )
    if _ir.is_store(sink):
        return _ir.make_node(
            "Update", ttype, [value, *indices], name=derived, op="store", edge=sink.field_text("edge"),
            axes=_ir.field_pairs(sink, "axes"),
        )
    if _ir.is_a(sink, "sample_sum"):
        if sink.field_text("op") != "sum":
            raise HawkError(
                _TIE_TEXT.format(name=sink.field_text("name"), op=sink.field_text("op"))
            )
        return _ir.make_node(
            "SampleSum", ttype, [value], name=derived, op="sum", cell=derived, columns=1
        )
    return _ir.make_node("Assign", ttype, [value], name=derived)


# --------------------------------------------------------------------------
# accumulation and un-broadcasting
