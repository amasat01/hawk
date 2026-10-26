# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``jump=`` through a branch that decides a future: the alternate path.

The jump rule of :mod:`hawk.diff.jump` hands ``K(d) (a - b)`` to the operands
of a switching comparison and lets the TAKEN path carry it on. That is the
derivative of the kernel-smoothed loss only while the switched value reaches
the outputs affinely. When it reaches another comparison, a select condition,
a ``break`` or a nonlinear op, the loss on the other side of the switch is not
the taken path's linearisation, so the plan is ENGAGED (:func:`engaged`) and
the derivative is taken by simulation instead:

* every live comparison instance (a site in a given loop iteration) has the
  rate ``rho = K(d)``; ``S`` is the sum of the rates over the sample, found by
  a first sweep that is a clone of the primal;
* one instance is drawn with probability ``rho / S`` (``U ~ uniform(0, S)``
  from the counter RNG, keyed on the sample and ``jump_seed``) and the ALTERNATE
  path is the primal with that comparison negated, evaluated naturally
  afterwards;
* ``jvp = delta + G (out_alt - out)`` with ``delta`` the ordinary piecewise
  tangent and ``G = S sigma phi d'`` at the drawn instance (``sigma`` the
  comparison's side, ``phi = -1`` when it was true, ``d'`` the tangent of
  ``l - r``). Averaged over ``U`` this is ``sum_i K(d_i) d'_i (L(flip_i) - L)``.

The alternate equals the primal until the fork, so the second sweep runs
primal, tangent and alternate in lockstep, with ``(cum, G)`` carried: a
comparison of the alternate is ``c_alt xor fork_here`` for
``fork_here = cum < U <= cum + rho``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..ir import (
    Const,
    Dispatch,
    HawkError,
    Leaf,
    Loop,
    LoopCarry,
    LoopIndex,
    LoopValue,
    Node,
    SampleIndex,
    Select,
    Sink,
    TapeRead,
    build_loop,
)
from ..ir import make as _mk
from ..ir.nodes import AccumWrite, At, ReduceSamples, WideWrite
from ..ir.slotwise import slotwise
from ..ir.walk import _rebuild
from ..types import INT_PLANE_DTYPES, TensorType
from .jump import _F64, _I32, _ORDERED, SEED, JumpPlan
from .loops import _TERMINALS, _post_order
from .rules import zero_like

#: The counter of that draw: a stream a kernel's own ``random_uniform`` calls
#: do not use.
STREAM = 0x4A554D50
_BOOL = TensorType((), "bool")
#: Ops a tainted value passes through affinely.
_AFFINE = ("add", "sub", "neg", "vec", "component", "component_at", "splat",
           "sum", "set_component_at", "add_at")
_BILINEAR = ("mul", "dot")
_SIGHT = (Leaf, Const, SampleIndex, LoopIndex, LoopCarry)


# --------------------------------------------------------------------------
# engagement: where a switched value goes


def _step(node: Node, get, live) -> tuple:
    """``(tainted, engages)`` for ``node``: whether it carries a switched value
    and whether it feeds one into control or a nonlinear op. ``live(cond)`` says
    a select's condition switches."""
    if isinstance(node, _SIGHT):
        return False, False
    ops = node.operands
    if isinstance(node, Select):
        cond, a, b = ops
        if get(cond):
            return True, True
        return live(cond) or get(a) or get(b), False
    if isinstance(node, Dispatch):
        if get(ops[0]):
            return True, True
        return any(get(b) for b in node.branches), False
    if isinstance(node, At):
        return get(ops[0]), get(ops[1])
    tainted = [get(o) for o in ops]
    if not any(tainted):
        return False, False
    kind = node.kind
    if kind in _AFFINE:
        return True, False
    if kind in _BILINEAR:
        return True, sum(tainted) > 1
    if kind == "div":
        return True, bool(tainted[1])
    return True, True


def engaged(plan: JumpPlan, seq: Sequence[Node]) -> bool:
    """Whether a live select's value reaches a comparison, a select condition,
    a ``break`` or a non-affine op (a product of two switched values counts)."""
    taint: dict = {}
    slots: dict = {}

    def get(n: Node) -> bool:
        return taint.get(id(n), False)

    for node in seq:
        if isinstance(node, Loop):
            carried, hit = _loop_taint(plan, node, get)
            if hit:
                return True
            slots[id(node)] = carried
        elif isinstance(node, LoopValue):
            taint[id(node)] = bool(slots[id(node.loop)][node.slot])
        elif not isinstance(node, Sink):
            tainted, hit = _step(node, get, plan.is_live)
            if hit:
                return True
            taint[id(node)] = tainted
    return False


def _loop_taint(plan: JumpPlan, loop: Loop, outer) -> tuple:
    """``(switched per carried slot, engages)`` for one loop, the carries'
    taint iterated to a fixed point."""
    roots = tuple(loop.nexts) + tuple(loop.exit_values) + (
        () if loop.exit_cond is None else (loop.exit_cond,))
    body = plan.body(loop, roots)
    live = body.is_live if body is not None else (lambda n: False)
    stop = {id(o) for o in loop.operands} | {id(loop.index)}
    order = _post_order(roots, stop)
    slot = {id(c): j for j, c in enumerate(loop.carries)}
    carried = [outer(i) for i in loop.inits]
    for _ in range(len(carried) + 1):
        taint: dict = {}

        def get(n: Node) -> bool:
            if id(n) in slot:
                return carried[slot[id(n)]]
            return taint.get(id(n), outer(n) if id(n) in stop else False)

        for node in order:
            if id(node) in stop or id(node) in slot or isinstance(node, _TERMINALS):
                continue
            tainted, hit = _step(node, get, live)
            if hit:
                return carried, True
            taint[id(node)] = tainted
        new = [c or get(n) for c, n in zip(carried, loop.nexts)]
        if new == carried:
            break
        carried = new
    return carried, False


# --------------------------------------------------------------------------
# the alternate clone


def _alt_pass(order: Sequence[Node], mapping: dict, wrap: dict) -> dict:
    """``{id(node): alternate}`` over ``order`` (operands first): a node whose
    operands changed is rebuilt, one in ``wrap`` is passed through its function
    after that. Unchanged nodes stay shared with the primal."""
    memo = dict(mapping)
    for node in order:
        if id(node) in memo or not node.operands:
            continue
        new = [memo.get(id(c), c) for c in node.operands]
        out = (node if all(a is b for a, b in zip(new, node.operands))
               else _rebuild(node, new))
        if id(node) in wrap:
            out = wrap[id(node)](out)
        if out is not node:
            memo[id(node)] = out
    return memo


def _and(a, b: Node) -> Node:
    return b if a is True else _mk("land", (a, b))


def _guards(order: Sequence[Node], roots: Sequence[Node], stop=()) -> dict:
    """``{id(comparison): guard}`` over ``order`` (operands first): the
    condition under which the comparison's outcome reaches a root, ``None`` when
    always. A select passes its own to a branch under that branch's condition;
    ``and``/``or`` pass it to an operand under the other operand's value."""
    sel: dict = {id(r): True for r in roots}
    for node in reversed(order):
        held = sel.get(id(node))
        if held is None or id(node) in stop:
            continue
        ops = node.operands
        if isinstance(node, Select):
            cond, a, b = ops
            given = ((cond, held), (a, _and(held, cond)),
                     (b, _and(held, _mk("lnot", (cond,)))))
        elif node.kind in ("land", "lor"):
            first, second = ops
            by = (lambda o: o) if node.kind == "land" else (lambda o: _mk("lnot", (o,)))
            given = ((first, _and(held, by(second))), (second, _and(held, by(first))))
        else:
            given = tuple((o, held) for o in ops)
        for o, g in given:
            was = sel.get(id(o))
            sel[id(o)] = True if was is True or g is True else (
                g if was is None else _mk("lor", (was, g)))
    return {id(n): (None if sel.get(id(n), True) is True else sel[id(n)])
            for n in order if n.kind in _ORDERED}


def _flip(hit: Node):
    return lambda cond: Select(hit, _mk("lnot", (cond,)), cond, _BOOL)


def _sum(terms: Sequence[Node]) -> Node:
    out = terms[0]
    for term in terms[1:]:
        out = _mk("add", (out, term))
    return out


def _zero() -> Const:
    return Const(0.0, _F64)


@dataclass(frozen=True)
class Parts:
    """What :func:`hawk.diff.loops.forward_loop` hands :meth:`AltPath.joint`:
    the augmented loop's pieces before it is built."""

    carries: tuple
    dots: tuple
    inits: tuple
    nexts: tuple
    dot_nexts: tuple
    exits: tuple
    dot_exits: tuple
    tangent: object


class _Body:
    """One loop's switching comparisons and its first sweep."""

    def __init__(self, sites: tuple, sweep: Loop | None) -> None:
        self.sites, self.sweep = sites, sweep


class AltPath:
    """The alternate-path derivative of one engaged :class:`JumpPlan`."""

    @classmethod
    def of(cls, plan: JumpPlan | None, seq: Sequence[Node],
           reverse: bool = False) -> AltPath | None:
        if plan is None or not engaged(plan, seq):
            return None
        plan.engaged = True
        return cls(plan, seq, reverse)

    def __init__(self, plan: JumpPlan, seq: Sequence[Node], reverse: bool = False) -> None:
        self.plan = plan
        #: reverse mode: ``top`` holds the straight-line sites' injections,
        #: ``tapes`` each joint loop's per-iteration injection vector, ``g``
        #: the seed ``sum bar (alt - out)``
        self.reverse = reverse
        self.top: list = []
        self.tapes: dict = {}
        self.alt_at: dict = {}
        self.g: Node | None = None
        self.amap: dict = {}
        self.joints: dict = {}
        self.seed = Leaf("uniform", "uniform", SEED, _I32)
        top = [n for n in seq if n.kind in _ORDERED and plan.is_live(n)
               and not isinstance(n, Loop)]
        self.sites = tuple(top)
        self.gamma = _guards(seq, [n for n in seq if isinstance(n, Sink)])
        self.bodies: dict = {}
        offset = [_sum([self.rate(s) for s in top])] if top else [_zero()]
        self.cum0: dict = {}
        totals = list(offset)
        for node in seq:
            if isinstance(node, Loop):
                body = self._body(node)
                self.bodies[id(node)] = body
                if body.sweep is not None:
                    self.cum0[id(node)] = _sum(totals)
                    totals.append(LoopValue(body.sweep, node.carry_count))
        self.total = _sum(totals)
        uniform = _mk("random_uniform", (self.seed, Const(STREAM, _I32)))
        self.fork = _mk("mul", (uniform, self.total))
        self.cum = _zero()
        self.gains: list = []

    # -- sites and rates ----------------------------------------------------
    def _body(self, loop: Loop) -> _Body:
        for node in loop.body_nodes():
            if isinstance(node, Loop):
                raise HawkError(
                    f"jump= in kernel {self.plan.kernel!r}: the alternate path "
                    "through a loop holding a nested `for` is not built")
        if self.reverse and any(isinstance(n, Dispatch) for n in loop.body_nodes()):
            raise HawkError(
                f"jump= in kernel {self.plan.kernel!r}: a `dispatch` inside a loop "
                "body of a `jump=` kernel differentiated in REVERSE: the switching "
                "comparisons of the plan are not applied under a branch; this is "
                "not built")
        exit_roots = () if loop.exit_cond is None else (loop.exit_cond,
                                                      *loop.exit_values)
        roots = tuple(loop.nexts) + exit_roots
        body = self.plan.body(loop, roots)
        if body is None:
            return _Body((), None)
        stop = {id(o) for o in loop.operands} | {id(loop.index)}
        order = _post_order(roots, stop)
        self.gamma.update(_guards(order, roots, stop))
        sites = tuple(n for n in order if n.kind in _ORDERED and body.is_live(n))
        if not sites:
            return _Body((), None)
        if loop.exit_form == "middle":
            raise HawkError(
                f"jump= in kernel {self.plan.kernel!r}: a `break` in the middle "
                "of a loop body with a switching comparison: the alternate "
                "path needs the exit at the head or the tail of the body")
        return _Body(sites, self._sweep(loop, sites))

    def rate(self, site: Node, guard=False) -> Node:
        """``K(d)`` of one comparison, exactly zero outside its support and
        where its outcome reaches no output (``guard``: its condition, else the
        one :func:`_guards` found)."""
        k = self.plan._supported(site, lambda kern: kern, 1.0)
        guard = self.gamma.get(id(site)) if guard is False else guard
        return k if guard is None else Select(guard, k, _zero(), _F64)

    def _sweep(self, loop: Loop, sites: tuple) -> Loop:
        """The first sweep: the primal's loop plus the sum of the rates."""
        m = loop.carry_count
        acc = LoopCarry(loop.index.depth, m, "jump_rate", _F64)
        step = _mk("add", (acc, _sum([self.rate(s) for s in sites])))
        extra: dict = {}
        if loop.exit_cond is not None:
            held = acc if loop.exit_form == "head" else step
            extra = {"exit_cond": loop.exit_cond,
                     "exit_values": tuple(loop.exit_values) + (held,)}
        return build_loop(loop.index, (*loop.carries, acc), (*loop.inits, _zero()),
                          (*loop.nexts, step), start=loop.start, stop=loop.stop,
                          step=loop.step, **extra)

    def weight(self, site: Node, t) -> Node | None:
        """``S sigma phi d'`` of one comparison, ``None`` without a tangent;
        in reverse mode (``t`` is ``None``) the part without ``d'``."""
        side = _ORDERED[site.kind]
        scale = Select(site, Const(-side, _F64), Const(side, _F64), _F64)
        base = _mk("mul", (self.total, scale))
        if t is None:
            return base
        lhs, rhs = site.operands
        tl, tr = t(lhs), t(rhs)
        if tl is None and tr is None:
            return None
        diff = (_mk("neg", (tr,)) if tl is None
                else tl if tr is None else _mk("sub", (tl, tr)))
        return _mk("mul", (base, diff))

    def _fork_here(self, cum: Node, rate: Node) -> tuple:
        """``(cum < U <= cum + rate, cum + rate)``."""
        after = _mk("add", (cum, rate))
        hit = _mk("land", (_mk("lt", (cum, self.fork)), _mk("ge", (after, self.fork))))
        return hit, after

    # -- straight-line nodes ---------------------------------------------------
    def visit(self, node: Node, t) -> None:
        """Record ``node``'s alternate, operands first (``t`` is the ordinary
        tangent)."""
        if isinstance(node, LoopValue):
            joint = self.joints.get(id(node.loop))
            if joint is not None:
                self.amap[id(node)] = LoopValue(joint, self.alt_at[id(node.loop)] + node.slot)
            return
        if isinstance(node, Loop):
            if self.reverse and self.affects(node):
                self.joint(node, Parts(tuple(node.carries), (), tuple(node.inits),
                                       tuple(node.nexts), (), tuple(node.exit_values),
                                       (), None))
            return
        new = [self.amap.get(id(o), o) for o in node.operands]
        changed = any(a is not b for a, b in zip(new, node.operands))
        if isinstance(node, ReduceSamples) and changed:
            raise HawkError(
                f"jump= in kernel {self.plan.kernel!r}: a read across samples "
                "downstream of a switching comparison cannot run on the "
                "alternate path of one sample")
        out = _rebuild(node, new) if changed else node
        if any(node is s for s in self.sites):
            hit, self.cum = self._fork_here(self.cum, self.rate(node))
            g = self.weight(node, t)
            if g is not None:
                (self.top if self.reverse else self.gains).append(
                    (node, Select(hit, g, _zero(), _F64)) if self.reverse
                    else Select(hit, g, _zero(), _F64))
            out = _flip(hit)(out)
        if out is not node:
            self.amap[id(node)] = out

    def affects(self, loop: Loop) -> bool:
        return (id(loop) in self.cum0
                or any(id(o) in self.amap for o in loop.operands))

    # -- the joint loop -----------------------------------------------------------
    def joint(self, loop: Loop, parts: Parts) -> Loop:
        """The augmented loop with the alternate beside the primal and tangent:
        slots ``[0, m)`` primal, ``[m, 2m)`` tangent, ``[2m, 3m)`` alternate,
        then ``cum`` and ``G`` (and, for a ``break``, a done flag each for the
        primal and the alternate)."""
        m, depth = loop.carry_count, loop.index.depth
        body = self.bodies[id(loop)]
        nb = m + len(parts.dots)
        alts = tuple(LoopCarry(depth, nb + j, f"alt_{c.name}", c.ttype)
                     for j, c in enumerate(parts.carries))
        cum = LoopCarry(depth, nb + m, "jump_cum", _F64)
        reverse = parts.tangent is None
        # forward mode also carries the gain G; reverse mode re-draws the fork off
        # the taped ``cum`` of each iteration instead (:meth:`inject`)
        gain = () if reverse else (LoopCarry(depth, nb + m + 1, "jump_gain", _F64),)
        brk = loop.exit_cond is not None
        first = nb + m + 1 + len(gain)
        flags = (LoopCarry(depth, first, "jump_done", _F64),
                 LoopCarry(depth, first + 1, "jump_done_alt", _F64)) if brk else ()
        half = Const(0.5, _F64)
        done = [_mk("gt", (f, half)) for f in flags]
        active = None
        if brk:
            active = _mk("lnot", (done[0],))
            if loop.exit_form == "head":
                active = _mk("land", (active, _mk("lnot", (loop.exit_cond,))))
        # the fork, along the primal's comparisons
        c, terms, hits = cum, [], {}
        for site in body.sites:
            rate = self.rate(site)
            if active is not None:
                rate = Select(active, rate, _zero(), _F64)
            hit, after = self._fork_here(c, rate)
            hits[id(site)] = _flip(hit)
            g = self.weight(site, parts.tangent)
            if g is not None and not reverse:
                terms.append(Select(hit, g, _zero(), _F64))
            c = after
        gain_next = ()
        if gain:
            gain_next = (_mk("add", (gain[0], _sum(terms))) if terms else gain[0],)
        # the alternate body
        mapping = {id(p): a for p, a in zip(parts.carries, alts)}
        mapping.update({id(o): self.amap[id(o)] for o in loop.operands
                        if id(o) in self.amap})
        stop = {id(o) for o in loop.operands} | {id(loop.index)} | {id(p) for p in parts.carries}
        roots = parts.nexts + parts.exits + ((loop.exit_cond,) if brk else ())
        memo = _alt_pass(_post_order(roots, stop), mapping, hits)

        def alt(n: Node) -> Node:
            return memo.get(id(n), n)

        a_nexts = tuple(alt(n) for n in parts.nexts)
        p_nexts, d_nexts = parts.nexts, parts.dot_nexts
        extra: dict = {}
        flag_nexts: tuple = ()
        if brk:
            ep, ea = loop.exit_cond, alt(loop.exit_cond)

            def freeze(is_done, now, held, at_exit, ahead):
                return Select(is_done, held, Select(now, at_exit, ahead, held.ttype),
                              held.ttype)

            p_nexts = tuple(freeze(done[0], ep, p, e, n) for p, e, n in
                            zip(parts.carries, parts.exits, parts.nexts))
            d_nexts = tuple(freeze(done[0], ep, d, e, n) for d, e, n in
                            zip(parts.dots, parts.dot_exits, parts.dot_nexts))
            a_nexts = tuple(freeze(done[1], ea, a, alt(e), n) for a, e, n in
                            zip(alts, parts.exits, a_nexts))
            stopped = (_mk("lor", (done[0], ep)), _mk("lor", (done[1], ea)))
            flag_nexts = tuple(Select(s, Const(1.0, _F64), _zero(), _F64) for s in stopped)
            extra["exit_cond"] = _mk("land", stopped)
        nexts = (*p_nexts, *d_nexts, *a_nexts, c, *gain_next, *flag_nexts)
        if brk:
            extra["exit_values"] = nexts
        inits = (*parts.inits, *(self.amap.get(id(i), i) for i in loop.inits),
                 self.cum0.get(id(loop), _zero()), *(_zero() for _ in gain),
                 *(_zero() for _ in flags))
        rooted = slotwise((*nexts, *extra.get("exit_values", ()),
                           *([extra["exit_cond"]] if brk else [])))
        n = len(nexts)
        if brk:
            extra["exit_values"], extra["exit_cond"] = rooted[n:2 * n], rooted[-1]
        built = build_loop(loop.index, (*parts.carries, *parts.dots, *alts, cum, *gain, *flags),
                           inits, rooted[:n], start=loop.start, stop=loop.stop,
                           step=loop.step, **extra)
        self.joints[id(loop)] = built
        self.alt_at[id(loop)] = nb
        if reverse:
            if body.sites:
                self.tapes[id(loop)] = (built, nb + m)
        else:
            self.gains.append(LoopValue(built, nb + m + 1))
        return built

    # -- reverse mode ---------------------------------------------------------------
    def seed_gain(self, sinks: Sequence[Node], seed) -> Node | None:
        """``g = sum bar (alt - out)`` over the sinks (``seed(sink)`` is the
        incoming adjoint), ``None`` when no output moves on the alternate."""
        terms = []
        for sink in sinks:
            if sink.ttype.dtype in INT_PLANE_DTYPES:
                continue
            bar = seed(sink)
            alt_value = self.amap.get(id(sink.value), sink.value)
            keyed = isinstance(sink, (AccumWrite, WideWrite)) and sink.index is not None
            alt_index = self.amap.get(id(sink.index), sink.index) if keyed else None
            if keyed and alt_index is not sink.index:
                there = At(bar.operands[0], alt_index, bar.ttype)
                term = _mk("sub", (_mk("mul", (there, alt_value)),
                                   _mk("mul", (bar, sink.value))))
            elif alt_value is not sink.value:
                term = _mk("mul", (bar, _mk("sub", (alt_value, sink.value))))
            else:
                continue
            terms.append(_mk("sum", (term,)) if term.ttype.shape else term)
        return _sum(terms) if terms else None

    def injections(self) -> tuple:
        """``(site, weight)`` of the straight-line sites, ``weight`` already
        times ``g``."""
        if self.g is None:
            return ()
        return tuple((site, _mk("mul", (coef, self.g))) for site, coef in self.top)

    def sites_of(self, loop: Loop) -> tuple:
        got = self.tapes.get(id(loop))
        return self.bodies[id(loop)].sites if got is not None else ()

    def inject(self, loop: Loop, sites: Sequence[Node], roots: Sequence[Node],
               stop) -> tuple:
        """``(site, weight)`` for the reverse body's ``sites`` (the forward
        ones, in order, rewritten onto the tape): the fork is drawn again from
        the ``cum`` the joint loop held entering this iteration, and the weight
        is the site's ``S sigma phi`` where it falls, times ``g``. The primal
        ran every iteration a reverse loop visits, so no site is masked."""
        got = self.tapes.get(id(loop))
        if got is None or self.g is None:
            return ()
        built, slot = got
        cum, out = TapeRead(built, slot, loop.index), []
        guard = _guards(_post_order(roots, stop), roots, stop)
        for site in sites:
            hit, after = self._fork_here(cum, self.rate(site, guard[id(site)]))
            out.append((site, _mk("mul", (Select(hit, self.weight(site, None), _zero(), _F64),
                                           self.g))))
            cum = after
        return tuple(out)

    def _events(self):
        """``f(value)``: ``value`` where the sample has a switching instance
        (``S > 0``), else zero: a sample without one runs the plain tangent."""
        selector = Select(_mk("gt", (self.total, _zero())), Const(0, _I32),
                          Const(1, _I32), _I32)
        return lambda value: Dispatch(selector, (value, zero_like(value.ttype)),
                                      "switch", value.ttype)

    # -- the sinks ----------------------------------------------------------------
    def sinks(self, sinks: Sequence[Node], tangent, derived) -> list:
        """The tangent sinks of ``sinks``: ``delta + G (alt - out)`` per value
        (a keyed scatter commits the alternate beside the primal at its own
        key). ``tangent(node)`` is the ordinary tangent and ``derived`` mints
        the sink."""
        gain = _sum(self.gains) if self.gains else None
        events = self._events()
        out: list = []
        for sink in sinks:
            if sink.ttype.dtype in INT_PLANE_DTYPES:
                continue
            value = tangent(sink.value)
            plain = value if value is not None else zero_like(sink.ttype)
            alt_value = self.amap.get(id(sink.value), sink.value)
            keyed = isinstance(sink, (AccumWrite, WideWrite)) and sink.index is not None
            alt_index = self.amap.get(id(sink.index), sink.index) if keyed else None
            moved = keyed and alt_index is not sink.index
            if gain is None or (alt_value is sink.value and not moved):
                out.append(derived(sink, plain))
            elif moved:
                out.append(derived(sink, plain))
                out.append(derived(sink, events(_mk("mul", (gain, alt_value))), alt_index))
                out.append(derived(sink, events(_mk("neg", (_mk("mul", (gain, sink.value)),)))))
            else:
                diff = _mk("sub", (alt_value, sink.value))
                out.append(derived(sink, _mk("add", (plain, events(_mk("mul", (gain, diff)))))))
        return out
