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

from .. import _ir
from .._errors import HawkError
from .jump import (
    _F64,
    _I32,
    _ORDERED,
    SEED,
    JumpPlan,
    _switch,
)
from .loops import _TERMINALS, _bounds, _index_of, _post_order, _select
from .tape import _carry

#: The counter of that draw: a stream a kernel's own ``random_uniform`` calls
#: do not use.
STREAM = 0x4A554D50
_BOOL = _ir.TensorType((), "bool")
#: Ops a tainted value passes through affinely.
_AFFINE = (
    "add",
    "sub",
    "neg",
    "vec",
    "component",
    "component_at",
    "splat",
    "sum",
    "set_component_at",
    "add_at",
)
_BILINEAR = ("mul", "dot")
_SIGHT = ("leaf", "const", "sample_index", "plane_extent", "loop_index", "loop_carry")


# --------------------------------------------------------------------------
# engagement: where a switched value goes


def _step(node, get, live) -> tuple:
    """``(tainted, engages)`` for ``node``: whether it carries a switched value
    and whether it feeds one into control or a nonlinear op. ``live(cond)`` says
    a select's condition switches."""
    if _ir.is_a(node, *_SIGHT):
        return False, False
    ops = node.operands
    if _ir.is_a(node, "select"):
        cond, a, b = ops
        if get(cond):
            return True, True
        return live(cond) or get(a) or get(b), False
    if _ir.is_a(node, "dispatch"):
        if get(ops[0]):
            return True, True
        return any(get(b) for b in ops[1:]), False
    if _ir.is_a(node, "at"):
        return get(ops[0]), get(ops[1])
    tainted = [get(o) for o in ops]
    if not any(tainted):
        return False, False
    kind = node.field_text("kind")
    if kind in _AFFINE:
        return True, False
    if kind in _BILINEAR:
        return True, sum(tainted) > 1
    if kind == "div":
        return True, bool(tainted[1])
    return True, True


def engaged(plan: JumpPlan, seq: Sequence) -> bool:
    """Whether a live select's value reaches a comparison, a select condition,
    a ``break`` or a non-affine op (a product of two switched values counts)."""
    taint: dict = {}
    slots: dict = {}

    def get(n) -> bool:
        return taint.get(n, False)

    for node in seq:
        if _ir.is_a(node, "loop"):
            carried, hit = _loop_taint(plan, node, get)
            if hit:
                return True
            slots[node] = carried
        elif _ir.is_a(node, "loop_value"):
            loop, slot = _ir.loop_value_of(node)
            taint[node] = bool(slots[loop][slot])
        elif not _ir.is_a(node, "sink"):
            tainted, hit = _step(node, get, plan.is_live)
            if hit:
                return True
            taint[node] = tainted
    return False


def _loop_taint(plan: JumpPlan, loop, outer) -> tuple:
    """``(switched per carried slot, engages)`` for one loop, the carries'
    taint iterated to a fixed point."""
    nexts = loop.field_nodes("nexts")
    exits = _ir.exit_kwargs(loop)
    roots = (*nexts, *exits["exit_values"])
    if exits["exit_cond"] is not None:
        roots += (exits["exit_cond"],)
    body = plan.body(loop, roots)
    live = body.is_live if body is not None else (lambda n: False)
    stop = {*loop.operands, _index_of(loop)}
    order = _post_order(roots, stop)
    slot = {c: j for j, c in enumerate(loop.field_nodes("carries"))}
    carried = [outer(i) for i in loop.field_nodes("inits")]
    for _ in range(len(carried) + 1):
        taint: dict = {}

        def get(n) -> bool:
            if n in slot:
                return carried[slot[n]]
            return taint.get(n, outer(n) if n in stop else False)

        for node in order:
            if node in stop or node in slot or _ir.is_a(node, *_TERMINALS):
                continue
            tainted, hit = _step(node, get, live)
            if hit:
                return carried, True
            taint[node] = tainted
        new = [c or get(n) for c, n in zip(carried, nexts)]
        if new == carried:
            break
        carried = new
    return carried, False


# --------------------------------------------------------------------------
# the alternate clone


def _alt_pass(order: Sequence, mapping: dict, wrap: dict) -> dict:
    """``{node: alternate}`` over ``order`` (operands first): a node whose
    operands changed is rebuilt, one in ``wrap`` is passed through its function
    after that. Unchanged nodes stay shared with the primal."""
    memo = dict(mapping)
    for node in order:
        if node in memo or not node.operands:
            continue
        new = [memo.get(c, c) for c in node.operands]
        out = node if all(a == b for a, b in zip(new, node.operands)) else _ir.rebuild(node, new)
        if node in wrap:
            out = wrap[node](out)
        if out != node:
            memo[node] = out
    return memo


def _and(a, b):
    return b if a is True else _ir.make("land", (a, b))


def _guards(order: Sequence, roots: Sequence, stop=()) -> dict:
    """``{comparison: guard}`` over ``order`` (operands first): the
    condition under which the comparison's outcome reaches a root, ``None`` when
    always. A select passes its own to a branch under that branch's condition;
    ``and``/``or`` pass it to an operand under the other operand's value."""
    sel: dict = {r: True for r in roots}
    for node in reversed(order):
        held = sel.get(node)
        if held is None or node in stop:
            continue
        ops = node.operands
        if _ir.is_a(node, "select"):
            cond, a, b = ops
            given = (
                (cond, held),
                (a, _and(held, cond)),
                (b, _and(held, _ir.make("lnot", (cond,)))),
            )
        elif node.field_text("kind") in ("land", "lor"):
            first, second = ops
            by = (
                (lambda o: o)
                if node.field_text("kind") == "land"
                else (lambda o: _ir.make("lnot", (o,)))
            )
            given = ((first, _and(held, by(second))), (second, _and(held, by(first))))
        else:
            given = tuple((o, held) for o in ops)
        for o, g in given:
            was = sel.get(o)
            sel[o] = (
                True
                if was is True or g is True
                else (g if was is None else _ir.make("lor", (was, g)))
            )
    return {
        n: (None if sel.get(n, True) is True else sel[n])
        for n in order
        if n.field_text("kind") in _ORDERED
    }


def _flip(hit):
    """Return the function negating a comparison where ``hit`` holds."""
    return lambda cond: _select(hit, _ir.make("lnot", (cond,)), cond)


def _sum(terms: Sequence):
    out = terms[0]
    for term in terms[1:]:
        out = _ir.make("add", (out, term))
    return out


def _zero(context):
    """Return the constant ``0.0`` of ``context``."""
    return _ir.const(context, 0.0, _F64)


def _keyed_index(sink):
    """Return the key a keyed scatter commits at, or ``None``."""
    if _ir.is_a(sink, "accum_write", "wide_write") and len(sink.operands) > 1:
        return sink.operands[1]
    return None


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

    def __init__(self, sites: tuple, sweep) -> None:
        self.sites, self.sweep = sites, sweep


class AltPath:
    """The alternate-path derivative of one engaged :class:`JumpPlan`."""

    @classmethod
    def of(cls, plan: JumpPlan | None, seq: Sequence, reverse: bool = False) -> AltPath | None:
        if plan is None or not engaged(plan, seq):
            return None
        plan.engaged = True
        return cls(plan, seq, reverse)

    def __init__(self, plan: JumpPlan, seq: Sequence, reverse: bool = False) -> None:
        self.plan = plan
        #: the context every node the plan builds belongs to
        self.ctx = seq[0].ctx
        #: reverse mode: ``top`` holds the straight-line sites' injections,
        #: ``tapes`` each joint loop's per-iteration injection vector, ``g``
        #: the seed ``sum bar (alt - out)``
        self.reverse = reverse
        self.top: list = []
        self.tapes: dict = {}
        self.alt_at: dict = {}
        self.g = None
        self.amap: dict = {}
        self.joints: dict = {}
        self.seed = _ir.leaf(self.ctx, "uniform", "uniform", SEED, _I32)
        top = [
            n
            for n in seq
            if n.field_text("kind") in _ORDERED and plan.is_live(n) and not _ir.is_a(n, "loop")
        ]
        self.sites = tuple(top)
        self.gamma = _guards(seq, [n for n in seq if _ir.is_a(n, "sink")])
        self.bodies: dict = {}
        offset = [_sum([self.rate(s) for s in top])] if top else [_zero(self.ctx)]
        self.cum0: dict = {}
        totals = list(offset)
        for node in seq:
            if _ir.is_a(node, "loop"):
                body = self._body(node)
                self.bodies[node] = body
                if body.sweep is not None:
                    self.cum0[node] = _sum(totals)
                    totals.append(_ir.loop_value(body.sweep, node.field_int("carry_count")))
        self.total = _sum(totals)
        uniform = _ir.make("random_uniform", (self.seed, _ir.const(self.ctx, STREAM, _I32)))
        self.fork = _ir.make("mul", (uniform, self.total))
        self.cum = _zero(self.ctx)
        self.gains: list = []

    # -- sites and rates ----------------------------------------------------
    def _body(self, loop) -> _Body:
        body_nodes = _ir.body_nodes(loop)
        if any(_ir.is_a(node, "loop") for node in body_nodes):
            raise HawkError(
                f"jump= in kernel {self.plan.kernel!r}: the alternate path "
                "through a loop holding a nested `for` is not built"
            )
        if self.reverse and any(_ir.is_a(n, "dispatch") for n in body_nodes):
            raise HawkError(
                f"jump= in kernel {self.plan.kernel!r}: a `dispatch` inside a loop "
                "body of a `jump=` kernel differentiated in REVERSE: the switching "
                "comparisons of the plan are not applied under a branch; this is "
                "not built"
            )
        exits = _ir.exit_kwargs(loop)
        exit_roots = ()
        if exits["exit_cond"] is not None:
            exit_roots = (exits["exit_cond"], *exits["exit_values"])
        roots = (*loop.field_nodes("nexts"), *exit_roots)
        body = self.plan.body(loop, roots)
        if body is None:
            return _Body((), None)
        stop = {*loop.operands, _index_of(loop)}
        order = _post_order(roots, stop)
        self.gamma.update(_guards(order, roots, stop))
        sites = tuple(n for n in order if n.field_text("kind") in _ORDERED and body.is_live(n))
        if not sites:
            return _Body((), None)
        if _ir.exit_form(loop) == "middle":
            raise HawkError(
                f"jump= in kernel {self.plan.kernel!r}: a `break` in the middle "
                "of a loop body with a switching comparison: the alternate "
                "path needs the exit at the head or the tail of the body"
            )
        return _Body(sites, self._sweep(loop, sites))

    def rate(self, site, guard=False):
        """``K(d)`` of one comparison, exactly zero outside its support and
        where its outcome reaches no output (``guard``: its condition, else the
        one :func:`_guards` found)."""
        k = self.plan._supported(site, lambda kern: kern, 1.0)
        guard = self.gamma.get(site) if guard is False else guard
        return k if guard is None else _select(guard, k, _zero(self.ctx))

    def _sweep(self, loop, sites: tuple):
        """The first sweep: the primal's loop plus the sum of the rates."""
        m = loop.field_int("carry_count")
        index = _index_of(loop)
        acc = _carry(self.ctx, index.field_int("depth"), m, "jump_rate", _F64)
        step = _ir.make("add", (acc, _sum([self.rate(s) for s in sites])))
        exits = _ir.exit_kwargs(loop)
        extra: dict = {}
        if exits["exit_cond"] is not None:
            held = acc if _ir.exit_form(loop) == "head" else step
            extra = {
                "exit_cond": exits["exit_cond"],
                "exit_values": (*exits["exit_values"], held),
            }
        return _ir.loop(
            index,
            (*loop.field_nodes("carries"), acc),
            (*loop.field_nodes("inits"), _zero(self.ctx)),
            (*loop.field_nodes("nexts"), step),
            **_bounds(loop),
            **extra,
        )

    def weight(self, site, t):
        """``S sigma phi d'`` of one comparison, ``None`` without a tangent;
        in reverse mode (``t`` is ``None``) the part without ``d'``."""
        side = _ORDERED[site.field_text("kind")]
        scale = _select(site, _ir.const(self.ctx, -side, _F64), _ir.const(self.ctx, side, _F64))
        base = _ir.make("mul", (self.total, scale))
        if t is None:
            return base
        lhs, rhs = site.operands
        tl, tr = t(lhs), t(rhs)
        if tl is None and tr is None:
            return None
        diff = (
            _ir.make("neg", (tr,))
            if tl is None
            else tl
            if tr is None
            else _ir.make("sub", (tl, tr))
        )
        return _ir.make("mul", (base, diff))

    def _fork_here(self, cum, rate) -> tuple:
        """``(cum < U <= cum + rate, cum + rate)``."""
        after = _ir.make("add", (cum, rate))
        hit = _ir.make(
            "land", (_ir.make("lt", (cum, self.fork)), _ir.make("ge", (after, self.fork)))
        )
        return hit, after

    # -- straight-line nodes ---------------------------------------------------
    def visit(self, node, t) -> None:
        """Record ``node``'s alternate, operands first (``t`` is the ordinary
        tangent)."""
        if _ir.is_a(node, "loop_value"):
            loop, node_slot = _ir.loop_value_of(node)
            joint = self.joints.get(loop)
            if joint is not None:
                slot = self.alt_at[loop] + node_slot
                self.amap[node] = _ir.loop_value(joint, slot)
            return
        if _ir.is_a(node, "loop"):
            if self.reverse and self.affects(node):
                exits = _ir.exit_kwargs(node)
                self.joint(
                    node,
                    Parts(
                        tuple(node.field_nodes("carries")),
                        (),
                        tuple(node.field_nodes("inits")),
                        tuple(node.field_nodes("nexts")),
                        (),
                        tuple(exits["exit_values"]),
                        (),
                        None,
                    ),
                )
            return
        new = [self.amap.get(o, o) for o in node.operands]
        changed = any(a != b for a, b in zip(new, node.operands))
        if _ir.has_feather(node, "reduces_samples") and changed:
            raise HawkError(
                f"jump= in kernel {self.plan.kernel!r}: a read across samples "
                "downstream of a switching comparison cannot run on the "
                "alternate path of one sample"
            )
        out = _ir.rebuild(node, new) if changed else node
        if node in self.sites:
            hit, self.cum = self._fork_here(self.cum, self.rate(node))
            g = self.weight(node, t)
            if g is not None:
                gained = _select(hit, g, _zero(self.ctx))
                if self.reverse:
                    self.top.append((node, gained))
                else:
                    self.gains.append(gained)
            out = _flip(hit)(out)
        if out != node:
            self.amap[node] = out

    def affects(self, loop) -> bool:
        return loop in self.cum0 or any(o in self.amap for o in loop.operands)

    # -- the joint loop -----------------------------------------------------------
    def joint(self, loop, parts: Parts):
        """The augmented loop with the alternate beside the primal and tangent:
        slots ``[0, m)`` primal, ``[m, 2m)`` tangent, ``[2m, 3m)`` alternate,
        then ``cum`` and ``G`` (and, for a ``break``, a done flag each for the
        primal and the alternate)."""
        m = loop.field_int("carry_count")
        index = _index_of(loop)
        depth = index.field_int("depth")
        body = self.bodies[loop]
        nb = m + len(parts.dots)
        alts = tuple(
            _carry(self.ctx, depth, nb + j, f"alt_{c.field_text('name')}", _ir.type_of(c))
            for j, c in enumerate(parts.carries)
        )
        cum = _carry(self.ctx, depth, nb + m, "jump_cum", _F64)
        reverse = parts.tangent is None
        # forward mode also carries the gain G; reverse mode re-draws the fork off
        # the taped ``cum`` of each iteration instead (:meth:`inject`)
        gain = () if reverse else (_carry(self.ctx, depth, nb + m + 1, "jump_gain", _F64),)
        exits = _ir.exit_kwargs(loop)
        exit_cond = exits["exit_cond"]
        brk = exit_cond is not None
        first = nb + m + 1 + len(gain)
        flags = (
            (
                _carry(self.ctx, depth, first, "jump_done", _F64),
                _carry(self.ctx, depth, first + 1, "jump_done_alt", _F64),
            )
            if brk
            else ()
        )
        half = _ir.const(self.ctx, 0.5, _F64)
        done = [_ir.make("gt", (f, half)) for f in flags]
        active = None
        if brk:
            active = _ir.make("lnot", (done[0],))
            if _ir.exit_form(loop) == "head":
                active = _ir.make("land", (active, _ir.make("lnot", (exit_cond,))))
        # the fork, along the primal's comparisons
        c, terms, hits = cum, [], {}
        for site in body.sites:
            rate = self.rate(site)
            if active is not None:
                rate = _select(active, rate, _zero(self.ctx))
            hit, after = self._fork_here(c, rate)
            hits[site] = _flip(hit)
            g = self.weight(site, parts.tangent)
            if g is not None and not reverse:
                terms.append(_select(hit, g, _zero(self.ctx)))
            c = after
        gain_next = ()
        if gain:
            gain_next = (_ir.make("add", (gain[0], _sum(terms))) if terms else gain[0],)
        # the alternate body
        mapping = dict(zip(parts.carries, alts))
        mapping.update({o: self.amap[o] for o in loop.operands if o in self.amap})
        stop = {*loop.operands, index, *parts.carries}
        roots = parts.nexts + parts.exits + ((exit_cond,) if brk else ())
        memo = _alt_pass(_post_order(roots, stop), mapping, hits)

        def alt(n):
            return memo.get(n, n)

        a_nexts = tuple(alt(n) for n in parts.nexts)
        p_nexts, d_nexts = parts.nexts, parts.dot_nexts
        extra: dict = {}
        flag_nexts: tuple = ()
        if brk:
            ep, ea = exit_cond, alt(exit_cond)

            def freeze(is_done, now, held, at_exit, ahead):
                return _select(is_done, held, _select(now, at_exit, ahead))

            p_nexts = tuple(
                freeze(done[0], ep, p, e, n)
                for p, e, n in zip(parts.carries, parts.exits, parts.nexts)
            )
            d_nexts = tuple(
                freeze(done[0], ep, d, e, n)
                for d, e, n in zip(parts.dots, parts.dot_exits, parts.dot_nexts)
            )
            a_nexts = tuple(
                freeze(done[1], ea, a, alt(e), n) for a, e, n in zip(alts, parts.exits, a_nexts)
            )
            stopped = (_ir.make("lor", (done[0], ep)), _ir.make("lor", (done[1], ea)))
            flag_nexts = tuple(
                _select(s, _ir.const(self.ctx, 1.0, _F64), _zero(self.ctx)) for s in stopped
            )
            extra["exit_cond"] = _ir.make("land", stopped)
        nexts = (*p_nexts, *d_nexts, *a_nexts, c, *gain_next, *flag_nexts)
        if brk:
            extra["exit_values"] = nexts
        inits = (
            *parts.inits,
            *(self.amap.get(i, i) for i in loop.field_nodes("inits")),
            self.cum0.get(loop, _zero(self.ctx)),
            *(_zero(self.ctx) for _ in gain),
            *(_zero(self.ctx) for _ in flags),
        )
        rooted = _ir.slotwise(
            (*nexts, *extra.get("exit_values", ()), *([extra["exit_cond"]] if brk else []))
        )
        n = len(nexts)
        if brk:
            extra["exit_values"], extra["exit_cond"] = rooted[n : 2 * n], rooted[-1]
        built = _ir.loop(
            index,
            (*parts.carries, *parts.dots, *alts, cum, *gain, *flags),
            inits,
            rooted[:n],
            **_bounds(loop),
            **extra,
        )
        self.joints[loop] = built
        self.alt_at[loop] = nb
        if reverse:
            if body.sites:
                self.tapes[loop] = (built, nb + m)
        else:
            self.gains.append(_ir.loop_value(built, nb + m + 1))
        return built

    # -- reverse mode ---------------------------------------------------------------
    def seed_gain(self, sinks: Sequence, seed):
        """``g = sum bar (alt - out)`` over the sinks (``seed(sink)`` is the
        incoming adjoint), ``None`` when no output moves on the alternate."""
        terms = []
        for sink in sinks:
            if _ir.type_of(sink).dtype in _ir.INT_PLANE_DTYPES:
                continue
            bar = seed(sink)
            value = sink.operands[0]
            alt_value = self.amap.get(value, value)
            index = _keyed_index(sink)
            alt_index = None if index is None else self.amap.get(index, index)
            if index is not None and alt_index != index:
                there = _ir.make_node("At", _ir.type_of(bar), [bar.operands[0], alt_index])
                term = _ir.make(
                    "sub",
                    (_ir.make("mul", (there, alt_value)), _ir.make("mul", (bar, value))),
                )
            elif alt_value != value:
                term = _ir.make("mul", (bar, _ir.make("sub", (alt_value, value))))
            else:
                continue
            terms.append(_ir.make("sum", (term,)) if _ir.type_of(term).shape else term)
        return _sum(terms) if terms else None

    def injections(self) -> tuple:
        """``(site, weight)`` of the straight-line sites, ``weight`` already
        times ``g``."""
        if self.g is None:
            return ()
        return tuple((site, _ir.make("mul", (coef, self.g))) for site, coef in self.top)

    def sites_of(self, loop) -> tuple:
        got = self.tapes.get(loop)
        return self.bodies[loop].sites if got is not None else ()

    def inject(self, loop, sites: Sequence, roots: Sequence, stop) -> tuple:
        """``(site, weight)`` for the reverse body's ``sites`` (the forward
        ones, in order, rewritten onto the tape): the fork is drawn again from
        the ``cum`` the joint loop held entering this iteration, and the weight
        is the site's ``S sigma phi`` where it falls, times ``g``. The primal
        ran every iteration a reverse loop visits, so no site is masked."""
        got = self.tapes.get(loop)
        if got is None or self.g is None:
            return ()
        built, slot = got
        ttype = _ir.type_of(built.field_nodes("carries")[slot])
        cum = _ir.make_node("TapeRead", ttype, [built, _index_of(loop)], slot=slot, literal=slot)
        out = []
        guard = _guards(_post_order(roots, stop), roots, stop)
        for site in sites:
            hit, after = self._fork_here(cum, self.rate(site, guard[site]))
            weight = _select(hit, self.weight(site, None), _zero(self.ctx))
            out.append((site, _ir.make("mul", (weight, self.g))))
            cum = after
        return tuple(out)

    def _events(self):
        """``f(value)``: ``value`` where the sample has a switching instance
        (``S > 0``), else zero: a sample without one runs the plain tangent."""
        selector = _select(
            _ir.make("gt", (self.total, _zero(self.ctx))),
            _ir.const(self.ctx, 0, _I32),
            _ir.const(self.ctx, 1, _I32),
        )
        return lambda value: _switch(selector, (value, _ir.zero_like(self.ctx, _ir.type_of(value))))

    # -- the sinks ----------------------------------------------------------------
    def sinks(self, sinks: Sequence, tangent, derived) -> list:
        """The tangent sinks of ``sinks``: ``delta + G (alt - out)`` per value
        (a keyed scatter commits the alternate beside the primal at its own
        key). ``tangent(node)`` is the ordinary tangent and ``derived`` mints
        the sink."""
        gain = _sum(self.gains) if self.gains else None
        events = self._events()
        out: list = []
        for sink in sinks:
            if _ir.type_of(sink).dtype in _ir.INT_PLANE_DTYPES:
                continue
            sink_value = sink.operands[0]
            value = tangent(sink_value)
            plain = value if value is not None else _ir.zero_like(self.ctx, _ir.type_of(sink))
            alt_value = self.amap.get(sink_value, sink_value)
            index = _keyed_index(sink)
            alt_index = None if index is None else self.amap.get(index, index)
            moved = index is not None and alt_index != index
            if gain is None or (alt_value == sink_value and not moved):
                out.append(derived(sink, plain))
            elif moved:
                out.append(derived(sink, plain))
                out.append(derived(sink, events(_ir.make("mul", (gain, alt_value))), alt_index))
                out.append(
                    derived(sink, events(_ir.make("neg", (_ir.make("mul", (gain, sink_value)),))))
                )
            else:
                diff = _ir.make("sub", (alt_value, sink_value))
                out.append(
                    derived(sink, _ir.make("add", (plain, events(_ir.make("mul", (gain, diff))))))
                )
        return out
