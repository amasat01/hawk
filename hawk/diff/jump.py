# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The derivative through a parameter-dependent branch: ``@kernel(jump=...)``.

A branch ``if x < t: y = a else: y = b`` is a staircase in ``t``: its
derivative is zero almost everywhere, so a loss fitting ``t`` never moves.
A kernel declaring ``jump=`` is differentiated as the KERNEL-SMOOTHED loss
instead: each switching comparison ``l < r`` (``d = l - r``) passes
``K(d) * (a - b)`` to its operands, with ``K`` the Epanechnikov kernel of
standard deviation ``h`` (half-width ``w = sqrt(5) h``, exactly zero for
``|d| >= w``).

The primal IR is untouched. The transform consults a second rule table for
the boolean ops, keyed on the comparisons that switch (:class:`JumpPlan`):

* ``select(c, a, b)``: ``bar_c += sum(bar_y * (a - b))`` (entrywise for a
  vector ``c``);
* ``c1 and c2``: ``bar_c1 += bar_c * c2``, ``bar_c2 += bar_c * c1``;
  ``or``: the same with ``not c``; ``not c``: ``bar_c -= bar``;
* ``l < r`` / ``l <= r``: ``bar_l -= K(d) bar_c``, ``bar_r += K(d) bar_c``;
  ``>``/``>=`` with the signs swapped.

Forward mode is the transpose. The ``K(d)`` product sits inside one
``switch`` on ``|d| < w``: outside the support a sample evaluates the
comparison and nothing else.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ..ir import Const, Dispatch, HawkError, Leaf, Node, Select
from ..ir import make as _mk
from ..ir.loop_nodes import Loop, LoopCarry, LoopIndex
from ..ir.nodes import ReduceSamples, SampleIndex
from ..types import TensorType
from .tape import _reachable

#: The kernel and the rule names the sidecar records.
KERNEL = "epanechnikov"
#: ``jump="auto"``'s width per comparison: ``h = SPREAD * sd(d)``.
SPREAD = 0.02 * math.sqrt(12.0)
AUTO = "auto"

_ORDERED = {"lt": -1.0, "le": -1.0, "gt": 1.0, "ge": 1.0}
_EQUALITY = ("eq", "ne")
_LOGIC = ("land", "lor", "lnot")
_FLOAT = ("f64", "f32")
_F64 = TensorType((), "f64")
_I32 = TensorType((), "i32")
_SQRT5 = math.sqrt(5.0)


# --------------------------------------------------------------------------
# the declaration: @hawk.kernel(jump=...)


def check_jump(name: str, jump: Any, planes: dict) -> Any:
    """Refuse a ``jump=`` the kernel cannot honour, at decoration; returns it."""
    if any(p == AUTO and getattr(plane, "form", None) == "uniform"
           for p, plane in planes.items()):
        raise HawkError(
            f"kernel {name!r}: a Param named 'auto' collides with jump=\"auto\"; "
            "rename the parameter")
    if isinstance(jump, str):
        if jump == AUTO:
            return jump
        plane = planes.get(jump)
        if (plane is None or getattr(plane, "form", None) != "uniform"
                or plane.ttype.shape != () or plane.ttype.dtype not in _FLOAT):
            raise HawkError(
                f"kernel {name!r}: jump={jump!r} names no declared Param[Scalar]; "
                "jump= is a positive number, \"auto\" or the name of a "
                "Param[Scalar] holding the width")
        return jump
    if (isinstance(jump, bool) or not isinstance(jump, (int, float))
            or not math.isfinite(jump) or jump <= 0):
        raise HawkError(
            f"kernel {name!r}: jump={jump!r} must be a finite positive number "
            "(the width, in the units of the compared values), \"auto\" or the "
            "name of a Param[Scalar]")
    return float(jump)


def jump_of(primal: Any) -> Any:
    """``primal``'s declared ``jump=``, else ``None``; refuses a second order."""
    if getattr(getattr(primal, "sinks", primal), "jump", None) is not None:
        raise HawkError(
            "this kernel is a jump= derivative: differentiating it again (a "
            "second order through a smoothed switch) is not built")
    return getattr(primal, "jump", None)


def width_param(jump: Any) -> str | None:
    """The Param naming the width, or ``None`` for a fixed width."""
    return jump if isinstance(jump, str) and jump != AUTO else None


def drop_width(inputs: tuple, wrt: Sequence[str] | None, jump: Any) -> tuple:
    """``inputs`` without the width Param; refused when ``wrt`` lists it."""
    param = width_param(jump)
    if param is None:
        return inputs
    if wrt is not None and param in tuple(wrt):
        raise HawkError(
            f"wrt names {param!r}: the jump width is not differentiable; drop "
            "it from wrt=")
    return tuple(leaf for leaf in inputs if leaf.name != param)


# --------------------------------------------------------------------------
# which comparisons switch


class JumpPlan:
    """The switching comparisons of one primal, for one ``wrt`` set.

    ``sites`` are the ordered comparisons with an operand on a ``wrt`` path
    that a select's condition reaches through ``and``/``or``/``not``;
    ``live`` holds the keys of every condition node that reaches one. The
    plan of a loop body (:meth:`body`) is keyed on node identity and records
    its sites into the kernel's plan, so one sidecar lists them all."""

    def __init__(self, seq: Sequence[Node], key, inputs: Sequence[Leaf],
                 jump: Any, kernel: str | None, *, root: JumpPlan | None = None,
                 external=None, loop: str | None = None,
                 record: bool = True) -> None:
        self.key, self.jump, self.kernel = key, jump, kernel or "<sinks>"
        self.root, self.loop, self.record_sites = root or self, loop, record
        if root is None:
            self.names = {leaf.name for leaf in inputs}
            self.sites: list[Node] = []
            self.info: list[dict] = []
        wanted = {key(leaf) for leaf in inputs} if root is None else None
        self.reach: dict = {}
        for node in seq:
            got = external(node) if external is not None else None
            self.reach[key(node)] = (got if got is not None else _reach_of(
                node, lambda o: self.reach.get(key(o), 0),
                lambda leaf: self._leaf(leaf, wanted)))
        self.live: set = set()
        self._seen: set = set()
        for node in seq:
            if isinstance(node, Select):
                self._classify(node.operands[0], None)
        if root is None:
            self.width = None if jump == AUTO else self._width(seq)
            for node in seq:
                if isinstance(node, Loop):
                    self._refuse_exit(node)
                    self.body(node, tuple(node.nexts) + tuple(node.exit_values),
                              record=True)
            if jump == AUTO:
                self._refuse_auto()

    def _leaf(self, leaf: Leaf, wanted: set | None) -> int:
        if wanted is not None:
            return 1 if self.key(leaf) in wanted else 0
        return 1 if (leaf.name in self.root.names
                     and leaf.ttype.dtype in _FLOAT) else 0

    def _refuse(self, cond: Node, why: str) -> HawkError:
        return HawkError(f"jump= in kernel {self.kernel!r}: {spell(cond)}: {why}")

    def _classify(self, cond: Node, group: int | None) -> bool:
        """Whether ``cond`` reaches a switching comparison (recording it).
        ``group`` is the original comparison a scalarised vector comparison's
        entries share."""
        k = self.key(cond)
        if k in self.live:
            return True
        kind = getattr(cond, "kind", "")
        hit = False
        if kind == "vec" and cond.ttype.dtype == "bool":
            group = self.root._next_group() if group is None else group
            for o in cond.operands:
                hit = self._classify(o, group) or hit
        elif kind in _LOGIC:
            for o in cond.operands:
                hit = self._classify(o, group) or hit
        elif kind in _ORDERED or kind in _EQUALITY:
            reach = [self.reach.get(self.key(o), 0) for o in cond.operands]
            if 2 in reach:
                raise self._refuse(
                    cond, "a rounding, sign or integer cast sits between a "
                    "differentiated input and this comparison; a jump needs the "
                    "compared values to move smoothly")
            floats = all(o.ttype.dtype in _FLOAT for o in cond.operands)
            if 1 in reach and floats:
                if kind in _EQUALITY:
                    raise self._refuse(
                        cond, "a jump needs an ordered comparison; `==` has no "
                        "side to cross")
                hit = True
                if k not in self._seen and self.record_sites:
                    self._seen.add(k)
                    self.root.sites.append(cond)
                    self.root.info.append({
                        "group": self.root._next_group() if group is None else group,
                        "loop": self.loop})
        if hit:
            self.live.add(k)
        return hit

    def _group_of(self, site: Node) -> int:
        k = self.key(site)
        return next(i["group"] for s, i in zip(self.sites, self.info)
                    if self.key(s) == k)

    def _next_group(self) -> int:
        self._groups = getattr(self, "_groups", -1) + 1
        return self._groups

    def _refuse_auto(self) -> None:
        """``"auto"`` on a switch inside a loop: no per-sample statistic."""
        inside = [s for s, i in zip(self.sites, self.info) if i["loop"] is not None]
        if inside:
            raise self._refuse(
                inside[0], "'auto' needs a width from statistics over samples; "
                "switches inside a `for` take a number in this release")

    # -- the spread of each comparison over the batch ----------------------
    def _spreads(self) -> dict:
        """``group -> (sum d, sum d^2, count)`` per sample, the components of
        a vector comparison pooled; built once, so the cells and ``auto``'s
        reductions share their nodes (the split then reads the stored cells)."""
        got = getattr(self, "_spread", None)
        if got is None:
            got = self._spread = {}
            for site, info in zip(self.sites, self.info):
                if info["loop"] is not None:
                    continue           # a value per iteration: no per-sample cell
                d = self._d(site)
                if d.ttype.shape:
                    terms = (_mk("sum", (d,)), _mk("dot", (d, d)),
                             float(d.ttype.shape[0]))
                else:
                    terms = (d, _mk("mul", (d, d)), 1.0)
                held = got.get(info["group"])
                got[info["group"]] = terms if held is None else (
                    _mk("add", (held[0], terms[0])), _mk("add", (held[1], terms[1])),
                    held[2] + terms[2])
            for g, (s1, s2, n) in got.items():
                got[g] = (s1, s2, Const(n, _F64))
        return got

    def _auto_width(self, group: int) -> Node:
        """``w_g = sqrt(5) * SPREAD * sd_g``: ``sd_g`` from the batch's sums,
        each a reduction over samples the split computes in its own stage."""
        cached = self._widths = getattr(self, "_widths", {})
        if group not in cached:
            s1, s2, n = (ReduceSamples("sum", v) for v in self._spreads()[group])
            mean = _mk("div", (s1, n))
            var = _mk("sub", (_mk("div", (s2, n)), _mk("mul", (mean, mean))))
            h = _mk("mul", (Const(SPREAD, _F64), _mk("sqrt", (var,))))
            cached[group] = (h, _mk("mul", (Const(_SQRT5, _F64), h)), n)
        return cached[group]

    def _outer(self, node: Node) -> int:
        """The reach of a node of the kernel's own walk (a loop's operand)."""
        try:
            return self.root.reach.get(self.root.key(node), 0)
        except KeyError:
            return 0

    def _carried(self, loop: Loop) -> tuple:
        """``(reach per carried slot, reach by id over the body)``: a carry is
        on a ``wrt`` path when its initial value or its update is."""
        from .loops import _post_order

        slot = {id(c): j for j, c in enumerate(loop.carries)}
        stop = {id(o) for o in loop.operands} | {id(loop.index)}
        roots = tuple(loop.nexts) + tuple(loop.exit_values) + (
            (loop.exit_cond,) if loop.exit_cond is not None else ())
        order = _post_order(roots, stop)
        carry = [self._outer(i) for i in loop.inits]
        for _ in range(len(carry) + 1):
            reach: dict = {}
            for node in order:
                if id(node) in slot:
                    reach[id(node)] = carry[slot[id(node)]]
                elif id(node) in stop:
                    reach[id(node)] = self._outer(node)
                else:
                    reach[id(node)] = _reach_of(
                        node, lambda o: reach.get(id(o), 0),
                        lambda leaf: self._leaf(leaf, None))
            new = [max(c, reach.get(id(n), 0)) for c, n in zip(carry, loop.nexts)]
            if new == carry:
                break
            carry = new
        return carry, reach

    def _refuse_exit(self, loop: Loop) -> None:
        """A ``break`` whose condition depends on a differentiated input."""
        if loop.exit_cond is None:
            return
        _, reach = self._carried(loop)
        if any(reach.get(id(o), 0) for o in _comparison_operands(loop.exit_cond)):
            raise self._refuse(
                loop.exit_cond, "a loop exit through a differentiated input "
                "is an event, not a jump")

    def body(self, loop: Loop, roots: Sequence[Node],
             record: bool = False) -> JumpPlan | None:
        """The plan of one loop body, over ``roots`` (the body as the
        derivative walks it: the forward ``nexts``, or the reverse body whose
        carries read the forward loop's tape); ``None`` when no switch in it
        moves with a ``wrt`` input. The kernel's plan records each body's
        sites once, from the forward body (``record``)."""
        from .loops import _post_order

        carry, _ = self._carried(loop)
        slot = {id(c): j for j, c in enumerate(loop.carries)}
        operands = {id(o) for o in loop.operands}

        def external(node: Node) -> int | None:
            if id(node) in slot:
                return carry[slot[id(node)]]
            if getattr(node, "loop", None) is loop and hasattr(node, "slot"):
                return carry[node.slot]          # a tape read of a carried slot
            if id(node) in operands:
                return self._outer(node)
            return None

        stop = operands | {id(loop.index)}
        plan = JumpPlan(_post_order(roots, stop), id, (), self.jump, self.kernel,
                        root=self.root, external=external, loop=loop.index.name,
                        record=record)
        plan.width = self.root.width
        return plan if plan.live else None

    def _width(self, seq: Sequence[Node]) -> Node:
        param = width_param(self.jump)
        if param is None:
            return Const(_SQRT5 * self.jump, _F64)
        leaf = next((n for n in seq if isinstance(n, Leaf) and n.name == param),
                    None)
        if leaf is None:
            leaf = Leaf("uniform", "uniform", param, _F64)
        return _mk("mul", (Const(_SQRT5, _F64), leaf))

    # -- applying the rules inside a pass ---------------------------------
    def apply_vjp(self, node: Node, g: Node, accumulate) -> bool:
        """Add ``node``'s jump contributions through ``accumulate(child,
        adjoint)``; ``True`` when they replace the ordinary rule."""
        if isinstance(node, Select):
            if self.is_live(node.operands[0]):
                accumulate(node.operands[0], self.select_cond_vjp(node, g))
            return False
        if not self.is_live(node):
            return False
        for child, contribution in self.vjp(node, g):
            accumulate(child, contribution)
        return True

    def apply_jvp(self, node: Node, t, plain) -> tuple:
        """``(handled, tangent)`` for ``node``; ``plain(node)`` is the
        ordinary rule's tangent of a select."""
        if isinstance(node, Select):
            tc = t(node.operands[0]) if self.is_live(node.operands[0]) else None
            if tc is None:
                return False, None
            _, a, b = node.operands
            ordinary = plain(node) if t(a) is not None or t(b) is not None else None
            return True, self.select_jvp(node, ordinary, tc)
        if not self.is_live(node):
            return False, None
        return True, self.jvp(node, t)

    # -- the boolean-op rules ---------------------------------------------
    def is_live(self, node: Node) -> bool:
        return self.key(node) in self.live

    def select_cond_vjp(self, node: Select, g: Node) -> Node:
        """``select``'s pseudo-adjoint to its condition: ``sum(g (a - b))``."""
        cond, a, b = node.operands
        part = _mk("mul", (g, _mk("sub", (a, b))))
        if cond.ttype.shape == () and part.ttype.shape != ():
            part = _mk("sum", (part,))
        return part

    def vjp(self, node: Node, g: Node) -> tuple:
        """A live boolean node's reverse contributions."""
        if node.kind in _ORDERED:
            return self.compare_vjp(node, g)
        if node.kind == "vec":        # a vector condition, one entry per comparison
            return tuple((o, _mk("component", (g,), i))
                         for i, o in enumerate(node.operands) if self.is_live(o))
        return self.logic_vjp(node, g)

    def jvp(self, node: Node, t) -> Node | None:
        """A live boolean node's tangent."""
        if node.kind in _ORDERED:
            return self.compare_jvp(node, t)
        if node.kind == "vec":
            parts = [t(o) for o in node.operands]
            if all(p is None for p in parts):
                return None
            return _mk("vec", tuple(p if p is not None else Const(0.0, _F64)
                                    for p in parts))
        return self.logic_jvp(node, t)

    def logic_vjp(self, node: Node, g: Node) -> tuple:
        ops = node.operands
        if node.kind == "lnot":
            return ((ops[0], _mk("neg", (g,))),)
        zero = _zero(g.ttype)
        out = []
        for i, o in enumerate(ops):
            if not self.is_live(o):
                continue
            other = ops[1 - i]
            if node.kind == "land":
                out.append((o, Select(other, g, zero, g.ttype)))
            else:
                out.append((o, Select(other, zero, g, g.ttype)))
        return tuple(out)

    def compare_vjp(self, node: Node, g: Node) -> tuple:
        lhs, rhs = node.operands
        sign = _ORDERED[node.kind]       # bar_l += sign * K bar_c
        out = []
        for operand, s in ((lhs, sign), (rhs, -sign)):
            if self.reach.get(self.key(operand), 0):
                out.append((operand, self._supported(
                    node, lambda kern: _mk("mul", (kern, g)), s)))
        return tuple(out)

    def compare_jvp(self, node: Node, t) -> Node | None:
        lhs, rhs = node.operands
        tl, tr = t(lhs), t(rhs)
        if tl is None and tr is None:
            return None
        sign = _ORDERED[node.kind]       # c' = sign * K (l' - r')
        if tl is None:
            diff, sign = tr, -sign
        else:
            diff = tl if tr is None else _mk("sub", (tl, tr))
        return self._supported(node, lambda kern: _mk("mul", (kern, diff)), sign)

    def logic_jvp(self, node: Node, t) -> Node | None:
        ops = node.operands
        if node.kind == "lnot":
            got = t(ops[0])
            return None if got is None else _mk("neg", (got,))
        terms = []
        for i, o in enumerate(ops):
            to = t(o)
            if to is None:
                continue
            zero = _zero(to.ttype)
            other = ops[1 - i]
            terms.append(Select(other, to, zero, to.ttype) if node.kind == "land"
                         else Select(other, zero, to, to.ttype))
        if not terms:
            return None
        return terms[0] if len(terms) == 1 else _mk("add", tuple(terms))

    def select_jvp(self, node: Select, plain: Node | None, tc: Node) -> Node:
        """``today's select of tangents + (a - b) c'``."""
        cond, a, b = node.operands
        jumped = _mk("mul", (_mk("sub", (a, b)), tc))
        if plain is None:
            return jumped
        return _mk("add", (plain, jumped))

    # -- K(d) under the |d| < w switch -------------------------------------
    def _d(self, node: Node) -> Node:
        lhs, rhs = node.operands
        return _mk("sub", (lhs, rhs))

    def _supported(self, node: Node, product, sign: float) -> Node:
        """``product(sign * K(d))`` under the ``switch`` on ``d^2 < w^2``.

        ``K = 3/(4w) - 3/(4w^3) d^2``; the sign is folded into the
        subtraction's order, so no negation is emitted."""
        d = self._d(node)
        dd = _mk("mul", (d, d))
        w = self.width
        if w is None:                      # "auto": the width of this site's group
            group = self.root._group_of(node)
            w = self.root._auto_width(group)[1]
        if isinstance(w, Const):
            scale = 0.75 / w.literal
            w2 = Const(w.literal * w.literal, _F64)
            c2, scale = Const(scale / w2.literal, _F64), Const(scale, _F64)
        else:
            w2 = _mk("mul", (w, w))
            scale = _mk("div", (Const(0.75, _F64), w))
            c2 = _mk("div", (scale, w2))
        shape = d.ttype.shape
        if shape:
            w2, scale = _mk("splat", (w2,), shape), _mk("splat", (scale,), shape)
        drop = _mk("mul", (c2, dd))
        kern = _mk("sub", (scale, drop) if sign > 0 else (drop, scale))
        inside = _mk("lt", (dd, w2))
        value = product(kern)
        zero = _zero(value.ttype)
        if shape:
            value = Select(inside, value, zero, value.ttype)
            ones = _mk("splat", (Const(1.0, _F64),), shape)
            count = _mk("sum", (Select(inside, ones, _zero(ones.ttype), ones.ttype),))
            inside = _mk("gt", (count, Const(0.5, _F64)))
        selector = Select(inside, Const(0, _I32), Const(1, _I32), _I32)
        return Dispatch(selector, (value, zero), "switch", value.ttype)

    # -- the record ---------------------------------------------------------
    def spread_cells(self) -> tuple:
        """``jump_sd<g>_0/1/2`` = ``(sum d, sum d^2, count)`` per comparison
        group, for a width Param a controller normalises and for ``"auto"``,
        which also stores the width it picked, ``jump_h<g>``; none for a
        fixed width."""
        if self.jump is None or not isinstance(self.jump, str):
            return ()
        out: list = []
        for g, terms in self._spreads().items():
            for j, term in enumerate(terms):
                out.extend(ReduceSamples("sum", term).stored(f"jump_sd{g}_{j}"))
            if self.jump == AUTO:
                # the mean of the one width every sample holds, as a sum cell
                # (the op a gradient's own cells use, so one stage stores
                # both): each sample adds h * its count / the count cell
                h, _, count = self._auto_width(g)
                share = _mk("div", (_mk("mul", (h, self._spreads()[g][2])), count))
                out.extend(ReduceSamples("sum", share).stored(f"jump_h{g}"))
        return tuple(out)

    def record(self, wrt: Sequence[str]) -> dict | None:
        if not self.sites:
            return None
        param = width_param(self.jump)
        sites = []
        wanted = set(wrt)
        for i, (site, info) in enumerate(zip(self.sites, self.info)):
            names = sorted({n.name for n in _reachable((site,))
                            if isinstance(n, Leaf) and n.name in wanted})
            entry = {"name": f"jump{i}", "cond": spell(site), "reaches": names,
                     "group": info["group"]}
            if info["loop"] is not None:
                entry["loop"] = info["loop"]
            sites.append(entry)
        auto = self.jump == AUTO
        rule = "spread_v1" if auto else "param" if param else "fixed"
        return {"kernel": KERNEL, "rule": rule,
                "h": None if param or auto else self.jump,
                "constant": SPREAD if auto else None,
                "param": param, "sites": sites}


def _reach_of(node: Node, get, leaf) -> int:
    """0: no ``wrt`` input reaches ``node``; 1: one does, smoothly; 2: one does
    through a piecewise-constant op (a rounding, a sign, an integer cast)."""
    if isinstance(node, Leaf):
        return leaf(node)
    if isinstance(node, (Const, SampleIndex, LoopIndex, LoopCarry)):
        return 0
    kind = getattr(node, "kind", "")
    if kind in ("lt", "le", "gt", "ge", "eq", "ne", "land", "lor", "lnot",
                "isnan", "isinf", "isfinite"):
        return 0
    ops = node.operands
    if isinstance(node, Select):
        ops = ops[1:]
    elif isinstance(node, Dispatch):
        ops = node.branches
    got = max((get(o) for o in ops), default=0)
    from .event import _STEPWISE

    if got and (kind in _STEPWISE or kind == "cast"):
        return 2
    return got


def spell(node: Node) -> str:
    """A comparison's spelling in a refusal or the sidecar."""
    from .event import spell as spelled

    return spelled(node)


def _zero(t: TensorType) -> Node:
    from .rules import zero_like

    return zero_like(TensorType(t.shape, "f64" if t.dtype not in _FLOAT else t.dtype))


def _comparison_operands(cond: Node) -> list:
    kind = getattr(cond, "kind", "")
    if kind in _LOGIC:
        return [x for o in cond.operands for x in _comparison_operands(o)]
    if kind in _ORDERED or kind in _EQUALITY:
        return list(cond.operands)
    return []
