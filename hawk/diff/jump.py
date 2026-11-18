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

from .. import _ir
from .._errors import HawkError
from .loops import _select

#: The derivative kernel's uniform input keying the alternate path's draw
#: (:mod:`hawk.diff.altpath`).
SEED = "jump_seed"

#: The kernel and the rule names the sidecar records.
KERNEL = "epanechnikov"
#: ``jump="auto"``'s width per comparison: ``h = SPREAD * sd(d)``.
SPREAD = 0.02 * math.sqrt(12.0)
AUTO = "auto"

_ORDERED = {"lt": -1.0, "le": -1.0, "gt": 1.0, "ge": 1.0}
_EQUALITY = ("eq", "ne")
_LOGIC = ("land", "lor", "lnot")
_FLOAT = ("f64", "f32")
_F64 = _ir.TensorType((), "f64")
_I32 = _ir.TensorType((), "i32")
_SQRT5 = math.sqrt(5.0)


# --------------------------------------------------------------------------
# the declaration: @hawk.kernel(jump=...)


def check_jump(name: str, jump: Any, planes: dict) -> Any:
    """Refuse a ``jump=`` the kernel cannot honour, at decoration; returns it."""
    if any(p == AUTO and getattr(plane, "form", None) == "uniform" for p, plane in planes.items()):
        raise HawkError(
            f"kernel {name!r}: a Param named 'auto' collides with jump=\"auto\"; "
            "rename the parameter"
        )
    if isinstance(jump, str):
        if jump == AUTO:
            return jump
        plane = planes.get(jump)
        if (
            plane is None
            or getattr(plane, "form", None) != "uniform"
            or plane.ttype.shape != ()
            or plane.ttype.dtype not in _FLOAT
        ):
            raise HawkError(
                f"kernel {name!r}: jump={jump!r} names no declared Param[Scalar]; "
                'jump= is a positive number, "auto" or the name of a '
                "Param[Scalar] holding the width"
            )
        return jump
    if (
        isinstance(jump, bool)
        or not isinstance(jump, (int, float))
        or not math.isfinite(jump)
        or jump <= 0
    ):
        raise HawkError(
            f"kernel {name!r}: jump={jump!r} must be a finite positive number "
            '(the width, in the units of the compared values), "auto" or the '
            "name of a Param[Scalar]"
        )
    return float(jump)


def jump_of(primal: Any) -> Any:
    """``primal``'s declared ``jump=``, else ``None``; refuses a second order."""
    if getattr(getattr(primal, "sinks", primal), "jump", None) is not None:
        raise HawkError(
            "this kernel is a jump= derivative: differentiating it again (a "
            "second order through a smoothed switch) is not built"
        )
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
            f"wrt names {param!r}: the jump width is not differentiable; drop it from wrt="
        )
    return tuple(leaf for leaf in inputs if leaf.field_text("name") != param)


# --------------------------------------------------------------------------
# which comparisons switch


class JumpPlan:
    """The switching comparisons of one primal, for one ``wrt`` set.

    ``sites`` are the ordered comparisons with an operand on a ``wrt`` path
    that a select's condition reaches through ``and``/``or``/``not``;
    ``live`` holds the keys of every condition node that reaches one. ``key``
    names a node's table slot: the kernel's plan keys by the walk's position, so
    structurally equal nodes share one entry; the plan of a loop body
    (:meth:`body`) is keyed on node identity and records its sites into the
    kernel's plan, so one sidecar lists them all."""

    def __init__(
        self,
        seq: Sequence[Any],
        key,
        inputs: Sequence[Any],
        jump: Any,
        kernel: str | None,
        *,
        root: JumpPlan | None = None,
        external=None,
        loop: str | None = None,
        record: bool = True,
    ) -> None:
        self.key, self.jump, self.kernel = key, jump, kernel or "<sinks>"
        self.root, self.loop, self.record_sites = root or self, loop, record
        self.engaged = False
        if root is None:
            self.names = {leaf.field_text("name") for leaf in inputs}
            self.sites: list[Any] = []
            self.info: list[dict] = []
        wanted = {key(leaf) for leaf in inputs} if root is None else None
        self.reach: dict = {}
        for node in seq:
            got = external(node) if external is not None else None
            self.reach[key(node)] = (
                got
                if got is not None
                else _reach_of(
                    node,
                    lambda o: self.reach.get(key(o), 0),
                    lambda leaf: self._leaf(leaf, wanted),
                )
            )
        self.live: set = set()
        self._seen: set = set()
        if root is None:
            self.rounded: list = []
        self.root.rounded.extend(
            idx
            for idx in map(_runtime_index, seq)
            if idx is not None and self.reach.get(key(idx), 0) == 2 and self._from_input(idx)
        )
        for node in seq:
            if _ir.is_a(node, "select"):
                self._classify(node.operands[0], None)
        if root is None:
            self.width = None if jump == AUTO else self._width(seq)
            for node in seq:
                if _ir.is_a(node, "loop"):
                    self._refuse_exit(node)
                    exits = _ir.exit_kwargs(node)
                    self.body(
                        node, (*node.field_nodes("nexts"), *exits["exit_values"]), record=True
                    )
            if jump == AUTO:
                self._refuse_auto()
            if self.sites and self.rounded:
                raise self._refuse(
                    self.rounded[0],
                    "a rounding, sign or integer cast of a "
                    "differentiated input picks a row or a bin: its crossings are "
                    "switches the jump does not see, so the derivative would "
                    "silently leave them out. Key the row by a value that does not "
                    "depend on the differentiated input",
                )

    def _leaf(self, leaf: Any, wanted: set | None) -> int:
        if wanted is not None:
            return 1 if self.key(leaf) in wanted else 0
        return (
            1
            if (leaf.field_text("name") in self.root.names and _ir.type_of(leaf).dtype in _FLOAT)
            else 0
        )

    def _from_input(self, node: Any) -> bool:
        """Whether a differentiated input (or a carry on its path) reaches
        ``node`` without crossing a loop's result, whose reach is the loop's
        as a whole."""
        stack, seen = [node], set()
        while stack:
            n = stack.pop()
            if n in seen or _ir.is_a(n, "loop", "loop_value"):
                continue
            seen.add(n)
            if not n.operands:
                if self.reach.get(self.key(n), 0):
                    return True
                continue
            stack.extend(n.operands[1:] if _ir.is_a(n, "select") else n.operands)
        return False

    def _refuse(self, cond: Any, why: str) -> HawkError:
        return HawkError(f"jump= in kernel {self.kernel!r}: {spell(cond)}: {why}")

    def _classify(self, cond: Any, group: int | None) -> bool:
        """Whether ``cond`` reaches a switching comparison (recording it).
        ``group`` is the original comparison a scalarised vector comparison's
        entries share."""
        k = self.key(cond)
        if k in self.live:
            return True
        kind = cond.field_text("kind")
        hit = False
        if kind == "vec" and _ir.type_of(cond).dtype == "bool":
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
                    cond,
                    "a rounding, sign or integer cast sits between a "
                    "differentiated input and this comparison; a jump needs the "
                    "compared values to move smoothly",
                )
            floats = all(_ir.type_of(o).dtype in _FLOAT for o in cond.operands)
            if 1 in reach and floats:
                if kind in _EQUALITY:
                    raise self._refuse(
                        cond, "a jump needs an ordered comparison; `==` has no side to cross"
                    )
                hit = True
                if k not in self._seen and self.record_sites:
                    self._seen.add(k)
                    self.root.sites.append(cond)
                    self.root.info.append(
                        {
                            "group": self.root._next_group() if group is None else group,
                            "loop": self.loop,
                        }
                    )
        if hit:
            self.live.add(k)
        return hit

    def _group_of(self, site: Any) -> int:
        k = self.key(site)
        return next(i["group"] for s, i in zip(self.sites, self.info) if self.key(s) == k)

    def _next_group(self) -> int:
        self._groups = getattr(self, "_groups", -1) + 1
        return self._groups

    def _refuse_auto(self) -> None:
        """``"auto"`` on a switch inside a loop: no per-sample statistic."""
        inside = [s for s, i in zip(self.sites, self.info) if i["loop"] is not None]
        if inside:
            raise self._refuse(
                inside[0],
                "'auto' needs a width from statistics over samples; "
                "switches inside a `for` take a number in this release",
            )

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
                    continue  # a value per iteration: no per-sample cell
                d = self._d(site)
                shape = _ir.type_of(d).shape
                if shape:
                    terms = (_ir.make("sum", (d,)), _ir.make("dot", (d, d)), float(shape[0]))
                else:
                    terms = (d, _ir.make("mul", (d, d)), 1.0)
                held = got.get(info["group"])
                got[info["group"]] = (
                    terms
                    if held is None
                    else (
                        _ir.make("add", (held[0], terms[0])),
                        _ir.make("add", (held[1], terms[1])),
                        held[2] + terms[2],
                    )
                )
            for g, (s1, s2, n) in got.items():
                got[g] = (s1, s2, _ir.const(s1.ctx, n, _F64))
        return got

    def _auto_width(self, group: int) -> Any:
        """``w_g = sqrt(5) * SPREAD * sd_g``: ``sd_g`` from the batch's sums,
        each a reduction over samples the split computes in its own stage."""
        cached = self._widths = getattr(self, "_widths", {})
        if group not in cached:
            s1, s2, n = (_reduce_samples(v) for v in self._spreads()[group])
            mean = _ir.make("div", (s1, n))
            var = _ir.make("sub", (_ir.make("div", (s2, n)), _ir.make("mul", (mean, mean))))
            h = _ir.make("mul", (_ir.const(n.ctx, SPREAD, _F64), _ir.make("sqrt", (var,))))
            cached[group] = (h, _ir.make("mul", (_ir.const(n.ctx, _SQRT5, _F64), h)), n)
        return cached[group]

    def _outer(self, node: Any) -> int:
        """The reach of a node of the kernel's own walk (a loop's operand); a node outside the
        walk, such as a loop's own index, has none."""
        try:
            return self.root.reach.get(self.root.key(node), 0)
        except KeyError:
            return 0

    def _carried(self, loop: Any) -> tuple:
        """``(reach per carried slot, reach over the body)``: a carry is
        on a ``wrt`` path when its initial value or its update is."""
        from .loops import _post_order

        nexts = loop.field_nodes("nexts")
        slot = {c: j for j, c in enumerate(loop.field_nodes("carries"))}
        stop = {*loop.operands, loop.field_nodes("index")[0]}
        exits = _ir.exit_kwargs(loop)
        roots = (*nexts, *exits["exit_values"])
        if exits["exit_cond"] is not None:
            roots += (exits["exit_cond"],)
        order = _post_order(roots, stop)
        carry = [self._outer(i) for i in loop.field_nodes("inits")]
        for _ in range(len(carry) + 1):
            reach: dict = {}
            for node in order:
                if node in slot:
                    reach[node] = carry[slot[node]]
                elif node in stop:
                    reach[node] = self._outer(node)
                else:
                    reach[node] = _reach_of(
                        node, lambda o: reach.get(o, 0), lambda leaf: self._leaf(leaf, None)
                    )
            new = [max(c, reach.get(n, 0)) for c, n in zip(carry, nexts)]
            if new == carry:
                break
            carry = new
        return carry, reach

    def _refuse_exit(self, loop: Any) -> None:
        """A ``break`` whose condition depends on a differentiated input."""
        exit_cond = _ir.exit_kwargs(loop)["exit_cond"]
        if exit_cond is None:
            return
        _, reach = self._carried(loop)
        if any(reach.get(o, 0) for o in _comparison_operands(exit_cond)):
            raise self._refuse(
                exit_cond, "a loop exit through a differentiated input is an event, not a jump"
            )

    def body(self, loop: Any, roots: Sequence[Any], record: bool = False) -> JumpPlan | None:
        """The plan of one loop body, over ``roots`` (the body as the
        derivative walks it: the forward ``nexts``, or the reverse body whose
        carries read the forward loop's tape); ``None`` when no switch in it
        moves with a ``wrt`` input. The kernel's plan records each body's
        sites once, from the forward body (``record``)."""
        from .loops import _post_order

        carry, _ = self._carried(loop)
        slot = {c: j for j, c in enumerate(loop.field_nodes("carries"))}
        operands = set(loop.operands)
        index = loop.field_nodes("index")[0]

        def external(node: Any) -> int | None:
            if node in slot:
                return carry[slot[node]]
            if _ir.is_a(node, "tape_read", "loop_value") and node.operands[0] == loop:
                return carry[node.field_int("slot")]  # a tape read of a carried slot
            if node in operands:
                return self._outer(node)
            return None

        stop = operands | {index}
        plan = JumpPlan(
            _post_order(roots, stop),
            lambda node: node,
            (),
            self.jump,
            self.kernel,
            root=self.root,
            external=external,
            loop=index.field_text("name"),
            record=record,
        )
        plan.width = self.root.width
        return plan if plan.live else None

    def _width(self, seq: Sequence[Any]) -> Any:
        context = seq[0].ctx
        param = width_param(self.jump)
        if param is None:
            return _ir.const(context, _SQRT5 * self.jump, _F64)
        leaf = next((n for n in seq if _ir.is_a(n, "leaf") and n.field_text("name") == param), None)
        if leaf is None:
            leaf = _ir.leaf(context, "uniform", "uniform", param, _F64)
        return _ir.make("mul", (_ir.const(context, _SQRT5, _F64), leaf))

    # -- applying the rules inside a pass ---------------------------------
    def apply_vjp(self, node: Any, g: Any, accumulate) -> bool:
        """Add ``node``'s jump contributions through ``accumulate(child,
        adjoint)``; ``True`` when they replace the ordinary rule."""
        if _ir.is_a(node, "select"):
            if self.is_live(node.operands[0]):
                accumulate(node.operands[0], self.select_cond_vjp(node, g))
            return False
        if not self.is_live(node):
            return False
        for child, contribution in self.vjp(node, g):
            accumulate(child, contribution)
        return True

    def apply_jvp(self, node: Any, t, plain) -> tuple:
        """``(handled, tangent)`` for ``node``; ``plain(node)`` is the
        ordinary rule's tangent of a select."""
        if _ir.is_a(node, "select"):
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
    def is_live(self, node: Any) -> bool:
        return self.key(node) in self.live

    def select_cond_vjp(self, node: Any, g: Any) -> Any:
        """``select``'s pseudo-adjoint to its condition: ``sum(g (a - b))``."""
        cond, a, b = node.operands
        part = _ir.make("mul", (g, _ir.make("sub", (a, b))))
        if _ir.type_of(cond).shape == () and _ir.type_of(part).shape != ():
            part = _ir.make("sum", (part,))
        return part

    def vjp(self, node: Any, g: Any) -> tuple:
        """A live boolean node's reverse contributions."""
        kind = node.field_text("kind")
        if kind in _ORDERED:
            return self.compare_vjp(node, g)
        if kind == "vec":  # a vector condition, one entry per comparison
            return tuple(
                (o, _ir.make("component", (g,), i))
                for i, o in enumerate(node.operands)
                if self.is_live(o)
            )
        return self.logic_vjp(node, g)

    def jvp(self, node: Any, t) -> Any:
        """A live boolean node's tangent."""
        kind = node.field_text("kind")
        if kind in _ORDERED:
            return self.compare_jvp(node, t)
        if kind == "vec":
            parts = [t(o) for o in node.operands]
            if all(p is None for p in parts):
                return None
            zero = _ir.const(node.ctx, 0.0, _F64)
            return _ir.make("vec", tuple(p if p is not None else zero for p in parts))
        return self.logic_jvp(node, t)

    def logic_vjp(self, node: Any, g: Any) -> tuple:
        ops = node.operands
        kind = node.field_text("kind")
        if kind == "lnot":
            return ((ops[0], _ir.make("neg", (g,))),)
        zero = _zero(g)
        out = []
        for i, o in enumerate(ops):
            if not self.is_live(o):
                continue
            other = ops[1 - i]
            if kind == "land":
                out.append((o, _select(other, g, zero)))
            else:
                out.append((o, _select(other, zero, g)))
        return tuple(out)

    def compare_vjp(self, node: Any, g: Any) -> tuple:
        lhs, rhs = node.operands
        sign = _ORDERED[node.field_text("kind")]  # bar_l += sign * K bar_c
        out = []
        for operand, s in ((lhs, sign), (rhs, -sign)):
            if self.reach.get(self.key(operand), 0):
                out.append(
                    (operand, self._supported(node, lambda kern: _ir.make("mul", (kern, g)), s))
                )
        return tuple(out)

    def compare_jvp(self, node: Any, t) -> Any:
        lhs, rhs = node.operands
        tl, tr = t(lhs), t(rhs)
        if tl is None and tr is None:
            return None
        sign = _ORDERED[node.field_text("kind")]  # c' = sign * K (l' - r')
        if tl is None:
            diff, sign = tr, -sign
        else:
            diff = tl if tr is None else _ir.make("sub", (tl, tr))
        return self._supported(node, lambda kern: _ir.make("mul", (kern, diff)), sign)

    def logic_jvp(self, node: Any, t) -> Any:
        ops = node.operands
        kind = node.field_text("kind")
        if kind == "lnot":
            got = t(ops[0])
            return None if got is None else _ir.make("neg", (got,))
        terms = []
        for i, o in enumerate(ops):
            to = t(o)
            if to is None:
                continue
            zero = _zero(to)
            other = ops[1 - i]
            terms.append(_select(other, to, zero) if kind == "land" else _select(other, zero, to))
        if not terms:
            return None
        return terms[0] if len(terms) == 1 else _ir.make("add", tuple(terms))

    def select_jvp(self, node: Any, plain: Any, tc: Any) -> Any:
        """``today's select of tangents + (a - b) c'``."""
        cond, a, b = node.operands
        jumped = _ir.make("mul", (_ir.make("sub", (a, b)), tc))
        if plain is None:
            return jumped
        return _ir.make("add", (plain, jumped))

    # -- K(d) under the |d| < w switch -------------------------------------
    def _d(self, node: Any) -> Any:
        lhs, rhs = node.operands
        return _ir.make("sub", (lhs, rhs))

    def _supported(self, node: Any, product, sign: float) -> Any:
        """``product(sign * K(d))`` under the ``switch`` on ``d^2 < w^2``.

        ``K = 3/(4w) - 3/(4w^3) d^2``; the sign is folded into the
        subtraction's order, so no negation is emitted."""
        d = self._d(node)
        dd = _ir.make("mul", (d, d))
        w = self.width
        if w is None:  # "auto": the width of this site's group
            group = self.root._group_of(node)
            w = self.root._auto_width(group)[1]
        if _ir.is_a(w, "const"):
            width = w.field_real("literal")
            scale = 0.75 / width
            w2_value = width * width
            w2 = _ir.const(d.ctx, w2_value, _F64)
            c2, scale = _ir.const(d.ctx, scale / w2_value, _F64), _ir.const(d.ctx, scale, _F64)
        else:
            w2 = _ir.make("mul", (w, w))
            scale = _ir.make("div", (_ir.const(d.ctx, 0.75, _F64), w))
            c2 = _ir.make("div", (scale, w2))
        shape = _ir.type_of(d).shape
        if shape:
            w2, scale = _ir.make("splat", (w2,), shape), _ir.make("splat", (scale,), shape)
        drop = _ir.make("mul", (c2, dd))
        kern = _ir.make("sub", (scale, drop) if sign > 0 else (drop, scale))
        inside = _ir.make("lt", (dd, w2))
        value = product(kern)
        zero = _zero(value)
        if shape:
            value = _select(inside, value, zero)
            ones = _ir.make("splat", (_ir.const(d.ctx, 1.0, _F64),), shape)
            count = _ir.make("sum", (_select(inside, ones, _zero(ones)),))
            inside = _ir.make("gt", (count, _ir.const(d.ctx, 0.5, _F64)))
        selector = _select(inside, _ir.const(d.ctx, 0, _I32), _ir.const(d.ctx, 1, _I32))
        return _switch(selector, (value, zero))

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
                out.extend(_ir.cells_stored(_reduce_samples(term), f"jump_sd{g}_{j}"))
            if self.jump == AUTO:
                # the mean of the one width every sample holds, as a sum cell
                # (the op a gradient's own cells use, so one stage stores
                # both): each sample adds h * its count / the count cell
                h, _, count = self._auto_width(g)
                share = _ir.make("div", (_ir.make("mul", (h, self._spreads()[g][2])), count))
                out.extend(_ir.cells_stored(_reduce_samples(share), f"jump_h{g}"))
        return tuple(out)

    def record(self, wrt: Sequence[str]) -> dict | None:
        if not self.sites:
            return None
        param = width_param(self.jump)
        sites = []
        wanted = set(wrt)
        for i, (site, info) in enumerate(zip(self.sites, self.info)):
            names = sorted(
                {
                    n.field_text("name")
                    for n in _ir.post_order((site,))
                    if _ir.is_a(n, "leaf") and n.field_text("name") in wanted
                }
            )
            entry = {
                "name": f"jump{i}",
                "cond": spell(site),
                "reaches": names,
                "group": info["group"],
            }
            if info["loop"] is not None:
                entry["loop"] = info["loop"]
            sites.append(entry)
        auto = self.jump == AUTO
        rule = "spread_v1" if auto else "param" if param else "fixed"
        record = {
            "kernel": KERNEL,
            "rule": rule,
            "h": None if param or auto else self.jump,
            "constant": SPREAD if auto else None,
            "param": param,
            "sites": sites,
        }
        if self.engaged:
            record.update(rule="altpath_v1", seed_param=SEED, engaged=True, width_rule=rule)
        return record


def _runtime_index(node: Any) -> Any:
    """The runtime row or key ``node`` reads or commits at, else ``None``."""
    if _ir.is_a(node, "accum_write", "wide_write"):
        return node.operands[1] if len(node.operands) > 1 else None
    if node.field_text("kind") in ("at", "component_at", "set_component_at", "add_at"):
        return node.operands[1]
    return None


def _reach_of(node: Any, get, leaf) -> int:
    """0: no ``wrt`` input reaches ``node``; 1: one does, smoothly; 2: one does
    through a piecewise-constant op (a rounding, a sign, an integer cast)."""
    if _ir.is_a(node, "leaf"):
        return leaf(node)
    if _ir.is_a(node, "const", "sample_index", "plane_extent", "loop_index", "loop_carry"):
        return 0
    kind = node.field_text("kind")
    if kind in (
        "lt",
        "le",
        "gt",
        "ge",
        "eq",
        "ne",
        "land",
        "lor",
        "lnot",
        "isnan",
        "isinf",
        "isfinite",
    ):
        return 0
    ops = node.operands
    if _ir.is_a(node, "select", "dispatch"):
        ops = ops[1:]
    got = max((get(o) for o in ops), default=0)
    from .event import _STEPWISE

    if got and (kind in _STEPWISE or kind == "cast"):
        return 2
    return got


def spell(node: Any) -> str:
    """A comparison's spelling in a refusal or the sidecar."""
    from .event import spell as spelled

    return spelled(node)


def _switch(selector: Any, branches: Sequence[Any]) -> Any:
    """Make the dispatch of ``selector`` over ``branches``, typed as the first."""
    return _ir.make_node("Dispatch", _ir.type_of(branches[0]), [selector, *branches])


def _reduce_samples(value: Any) -> Any:
    """Make the sum of ``value`` over the samples of a launch."""
    return _ir.make_node(
        "ReduceSamples",
        _ir.type_of(value),
        [value],
        op="sum",
        contract="auto",
        key=None,
        count=None,
        kept=None,
    )


def _zero(like: Any) -> Any:
    """A structural zero shaped like ``like`` (float64 when ``like`` is not floating)."""
    shape, dtype, tag = _ir.type_of(like)
    return _ir.zero_like(like.ctx, _ir.TensorType(shape, dtype if dtype in _FLOAT else "f64", tag))


def _comparison_operands(cond: Any) -> list:
    kind = cond.field_text("kind")
    if kind in _LOGIC:
        return [x for o in cond.operands for x in _comparison_operands(o)]
    if kind in _ORDERED or kind in _EQUALITY:
        return list(cond.operands)
    return []
