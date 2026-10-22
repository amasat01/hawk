# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Slotwise form: a wide value built from ``vec``/``splat`` as a ``vec`` of
scalar parts.

A reverse-mode kernel builds each cotangent as a DENSE ``vec`` of the input's
full width (one part per component, nearly all structural zeros) and chains
them with ``add``. Rendered whole, every one is a width-wide aggregate. This
pass rewrites a node whose rank>=1 operands are all ``vec``/``splat`` into a
``vec`` of scalar parts, with the structural zeros folded, so a component that
only one case writes is one scalar, not a column of an aggregate.

Rewritten: ``component`` of a ``vec``; ``sum`` of a rank-1 ``vec`` (left-nested
``add``, aether's own fold order); ``add``/``sub``/``neg``/``mul``/``div``;
``select``; ``dispatch`` (one dispatch per slot, same selector, ``K`` and
policy, so the renderer still fuses them into ONE ``switch``). Everything else
over a ``vec`` keeps its node. Structural zero: a ``Const`` of value 0.

Per slot the value equals the dense form's, with the same summation order
minus the zeros. Admitted differences: the sign of an exact zero, and a
non-finite factor times a structural zero is 0 (the dense form gave NaN).

Pure, idempotent, and position-free. A node with nothing to rewrite comes back
as the SAME object. An existing :class:`~hawk.ir.loop_nodes.Loop` is a scope
and is never entered.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..types import TensorType
from .loop_nodes import Loop
from .nodes import (
    AccumWrite,
    Assign,
    Const,
    Dispatch,
    HawkError,
    Node,
    Op,
    SampleSum,
    Select,
    Sink,
    WideWrite,
)
from .ops import make
from .walk import _rebuild

_FLOAT = ("f64", "f32")
_ELEMENTWISE = ("add", "sub", "mul", "div")


class _Bail(Exception):
    """This node keeps its form (a type the slot form cannot reproduce)."""


def slotwise(roots: Sequence[Node]) -> tuple:
    """``roots`` in slotwise form (see the module docstring), same length.

    A root may be a value, a sink (own column, scatter or cell) or a loop;
    a sink is rebuilt over its rewritten value."""
    roots = tuple(roots)
    memo: dict = {}
    for root in roots:
        _walk(root, memo)
    out = []
    for root in roots:
        out.append(_resink(root, memo) if isinstance(root, Sink) else memo[id(root)])
    return tuple(out)


def _walk(root: Node, memo: dict) -> None:
    """Fill ``memo`` (id -> rewritten node) for everything ``root`` reaches."""
    stack = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if id(node) in memo:
            continue
        if isinstance(node, Loop) or not node.operands:
            memo[id(node)] = node
            continue
        if not expanded:
            stack.append((node, True))
            stack.extend((c, False) for c in reversed(node.operands)
                         if id(c) not in memo)
            continue
        new = tuple(memo[id(c)] for c in node.operands)
        if isinstance(node, Sink):
            memo[id(node)] = node       # a sink is rebuilt by :func:`_resink`
            continue
        got = _rewrite(node, new)
        if got is None:
            changed = any(a is not b for a, b in zip(new, node.operands))
            got = _rebuild(node, new) if changed else node
        memo[id(node)] = got


def _resink(sink: Sink, memo: dict) -> Node:
    """``sink`` over the rewritten values of its operands."""
    new = tuple(memo[id(c)] for c in sink.operands)
    if all(a is b for a, b in zip(new, sink.operands)):
        return sink
    if isinstance(sink, SampleSum):
        return sink.rebuilt(new[0])
    if isinstance(sink, Assign):
        return Assign(sink.name, new[0], sink.ttype)
    if isinstance(sink, WideWrite):
        return WideWrite(sink.name, *new, ttype=sink.ttype)
    if isinstance(sink, AccumWrite):
        return AccumWrite(sink.name, *new, ttype=sink.ttype)
    return sink                                     # TapeWrite/Finish: untouched


# --------------------------------------------------------------------------
# the rewrite of one node, over its already-rewritten operands


def _rewrite(node: Node, ops: tuple) -> Node | None:
    """The slotwise replacement of ``node`` over ``ops``, or ``None``."""
    try:
        if any(_is_chain(op) for op in ops):
            return _runtime(node, ops)
        if isinstance(node, Dispatch):
            return _dispatch(node, ops)
        if isinstance(node, Select):
            return _select(node, ops)
        if isinstance(node, Op):
            if node.kind == "component":
                return _component(node, ops)
            if node.kind == "sum":
                return _sum(node, ops)
            if node.kind in _ELEMENTWISE or node.kind == "neg":
                return _elementwise(node, ops)
    except (_Bail, HawkError):
        return None
    return None


def _slots(node: Node) -> list | None:
    """The scalar parts of a ``vec``/``splat`` of floats (or bools), row-major."""
    if not isinstance(node, Op) or node.ttype.dtype not in _FLOAT + ("bool",):
        return None
    if node.kind == "vec":
        return list(node.operands)
    if node.kind == "splat":
        return [node.operands[0]] * math.prod(node.ttype.shape)
    return None


def _is_zero(node: Node) -> bool:
    return (isinstance(node, Const) and node.ttype.shape == ()
            and node.ttype.dtype in _FLOAT and node.literal == 0)


def _assemble(shape: tuple, parts: list, template: TensorType) -> Node:
    """``parts`` as a ``vec`` of ``shape``, refusing a type the node never had."""
    out = make("vec", parts, literal=shape if len(shape) == 2 else None)
    if out.ttype != template:
        raise _Bail
    return out


def _per_slot(node: Node, ops: tuple, build) -> Node | None:
    """``vec(build(slot k of each operand) ...)`` when every rank>=1 operand is
    a ``vec``/``splat`` of ``node``'s shape (a rank-0 one broadcasts)."""
    shape = node.ttype.shape
    if (not shape or node.ttype.dtype not in _FLOAT or node.ttype.tag is not None):
        return None
    columns = []
    for op in ops:
        if op.ttype.shape == ():
            columns.append(None)
            continue
        slots = _slots(op) if op.ttype.shape == shape else None
        if slots is None:
            return None
        columns.append(slots)
    if all(c is None for c in columns):
        return None
    width = math.prod(shape)
    parts = [build(*(op if col is None else col[k] for op, col in zip(ops, columns)))
             for k in range(width)]
    return _assemble(shape, parts, node.ttype)


def _elementwise(node: Op, ops: tuple) -> Node | None:
    build = {"add": _add, "sub": _sub, "mul": _mul, "div": _div, "neg": _neg}[node.kind]
    return _per_slot(node, ops, build)


def _select(node: Select, ops: tuple) -> Node | None:
    return _per_slot(node, ops, _where)


def _dispatch(node: Dispatch, ops: tuple) -> Node | None:
    selector, branches = ops[0], ops[1:]

    def build(*parts):
        return _switch(selector, parts, node.policy)

    return _per_slot(node, branches, build) if all(
        b.ttype.shape == node.ttype.shape for b in branches) else None


def _component(node: Op, ops: tuple) -> Node | None:
    slots = _slots(ops[0])
    if slots is None:
        return None
    shape, at = ops[0].ttype.shape, node.literal
    part = slots[at if isinstance(at, int) else at[0] * shape[1] + at[1]]
    if part.ttype != node.ttype:
        raise _Bail
    return part


def _sum(node: Op, ops: tuple) -> Node | None:
    slots = _slots(ops[0])
    if slots is None or len(ops[0].ttype.shape) != 1:
        return None
    acc = slots[0]
    for part in slots[1:]:
        acc = _add(acc, part)
    if acc.ttype != node.ttype:
        raise _Bail
    return acc


# --------------------------------------------------------------------------
# scalar builders: the structural-zero folds


def _add(a: Node, b: Node) -> Node:
    if _is_zero(a):
        return b
    if _is_zero(b):
        return a
    return make("add", (a, b))


def _sub(a: Node, b: Node) -> Node:
    if _is_zero(b):
        return a
    if _is_zero(a):
        return _neg(b)
    return make("sub", (a, b))


def _neg(a: Node) -> Node:
    return a if _is_zero(a) else make("neg", (a,))


def _mul(a: Node, b: Node) -> Node:
    if _is_zero(a):
        return a
    if _is_zero(b):
        return b
    return make("mul", (a, b))


def _div(a: Node, b: Node) -> Node:
    return a if _is_zero(a) else make("div", (a, b))


def _where(cond: Node, a: Node, b: Node) -> Node:
    if a is b or (_is_zero(a) and _is_zero(b)):
        return a
    if a.ttype != b.ttype:
        raise _Bail
    return Select(cond, a, b, a.ttype)


def _switch(selector: Node, parts: Sequence[Node], policy: str) -> Node:
    if all(p is parts[0] for p in parts) or all(_is_zero(p) for p in parts):
        return parts[0]
    return Dispatch(selector, parts, policy, parts[0].ttype)


# --------------------------------------------------------------------------
# the runtime form: an ``add_at`` chain over a slot-form base
#
# ``add_at(base, k, e)`` is ``base`` with ``e`` added at the RUNTIME index
# ``k`` (the adjoint of ``component_at``). A chain of them over a ``vec``/
# ``splat`` base is kept as such — an own-column commit renders each link as
# ``bar[i].at(k) += e`` — and pushed through what the dense form is pushed
# through: ``add``/``sub``/``neg``, ``mul``/``div`` by a rank-0 value, a
# rank-0 ``select`` and a ``dispatch`` (index and value become scalar members
# of the same switch; a case without the link carries index 0 and value 0, a
# no-op add). Where a static slot is forced (``component``, ``sum``, an
# elementwise op against another vector), the chain is lowered: slot ``c``
# gets ``select(k == c, e, 0)`` per link, in chain order.


def _is_chain(node: Node) -> bool:
    return isinstance(node, Op) and node.kind == "add_at"


def _chain(node: Node) -> tuple:
    """``(base, [(k, e), ...])`` of an ``add_at`` chain, links in order."""
    links = []
    while _is_chain(node):
        links.append((node.operands[1], node.operands[2]))
        node = node.operands[0]
    return node, links[::-1]


#: The public spelling of :func:`_chain`, for the loop transform and the emitter.
chain = _chain


def is_structural_zero(node: Node) -> bool:
    """A ``splat`` of a zero literal, or a ``vec`` of zero literals."""
    slots = _slots(node)
    return slots is not None and all(_is_zero(s) for s in slots)


def _rechain(base: Node, links: list) -> Node:
    for k, e in links:
        base = make("add_at", (base, k, e))
    return base


def _lower(node: Node) -> list | None:
    """The slots of ``node`` with every link of its chain as a select."""
    base, links = _chain(node)
    slots = _slots(base)
    if slots is None:
        return None
    zero = Const(0.0, TensorType((), node.ttype.dtype))
    out = list(slots)
    for k, e in links:
        for c in range(len(out)):
            hit = make("eq", (k, Const(c, k.ttype)))
            out[c] = _add(out[c], Select(hit, e, zero, e.ttype))
    return out


def _runtime(node: Node, ops: tuple) -> Node | None:
    """``node`` over operands of which at least one is a chain."""
    kind = getattr(node, "kind", None)
    shape = node.ttype.shape
    if kind == "add_at":
        return None                         # a longer chain: already the form
    if isinstance(node, Dispatch):
        return _runtime_dispatch(node, ops)
    if isinstance(node, Select):
        c, a, b = ops
        if c.ttype.shape != () or a.ttype.shape != shape:
            return _lowered(node, ops)
        (ab, al), (bb, bl) = _chain(a), _chain(b)
        zero = Const(0.0, TensorType((), node.ttype.dtype))
        links = ([(k, _gate(c, e, zero, e)) for k, e in al]
                 + [(k, _gate(c, zero, e, e)) for k, e in bl])
        inner = _select(node, (c, ab, bb)) or Select(c, ab, bb, node.ttype)
        return _rechain(inner, links)
    if not isinstance(node, Op):
        return None
    if kind == "component":
        slots = _lower(ops[0])
        return None if slots is None else _component(node, (_assemble(ops[0].ttype.shape, slots, ops[0].ttype),))
    if kind == "sum":
        base, links = _chain(ops[0])
        acc = _sum(node, (base,)) if _slots(base) is not None else make("sum", (base,))
        for _k, e in links:
            acc = _add(acc, e)
        return acc
    if kind in ("add", "sub"):
        a, b = ops
        (ab, al), (bb, bl) = _chain(a), _chain(b)
        if kind == "sub":
            bl = [(k, _neg(e)) for k, e in bl]
        inner = _elementwise(node, (ab, bb)) or make(kind, (ab, bb))
        return _rechain(inner, al + bl)
    if kind == "neg":
        base, links = _chain(ops[0])
        inner = _elementwise(node, (base,)) or make("neg", (base,))
        return _rechain(inner, [(k, _neg(e)) for k, e in links])
    if kind in ("mul", "div"):
        a, b = ops
        if _is_chain(a) and b.ttype.shape == ():
            base, links = _chain(a)
            build = _mul if kind == "mul" else _div
            inner = _elementwise(node, (base, b)) or make(kind, (base, b))
            return _rechain(inner, [(k, build(e, b)) for k, e in links])
        if kind == "mul" and _is_chain(b) and a.ttype.shape == ():
            base, links = _chain(b)
            inner = _elementwise(node, (a, base)) or make(kind, (a, base))
            return _rechain(inner, [(k, _mul(a, e)) for k, e in links])
        return _lowered(node, ops)
    return _lowered(node, ops)


def _gate(cond: Node, on_true: Node, on_false: Node, value: Node) -> Node:
    """``cond ? on_true : on_false`` where ``value`` is the link value in it. A
    link value that is a ``dispatch`` takes the gate into its cases, so it stays
    a member of its switch (and an accumulate of a case, see the emitter)."""
    if not isinstance(value, Dispatch):
        return Select(cond, on_true, on_false, value.ttype)
    zero = on_false if value is on_true else on_true
    return Dispatch(value.selector,
                    [_where(cond, b, zero) if value is on_true else _where(cond, zero, b)
                     for b in value.branches], value.policy, value.ttype)


def _lowered(node: Node, ops: tuple) -> Node | None:
    """``node`` over its operands with every chain lowered to slots."""
    new = []
    for op in ops:
        if _is_chain(op):
            slots = _lower(op)
            if slots is None:
                return None
            op = _assemble(op.ttype.shape, slots, op.ttype)
        new.append(op)
    got = _rewrite(node, tuple(new))
    return got if got is not None else _rebuild(node, new)


def _runtime_dispatch(node: Dispatch, ops: tuple) -> Node | None:
    selector, branches = ops[0], ops[1:]
    if any(b.ttype.shape != node.ttype.shape for b in branches):
        return None
    bases, links = [], []
    for i, b in enumerate(branches):
        base, bl = _chain(b)
        bases.append(base)
        for k, e in bl:
            zk = Const(0, k.ttype)
            ze = Const(0.0, e.ttype)
            links.append((
                Dispatch(selector, [k if j == i else zk for j in range(len(branches))],
                         node.policy, k.ttype),
                Dispatch(selector, [e if j == i else ze for j in range(len(branches))],
                         node.policy, e.ttype)))
    inner = _dispatch(node, (selector, *bases)) or Dispatch(selector, bases, node.policy, node.ttype)
    return _rechain(inner, links)
