# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Access-class inference over a :class:`~hawk.ir.walk.Walk` (…).

HAWK DECLARES an access class; eagle's ``eagle.exec.check_placement`` refuses
an illegal placement. The classifier reads the walk's access FORMS, never
node classes: an :class:`~hawk.ir.nodes.At` read of a ``lookup`` plane is
``cross_sample_read``, a wide/accum sink carrying a scatter index is
``cross_sample_write``, everything else is ``sample_local``. The MOST
RESTRICTIVE of the three wins — over-restriction costs only a placement,
under-restriction is a wrong answer under ``RankPartition``. ``mapreduce``
is DISJOINT (an un-folded partial is a wrong answer, not a placement cost).
No annotation surface exists except for injected raw text (``@raw_device``):
a traced body can never carry a hand-written class.

The same module answers a finer question: how ONE index expression moves
with the sample index (:func:`index_form`) — a contraction's reduce-axis
stride, a scatter whose lanes share a key — so the descent over index
arithmetic exists once.
"""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
from typing import TYPE_CHECKING, NamedTuple

from .loop_nodes import Loop
from .nodes import (
    FUSED_STEPS_PLANE,
    RAW_KIND,
    AccumWrite,
    At,
    Const,
    HawkError,
    MapreducePartial,
    Node,
    Op,
    RoleConst,
    SampleIndex,
    WideWrite,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .walk import Walk


#: Access classes, LEAST to MOST restrictive (``mapreduce`` is disjoint).
ACCESS_CLASSES = ("sample_local", "cross_sample_read", "cross_sample_write",
                  "mapreduce")

_RANK = {"sample_local": 0, "cross_sample_read": 1, "cross_sample_write": 2}


class Access(NamedTuple):
    """The ``(class, op|None)``; ``op`` is set only for ``mapreduce``."""

    cls: str
    op: str | None = None


def infer_over(order: Iterable[Node]) -> Access:
    """The classifier proper, over a canonical node ORDER.

    A :class:`~hawk.ir.loop_nodes.Loop`'s body is NOT in that order (it's a
    scope the emitter renders into), so the classifier descends into it
    here: a table read at the loop index is ``cross_sample_read`` whether
    the ``for`` was unrolled or lowered, and missing that would let eagle
    place a kernel under a ``RankPartition`` that splits the plane it
    reads — a wrong answer, not a placement cost.

    One read is NOT a form of the body: a ``steps="auto"`` loop's trip
    count, the reserved ``fused_steps`` word the runner sets (one cell, the
    same for every sample). It is the loop's ``count`` operand, which no
    traced body can mint, so a kernel's class is its single step's."""
    order = tuple(order)
    seam = {id(_step_word(node.count)) for node in _with_loop_bodies(order)
            if isinstance(node, Loop) and _is_step_count(node.count)}
    cls = "sample_local"
    for node in _with_loop_bodies(order):
        if id(node) in seam:
            continue
        if isinstance(node, MapreducePartial):
            # guarantees this sink is the only one, so it decides alone.
            return Access("mapreduce", node.op)
        if isinstance(node, At):
            # An absolute index into a lookup plane — cross-sample by form,
            # regardless of what it evaluates to (over-restriction is cheap).
            cls = _more_restrictive(cls, "cross_sample_read")
        elif isinstance(node, (WideWrite, AccumWrite)) and node.index is not None:
            cls = _more_restrictive(cls, "cross_sample_write")
        elif isinstance(node, Op) and node.kind == RAW_KIND:
            #: the walk cannot see inside spliced text, so the block's own
            # DECLARED class folds in here; an unannotated one is refused.
            cls = _more_restrictive(cls, _declared(node))
    return Access(cls, None)


def _is_step_count(count: Node | None) -> bool:
    """Whether ``count`` is the read of the reserved ``fused_steps`` word
    (possibly clamped)."""
    return _step_word(count) is not None


def _step_word(count: Node | None) -> Node | None:
    """The ``fused_steps`` read inside a loop's ``count``, else ``None``."""
    stack, seen = [count] if count is not None else [], set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if (isinstance(node, At) and node.plane.role == "lookup"
                and node.plane.name == FUSED_STEPS_PLANE
                and isinstance(node.index, Const)):
            return node
        stack.extend(node.operands)
    return None


def _with_loop_bodies(order: Iterable[Node]) -> Iterable[Node]:
    """``order``, with every lowered loop's body spliced in after it."""
    for node in order:
        yield node
        if isinstance(node, Loop):
            yield from node.body_sinks
            yield from _with_loop_bodies(node.body_nodes())


def infer(walk: Walk) -> Access:
    """The entry point: the access class + op a manifest declares."""
    return infer_over(walk.order)


def _declared(node: Op) -> str:
    """The access class a raw block declares; refused when absent."""
    declared = getattr(node.literal, "access", None)
    if declared not in _RANK:
        raise HawkError(
            f"a spliced raw block reached the classifier declaring access="
            f"{declared!r}: @raw_device requires an explicit access= annotation "
            f"drawn from {tuple(_RANK)} — the walk cannot see inside "
            "the text, and a mapreduce class is a sink property, not a block's"
        )
    return declared


def _more_restrictive(a: str, b: str) -> str:
    return a if _RANK[a] >= _RANK[b] else b


# --------------------------------------------------------------------------- #
# The index-form classifier.
# --------------------------------------------------------------------------- #
class IndexForm(NamedTuple):
    """How one integer index expression moves with an index variable ``s``.

    ``coefficient``/``offset`` are exact rationals: integer coefficient
    means ``coefficient * s + offset`` for every ``s`` (``0`` = no
    dependence); non-integer means it depends on ``s`` only through
    ``floor(coefficient * s + offset)`` — lanes share a value in runs of
    about ``1 / coefficient`` (a quotient form).

    ``offset`` is ``None`` when the ``s``-free part isn't a compile-time
    integer. ``labels`` are the declaration-named constants scaled by;
    ``anonymous`` marks a bare-literal scaling factor instead."""

    coefficient: Fraction
    offset: Fraction | None
    labels: tuple = ()
    anonymous: bool = False


def index_form(node: Node, var: Node | None = None, *,
               quotients: bool = True) -> IndexForm | None:
    """Classify ``node`` as affine in the index ``var``, or return ``None``.

    ``var`` is matched by IDENTITY (a loop's
    :class:`~hawk.ir.loop_nodes.LoopIndex`); ``None`` means the sample
    index, matched by kind. A compile-time integer factor is a
    :class:`~hawk.ir.nodes.RoleConst` or an ``i32``/``i64``
    :class:`~hawk.ir.nodes.Const`.

    * free of ``var`` — coefficient ``0``, offset its value if compile-time,
      else ``None``; ``var`` itself — ``(1, 0)``;
    * ``a + b``/``a - b`` of two exact forms add; a quotient shifted by a
      ``var``-free term keeps its coefficient/offset;
    * ``a * k``/``k * a`` scales an exact form by compile-time ``k``; a
      quotient keeps its coefficient (``k == 0`` -> ``(0, 0)``);
    * with ``quotients`` (default), ``a / k`` of a non-negative exact form
      by positive compile-time ``k`` is ``(c/k, d/k)`` (truncating ``/``
      then equals ``floor``) — the major half of
      :func:`hawk.math.split_index`.

    Anything else mentioning ``var`` is ``None`` (a product of two
    ``var``-carrying terms, a runtime factor/divisor, the periodic minor
    half, a nested quotient, a bad division, any other op). Conservative:
    ``None`` claims nothing."""
    if not _carries(node, var):
        return IndexForm(Fraction(0), _int_value(node))
    if _is_var(node, var):
        return IndexForm(Fraction(1), Fraction(0))
    if not isinstance(node, Op):
        return None
    if node.kind in ("add", "sub"):
        return _sum_form(node, var, quotients)
    if node.kind == "mul":
        return _product_form(node, var, quotients)
    if node.kind == "div" and quotients:
        return _quotient_form(node, var)
    return None


def _is_var(node: Node, var: Node | None) -> bool:
    return isinstance(node, SampleIndex) if var is None else node is var


def _carries(node: Node, var: Node | None) -> bool:
    """Whether ``var`` appears anywhere under ``node`` (this node included)."""
    pending = [node]
    while pending:
        current = pending.pop()
        if _is_var(current, var):
            return True
        pending.extend(current.operands)
    return False


def _int_factor(node: Node) -> tuple:
    """``(value, label)`` for a compile-time integer, ``(None, None)``
    otherwise: a :class:`~hawk.ir.nodes.RoleConst` carries its label; a
    plain integer :class:`~hawk.ir.nodes.Const` the same number with none."""
    if isinstance(node, RoleConst):
        return int(node.literal), node.label
    if isinstance(node, Const) and node.ttype.dtype in ("i32", "i64"):
        return int(node.literal), None
    return None, None


def _int_value(node: Node) -> Fraction | None:
    value, _label = _int_factor(node)
    return None if value is None else Fraction(value)


def _exact(form: IndexForm) -> bool:
    return form.coefficient.denominator == 1


def _sum_form(node: Op, var: Node | None, quotients: bool) -> IndexForm | None:
    left, right = (index_form(x, var, quotients=quotients) for x in node.operands)
    if left is None or right is None:
        return None
    sign = 1 if node.kind == "add" else -1
    if _exact(left) and _exact(right):
        offset = (None if left.offset is None or right.offset is None
                  else left.offset + sign * right.offset)
        return IndexForm(left.coefficient + sign * right.coefficient, offset,
                         left.labels + right.labels,
                         left.anonymous or right.anonymous)
    # a quotient form survives only a shift by a term free of ``var``
    if left.coefficient == 0:
        quotient = right
    elif right.coefficient == 0:
        quotient = left
    else:
        return None
    if sign == -1 and quotient is right:
        return None                     # b - q: outside the named shapes
    return quotient


def _product_form(node: Op, var: Node | None,
                  quotients: bool) -> IndexForm | None:
    a, b = node.operands
    a_has, b_has = _carries(a, var), _carries(b, var)
    if a_has and b_has:
        return None                     # var * var: not a strided walk
    term, factor_node = (a, b) if a_has else (b, a)
    factor, label = _int_factor(factor_node)
    if factor is None:
        return None                     # scaled by a runtime quantity
    inner = index_form(term, var, quotients=quotients)
    if inner is None:
        return None
    labels = inner.labels + ((label,) if label is not None else ())
    anonymous = inner.anonymous or label is None
    if _exact(inner):
        offset = None if inner.offset is None else inner.offset * factor
        return IndexForm(inner.coefficient * factor, offset, labels, anonymous)
    if factor == 0:
        return IndexForm(Fraction(0), Fraction(0), labels, anonymous)
    return IndexForm(inner.coefficient, inner.offset, labels, anonymous)


def _quotient_form(node: Op, var: Node | None) -> IndexForm | None:
    if node.ttype.dtype not in ("i32", "i64"):
        return None                     # a real division is not an index form
    numerator, divisor = node.operands
    if _carries(divisor, var):
        return None
    k, label = _int_factor(divisor)
    if k is None or k <= 0:
        return None
    inner = index_form(numerator, var, quotients=True)
    if inner is None or not _exact(inner) or inner.coefficient <= 0\
            or inner.offset is None or inner.offset < 0:
        return None
    labels = inner.labels + ((label,) if label is not None else ())
    anonymous = inner.anonymous or label is None
    coefficient = inner.coefficient / k
    if coefficient.denominator == 1:
        return IndexForm(coefficient, Fraction(int(inner.offset) // k), labels,
                         anonymous)
    return IndexForm(coefficient, inner.offset / k, labels, anonymous)
