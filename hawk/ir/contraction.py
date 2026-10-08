# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Recognising a sum of products as a CONTRACTION — an IR analysis.

Reads contraction semantics off a HAWK Walk into a PLAIN record (ints,
strings, tuples) that :func:`eagle.gemm.plan.lower_dense_gemm` consumes by
DUCK-TYPED attribute access — the seam between HAWK and eagle, neither of
which imports the other.

WHAT IT MATCHES: a bounded ``for`` lowers to a
:class:`~hawk.ir.loop_nodes.Loop`; the matched shape is ONE loop, one
additive carry, and a reduce VARIABLE that is a real symbolic index, its
header giving ``reduce_start``/``reduce_step``/``reduce_extent`` directly
rather than by counting emitted terms. Each operand's reduce-axis stride is
the loop index's COEFFICIENT in its flat index — ANONYMOUS and not
derivable when built from a bare literal (``t.at(r * cols + c)`` folds
``cols`` into Python) rather than a declared
:class:`~hawk.ir.nodes.RoleConst` (``t.at(row=r, col=c)``). An index that
READS data (``t.at(other.at(k))``) walks a runtime-chosen row — a banded
gather, not a matrix. ``term_form`` is eagle's VALUE vocabulary
(``bare_read``/``product_of_reads``/``transformed``), matched against
``eagle.gemm.plan._TERM_FORM_FOR_CLASS``.

CONSERVATIVE BY CONSTRUCTION: an unanswerable question from the DECLARATION
is "no" — an out-of-family shape is ``None``, an underivable index is
``dense=False`` with the reason recorded. It only reads; it builds no IR.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .access import IndexForm, index_form
from .loop_nodes import Loop, LoopIndex, LoopValue
from .loops import addends as _addends
from .nodes import (
    AccumWrite,
    Assign,
    At,
    Const,
    Node,
    Op,
    SampleIndex,
    Sink,
    WideWrite,
)

#: The rank -> declared-element spelling a committed output reports.
_OUTPUT_KIND = {0: "scalar", 1: "vector", 2: "matrix"}


@dataclass(frozen=True)
class ContractionOperand:
    """One plane a recognised contraction reads, described from its
    DECLARATION (field-for-field ``eagle.gemm.contraction.ContractionOperand``).

    ``dims`` is ``((label, size, stride), …)`` in declared order (empty for
    a positional plane); ``reduce_stride`` is the loop index's COEFFICIENT
    in this plane's flat index; ``reduce_dim`` names the declared axis that
    stride belongs to, only when it IS that axis's constant."""

    name: str
    role: str  # "staged" | "named" | "flat"
    dims: tuple
    bound: int
    reads: int
    reduce_stride: int | None
    reduce_dim: str | None
    stride_labels: tuple
    dense: bool
    refusals: tuple


@dataclass(frozen=True)
class RecognizedContraction:
    """An unrolled sum of products described as a contraction — frozen.
    Field-for-field ``eagle.gemm.contraction.RecognizedContraction``.

    ``carry`` is the sink the fold commits to; ``reduce_var`` is the loop's
    own index variable, a real symbolic index read off its header."""

    carry: str
    reduce_var: str
    reduce_extent: int | None
    reduce_start: int | None
    reduce_step: int
    operands: tuple
    operand_class: str
    term_form: str
    outputs: tuple
    dense: bool
    refusals: tuple
    staged_fanin: tuple
    fanin_source: str
    adjoint_accumulates: tuple

    @property
    def operand_count(self) -> int:
        """How many distinct planes the terms read (1 or 2 — ``operand_class``)."""
        return len(self.operands)

    @property
    def operand_names(self) -> tuple:
        """The read planes' declared names, in first-read order."""
        return tuple(op.name for op in self.operands)


# --------------------------------------------------------------------------- #
# The reduce-axis stride: the loop index's coefficient (``index_form``).
# --------------------------------------------------------------------------- #
def _stride_form(index: Node, loop_index: LoopIndex) -> IndexForm | None:
    """The loop index's coefficient in one flat index expression, or
    ``None`` when the expression is not affine in it.

    Read through :func:`hawk.ir.access.index_form` with quotient forms OFF:
    a reduce-axis stride is an exact affine walk, and a dividing index walks
    no matrix axis. An index free of the loop index has coefficient ``0``."""
    return index_form(index, loop_index, quotients=False)


def _walk(node: Node):
    """Every node of one expression, this one included (order-insensitive uses)."""
    yield node
    for child in node.operands:
        yield from _walk(child)


def _reads_data(index: Node) -> bool:
    """Whether an index expression depends on a value READ from a plane — the
    gather that makes the walk's base a runtime choice."""
    return any(isinstance(n, At) for n in _walk(index))


# --------------------------------------------------------------------------- #
# The addend family.
# --------------------------------------------------------------------------- #
def _is_zero(node: Node) -> bool:
    """A literal zero — the identity Python's own ``sum(...)`` seeds a generator
    fold with, and therefore an addend every unrolled contraction carries."""
    return isinstance(node, Const) and node.ttype.shape == ()\
        and float(node.literal) == 0.0


def _reads_of(term: Node) -> list:
    """Every plane read in one term, in first-seen order."""
    return [n for n in _walk(term) if isinstance(n, At)]


def _term_form(terms: Sequence[Node], operands: tuple) -> str:
    """Which of eagle's three term shapes every term is.

    Density says the operands can be WALKED as matrices; this says a matmul
    over them computes what the sum computes — ``sum(x[s,j])`` and
    ``sum(exp(x[s,j] - shift))`` are equally dense but only the first is a
    row sum."""
    if len(operands) == 1 and all(isinstance(t, At) for t in terms):
        return "bare_read"
    if len(operands) != 2:
        return "transformed"
    for term in terms:
        if not (isinstance(term, Op) and term.kind == "mul"):
            return "transformed"
        factors = term.operands
        if not all(isinstance(f, At) for f in factors):
            return "transformed"
        if {f.plane.name for f in factors} != set(operands):
            return "transformed"
    return "product_of_reads"


# --------------------------------------------------------------------------- #
# Declarations, staged fan-in, and the per-operand record.
# --------------------------------------------------------------------------- #
def _declarations(kernel: Any) -> dict:
    """``{plane name: ((label, size, stride), …)}`` off the kernel's own
    declarations, read by ATTRIBUTE (``hawk/ir/`` imports nothing from
    ``hawk/trace/``): anything carrying ``planes`` whose values carry
    ``dims`` drives this identically."""
    out = {}
    for name, plane in (getattr(kernel, "planes", None) or {}).items():
        dims = getattr(plane, "dims", ())
        if dims:
            out[name] = tuple(dims)
    return out


def _staged_names(kernel: Any) -> frozenset:
    """The planes declared as STAGED — a producing launch's buffer read by a
    consumer, which is the only role whose adjoint fan-in is a declared number."""
    return frozenset(
        name for name, plane in (getattr(kernel, "planes", None) or {}).items()
        if getattr(plane, "staged", False)
    )


def _fanin(indices: Sequence[Node]) -> int | None:
    """How many lanes read ONE slot of a staged plane, or ``None``.

    Derived only from the address, and only when it is a fact rather than a
    guess: the plane addressed by the MAJOR half of ``split_index(m)`` means
    the ``m`` consecutive lanes of one major share the slot. Any other
    appearance of the lane index is silence (``None``), since
    ``eagle.gemm.plan.verify_staged_fanin`` treats a WRONG number as a hard
    refusal."""
    divisors, covered, reached = set(), set(), set()
    for index in indices:
        for node in _walk(index):
            if isinstance(node, SampleIndex):
                reached.add(id(node))
            if isinstance(node, Op) and node.kind == "div"\
                    and isinstance(node.operands[0], SampleIndex):
                # a divisor free of the lane index reads back as a constant
                # form whose offset is its value when it is a compile-time int
                divisor = index_form(node.operands[1])
                if divisor is None or divisor.coefficient != 0\
                        or divisor.offset is None:
                    continue
                divisors.add(int(divisor.offset))
                covered.add(id(node.operands[0]))
    if not reached or reached - covered or len(divisors) != 1:
        return None
    return divisors.pop()


def _describe(name: str, indices: list, dims: tuple, staged: bool,
              loop_index: LoopIndex, trip: int) -> ContractionOperand:
    """One operand's record: its declared layout, how its reads walk the
    reduce axis, and every reason that walk is not dense.

    ``indices`` are the flat index expressions WITHIN ONE ITERATION — the
    loop traces once, so one iteration is the whole description."""
    refusals: list = []
    strides: set = set()
    labels: tuple = ()
    anonymous = False

    for index in indices:
        if _reads_data(index):
            refusals.append(
                f"{name}: the index depends on a value read from a plane, so the "
                "row it walks is chosen at runtime"
            )
            break
    for index in indices:
        form = _stride_form(index, loop_index)
        if form is None:
            refusals.append(f"{name}: the index is not affine in the reduce axis")
            continue
        value = int(form.coefficient)
        if form.anonymous and value != 0:
            anonymous = True
            refusals.append(
                f"{name}: the reduce-axis stride is built from an anonymous "
                "integer literal, so it is not derivable from the declaration"
            )
        if value != 0:
            strides.add(value)
            labels = labels or form.labels
    if len(strides) > 1:
        refusals.append(
            f"{name}: its reads disagree on the reduce-axis stride "
            f"({sorted(strides)})")

    stride: int | None
    if not strides:
        stride = 0
    elif len(strides) == 1:
        stride = next(iter(strides))
    else:
        stride = None

    reduce_dim = None
    if len(labels) == 1 and not anonymous:
        reduce_dim = next(
            (d for d, _size, declared in dims
             if labels[0] == f"{name}_{d}_stride" and declared == stride),
            None,
        )
    bound = 1
    for _d, size, _s in dims:
        bound *= int(size) if size else 0
    unique = tuple(dict.fromkeys(refusals))
    return ContractionOperand(
        name=name,
        role="staged" if staged else ("named" if dims else "flat"),
        dims=dims,
        bound=bound,
        reads=len(indices) * trip,
        reduce_stride=stride,
        reduce_dim=reduce_dim,
        stride_labels=labels,
        dense=not unique,
        refusals=unique,
    )


# --------------------------------------------------------------------------- #
# The recogniser.
# --------------------------------------------------------------------------- #
def _sinks_of(obj: Any) -> tuple:
    """``obj``'s sink set, or ``()`` for anything that is not one.

    Duck-typed and total: a kernel carries ``sinks``, a derivative IR IS a
    sink tuple, and anything else is simply not a subject — a ``None``
    recognition, never a ``TypeError``."""
    candidate = getattr(obj, "sinks", obj)
    try:
        sinks = tuple(candidate or ())
    except TypeError:
        return ()
    return sinks if sinks and all(isinstance(s, Sink) for s in sinks) else ()


def _accumulates(obj: Any) -> tuple:
    """Every accumulate sink a derived IR commits to, TOP LEVEL OR IN A LOOP.

    A derived kernel's scatters live inside the mirrored loop when the
    primal gathered inside one; "which planes the adjoint accumulates
    into" is a fact about the derivative, not the scope the write sits in."""
    out: list = []
    pending = list(getattr(obj, "sinks", obj) or ())
    for node in pending:
        if isinstance(node, Loop):
            out.extend(s for s in node.body_sinks if isinstance(s, AccumWrite))
        elif isinstance(node, AccumWrite):
            out.append(node)
    return tuple(out)


def _fold(sink: Sink) -> tuple:
    """``(loop, slot, per-iteration addends)`` for a sink that commits ONE
    lowered additive fold, or ``None``: a carried slot seeded with a
    literal zero (the sum's identity), advanced as ``acc = acc + <terms>``
    with the carry appearing exactly once as a bare addend. A max fold, a
    product recurrence or a carry read twice is not recognised."""
    value = sink.value
    if isinstance(value, Op) and value.kind == "vec":
        # one live component of a vector output padded with literal zeros
        # (a banded KAN stage); two live components are not this family.
        live = [c for c in value.operands if not _is_zero(c)]
        if len(live) != 1:
            return None
        value = live[0]
    if not isinstance(value, LoopValue):
        return None
    loop, slot = value.loop, value.slot
    if loop.trip < 2 or loop.exit_cond is not None or loop.count is not None:
        return None
    if not _is_zero(loop.inits[slot]):
        return None
    carry = loop.carries[slot]
    terms = _addends(loop.nexts[slot])
    if sum(1 for t in terms if t is carry) != 1:
        return None
    rest = [t for t in terms if t is not carry]
    if not rest or any(_touches(t, carry) for t in rest):
        return None
    # no OTHER carried slot may feed this one: a two-slot recurrence whose
    # second slot happens to accumulate is not a contraction over one axis.
    for other in loop.carries:
        if other is not carry and any(_touches(t, other) for t in rest):
            return None
    return loop, slot, rest


def _touches(node: Node, carry: Node) -> bool:
    return any(n is carry for n in _walk(node))


def recognize(kernel: Any, vjp: Any = None) -> RecognizedContraction | None:
    """Describe ``kernel``'s lowered additive fold as a contraction, or
    ``None`` for anything outside the single-additive-fold family.

    ``kernel`` is a traced :class:`hawk.trace.Kernel` (anything carrying
    ``sinks`` and, for declared axes, ``planes``). ``vjp`` is the optional
    DERIVED reverse IR: the accumulate outputs the adjoint contributes to,
    telling a consumer whether the reverse direction lands in shared slots.
    A recognised-but-not-dense contraction still returns, ``dense=False``
    and the reasons named."""
    sinks = _sinks_of(kernel)
    if len(sinks) != 1 or not isinstance(sinks[0], (Assign, WideWrite)):
        return None
    sink = sinks[0]
    fold = _fold(sink)
    if fold is None:
        return None
    loop, _slot, terms = fold

    order: list = []
    per_plane: dict = {}
    for term in terms:
        for read in _reads_of(term):
            name = read.plane.name
            if name not in per_plane:
                order.append(name)
                per_plane[name] = []
            per_plane[name].append(read.index)
    if not 1 <= len(order) <= 2:
        return None

    dims = _declarations(kernel)
    staged = _staged_names(kernel)
    operands = tuple(
        _describe(name, per_plane[name], dims.get(name, ()), name in staged,
                  loop.index, loop.trip)
        for name in order
    )
    fanin = tuple((name, _fanin(per_plane[name]))
                  for name in order if name in staged)
    adjoint = tuple((s.name, False) for s in _accumulates(vjp))
    return RecognizedContraction(
        carry=sink.name,
        reduce_var=loop.index.name,
        reduce_extent=loop.trip,
        reduce_start=loop.start,
        reduce_step=loop.step,
        operands=operands,
        operand_class="gemv" if len(operands) == 1 else "gemm",
        term_form=_term_form(terms, tuple(order)),
        outputs=((sink.name, _OUTPUT_KIND[len(sink.ttype.shape)]),),
        dense=all(op.dense for op in operands),
        refusals=tuple(r for op in operands for r in op.refusals),
        staged_fanin=fanin,
        fanin_source="primal_walk" if fanin else "none",
        adjoint_accumulates=adjoint,
    )
