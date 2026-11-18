# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Recognising a sum of products as a CONTRACTION — an IR analysis.

Reads contraction semantics off a HAWK Walk into a PLAIN record (ints,
strings, tuples) that :func:`eagle.gemm.plan.lower_dense_gemm` consumes by
DUCK-TYPED attribute access — the seam between HAWK and eagle, neither of
which imports the other.

WHAT IT MATCHES: a bounded ``for`` lowers to a
``loop`` node; the matched shape is ONE loop, one
additive carry, and a reduce VARIABLE that is a real symbolic index, its
header giving ``reduce_start``/``reduce_step``/``reduce_extent`` directly
rather than by counting emitted terms. Each operand's reduce-axis stride is
the loop index's COEFFICIENT in its flat index — ANONYMOUS and not
derivable when built from a bare literal (``t.at(r * cols + c)`` folds
``cols`` into Python) rather than a declared ``role_const``
(``t.at(row=r, col=c)``). An index that
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

from .. import _ir

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
    reduce_stride: int | str | None
    reduce_dim: str | None
    stride_labels: tuple
    dense: bool
    refusals: tuple


@dataclass(frozen=True)
class RecognizedContraction:
    """An unrolled sum of products described as a contraction — frozen.
    Field-for-field ``eagle.gemm.contraction.RecognizedContraction``.

    ``carry`` is the sink the fold commits to; ``reduce_var`` is the loop's
    own index variable, a real symbolic index read off its header.

    ``lane_major`` is the NAME of the operand the lanes are major in, or ``None``:
    a lane is one point ``(i, j)`` of a grid of ``m`` columns (``lane = i * m + j``),
    one operand is read through the major half ``lane // m`` alone and the other
    through the minor half ``lane - (lane // m) * m``. For each operand the kinds of the
    operations applied DIRECTLY to the lane index in its read indices are
    collected. Operand X is lane-major when its kinds are exactly ``{"div"}``,
    the other operand's kinds are a strict superset of ``{"div"}``, and ``m``, the
    divisor of both, is X's staged fan-in and the other operand's free extent (its one
    declared axis that is not the reduced one). Any other shape of the reads is ``None``."""

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
    lane_major: str | None

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
def _stride_form(index: Any, loop_index: Any) -> _ir.IndexForm | None:
    """The loop index's coefficient in one flat index expression, or
    ``None`` when the expression is not affine in it.

    Read through :func:`hawk._ir.index_form` with quotient forms OFF:
    a reduce-axis stride is an exact affine walk, and a dividing index walks
    no matrix axis. An index free of the loop index has coefficient ``0``."""
    return _ir.index_form(index, loop_index, quotients=False)


def _walk(node: Any):
    """Every node of one expression, this one included (order-insensitive uses)."""
    yield node
    for child in node.operands:
        yield from _walk(child)


def _reads_data(index: Any) -> bool:
    """Whether an index expression depends on a value READ from a plane — the
    gather that makes the walk's base a runtime choice."""
    return any(_ir.is_a(n, "at") for n in _walk(index))


def _op_kind(node: Any) -> str | None:
    """The operation of an ``op`` node, ``None`` for any other node."""
    return node.field_text("kind") if _ir.is_a(node, "op") else None


# --------------------------------------------------------------------------- #
# The addend family.
# --------------------------------------------------------------------------- #
def _is_zero(node: Any) -> bool:
    """A literal zero — the identity Python's own ``sum(...)`` seeds a generator
    fold with, and therefore an addend every unrolled contraction carries."""
    if not _ir.is_a(node, "const"):
        return False
    ttype = _ir.type_of(node)
    return ttype.shape == () and ttype.dtype != "bool" and node.field_real("literal") == 0.0


def _is_component(node: Any) -> bool:
    """A runtime component read of a wide value: ``bf[j]`` with ``j`` traced."""
    return _op_kind(node) == "component_at"


def _is_read(node: Any) -> bool:
    """A plane read (``at``) or a wide value's component read — the two
    shapes one term of a contraction multiplies."""
    return _ir.is_a(node, "at") or _is_component(node)


def _reads_of(term: Any) -> list:
    """Every read in one term, in first-seen order: plane reads and
    component reads of a wide value.

    A component read is ONE read of its value: the value (a ``Vector``
    input, or a computed wide value built outside the loop) is not walked,
    only the index is, since a gather in the index is still a gather."""
    out: list = []

    def visit(node: Any) -> None:
        if _is_component(node):
            out.append(node)
            visit(node.operands[1])
            return
        if _ir.is_a(node, "at"):
            out.append(node)
        for child in node.operands:
            visit(child)

    visit(term)
    return out


def _read_name(read: Any, names: dict) -> str:
    """The operand a read belongs to: the plane's declared name, the wide
    input's own name, or ``computed#k`` (first-seen order) for a wide value
    built in the kernel. ``names`` is the call's ``{value: name}`` memo."""
    if _ir.is_a(read, "at"):
        return read.operands[0].field_text("name")
    source = read.operands[0]
    if _ir.is_a(source, "leaf"):
        return source.field_text("name")
    return names.setdefault(source, f"computed#{len(names)}")


def _read_index(read: Any) -> Any:
    """The flat index a read walks: ``at``'s index without the clamp a table read holds it in, or a
    component's."""
    index = read.operands[1]
    if _ir.is_a(read, "at") and _op_kind(index) == "clamp_index":
        return index.operands[0]
    return index


def _term_form(terms: Sequence, operands: tuple, names: dict) -> str:
    """Which of eagle's three term shapes every term is.

    Density says the operands can be WALKED as matrices; this says a matmul
    over them computes what the sum computes — ``sum(x[s,j])`` and
    ``sum(exp(x[s,j] - shift))`` are equally dense but only the first is a
    row sum."""
    if len(operands) == 1 and all(_is_read(t) for t in terms):
        return "bare_read"
    if len(operands) != 2:
        return "transformed"
    for term in terms:
        if _op_kind(term) != "mul":
            return "transformed"
        factors = term.operands
        if not all(_is_read(f) for f in factors):
            return "transformed"
        if {_read_name(f, names) for f in factors} != set(operands):
            return "transformed"
    return "product_of_reads"


# --------------------------------------------------------------------------- #
# Declarations, staged fan-in, and the per-operand record.
# --------------------------------------------------------------------------- #
def _declarations(kernel: Any) -> dict:
    """``{plane name: ((label, size, stride), …)}`` off the kernel's own
    declarations, read by ATTRIBUTE (``raptor/ir/`` imports nothing from
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


def _fanin(indices: Sequence) -> int | None:
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
            if _ir.is_a(node, "sample_index"):
                reached.add(node)
            divisor = _lane_divisor(node)
            if divisor is not None:
                divisors.add(divisor)
                covered.add(node.operands[0])
    if not reached or reached - covered or len(divisors) != 1:
        return None
    return divisors.pop()


def _lane_divisor(node: Any) -> int | None:
    """``m`` of a node ``lane // m`` with a compile-time integer ``m``, else ``None``."""
    if _op_kind(node) != "div" or not _ir.is_a(node.operands[0], "sample_index"):
        return None
    # a divisor free of the lane index reads back as a constant form whose offset is its value when it is a compile-time int
    divisor = _ir.index_form(node.operands[1])
    if divisor is None or divisor.coefficient != 0 or divisor.offset is None:
        return None
    return int(divisor.offset)


def _lane_ops(indices: Sequence) -> tuple:
    """``(kinds, divisors)`` of the lane index in ``indices``: the kinds of the operations applied directly to it and the
    divisors of its integer divisions."""
    kinds, divisors = set(), set()
    for index in indices:
        for node in _walk(index):
            if any(_ir.is_a(child, "sample_index") for child in node.operands):
                kinds.add(_op_kind(node))
            divisor = _lane_divisor(node)
            if divisor is not None:
                divisors.add(divisor)
    return kinds, divisors


def _free_extent(operand: ContractionOperand) -> int | None:
    """The extent of the one declared axis of ``operand`` that is not the reduced one, or ``None``."""
    free = [size for label, size, _stride in operand.dims if label != operand.reduce_dim]
    return free[0] if len(free) == 1 else None


def _lane_major(operands: tuple, per_plane: dict) -> str | None:
    """The operand the lanes are major in (:class:`RecognizedContraction`'s ``lane_major``), or ``None``."""
    if len(operands) != 2:
        return None
    for major, minor in (operands, operands[::-1]):
        kinds, divisors = _lane_ops(per_plane[major.name])
        other_kinds, other_divisors = _lane_ops(per_plane[minor.name])
        fanin = _fanin(per_plane[major.name])
        if (kinds == {"div"} and other_kinds > {"div"} and fanin is not None
                and divisors == other_divisors == {fanin} and _free_extent(minor) == fanin):
            return major.name
    return None


def _describe(name: str, indices: list, dims: tuple, staged: bool,
              loop_index: Any, trip: int, wide: str | None = None,
              width: int = 0) -> ContractionOperand:
    """One operand's record: its declared layout, how its reads walk the
    reduce axis, and every reason that walk is not dense.

    ``indices`` are the flat index expressions WITHIN ONE ITERATION — the
    loop traces once, so one iteration is the whole description. ``wide`` is
    ``"wide"`` (a vector input) or ``"computed"`` (a value built in the
    kernel) for component reads, whose single axis is ``width`` long."""
    refusals: list = []
    symbolic = False
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
        if form.symbol:                 # a run-time stride: the sample count, once per step of the axis
            symbolic = True
            if value == 1:
                strides.add(form.symbol)
            else:
                refusals.append(f"{name}: the reduce-axis stride is a multiple of the sample count")
            continue
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
            f"({sorted(strides, key=str)})")

    stride: int | str | None
    if not strides:
        stride = 0
    elif len(strides) == 1:
        stride = next(iter(strides))
    else:
        stride = None

    reduce_dim = None
    if symbolic and stride is not None:
        reduce_dim = next((d for d, _size, declared in dims if declared == stride), None)
    elif len(labels) == 1 and not anonymous:
        reduce_dim = next(
            (d for d, _size, declared in dims
             if labels[0] == f"{name}_{d}_stride" and declared == stride),
            None,
        )
    bound = 1
    for _d, size, _s in dims:
        bound *= int(size) if size else 0
    unique = tuple(dict.fromkeys(refusals))
    if wide is not None:
        # a wide value is addressed by component alone: one declared axis,
        # unit stride, and "the axis" is the component axis only when the
        # walk is dense.
        return ContractionOperand(
            name=name, role=wide, dims=(("component", width, 1),), bound=width,
            reads=len(indices) * trip, reduce_stride=stride,
            reduce_dim="component" if stride == 1 and not unique else None,
            stride_labels=labels, dense=not unique, refusals=unique)
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
    is_sink = all(isinstance(s, _ir.Node) and _ir.is_a(s, "sink") for s in sinks)
    return sinks if sinks and is_sink else ()


def _accumulates(obj: Any) -> tuple:
    """Every accumulate sink a derived IR commits to, TOP LEVEL OR IN A LOOP.

    A derived kernel's scatters live inside the mirrored loop when the
    primal gathered inside one; "which planes the adjoint accumulates
    into" is a fact about the derivative, not the scope the write sits in."""
    out: list = []
    pending = list(getattr(obj, "sinks", obj) or ())
    for node in pending:
        if _ir.is_a(node, "loop"):
            out.extend(s for s in node.field_nodes("body_sinks") if _ir.is_a(s, "accum_write"))
        elif _ir.is_a(node, "accum_write"):
            out.append(node)
    return tuple(out)


def _fold(sink: Any) -> tuple:
    """``(loop, slot, per-iteration addends)`` for a sink that commits ONE
    lowered additive fold, or ``None``: a carried slot seeded with a
    literal zero (the sum's identity), advanced as ``acc = acc + <terms>``
    with the carry appearing exactly once as a bare addend. A max fold, a
    product recurrence or a carry read twice is not recognised."""
    value = sink.operands[0]
    if _op_kind(value) == "vec":
        # one live component of a vector output padded with literal zeros
        # (a banded spline stage); two live components are not this family.
        live = [c for c in value.operands if not _is_zero(c)]
        if len(live) != 1:
            return None
        value = live[0]
    if not _ir.is_a(value, "loop_value"):
        return None
    loop, slot = _ir.loop_value_of(value)
    exits = _ir.exit_kwargs(loop)
    if loop.field_int("trip") < 2 or exits["exit_cond"] is not None or exits["count"] is not None:
        return None
    if not _is_zero(loop.operands[slot]):      # the first operands are the carries' initial values
        return None
    carries = loop.field_nodes("carries")
    carry = carries[slot]
    terms = _ir.addends(loop.field_nodes("nexts")[slot])
    if sum(1 for t in terms if t == carry) != 1:
        return None
    rest = [t for t in terms if t != carry]
    if not rest or any(_touches(t, carry) for t in rest):
        return None
    # no OTHER carried slot may feed this one: a two-slot recurrence whose
    # second slot happens to accumulate is not a contraction over one axis.
    for other in carries:
        if other != carry and any(_touches(t, other) for t in rest):
            return None
    return loop, slot, rest


def _touches(node: Any, carry: Any) -> bool:
    return any(n == carry for n in _walk(node))


def recognize(kernel: Any, vjp: Any = None, *, mapping: dict | None = None
              ) -> RecognizedContraction | None:
    """Describe ``kernel``'s lowered additive fold as a contraction, or
    ``None`` for anything outside the additive-fold family: one or several
    folds over ONE loop, reading planes and components of wide values.

    ``kernel`` is a traced :class:`hawk.trace.Kernel` (anything carrying
    ``sinks`` and, for declared axes, ``planes``). ``vjp`` is the optional
    DERIVED reverse IR: the accumulate outputs the adjoint contributes to,
    telling a consumer whether the reverse direction lands in shared slots.
    A recognised-but-not-dense contraction still returns, ``dense=False``
    and the reasons named.

    ``mapping`` is :func:`raptor.wings.simt.lanes.choose_mapping`'s record: when the
    device card chose ``"direct"`` for the scalar type, the fold stays the
    kernel's own loop and nothing is recognised (``None``)."""
    if mapping is not None and mapping.get("contraction") == "direct":
        return None
    sinks = _sinks_of(kernel)
    if not sinks or not all(_ir.is_a(s, "assign", "wide_write") for s in sinks):
        return None
    folds = [_fold(sink) for sink in sinks]
    if any(fold is None for fold in folds):
        return None
    # several sinks are ONE contraction only when they fold over the SAME
    # loop (one reduce axis, one header); two loops are two contractions.
    loop = folds[0][0]
    if any(fold[0] != loop for fold in folds):
        return None
    terms = [t for _loop, _slot, rest in folds for t in rest]

    names: dict = {}
    order: list = []
    per_plane: dict = {}
    wide: dict = {}
    for term in terms:
        for read in _reads_of(term):
            name = _read_name(read, names)
            if name not in per_plane:
                order.append(name)
                per_plane[name] = []
            per_plane[name].append(_read_index(read))
            if _is_component(read):
                source = read.operands[0]
                wide[name] = ("wide" if _ir.is_a(source, "leaf") else "computed",
                              _ir.type_of(source).shape[0])
    if not 1 <= len(order) <= 2:
        return None

    dims = _declarations(kernel)
    staged = _staged_names(kernel)
    loop_index = loop.field_nodes("index")[0]
    trip = loop.field_int("trip")
    operands = tuple(
        _describe(name, per_plane[name], dims.get(name, ()), name in staged,
                  loop_index, trip, *wide.get(name, (None, 0)))
        for name in order
    )
    fanin = tuple((name, _fanin(per_plane[name]))
                  for name in order if name in staged)
    adjoint = tuple((s.field_text("name"), False) for s in _accumulates(vjp))
    return RecognizedContraction(
        carry=sinks[0].field_text("name"),
        reduce_var=loop_index.field_text("name"),
        reduce_extent=trip,
        reduce_start=loop.field_int("start"),
        reduce_step=loop.field_int("step"),
        operands=operands,
        operand_class="gemv" if len(operands) == 1 else "gemm",
        term_form=_term_form(terms, tuple(order), names),
        outputs=tuple(
            (s.field_text("name"), _OUTPUT_KIND[len(_ir.type_of(s).shape)]) for s in sinks
        ),
        dense=all(op.dense for op in operands),
        refusals=tuple(r for op in operands for r in op.refusals),
        staged_fanin=fanin,
        fanin_source="primal_walk" if fanin else "none",
        adjoint_accumulates=adjoint,
        lane_major=_lane_major(operands, per_plane),
    )
