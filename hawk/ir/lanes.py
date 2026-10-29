# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Lane-space FACTS of a kernel — an IR analysis.

Answers, per axis of the lane space, how every plane read and every sink KEY
moves with that axis, into a PLAIN record (strings, ints, tuples) that a
consumer reads by duck-typed attribute access — the seam style of
:mod:`hawk.ir.contraction`. It only reads: it builds no IR and changes no
emitted text.

The AXES come from the kernel's own ``split_index(minor)`` uses (one
``lane / k`` divisor each), in numpy order: an undeclared kernel has ONE axis;
one divisor ``k`` gives a runtime first axis and a static last axis of extent
``k``; nested divisors that divide one another give a chain. The flat lane is
``sum(axis_i * stride_i)`` (strides are the divisors; the last is 1), the
fastest axis last. A divisor set that is not such a chain falls back to the
single flat axis (conservative). ``labels`` name the axes; ``group`` is the
kernel's declared group, a lane count.

PER-AXIS CLASS of an index expression, from an affine algebra over the axes:

* ``invariant``      the index does not move along the axis;
* ``affine``         moves as ``coefficient * axis + rest`` (coefficient = stride);
* ``quotient``       moves through ``floor(coefficient * axis + ...)``, a
                     fractional coefficient (lanes share a value in runs);
* ``data_dependent`` moves because it carries a value READ from a plane (gather);
* ``non_affine``     any other dependence (a product of lane terms, a real
                     division, a ``floor``, a loop carry, ...).

Unknown means NOT invariant. ``gather`` on a site is True when the index
carries a plane-read value that varies with a lane axis; its per-axis ``base`` forms are then those of
the data-free part (a gather into a row the lane chooses by its coordinates).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import lcm
from typing import Any

from .loop_nodes import Loop, LoopCarry, LoopCount, LoopIndex
from .nodes import KEYED_AUTO, At, Const, Dispatch, Finish, Leaf, Node, Op, SampleIndex, Sink

INVARIANT, AFFINE, QUOTIENT = "invariant", "affine", "quotient"
DATA_DEPENDENT, NON_AFFINE = "data_dependent", "non_affine"

#: Weakest-to-strongest dependence, for joining sites of one plane.
_ORDER = (INVARIANT, AFFINE, QUOTIENT, NON_AFFINE, DATA_DEPENDENT)


@dataclass(frozen=True)
class LaneAxis:
    """One axis of the lane space: ``label`` is ``None`` when undeclared,
    ``extent`` is ``None`` for the runtime first axis, ``stride`` is the
    flat-lane distance of one step (the ``lane_<label>_stride``)."""

    label: str | None
    extent: int | None
    stride: int


@dataclass(frozen=True)
class AxisForm:
    """How one index moves along one axis. ``coefficient`` is set for
    ``affine`` / ``quotient`` (an exact ``Fraction``)."""

    cls: str
    coefficient: Fraction | None = None


@dataclass(frozen=True)
class ReadFact:
    """One ``At`` read site. ``forms``/``base`` are per-axis (axis order).
    ``base`` is the data-free part of a gather. ``offset`` is the index's
    constant term (``None`` unknown); ``lane_offset`` is that constant when
    the index is exactly ``lane + offset`` (a neighbour/stencil read)."""

    plane: str
    role: str
    forms: tuple
    base: tuple
    gather: bool
    offset: Fraction | None
    lane_offset: int | None
    #: ``(row, offset, slot)`` when the index is ``row + offset [+ clamp]`` with the
    #: data-free ``row`` invariant along the fastest axis (see :func:`_tile_site`):
    #: ``row`` names its non-constant terms, ``slot`` is the clamp's static bound or ``None``.
    tile_site: tuple | None = None


@dataclass(frozen=True)
class SinkFact:
    """One sink. ``keyed`` is False for an own-column write (it moves with
    every axis); a keyed sink's ``forms`` are its KEY's."""

    name: str
    kind: str
    keyed: bool
    forms: tuple
    gather: bool
    #: keyed sinks only (``None`` for an own-column sink); see :func:`_key_facts`.
    #: ``key_origin``: invariant_axis | index_plane_sorted | index_plane_unsorted | data_dependent.
    key_origin: str | None = None
    #: static EXCLUSIVE upper bound of the key, or ``None`` (unknown = no).
    key_extent: int | None = None
    #: dense-slot form ``key = base + slot``: the slot's static bound G, else ``None``.
    slot_extent: int | None = None
    #: per-axis forms of ``base`` whenever the key has the ``base + clamp`` shape.
    slot_base_forms: tuple = ()
    #: the sum contract: the sink's own policy when it names one, else
    #: ``"compensated"`` when the sink carries a companion plane, else ``"auto"``.
    policy: str = "auto"

    def invariant_axes(self, facts: "LaneFacts") -> tuple:
        return tuple(a.label for a, f in zip(facts.axes, self.forms) if f.cls == INVARIANT)


@dataclass(frozen=True)
class BoxCandidate:
    """A non-fastest axis ``axis`` along which every site of each of ``planes``
    is invariant: lanes that differ only in that axis read the same values, so
    one staged copy serves ``reuse`` of them (``None`` = the runtime first
    axis). ``fastest_extent`` is the static width the staged run spans."""

    axis: str
    planes: tuple
    reuse: int | None
    fastest_extent: int | None
    static: bool
    #: the axis's position in :attr:`LaneFacts.axes` (a label may be ``None``).
    index: int = 0


#: Words a row of shared memory serves without a bank conflict.
BANK_WORDS = 32

#: What a card may choose for a contraction, and the fields of its ``tile`` block.
CONTRACTIONS = ("gemm", "direct")
TILE_FIELDS = ("kind", "SB", "T", "R", "EB", "smem")


@dataclass(frozen=True)
class GatherTile:
    """A plane read as ``base + slot`` (a clamped data-dependent slot of static
    bound ``slot_extent``) whose ``base`` does not move along the fastest axis:
    every lane of a fastest-axis run reads the same row of ``row_words`` words
    of ``plane``, so a block can stage that row once."""

    plane: str
    slot_extent: int
    row_words: int

    @property
    def fits_warp_row(self) -> bool:
        return self.row_words <= BANK_WORDS


@dataclass(frozen=True)
class LaneFacts:
    axes: tuple
    reads: tuple
    sinks: tuple
    notes: tuple = ()
    #: thread/block box candidates, one per non-fastest axis with an invariant read.
    box_candidates: tuple = ()
    #: True only when the walk holds no per-lane loop exit, count, finish,
    #: terminated read or lane-varying dispatch; unknown = False. It does NOT
    #: see the launch's own tail guard or an external active set.
    uniform_control: bool = False
    #: the kernel's declared group (a lane count, default 1).
    group: int = 1
    #: the least lane count every proven key run and the declared ``group`` divide.
    required_group: int = 1
    #: planes a block could stage as a gather tile, one per plane.
    gather_tiles: tuple = ()
    #: the chosen lowering per pattern (see :func:`choose_mapping`).
    mapping: dict = field(default_factory=dict, compare=False)

    def axis_index(self, axis: Any) -> int:
        if isinstance(axis, int):
            return axis
        labels = [a.label for a in self.axes]
        if axis is None or axis not in labels:
            raise ValueError(f"no axis labelled {axis!r} in {labels}")
        return labels.index(axis)

    def read_classes(self, plane: str, axis: Any) -> tuple:
        """The set of per-axis classes over the read sites of ``plane``."""
        i = self.axis_index(axis)
        return tuple(sorted({r.forms[i].cls for r in self.reads if r.plane == plane},
                            key=_ORDER.index))

    def invariant_reads(self, axis: Any) -> tuple:
        """Planes ALL of whose read sites are invariant along ``axis``."""
        i = self.axis_index(axis)
        names = sorted({r.plane for r in self.reads})
        return tuple(n for n in names
                     if all(r.forms[i].cls == INVARIANT for r in self.reads if r.plane == n))


# --------------------------------------------------------------------------- #
# The value algebra: const + sum(coef_a * axis_a), plus "varies in some other way".
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _V:
    lin: tuple = ()                 # sorted ((axis, Fraction), ...), no zeros
    const: Fraction | None = Fraction(0)
    bad: frozenset = frozenset()    # axes it varies along non-affinely
    data: bool = False              # carries a plane-read value
    dbad: frozenset = frozenset()   # axes along which a carried read value varies

    def coefs(self) -> dict:
        return dict(self.lin)

    def varies(self) -> frozenset:
        return frozenset(a for a, _ in self.lin) | self.bad | self.dbad


def _mk(coefs: dict, const, bad=frozenset(), data=False, dbad=frozenset()) -> _V:
    return _V(tuple(sorted((a, c) for a, c in coefs.items() if c != 0)), const,
              frozenset(bad), data, frozenset(dbad))


def _nodes_of(obj: Any) -> list:
    """Every node reachable from ``obj``'s sinks (loop bodies included)."""
    candidate = getattr(obj, "sinks", None)
    if candidate is None:
        order = getattr(obj, "order", None)
        candidate = order if order is not None else obj
    roots = list(candidate)
    seen: set = set()
    out: list = []
    stack = roots[::-1]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        out.append(node)
        stack.extend(node.operands)
        if isinstance(node, Loop):
            stack.extend(node.body_sinks)
            stack.extend(node.body_nodes())
    return out


def _divisors(nodes: list) -> list:
    """Distinct ``(k, label|None)`` of every ``lane / k`` in the nodes."""
    found: dict = {}
    for n in nodes:
        if isinstance(n, Op) and n.kind == "div" and len(n.operands) == 2 \
                and isinstance(n.operands[0], SampleIndex) \
                and isinstance(n.operands[1], Const) \
                and n.operands[1].ttype.dtype in ("i32", "i64"):
            k = int(n.operands[1].literal)
            if k > 0:
                found.setdefault(k, getattr(n.operands[1], "label", None))
    return sorted(found.items(), reverse=True)


def _axes_from(divs: list) -> tuple:
    """(axes, notes) for the divisor set; a non-chain falls back to one axis."""
    ks = [k for k, _ in divs]
    chain = all(a % b == 0 for a, b in zip(ks, ks[1:])) if ks else True
    if not ks:
        return (LaneAxis(None, None, 1),), ()
    notes = ()
    if not chain:
        notes = (f"divisors {tuple(ks)} are not a chain: one flat axis (conservative)",)
        return (LaneAxis(None, None, 1),), notes
    strides = ks + [1]
    def _name(i: int) -> str | None:
        tag = divs[i][1] if i < len(divs) else None      # "lane_<label>_stride" of a declared axis
        if tag and tag.startswith("lane_") and tag.endswith("_stride"):
            return tag[len("lane_"):-len("_stride")]
        return None

    axes = [LaneAxis(_name(i), None if i == 0 else strides[i - 1] // strides[i], strides[i])
            for i in range(len(strides))]
    return tuple(axes), notes


class _Eval:
    def __init__(self, axes: tuple) -> None:
        self.axes = axes
        self.n = len(axes)
        self.memo: dict = {}
        self.all = frozenset(range(self.n))

    def lane(self) -> _V:
        return _mk({i: Fraction(a.stride) for i, a in enumerate(self.axes)}, Fraction(0))

    def value(self, root: Node) -> _V:
        stack = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if id(node) in self.memo:
                continue
            if not expanded:
                stack.append((node, True))
                for child in self._children(node):
                    if id(child) not in self.memo:
                        stack.append((child, False))
                continue
            self.memo[id(node)] = self._eval(node)
        return self.memo[id(root)]

    @staticmethod
    def _children(node: Node):
        if isinstance(node, At):
            return (node.index,)        # the plane leaf is not a lane quantity
        if isinstance(node, (LoopIndex, LoopCarry, Leaf)):
            return ()
        return node.operands

    def _eval(self, node: Node) -> _V:
        if isinstance(node, SampleIndex):
            return self.lane()
        if isinstance(node, Const):
            if node.ttype.dtype in ("i32", "i64"):
                return _mk({}, Fraction(int(node.literal)))
            return _mk({}, None)
        if isinstance(node, LoopIndex):
            return _mk({}, None)                       # same for every lane
        if isinstance(node, LoopCarry):
            return _mk({}, None, self.all)             # conservative
        if isinstance(node, Leaf):
            if node.role in ("uniform", "nsamples"):
                return _mk({}, None)                   # the same for every lane
            return _mk({}, None, frozenset(), True, self.all)   # a per-lane input value
        if isinstance(node, At):
            idx = self.memo[id(node.index)]
            return _mk({}, None, frozenset(), True, idx.varies())   # a read value
        if isinstance(node, Op):
            ops = [self.memo[id(o)] for o in node.operands]
            if node.kind in ("add", "sub") and len(ops) == 2:
                return self._addsub(node.kind, *ops)
            if node.kind == "mul" and len(ops) == 2:
                return self._mul(*ops)
            if node.kind == "div" and len(ops) == 2 and node.ttype.dtype in ("i32", "i64"):
                return self._div(*ops)
        # any other node: invariant iff every operand is; else non-affine
        ops = [self.memo[id(o)] for o in node.operands if id(o) in self.memo]
        lin = frozenset().union(*(frozenset(a for a, _ in o.lin) | o.bad for o in ops)) if ops else frozenset()
        return _mk({}, None, lin, any(o.data for o in ops),
                   frozenset().union(*(o.dbad for o in ops)) if ops else frozenset())

    @staticmethod
    def _lin(v: _V) -> frozenset:
        """The axes ``v`` moves along WITHOUT counting its data axes."""
        return frozenset(a for a, _ in v.lin) | v.bad

    def _addsub(self, kind: str, a: _V, b: _V) -> _V:
        sign = 1 if kind == "add" else -1
        coefs = a.coefs()
        for ax, c in b.lin:
            coefs[ax] = coefs.get(ax, Fraction(0)) + sign * c
        const = None if a.const is None or b.const is None else a.const + sign * b.const
        # a quotient survives only a shift by an axis-free term (index_form's rule)
        if any(c.denominator != 1 for c in coefs.values()) and (len(a.lin) and len(b.lin)):
            return _mk({}, None, self._lin(a) | self._lin(b), a.data or b.data, a.dbad | b.dbad)
        return _mk(coefs, const, a.bad | b.bad, a.data or b.data, a.dbad | b.dbad)

    def _mul(self, a: _V, b: _V) -> _V:
        for x, y in ((a, b), (b, a)):
            if not x.varies() and x.const is not None:    # x is a compile-time number
                k = x.const
                if k == 0:
                    return _mk({}, Fraction(0))
                if any(c.denominator != 1 for _, c in y.lin) and k.denominator != 1:
                    break
                return _mk({ax: c * k for ax, c in y.lin},
                           None if y.const is None else y.const * k, y.bad, y.data or x.data, y.dbad)
        varies = a.varies() | b.varies()
        if not varies:
            const = a.const * b.const if a.const is not None and b.const is not None else None
            return _mk({}, const, frozenset(), a.data or b.data, a.dbad | b.dbad)
        return _mk({}, None, self._lin(a) | self._lin(b), a.data or b.data, a.dbad | b.dbad)

    def _div(self, a: _V, b: _V) -> _V:
        if b.varies() or b.const is None or b.const <= 0:
            return _mk({}, None, self._lin(a) | self._lin(b), a.data or b.data, a.dbad | b.dbad)
        k = b.const
        if not a.varies():
            const = None if a.const is None or a.const < 0 else Fraction(int(a.const) // int(k))
            return _mk({}, const, frozenset(), a.data, a.dbad)
        if a.bad or a.dbad or a.const is None or a.const < 0:
            return _mk({}, None, self._lin(a), a.data, a.dbad)
        # exact part: axes whose coefficient k divides; the rest must stay below k
        whole, rest_max = {}, Fraction(0)
        for ax, c in a.lin:
            if c.denominator == 1 and c % k == 0:
                whole[ax] = c / k
                continue
            ext = self.axes[ax].extent
            if ext is None or c < 0 or c.denominator != 1:
                whole = None
                break
            rest_max += c * (ext - 1)
        if whole is not None and a.const + rest_max < k:
            return _mk(whole, Fraction(0) if a.const < k else None, frozenset(), a.data, a.dbad)
        if len(a.lin) == 1:                              # a lone axis: quotient form
            (ax, c), = a.lin
            return _mk({ax: c / k}, Fraction(int(a.const) // int(k)), frozenset(), a.data, a.dbad)
        return _mk({}, None, self._lin(a), a.data, a.dbad)


ORIGIN_AXIS, ORIGIN_SORTED = "invariant_axis", "index_plane_sorted"
ORIGIN_UNSORTED, ORIGIN_DATA = "index_plane_unsorted", "data_dependent"


def _clamp_bound(node: Node) -> int | None:
    """G when ``node`` is ``hawk.math.index(x, G)``: ``cast_i32(min(max(_, 0), G-1))``."""
    if not (isinstance(node, Op) and node.kind == "cast" and node.literal == "i32"):
        return None
    mn = node.operands[0]
    if not (isinstance(mn, Op) and mn.kind == "min" and isinstance(mn.operands[1], Const)):
        return None
    mx = mn.operands[0]
    if not (isinstance(mx, Op) and mx.kind == "max" and isinstance(mx.operands[1], Const)
            and mx.operands[1].literal == 0.0):
        return None
    top = mn.operands[1].literal
    return int(top) + 1 if float(top) == int(top) and top >= 0 else None


def _base_extent(base: _V, axes: tuple) -> int | None:
    """Static EXCLUSIVE bound of a data-free, non-negative affine ``base``
    (every axis it moves along static), else ``None``."""
    if base.bad or base.const is None or base.const < 0 or base.const.denominator != 1:
        return None
    hi = base.const
    for ax, c in base.lin:
        ext = axes[ax].extent
        if ext is None or c < 0 or c.denominator != 1:
            return None
        hi += c * (ext - 1)
    return int(hi) + 1


def _stored(node: Node) -> Node:
    """The plane read under the widening cast of an ``Index`` read (``int32`` on the wire)."""
    if isinstance(node, Op) and node.kind == "cast" and node.ttype.dtype == "i32" \
            and node.operands[0].ttype.dtype == "int32":
        return node.operands[0]
    return node


def _index_plane_read(node: Node, ev: "_Eval") -> bool:
    """Whether ``node`` reads an ``Index`` plane: the own-lane leaf, or a ``Table[Index]`` read at a
    data-free index (either through the widening cast of its ``int32`` storage)."""
    stored = _stored(node)
    dtypes = ("i32", "i64") if stored is node else ("int32",)
    if isinstance(stored, Leaf):
        return stored.role == "per_sample" and stored.ttype.dtype in dtypes
    return (isinstance(stored, At) and stored.plane.ttype.dtype in dtypes
            and stored.plane.role == "lookup" and not ev.value(stored.index).data)


def _split_terms(ev: "_Eval", key: Node) -> tuple:
    """The additive terms of an index: ``(free, slots, planes, other)``. ``free``
    holds ``(node, value, sign)`` of data-free terms, ``slots`` ``(bound, sign)`` of
    ``hawk.math.index`` clamps, ``planes`` ``(node, sign)`` of Index-plane reads,
    ``other`` counts the rest."""
    free, slots, planes, other = [], [], [], 0
    stack = [(key, 1)]
    while stack:
        node, sign = stack.pop()
        if isinstance(node, Op) and node.kind == "add" and len(node.operands) == 2:
            stack += [(node.operands[1], sign), (node.operands[0], sign)]
            continue
        if isinstance(node, Op) and node.kind == "sub" and len(node.operands) == 2 \
                and not ev.value(node.operands[1]).data:
            stack += [(node.operands[1], -sign), (node.operands[0], sign)]
            continue
        v = ev.value(node)
        g = _clamp_bound(node)
        if not v.data:
            free.append((node, v, sign))
        elif g is not None:
            slots.append((g, sign))
        elif _index_plane_read(node, ev):
            planes.append((_stored(node), sign))
        else:
            other += 1
    return free, slots, planes, other


def _tile_site(ev: "_Eval", index: Node, axes: tuple) -> tuple | None:
    """``(row, offset, slot)`` of a read index shaped ``row + offset`` or
    ``row + offset + clamp``, with ``row`` (its non-constant data-free terms, by
    node identity) invariant along the fastest axis; else ``None``."""
    free, slots, planes, other = _split_terms(ev, index)
    if planes or other or len(slots) > 1 or (slots and slots[0][1] < 0):
        return None
    row, offset, base = [], 0, _mk({}, Fraction(0))
    for node, v, sign in free:
        base = ev._addsub("add" if sign > 0 else "sub", base, v)
        if v.varies() or v.const is None:
            row.append((id(node), sign))
        elif v.const.denominator == 1:
            offset += sign * int(v.const)
        else:
            return None
    if base.bad or _forms(base, len(axes))[-1].cls != INVARIANT:
        return None
    return tuple(sorted(row)), offset, slots[0][0] if slots else None


def _key_facts(ev: "_Eval", key: Node, axes: tuple) -> tuple:
    """``(key_origin, key_extent, slot_extent, slot_base_forms)`` of one sink key.

    The key is split into its additive terms; each is data-free (folded into
    ``base``), a ``hawk.math.index`` clamp (a SLOT of static bound G), a read of
    an Index plane (``Leaf`` per-sample at the lane's own index, or a ``Table[Index]``
    read at a data-free index), or other plane data. Unknown = no."""
    free, slots, planes, other = _split_terms(ev, key)
    base = _mk({}, Fraction(0))
    for _, v, sign in free:
        base = ev._addsub("add" if sign > 0 else "sub", base, v)
    base_forms = _forms(base, len(axes))
    if not slots and not planes and not other:
        if base.bad:
            return ORIGIN_DATA, None, None, ()
        return ORIGIN_AXIS, _base_extent(base, axes), None, ()
    if planes and not slots and not other and len(planes) == 1 and not base.bad:
        leaf, sign = planes[0]
        invariant = all(f.cls == INVARIANT for f in base_forms)
        if isinstance(leaf, Leaf) and leaf.sorted and sign > 0 and invariant:
            return ORIGIN_SORTED, None, None, ()
        return ORIGIN_UNSORTED, None, None, ()
    if len(slots) == 1 and not planes and not other and slots[0][1] > 0 and not base.bad:
        g = slots[0][0]
        be = _base_extent(base, axes)
        slot_ext = g if base_forms[-1].cls == INVARIANT else None
        return ORIGIN_DATA, (None if be is None else be - 1 + g), slot_ext, base_forms
    return ORIGIN_DATA, None, None, ()


def _forms(v: _V, n: int) -> tuple:
    out = []
    coefs = v.coefs()
    for i in range(n):
        if i in v.dbad:
            out.append(AxisForm(DATA_DEPENDENT))
        elif i in v.bad:
            out.append(AxisForm(NON_AFFINE))
        elif i in coefs:
            c = coefs[i]
            out.append(AxisForm(AFFINE if c.denominator == 1 else QUOTIENT, c))
        else:
            out.append(AxisForm(INVARIANT))
    return tuple(out)


def _compensated_names(obj: Any) -> tuple:
    """``(on, companion)``: whether the kernel's sink kind is compensated and
    the name of the companion sink it writes the correction to."""
    sink = getattr(getattr(obj, "kind", None), "sink", None)
    if getattr(sink, "id", None) != "compensated":
        return False, None
    return True, getattr(sink, "compensation", None)


def facts(obj: Any, labels: tuple | None = None, group: int = 1) -> LaneFacts:
    """The lane-space facts of a traced kernel (anything with ``sinks``), a
    :class:`~hawk.ir.walk.Walk`, or a sink tuple. ``labels`` are the declared
    axis labels in numpy order (``None`` entries stay unlabeled); ``group`` is
    the kernel's declared group, a lane count."""
    nodes = _nodes_of(obj)
    axes, notes = _axes_from(_divisors(nodes))
    if labels is not None:
        if len(labels) != len(axes):
            raise ValueError(f"labels {labels!r} do not match {len(axes)} lane axes")
        axes = tuple(LaneAxis(lb, a.extent, a.stride) for lb, a in zip(labels, axes))
    if not isinstance(group, int) or isinstance(group, bool) or group < 1:
        raise ValueError(f"group must be a lane count >= 1, got {group!r}")
    ev = _Eval(axes)
    n = len(axes)
    comp_on, companion = _compensated_names(obj)
    reads, sinks, widths = [], [], [group]

    def policy(s: Any) -> str:
        declared = getattr(s, "policy", "auto")
        if declared == KEYED_AUTO:
            return "compensated"
        if declared not in ("auto", "plain"):
            return declared
        carries = (comp_on and s.name != companion) or (
            s.kind == "sample_sum" and getattr(s, "op", None) == "sum")
        return "compensated" if carries else "auto"

    def add_sink(s: Any) -> None:
        if s.index is None:
            sinks.append(SinkFact(s.name, s.kind, False,
                                  tuple(AxisForm(AFFINE, Fraction(a.stride)) for a in axes), False,
                                  policy=policy(s)))
            return
        v = ev.value(s.index)
        sinks.append(SinkFact(s.name, s.kind, True, _forms(v, n), bool(v.dbad),
                              *_key_facts(ev, s.index, axes), policy=policy(s)))
        run = _key_run(ev, axes, s.index)
        if run is not None and run.form == PROVEN:
            widths.append(run.width)

    for node in nodes:
        if isinstance(node, At):
            v = ev.value(node.index)
            plane = node.plane
            base = _forms(_mk(v.coefs(), v.const, v.bad), n)       # the data-free part
            lane_off = None
            if not v.dbad and not v.bad and v.const is not None and v.lin == ev.lane().lin:
                lane_off = int(v.const) if v.const.denominator == 1 else None
            reads.append(ReadFact(plane.name, plane.role, _forms(v, n), base, bool(v.dbad),
                                  v.const, lane_off, _tile_site(ev, node.index, axes)))
        elif isinstance(node, Sink):
            add_sink(node)
        elif isinstance(node, Loop):
            for s in node.body_sinks:
                if isinstance(s, Sink):
                    add_sink(s)
    # deterministic order
    reads.sort(key=lambda r: (r.plane, repr(r.forms), repr(r.offset)))
    out = LaneFacts(axes, tuple(reads), tuple(sinks), notes)
    return LaneFacts(axes, out.reads, out.sinks, notes, _box_candidates(out),
                     _uniform_control(nodes, ev), group, lcm(*widths),
                     _gather_tiles(out.reads))


def varies_along(obj: Any, axis: int):
    """A predicate over ``obj``'s nodes (loop bodies included): whether a
    node's value may differ between lanes that differ only in lane axis
    ``axis`` (by position). Unknown = it may."""
    ev = _Eval(_axes_from(_divisors(_nodes_of(obj)))[0])
    return lambda node: axis in ev.value(node).varies()


def _box_candidates(f: LaneFacts) -> tuple:
    last = f.axes[-1].extent
    return tuple(BoxCandidate(a.label, planes, a.extent, last, a.extent is not None and last is not None, i)
                 for i, a in enumerate(f.axes[:-1]) if (planes := f.invariant_reads(i)))


def _gather_tiles(reads: tuple) -> tuple:
    """One :class:`GatherTile` per plane with a slot read: its rows are the sites
    sharing one ``row`` term, and a row's words span its lowest to its highest
    offset (a slot adds its bound)."""
    out = []
    for plane in sorted({r.plane for r in reads if r.tile_site and r.tile_site[2] is not None}):
        rows: dict = {}
        for r in reads:
            if r.plane == plane and r.tile_site:
                row, off, slot = r.tile_site
                lo, hi = rows.get(row, (off, off))
                rows[row] = (min(lo, off), max(hi, off + (slot or 1) - 1))
        slot = max(r.tile_site[2] for r in reads if r.plane == plane and r.tile_site
                   and r.tile_site[2] is not None)
        out.append(GatherTile(plane, slot, max(hi - lo + 1 for lo, hi in rows.values())))
    return tuple(out)


def _uniform_control(nodes: list, ev: "_Eval") -> bool:
    """No node a lane can leave, skip or finish on its own (conservative)."""
    for n in nodes:
        if isinstance(n, (LoopCount, Finish)) or (isinstance(n, Loop) and (n.exit_cond is not None
                                                                           or n.count is not None)):
            return False
        if isinstance(n, Leaf) and n.role == "terminated":
            return False
        if isinstance(n, Dispatch) and n.policy != "predicated" and ev.value(n.operands[0]).varies():
            return False
    return True


# --------------------------------------------------------------------------- #
# The block box: the candidate plus the conditions that make staging it safe.
# --------------------------------------------------------------------------- #
def _tile_blocker(f: LaneFacts, fastest: int | None, tile: int, required_group: int | None,
                  uniform_control: bool | None, between: str | None = None) -> str | None:
    """Why a block of ``tile`` lanes cannot stage along the fastest axis, else
    ``None``. ``between`` (a reason the caller found) slots in after the extent
    checks and before the control and group ones."""
    if required_group is None:
        required_group = f.required_group
    if uniform_control is None:
        uniform_control = f.uniform_control
    if fastest is None:
        return "fastest extent is runtime"
    if fastest % tile:
        return f"fastest extent {fastest} not a multiple of tile {tile}"
    if between:
        return between
    if not uniform_control:
        return "control diverges: masked-alive path"
    if required_group > 1 and required_group % tile:
        return (f"group {required_group} < box {tile}" if required_group < tile
                else f"group {required_group} not a multiple of box {tile}")
    return None


def block_box_eligible(f: LaneFacts, *, required_group: int | None = None,
                       uniform_control: bool | None = None, tile: int = 32,
                       reuse: int | None = None) -> tuple:
    """``(ok, reason)``: may a block stage one candidate for its lanes?

    ``reuse`` is the plan-time extent of a runtime first axis (lanes // stride);
    ``required_group`` and ``uniform_control`` default to ``f``'s own, a caller
    may pass a stronger proof of the latter."""
    if not f.box_candidates:
        return False, "no invariant-axis read"
    reasons = []
    for c in f.box_candidates:
        r = c.reuse if c.reuse is not None else reuse
        short = ("reuse extent is runtime" if r is None
                 else f"reuse {r} < 2" if r < 2 else None)
        why = _tile_blocker(f, c.fastest_extent, tile, required_group, uniform_control, short)
        if why is None:
            return True, f"axis {c.axis}: reuse {r}, fastest {c.fastest_extent}"
        reasons.append(why)
    return False, reasons[0]


def gather_tile_eligible(f: LaneFacts, *, required_group: int | None = None,
                         uniform_control: bool | None = None, tile: int = 32) -> tuple:
    """``(ok, reason)``: may a block of ``tile`` lanes stage a gather row? The
    row must fit one bank row (:data:`BANK_WORDS`); the rest is the block box's
    conditions (:func:`block_box_eligible`), with the same reason strings."""
    if not f.gather_tiles:
        return False, "no gather-tile read"
    reasons = []
    for g in f.gather_tiles:
        wide = None if g.fits_warp_row else f"row {g.row_words} words exceeds {BANK_WORDS}-word bank width"
        why = _tile_blocker(f, f.axes[-1].extent, tile, required_group, uniform_control, wide)
        if why is None:
            return True, f"plane {g.plane}: row {g.row_words} words, slot {g.slot_extent}"
        reasons.append(why)
    return False, reasons[0]


def choose_mapping(f: LaneFacts, *, card: dict | None, reuse: int | None = None) -> dict:
    """Record the lowering per pattern from ``f`` and a per-device ``card``
    (``T`` thread-box width, ``contraction`` ``"gemm"``/``"direct"``, ``itemsize``,
    and optionally a ``tile`` block: ``kind`` ``"BF"`` broadcast or ``"BT"`` gather,
    ``SB`` block width, ``T``, ``R`` rows, ``EB`` steps, ``smem`` bytes); no device
    constant lives here. A card without ``tile`` chooses no block box and no gather
    tile; no card chooses none of them and ``gemm``. The decision is stored in
    ``f.mapping`` and returned."""
    card = card or {}
    cand = f.box_candidates[0] if f.box_candidates else None
    p1 = None if cand is None or "T" not in card else {
        "axis": cand.axis, "index": cand.index, "T": min(card["T"], cand.reuse or card["T"])}
    contraction = card.get("contraction", "gemm")
    if contraction not in CONTRACTIONS:
        raise ValueError(f"card contraction must be one of {CONTRACTIONS}, got {contraction!r}")
    tile = card.get("tile")
    box = gather = None
    why = gather_why = "no tile on the card"
    if tile is not None:
        missing = [k for k in TILE_FIELDS if k not in tile]
        if missing or tile["kind"] not in ("BF", "BT"):
            raise ValueError(f"card tile needs {TILE_FIELDS} with kind BF or BT, got {tile!r}")
        item = card.get("itemsize", 8)
        if tile["kind"] == "BF":
            ok, why = block_box_eligible(f, tile=tile["SB"], reuse=reuse)
            if ok:
                cand = next(c for c in f.box_candidates if why.startswith(f"axis {c.axis}:"))
                sites = sum(1 for r in f.reads if r.plane in cand.planes)   # upper bound on the staged run
                if sites * tile["SB"] * item > tile["smem"]:
                    why = "staged footprint exceeds the smem budget"
                else:
                    box = {"axis": cand.axis, "planes": cand.planes, "tile": tile["SB"], "sites": sites}
            gather_why = "tile kind is BF"
        else:
            ok, gather_why = gather_tile_eligible(f, tile=tile["SB"])
            if ok:
                g = next(g for g in f.gather_tiles if gather_why.startswith(f"plane {g.plane}:"))
                if g.row_words * tile["R"] * 2 * item > tile["smem"]:     # double buffered
                    gather_why = "staged footprint exceeds the smem budget"
                else:
                    gather = {"plane": g.plane, "row_words": g.row_words, "slot_extent": g.slot_extent,
                              **{k: tile[k] for k in ("SB", "T", "R", "EB")}}
            why = "tile kind is BT"
    f.mapping.clear()
    f.mapping.update({"P1": p1, "block_box": box, "block_box_reason": why,
                      "gather_tile": gather, "gather_tile_reason": gather_why,
                      "contraction": contraction})
    return f.mapping


# --------------------------------------------------------------------------- #
# The key run: a keyed accumulate whose key is uniform over aligned lane runs.
# The IR PATTERN is "keyed accumulate, key invariant along the axes a run of W
# consecutive lanes spans"; the row commit is one lowering of it.
# --------------------------------------------------------------------------- #
PROVEN, TESTED, SORTED = "proven", "tested", "sorted"

#: The widest run the commit reduces over: the target's coalescing width (a row
#: of :data:`~hawk.ir.nodes.ROW_WIDTH` lanes on GPU; the host's twin ignores it).
RUN_MAX = 32


@dataclass(frozen=True)
class KeyRun:
    """How a keyed sink's key moves over runs of consecutive lanes.

    ``form`` is ``"proven"`` when the FACTS prove the key equal over every
    aligned run of ``width`` consecutive lanes (``width`` a power of two,
    2..32), ``"sorted"`` when the key is read from a plane declared sorted
    (runs are found at run time), ``"tested"`` when the key is free of plane
    data and invariant along the fastest axis but no static run of >= 2 is
    proven (the runtime test decides)."""

    form: str
    width: int = 1
    forms: tuple = ()


def _run_axes(axes: tuple, width: int) -> tuple:
    """The axes whose coordinate can change inside an aligned run of
    ``width`` consecutive flat lanes: an axis is constant over every such run
    iff its stride is a multiple of ``width``."""
    return tuple(i for i, a in enumerate(axes) if a.stride % width != 0)


def _key_run(ev: "_Eval", axes: tuple, key: Node) -> KeyRun | None:
    v = ev.value(key)
    forms = _forms(v, len(axes))
    if _key_facts(ev, key, axes)[0] == ORIGIN_SORTED:
        # a key read from a plane DECLARED sorted: runs of equal keys are found
        # at run time (head flags), one commit per run. The declaration only
        # promises speed: an unsorted plane still sums right.
        return KeyRun(SORTED, RUN_MAX, forms)
    if v.dbad or v.data and v.varies():
        return None                                 # a gather key: per lane, by policy
    width = RUN_MAX
    while width >= 2:
        if all(forms[i].cls == INVARIANT for i in _run_axes(axes, width)):
            return KeyRun(PROVEN, width, forms)
        width //= 2
    if forms[-1].cls in (INVARIANT, QUOTIENT):
        return KeyRun(TESTED, RUN_MAX, forms)
    return None                                     # moves along the fastest axis: per lane


def key_run(sinks: Any, sink: Any) -> KeyRun | None:
    """The key run of ONE keyed sink of the kernel whose sink set is
    ``sinks`` (the axes come from the whole kernel); ``None`` for an
    own-column sink or a key with no run. Unknown = no."""
    if getattr(sink, "index", None) is None:
        return None
    axes, _notes = _axes_from(_divisors(_nodes_of(sinks)))
    return _key_run(_Eval(axes), axes, sink.index)


# --------------------------------------------------------------------------- #
# Warp-uniform values: the same for every lane of an aligned run of WARP lanes.
# --------------------------------------------------------------------------- #
#: The lanes a warp spans (the width a uniform access is shared over).
WARP = 32


def warp_uniform(sinks: Any):
    """A predicate ``uniform(node)`` for the kernel whose sink set is ``sinks``:
    True when the node is PROVEN equal over every aligned run of :data:`WARP`
    consecutive lanes. A value uniform over the run is read at one address by the
    whole warp. Unknown is False.

    Proven: a constant, a ``uniform``/``nsamples`` value, a loop index, a read at
    a proven index, an op of proven operands, and ``floor(y / d)`` of a lane form
    ``y`` (or ``y * (1 / d)``) when ``d`` is a multiple of :data:`WARP` and the
    lanes of a run differ only in ``y``'s fastest axis (whose coefficient is 1), as ``lane / k`` and
    ``split_index`` write it."""
    from .ops import RANDOM_OPS

    axes, _notes = _axes_from(_divisors(_nodes_of(sinks)))
    ev = _Eval(axes)
    memo: dict = {}

    def run_quotient(num: Node, d: Node) -> bool:
        if not (isinstance(d, Const) and float(d.literal) > 0
                and float(d.literal) == int(d.literal) and int(d.literal) % WARP == 0):
            return False
        if (isinstance(num, Op) and num.kind == "add"
                and isinstance(num.operands[1], Const)
                and 0 <= float(num.operands[1].literal) < 1):
            num = num.operands[0]                       # y + h, 0 <= h < 1: same floor
        while isinstance(num, Op) and num.kind == "cast":
            num = num.operands[0]
        v = ev.value(num)
        if v.const is None or v.const % WARP or v.dbad or v.bad or v.data:
            return False
        for ax, coef in v.lin:
            inner = axes[ax].stride % WARP != 0
            if coef.denominator != 1 or (coef != 1 if inner else coef % WARP):
                return False
            if inner and ax != len(axes) - 1:
                return False
        return True

    def uniform(node: Node) -> bool:
        held = memo.get(id(node))
        if held is not None:
            return held
        if isinstance(node, (Const, LoopIndex)):
            out = True
        elif isinstance(node, Leaf):
            out = node.role in ("uniform", "nsamples")
        elif isinstance(node, At):
            out = uniform(node.index)
        elif isinstance(node, Op):
            if node.kind == "div" and len(node.operands) == 2 \
                    and run_quotient(node.operands[0], node.operands[1]):
                out = True
            elif node.kind == "mul" and any(
                    isinstance(c, Const) and 0 < float(c.literal) < 1
                    and float(c.literal) * round(1 / float(c.literal)) == 1.0
                    and run_quotient(o, Const(round(1 / float(c.literal)), c.ttype))
                    for c, o in (node.operands, node.operands[::-1])):
                out = True              # y * (1 / d), as a division by a constant folds
            else:
                out = (node.kind not in RANDOM_OPS
                       and all(uniform(o) for o in node.operands))
        else:
            out = False
        memo[id(node)] = out
        return out

    return uniform
