# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Lane queries of the SIMT component: whether a value is the same over a warp's aligned run, the run a key moves in, and the choice
of block box and gather tile from a device card. The index algebra (the lane facts) is :mod:`raptor.ir.lanes`; every machine number
here (bank width, run widths) is the variant's ``lanes`` data, passed in and refused by name when absent."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import lcm
from typing import Any

from raptor.ir import lanes as _facts
from raptor.errors import RaptorError
from raptor.ir.lanes import (INVARIANT, ORIGIN_SORTED, QUOTIENT, Eval, axes_from, axis_forms, divisors, key_facts,
                             nodes_of)
from raptor.ir.nodes import Loop, LoopCarry, LoopCount, LoopIndex
from raptor.ir.nodes import KEYED_AUTO, At, Const, Dispatch, Finish, Leaf, Node, Op, SampleIndex, Sink, has_feather, peel_clamp
from raptor.ir.ops import RANDOM_OPS


def _run_numbers(spec) -> tuple:
    return spec.need("lanes", "run_max"), spec.need("lanes", "block_run")


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


@dataclass(frozen=True)
class GatherTile:
    """A plane read as ``base + slot`` (a clamped data-dependent slot of static
    bound ``slot_extent``) whose ``base`` does not move along the fastest axis:
    every lane of a fastest-axis run reads the same row of ``row_words`` words
    of ``plane``, so a block can stage that row once."""

    plane: str
    slot_extent: int
    row_words: int


@dataclass(frozen=True)
class LaneFacts(_facts.LaneFacts):
    """The lane-space facts of :mod:`raptor.ir.lanes` and the candidates for staging one copy per block."""

    #: thread/block box candidates, one per non-fastest axis with an invariant read.
    box_candidates: tuple = ()
    #: True only when the walk holds no per-lane loop exit, count, finish,
    #: terminated read or lane-varying dispatch; unknown = False. It does NOT
    #: see the launch's own tail guard or an external active set.
    uniform_control: bool = False
    #: the least lane count every proven key run and the declared ``group`` divide.
    required_group: int = 1
    #: planes a block could stage as a gather tile, one per plane.
    gather_tiles: tuple = ()
    #: the chosen lowering per pattern, recorded by the consumer that chooses it.
    mapping: dict = field(default_factory=dict, compare=False)


def facts(obj: Any, labels: tuple | None = None, group: int = 1, *, spec=None) -> LaneFacts:
    """:func:`raptor.ir.lanes.facts` with the candidates for staging one copy per block. The proven key runs of the variant
    (``spec``) are folded into ``required_group``; without ``spec`` only ``group`` fixes it."""
    base = _facts.facts(obj, labels, group)
    nodes = nodes_of(obj)
    ev = Eval(base.axes)
    widths = [group]
    if spec is not None:
        run_max, block_run = _run_numbers(spec)
        keys = [s.index for n in nodes for s in ((n,) if isinstance(n, Sink) else n.body_sinks if isinstance(n, Loop) else ())
                if isinstance(s, Sink) and s.index is not None]
        for key in keys:
            run = find_key_run(ev, base.axes, key, run_max=run_max, block_run=block_run)
            if run is not None and run.form == PROVEN:
                widths.append(run.width)
    return LaneFacts(base.axes, base.reads, base.sinks, base.notes, group, _box_candidates(base), _uniform_control(nodes, ev),
                     lcm(*widths), _gather_tiles(base.reads))


def _box_candidates(f: _facts.LaneFacts) -> tuple:
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


def _uniform_control(nodes: list, ev: "Eval") -> bool:
    """No node a lane can leave, skip or finish on its own (conservative)."""
    for n in nodes:
        if isinstance(n, (LoopCount, Finish)) or (isinstance(n, Loop) and (n.exit_cond is not None
                                                                           or n.count is not None)):
            return False
        if isinstance(n, Leaf) and n.role == "terminated":
            return False
        if isinstance(n, Dispatch) and not has_feather(n, "evaluates_all_branches") and ev.value(n.operands[0]).varies():
            return False
    return True


def key_run(sinks: Any, sink: Any, spec) -> KeyRun | None:
    """The key run of ONE keyed sink of the kernel whose sink set is
    ``sinks`` (the axes come from the whole kernel; the run widths from the variant of ``spec``); ``None`` for an
    own-column sink or a key with no run. Unknown = no."""
    if getattr(sink, "index", None) is None:
        return None
    run_max, block_run = _run_numbers(spec)
    axes, _notes = axes_from(divisors(nodes_of(sinks)))
    return find_key_run(Eval(axes), axes, sink.index, run_max=run_max, block_run=block_run)


# --------------------------------------------------------------------------- #
# Warp-uniform values: the same for every lane of an aligned run of WARP lanes.
# --------------------------------------------------------------------------- #
def warp_uniform(sinks: Any, warp: int | None):
    """A predicate ``uniform(node)`` for the kernel whose sink set is ``sinks``:
    True when the node is PROVEN equal over every aligned run of ``warp``
    consecutive lanes (``warp``, the variant's run width; ``None``: it names none, so only what needs no run is proven). A value uniform over the run is read at one address by the
    whole warp. Unknown is False.

    Proven: a constant, a ``uniform``/``nsamples`` value, a loop index, a read at
    a proven index, an op of proven operands, and ``floor(y / d)`` of a lane form
    ``y`` (or ``y * (1 / d)``) when ``d`` is a multiple of ``warp`` and the
    lanes of a run differ only in ``y``'s fastest axis (whose coefficient is 1), as ``lane / k`` and
    ``split_index`` write it."""
    from raptor.ir.ops import RANDOM_OPS

    axes, _notes = axes_from(divisors(nodes_of(sinks)))
    ev = Eval(axes)
    memo: dict = {}

    def run_quotient(num: Node, d: Node) -> bool:
        if not warp:
            return False
        if not (isinstance(d, Const) and float(d.literal) > 0
                and float(d.literal) == int(d.literal) and int(d.literal) % warp == 0):
            return False
        if (isinstance(num, Op) and num.kind == "add"
                and isinstance(num.operands[1], Const)
                and 0 <= float(num.operands[1].literal) < 1):
            num = num.operands[0]                       # y + h, 0 <= h < 1: same floor
        while isinstance(num, Op) and num.kind == "cast":
            num = num.operands[0]
        v = ev.value(num)
        if v.const is None or v.const % warp or v.dbad or v.bad or v.data:
            return False
        for ax, coef in v.lin:
            inner = axes[ax].stride % warp != 0
            if coef.denominator != 1 or (coef != 1 if inner else coef % warp):
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
            out = uniform(node.raw_index)
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


# --------------------------------------------------------------------------- #
# What a card may choose for a contraction, and the fields of its ``tile`` block.
# --------------------------------------------------------------------------- #
CONTRACTIONS = ("gemm", "direct")
TILE_FIELDS = ("kind", "SB", "T", "R", "EB", "smem")

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


def block_box_eligible(f: LaneFacts, *, tile: int, required_group: int | None = None,
                       uniform_control: bool | None = None, reuse: int | None = None) -> tuple:
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


def gather_tile_eligible(f: LaneFacts, *, tile: int, bank_words: int | None, required_group: int | None = None,
                         uniform_control: bool | None = None) -> tuple:
    """``(ok, reason)``: may a block of ``tile`` lanes stage a gather row? The
    row must fit one bank row (``bank_words``, the variant's ``lanes.bank_words``); the rest is the block box's
    conditions (:func:`block_box_eligible`), with the same reason strings."""
    if not f.gather_tiles:
        return False, "no gather-tile read"
    if bank_words is None:
        raise RaptorError("the variant of the target being built has no lanes.bank_words: a gather tile cannot be chosen without it")
    reasons = []
    for g in f.gather_tiles:
        wide = None if g.row_words <= bank_words else f"row {g.row_words} words exceeds {bank_words}-word bank width"
        why = _tile_blocker(f, f.axes[-1].extent, tile, required_group, uniform_control, wide)
        if why is None:
            return True, f"plane {g.plane}: row {g.row_words} words, slot {g.slot_extent}"
        reasons.append(why)
    return False, reasons[0]


def choose_mapping(f: LaneFacts, *, card: dict | None, bank_words: int | None = None, reuse: int | None = None) -> dict:
    """Record the lowering per pattern from ``f`` and a per-device ``card``
    (``T`` thread-box width, ``contraction`` ``"gemm"``/``"direct"``, ``itemsize``,
    and optionally a ``tile`` block: ``kind`` ``"BF"`` broadcast or ``"BT"`` gather,
    ``SB`` block width, ``T``, ``R`` rows, ``EB`` steps, ``smem`` bytes); ``bank_words`` is the variant's ``lanes.bank_words`` (read only when a gather tile is chosen, refused by name if absent then). A card without ``tile`` chooses no block box and no gather
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
            ok, gather_why = gather_tile_eligible(f, tile=tile["SB"], bank_words=bank_words)
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
    #: the key is also equal over every aligned run of ``block_run`` lanes (one key per block of any size dividing it)
    block: bool = False


def _run_axes(axes: tuple, width: int) -> tuple:
    """The axes whose coordinate can change inside an aligned run of
    ``width`` consecutive flat lanes: an axis is constant over every such run
    iff its stride is a multiple of ``width``."""
    return tuple(i for i, a in enumerate(axes) if a.stride % width != 0)


def find_key_run(ev: "Eval", axes: tuple, key: Node, *, run_max: int, block_run: int) -> KeyRun | None:
    """``run_max`` is the widest run the commit reduces over (the variant's ``lanes.run_max``) and ``block_run`` the lanes a block
    key-run spans (``lanes.block_run``)."""
    v = ev.value(key)
    forms = axis_forms(v, len(axes))
    if key_facts(ev, key, axes)[0] == ORIGIN_SORTED:
        # a key read from a plane DECLARED sorted: runs of equal keys are found
        # at run time (head flags), one commit per run. The declaration only
        # promises speed: an unsorted plane still sums right.
        return KeyRun(SORTED, run_max, forms)
    if v.dbad or v.data and v.varies():
        return None                                 # a gather key: per lane
    width = run_max
    while width >= 2:
        if all(forms[i].cls == INVARIANT for i in _run_axes(axes, width)):
            return KeyRun(PROVEN, width, forms, width == run_max and all(
                forms[i].cls == INVARIANT for i in _run_axes(axes, block_run)))
        width //= 2
    if forms[-1].cls in (INVARIANT, QUOTIENT):
        return KeyRun(TESTED, run_max, forms)
    return None                                     # moves along the fastest axis: per lane
