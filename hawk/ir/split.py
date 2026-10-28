# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The split: a value reduced over samples and read later is a launch boundary.

``z = x / hawk.sum(x)`` needs every sample's ``x`` before any
sample's ``z``, so one kernel becomes stages run in sequence. Stage ``k`` holds
the nodes whose every reduction over samples is a cell an earlier stage
produced: stage 1 is the per-sample kernel with its reductions stored as rows
(one per 32 samples), and stage ``k >= 2`` reads each earlier cell as a
one-cell ``lookup`` plane, exactly as :func:`hawk.diff.param_planes` reads a
promoted parameter. Between two stages the runtime finishes the rows into the
cell. A reduction over SOME lane axes (``hawk.sum(x, over=...)``) is a keyed
plane instead: stage 1 accumulates into it by key, a later stage reads it at its
own lane's key (entry ``k`` of a vector payload at ``key + k * count``), and the
runtime owns the plane and zeroes it for every run. A per-sample value a later stage needs is computed again there from the
inputs (no plane carries it across); each stage records which values it
recomputes.

A kernel that reads no reduction stays ONE kernel, untouched. The split runs on
the finished IR, after any derivative is taken, so a derivative kernel is split
like any other.
"""

from __future__ import annotations

from dataclasses import dataclass

from .._cells import planes
from . import team
from .nodes import (
    COMPANION_SUFFIX,
    AccumWrite,
    At,
    Const,
    HawkError,
    Leaf,
    Node,
    ReduceSamples,
    RoleConst,
    SampleSum,
    compensates,
    extent_of,
    is_param_cell,
)

#: The prefix of a cell the split makes for a reduction the body reads but
#: never stores to a Mutable output.
CUT_PREFIX = "cut"


@dataclass(frozen=True)
class Keyed:
    """A keyed plane the runtime owns between stages: ``width`` payload entries
    over the kept lane axes ``kept`` (``((label, extent|None), ...)``)."""

    name: str
    width: int
    kept: tuple
    #: The shape of one key's payload (``()`` a scalar, ``(p,)`` a vector, ``(p, q)`` a block), so a
    #: runtime can return ``(*payload, keys)`` where ``width`` alone says ``(width, keys)``.
    payload: tuple = ()


@dataclass(frozen=True)
class Stage:
    """One launch of a split kernel: its sinks, the cells it reads (as one-cell
    ``lookup`` planes) and stores (as rows), the per-sample values it
    computes again because an earlier stage computed them too, and the keyed
    planes it accumulates into."""

    name: str
    sinks: tuple
    cells_in: tuple
    cells_out: tuple
    recomputed: tuple
    keyed: tuple = ()
    #: A team stage (:class:`hawk.ir.team.TeamSpec`): its sinks are markers that name the scratch
    #: planes it works on, and the emitter writes its body from this record.
    team: object = None


def _post_order(roots, stop=()) -> list:
    """Every node under ``roots``, children before parents, each once; a
    node whose id is in ``stop`` is kept but not entered."""
    out, seen = [], set()
    stack = [(r, False) for r in reversed(tuple(roots))]
    while stack:
        node, done = stack.pop()
        if id(node) in seen:
            continue
        if done:
            seen.add(id(node))
            out.append(node)
            continue
        stack.append((node, True))
        if id(node) not in stop:
            stack.extend((c, False) for c in reversed(node.operands) if id(c) not in seen)
    return out


def _keys(order, memo: dict) -> None:
    """Fill ``memo`` with a structural key per node of ``order`` (children
    first), so a value traced twice counts as one value."""
    for node in order:
        if id(node) not in memo:
            memo[id(node)] = (type(node).__name__, node.kind, node.ttype,
                              getattr(node, "literal", None), node.dedup_extra,
                              getattr(node, "name", None),
                              tuple(memo.get(id(c)) for c in node.operands))


def _depths(order, members=frozenset()) -> dict:
    """``id(node)`` -> the number of cuts on its deepest path: reductions over samples and the
    members of team-form linear algebra (``members``, their ids)."""
    depth: dict = {}
    for node in order:
        d = max((depth[id(c)] for c in node.operands), default=0)
        depth[id(node)] = d + 1 if isinstance(node, ReduceSamples) or id(node) in members else d
    return depth


def _fresh(taken: set, j: int) -> str:
    name = f"{CUT_PREFIX}{j}"
    while name in taken:
        name += "_"
    taken.add(name)
    return name


def _cell_read(cut: ReduceSamples, cell: str) -> Node:
    """The read of ``cut``'s finished cell: a one-cell ``lookup`` plane
    (``cell``; a vector's column ``k`` is the plane ``cell_k``)."""
    from ..diff.params import param_cell
    from ..types import TensorType
    from .ops import make

    ttype = cut.ttype
    if len(ttype.shape) == 2:
        raise HawkError(
            "a matrix reduced over samples is not read back whole in this release: "
            "reduce each entry the kernel reads (hawk.sum(m[i, j]))")
    if cut.key is not None:
        return _keyed_read(cut, cell)
    if not ttype.shape:
        return param_cell(cell, ttype)
    scalar = TensorType((), ttype.dtype)
    return make("vec", [param_cell(f"{cell}_{k}", scalar) for k in range(ttype.shape[0])])


def _keyed_read(cut: ReduceSamples, cell: str) -> Node:
    """The read of a keyed reduction at the lane's own key: the plane ``cell``
    holds entry ``k`` of the payload at ``key + k * count``."""
    from ..types import TensorType
    from .ops import make

    scalar = TensorType((), cut.ttype.dtype)
    plane = Leaf("table_read", "lookup", cell, scalar)
    if not cut.ttype.shape:
        return At(plane, cut.key, scalar)
    return make("vec", [At(plane, cut.key if k == 0 else make("add", (cut.key, make(
        "mul", (Const(k, cut.count.ttype), cut.count)))), scalar)
        for k in range(cut.ttype.shape[0])])


def _block_cells(sinks, blocks) -> dict:
    """``id(block) -> the output plane`` a :class:`~hawk.ir.team.BlockSumWrite` stores it to.
    A block sum is lowered as a team stage, so ``linalg_form="thread"`` refuses it by name."""
    from ..emit.policy import linalg_form

    if blocks and linalg_form() == "thread":
        raise HawkError("linalg_form='thread' keeps every sum in per-entry form: a sum over the "
                        "lanes of a key of per-row products into a dense block is a team stage "
                        "(build with linalg_form='auto' or 'team')")
    out: dict = {}
    for sink in sinks:
        if isinstance(sink, team.BlockSumWrite):
            if out.setdefault(id(sink.block), sink.name) != sink.name:
                raise HawkError(f"one sum over the lanes of a key is stored to two planes, "
                                f"{out[id(sink.block)]!r} and {sink.name!r}: store it once")
    return out


def _with_companions(sinks: tuple) -> tuple:
    """``sinks`` and, for a compensated keyed accumulate, the zero commit that
    binds its companion plane at the same key (as the tracer adds it for a
    declared one)."""
    return (*sinks, *(AccumWrite(s.name + COMPANION_SUFFIX, Const(0.0, s.ttype), s.index,
                                 s.ttype)
                      for s in sinks if isinstance(s, AccumWrite) and compensates(s.policy)))


def _reads(roots) -> set:
    """The names of the per-sample planes read under ``roots``."""
    return {n.name for n in _post_order(roots)
            if isinstance(n, Leaf) and n.role not in ("uniform", "lookup", "nsamples")}


def internal(found, declared) -> frozenset:
    """The cells of the stages ``found`` that only a later stage reads: each
    one a stage reads back that is not a plane the kernel declares (a
    reduction stored to a ``Mutable`` is the caller's). The runtime owns
    them and never hands them back."""
    return frozenset(c for st in found for c in st.cells_in if c not in declared)


def _ordered_scatters(name: str, sinks: tuple) -> tuple:
    """``(sinks, {target: slots per lane})``: each ``hawk.scatter(t, policy="ordered")[key] += v``
    replaced by the stores of its key and value into the lane's next log slots, one slot per add
    in program order (:func:`hawk.ir.team.scatter_specs` adds the logs up). A ``for`` that scatters is
    unrolled by the tracer, so each trip is an add of its own; a loop node holding one is refused."""
    from ..types import TensorType
    from .loop_nodes import Loop
    from .nodes import SampleIndex
    from .ops import make

    def ordered(s):
        return isinstance(s, AccumWrite) and s.policy == "ordered" and s.index is not None

    sinks = tuple(sinks)
    if any(isinstance(s, Loop) and any(ordered(b) or isinstance(b, Loop) for b in s.body_sinks) for s in sinks):
        raise HawkError(f"{name!r}: hawk.scatter(policy='ordered') inside a lowered loop is not built "
                        "(a `for` that scatters is unrolled by the trace, so this needs a loop node made by hand)")
    width: dict = {}
    for sink in sinks:
        if ordered(sink):
            width[sink.name] = width.get(sink.name, 0) + 1
    if not width:
        return sinks, {}
    wide = TensorType((), "i64")
    used: dict = {}
    out: list = []
    for sink in sinks:
        if not ordered(sink):
            out.append(sink)
            continue
        slot = used.get(sink.name, 0)
        used[sink.name] = slot + 1
        keys, values = team.scatter_names(sink.name)[:2]
        at = make("add", (make("mul", (SampleIndex(wide), Const(width[sink.name], wide))), Const(slot, wide)))
        key = sink.index if sink.index.ttype.dtype == "int32" else make("cast", (sink.index,), literal="int32")
        out += [team.ScatterLog(keys, key, at, sink.name), team.ScatterLog(values, sink.value, at, sink.name)]
    return tuple(out), width


def stages(name: str, sinks) -> tuple | None:
    """The stages of the kernel ``name`` with sink set ``sinks``, or ``None``
    when the body reads no reduction over samples (one kernel, unchanged)."""
    sinks, scattered = _ordered_scatters(name, sinks)
    order = _post_order(sinks)
    cuts = [n for n in order if isinstance(n, ReduceSamples)]
    team.refuse_in_loop(order)
    team.refuse_in_dispatch(sinks, order)
    groups = team.members_of(order)
    logs = {n.target: n for n in (*sinks, *(b for s in sinks if hasattr(s, "body_sinks") for b in s.body_sinks))
            if isinstance(n, team.LogStore)}
    if not cuts and not groups and not logs and not scattered:
        return None
    blocks = [c for c in cuts if isinstance(c, team.BlockSum)]
    stored_to = _block_cells(sinks, blocks)
    sinks = tuple(s for s in sinks if not isinstance(s, team.BlockSumWrite))
    for cut in cuts:
        if cut.key is not None and not isinstance(cut, team.BlockSum) and (
                cut.op != "sum" or cut.policy == "ordered"):
            raise HawkError(
                f"a {cut.op} over some lane axes (policy={cut.policy!r}) read back in "
                "the body is not in this release: read back hawk.sum/mean(x, over=...) "
                "with policy 'auto' or 'compensated'")
    member_group = {id(m): g for g in groups for m in g.members}
    depth = _depths(order, frozenset(member_group))
    # the values a team stage works on are stored by the per-lane stage that computes them
    stores = [(team.TeamStore(plane, value), g) for g in groups for value, plane in team.stores(g)]
    out_node = {id(m): team.output(g, m) for g in groups for m in g.members}
    bi = {id(b): i for i, b in enumerate(blocks)}
    read_back = {id(o) for n in order if not isinstance(n, team.BlockSumWrite) for o in n.operands
                 if isinstance(o, team.BlockSum)}
    sinks = (*sinks, *(s for s, _ in stores), *(s for b in blocks for s in team.factor_stores(bi[id(b)], b)))

    # a cut stored to a Mutable already has a cell: read that one
    # (a vector's columns are components of the reduced value)
    stored = {(id(s.value.operands[0] if s.columns > 1 else s.value), s.op): s.cell
              for s in sinks if isinstance(s, SampleSum)}
    taken = {getattr(n, "name", None) for n in order} - {None}
    cell_of, made = {}, {}
    for cut in cuts:
        if isinstance(cut, team.BlockSum):
            continue            # a block sum is read from its team stage's own plane
        cell = None if cut.key is not None else stored.get((id(cut.operands[0]), cut.op))
        if cell is None:
            cell = _fresh(taken, len(made))
            made[id(cut)] = cell
        cell_of[id(cut)] = cell
    cell_plane = {plane: cell_of[id(c)] for c in cuts if c.key is None
                  for plane in planes(cell_of[id(c)], c.ttype.shape[0] if c.ttype.shape else 1)}
    keyed = {cell_of[id(c)]: Keyed(cell_of[id(c)], c.ttype.shape[0] if c.ttype.shape else 1, c.kept, c.ttype.shape)
             for c in cuts if c.key is not None and not isinstance(c, team.BlockSum)}

    # the stage each sink lands in: after every reduction it reads
    place = {id(s): (depth[id(s.value)] if isinstance(s, team.TeamStore) else depth[id(s)]) + 1
             for s in sinks}
    count = max([max(place.values()), *(depth[id(c)] for c in cuts), *(depth[i] for i in member_group)])
    # a Mutable's earlier value read by a later stage: its store waits for it
    read_by = [(id(s), _reads((s,))) for s in sinks]
    read_by += [(id(c), _reads(c.operands)) for c in cuts if id(c) in made]
    stage_of = {**place, **{id(c): depth[id(c)] for c in cuts}}
    changed = True
    while changed:
        changed = False
        last: dict = {}
        for owner, names in read_by:
            for n in names:
                last[n] = max(last.get(n, 0), stage_of[owner])
        for s in sinks:
            later = last.get(getattr(s, "name", None), 0)
            if not isinstance(s, SampleSum) and later > stage_of[id(s)]:
                stage_of[id(s)] = place[id(s)] = later
                changed = True

    from ..diff.params import swap

    out, seen_before, keys, factored = [], set(), {}, set()
    for k in range(1, count + 1):
        own = [s for s in sinks if place[id(s)] == k]
        own += [s for c in cuts if id(c) in made and depth[id(c)] == k
                for s in _with_companions(c.stored(made[id(c)]))]
        mapping = {id(c): (team.block_read(c, bi[id(c)]) if isinstance(c, team.BlockSum)
                           else _cell_read(c, cell_of[id(c)])) for c in cuts if depth[id(c)] < k}
        mapping.update({i: node for i, node in out_node.items() if depth[i] < k})
        new = swap(own, mapping) if mapping else tuple(own)
        original = _post_order(own, stop=mapping)
        interior = [n for n in original
                    if not isinstance(n, (Leaf, Const, RoleConst, ReduceSamples))
                    and not is_param_cell(n) and n.operands
                    and not any(n is s for s in own)]
        _keys(original, keys)
        recomputed = tuple(n.kind for n in interior if keys[id(n)] in seen_before
                           and extent_of(n) == "per_sample")
        seen_before |= {keys[id(n)] for n in original}
        cells_out = tuple(dict.fromkeys(s.cell for s in new if isinstance(s, SampleSum)))
        cells_in = tuple(dict.fromkeys(cell_plane[n.plane.name] for n in _post_order(new)
                                       if is_param_cell(n) and n.plane.name in cell_plane))
        keyed_out = tuple(dict.fromkeys(keyed[s.name] for s in new
                                        if isinstance(s, AccumWrite) and s.name in keyed))
        out.append(Stage(name if not out else f"{name}_{len(out) + 1}", new, cells_in, cells_out,
                         recomputed, keyed_out))
        spec = team.spec_of(groups, k, depth, factored)
        if spec is not None:
            out.append(Stage(f"{name}_{len(out) + 1}", team.markers(spec), (), (), (), (), spec))
        for block in (b for b in blocks if depth[id(b)] == k):
            spec = team.block_spec(bi[id(block)], block, stored_to.get(id(block)),
                                   Keyed(f"{team.PLANE_PREFIX}b{bi[id(block)]}_g", team.size_of(block.ttype), block.kept, block.ttype.shape),
                                   id(block) in read_back)
            out.append(Stage(f"{name}_{len(out) + 1}", team.markers(spec), (), (), (), (), spec))
    for target, slots in scattered.items():     # an ordered scatter: add each target's logs after the stages that fill them
        for spec in team.scatter_specs(target, slots):
            out.append(Stage(f"{name}_{len(out) + 1}", team.markers(spec), (), (), (), (), spec))
    for log in logs.values():       # an ordered gradient: fold each log after the stages that fill it
        spec = team.fold_spec(log, Keyed(log.target, 1, log.kept))
        out.append(Stage(f"{name}_{len(out) + 1}", team.markers(spec), (), (), (), (), spec))
    return tuple(out)
