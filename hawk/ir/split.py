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
                              tuple(memo[id(c)] for c in node.operands))


def _depths(order) -> dict:
    """``id(node)`` -> the number of reductions over samples on its deepest path."""
    depth: dict = {}
    for node in order:
        d = max((depth[id(c)] for c in node.operands), default=0)
        depth[id(node)] = d + 1 if isinstance(node, ReduceSamples) else d
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


def stages(name: str, sinks) -> tuple | None:
    """The stages of the kernel ``name`` with sink set ``sinks``, or ``None``
    when the body reads no reduction over samples (one kernel, unchanged)."""
    sinks = tuple(sinks)
    order = _post_order(sinks)
    cuts = [n for n in order if isinstance(n, ReduceSamples)]
    if not cuts:
        return None
    for cut in cuts:
        if cut.key is not None and (cut.op != "sum" or cut.policy == "ordered"):
            raise HawkError(
                f"a {cut.op} over some lane axes (policy={cut.policy!r}) read back in "
                "the body is not in this release: read back hawk.sum/mean(x, over=...) "
                "with policy 'auto' or 'compensated'")
    depth = _depths(order)

    # a cut stored to a Mutable already has a cell: read that one
    # (a vector's columns are components of the reduced value)
    stored = {(id(s.value.operands[0] if s.columns > 1 else s.value), s.op): s.cell
              for s in sinks if isinstance(s, SampleSum)}
    taken = {getattr(n, "name", None) for n in order} - {None}
    cell_of, made = {}, {}
    for cut in cuts:
        cell = None if cut.key is not None else stored.get((id(cut.operands[0]), cut.op))
        if cell is None:
            cell = _fresh(taken, len(made))
            made[id(cut)] = cell
        cell_of[id(cut)] = cell
    cell_plane = {plane: cell_of[id(c)] for c in cuts if c.key is None
                  for plane in planes(cell_of[id(c)], c.ttype.shape[0] if c.ttype.shape else 1)}
    keyed = {cell_of[id(c)]: Keyed(cell_of[id(c)], c.ttype.shape[0] if c.ttype.shape else 1, c.kept)
             for c in cuts if c.key is not None}

    # the stage each sink lands in: after every reduction it reads
    place = {id(s): depth[id(s)] + 1 for s in sinks}
    count = max(max(place.values()), max(depth[id(c)] for c in cuts))
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

    out, seen_before, keys = [], set(), {}
    for k in range(1, count + 1):
        own = [s for s in sinks if place[id(s)] == k]
        own += [s for c in cuts if id(c) in made and depth[id(c)] == k
                for s in _with_companions(c.stored(made[id(c)]))]
        mapping = {id(c): _cell_read(c, cell_of[id(c)]) for c in cuts if depth[id(c)] < k}
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
        out.append(Stage(name if k == 1 else f"{name}_{k}", new, cells_in, cells_out,
                         recomputed, keyed_out))
    return tuple(out)
