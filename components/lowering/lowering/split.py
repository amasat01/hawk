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

A per-lane value wider than a thread's registers is placed in a plane by the same
slot (:func:`stages` with ``wide``, :mod:`lowering.wide`): the stage's sinks come back
with the value written to its plane where it is made, and the stage names the scratch
planes the runtime makes for it, as it does for a team stage's.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from raptor.ir.cells import planes
from . import team
from . import cells
from .block import BlockSum, BlockSumWrite, block_of, is_block, is_block_write
from raptor.ir.walk import post_order, structural_keys
from .logs import log_ordered
from raptor.ir.kinds import is_a, kind_name
from . import LogStore, ScatterLog, TeamCopy, TeamStore
from raptor.ir.nodes import COMPANION_SUFFIX, REDUCE_KIND, AccumWrite, Assign, At, Const, Leaf, Loop, Node, ReduceSamples, RoleConst, SampleSum, WideWrite, compensates, extent_of, has_feather, is_param_cell
from raptor.errors import RaptorError

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
    #: A team stage (:class:`lowering.team.TeamSpec`): its sinks are markers that name the scratch
    #: planes it works on, and the emitter writes its body from this record.
    team: object = None
    #: The scratch planes a wide value of this stage lives in (:func:`lowering.wide.place`): the
    #: runtime makes each once per bind, as it does a team stage's.
    planes: tuple = ()


def _depths(order, members=frozenset()) -> dict:
    """``id(node)`` -> the number of cuts on its deepest path: reductions over samples and the
    members of team-form linear algebra (``members``, their ids)."""
    depth: dict = {}
    for node in order:
        d = max((depth[id(c)] for c in node.operands), default=0)
        depth[id(node)] = d + 1 if has_feather(node, "reduces_samples") or id(node) in members else d
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
    from raptor.ir.cells import param_cell
    from raptor.ir.types import TensorType
    from raptor.ir.ops import make

    ttype = cut.ttype
    if len(ttype.shape) == 2:
        raise RaptorError(
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
    from raptor.ir.types import TensorType
    from raptor.ir.ops import make

    scalar = TensorType((), cut.ttype.dtype)
    plane = Leaf("table_read", "lookup", cell, scalar)
    if not cut.ttype.shape:
        return At(plane, cut.key, scalar)
    return make("vec", [At(plane, cut.key if k == 0 else make("add", (cut.key, make(
        "mul", (Const(k, cut.count.ttype), cut.count)))), scalar)
        for k in range(cut.ttype.shape[0])])


def _block_cells(sinks, blocks, linalg_form: str) -> dict:
    """``id(block) -> the output plane`` a :class:`~lowering.BlockSumWrite` stores it to.
    A block sum is lowered as a team stage, so ``linalg_form="thread"`` refuses it by name."""
    if blocks and linalg_form == "thread":
        raise RaptorError("linalg_form='thread' keeps every sum in per-entry form: a sum over the "
                        "lanes of a key of per-row products into a dense block is a team stage "
                        "(build with linalg_form='auto' or 'team')")
    out: dict = {}
    for sink in sinks:
        if is_block_write(sink):
            if out.setdefault(id(block_of(sink)), sink.name) != sink.name:
                raise RaptorError(f"one sum over the lanes of a key is stored to two planes, "
                                f"{out[id(block_of(sink))]!r} and {sink.name!r}: store it once")
    return out


def _with_companions(sinks: tuple) -> tuple:
    """``sinks`` and, for a compensated keyed accumulate, the zero commit that
    binds its companion plane at the same key (as the tracer adds it for a
    declared one)."""
    return (*sinks, *(AccumWrite(s.name + COMPANION_SUFFIX, Const(0.0, s.ttype), s.index,
                                 s.ttype)
                      for s in sinks if isinstance(s, AccumWrite) and compensates(s.contract)))


def _reads(roots) -> set:
    """The names of the per-sample planes read under ``roots``."""
    return {n.name for n in post_order(roots)
            if isinstance(n, Leaf) and n.role not in ("uniform", "lookup", "nsamples")}


def internal(found, declared) -> frozenset:
    """The cells of the stages ``found`` that only a later stage reads: each
    one a stage reads back that is not a plane the kernel declares (a
    reduction stored to a ``Mutable`` is the caller's). The runtime owns
    them and never hands them back."""
    return frozenset(c for st in found for c in st.cells_in if c not in declared)


# Whether a keyed accumulate with the default or compensated contract takes the ordered stages is the build's
# ``spec.settings.route_to_ordered``: a ``cells`` scalar mode has no atomic add on a packed cell and wants the compensated fold in
# a fixed order, so it sets it for its build: ONE rule, and the ordered stages are then the only path an accumulate takes there.


def takes_ordered_stages(sink, routed: bool = False) -> bool:
    """Whether the keyed accumulate ``sink`` is lowered to the ordered scatter stages."""
    return (isinstance(sink, AccumWrite) and sink.index is not None
            and (sink.contract == "ordered" or routed and sink.contract in ("auto", "compensated", "keyed_auto")))


def _lane_sums_as_block_sums(sinks: tuple, routed: bool, spec=None) -> tuple:
    """When ``routed`` (``route_to_ordered``), each sum over lane axes that a sink stores as a keyed accumulate becomes the
    block sum of the same payload (the ordered block stages: no scatter log), and the residual sink of a compensated
    contract is dropped (the accumulator carries its own compensation). Every other sink is returned as it was."""
    if not routed:
        return sinks
    residual = {s.name + "_c" for s in sinks if isinstance(s, AccumWrite) and s.contract in ("compensated", "keyed_auto")}
    out = []
    for sink in sinks:
        if isinstance(sink, AccumWrite) and sink.name in residual:
            continue
        reduced = getattr(sink, "reduced", None)
        if (isinstance(sink, AccumWrite) and reduced is not None and sink.contract in ("auto", "compensated", "keyed_auto")
                and team.ordered_wanted(reduced.operands[0], spec)):
            block = BlockSum(reduced.operands[0], key=reduced.key, count=reduced.count, contract="ordered", kept=reduced.kept)
            out.append(BlockSumWrite(sink.name, block))
        else:
            out.append(sink)
    return tuple(out)


def _ordered_scatters(name: str, sinks: tuple, routed: bool = False) -> tuple:
    """``(sinks, {target: slots per lane})``: each ``hawk.scatter(t, contract="ordered")[key] += v``
    replaced by the stores of its key and value into the lane's next log slots, one slot per add
    in program order (:func:`lowering.team.scatter_specs` adds the logs up). A ``for`` that scatters is
    unrolled by the tracer, so each trip is an add of its own; a loop node holding one is refused."""
    from raptor.ir.types import TensorType
    from raptor.ir.nodes import Loop
    from raptor.ir.nodes import SampleIndex
    from raptor.ir.ops import make

    def ordered(sink):
        return takes_ordered_stages(sink, routed)

    sinks = tuple(sinks)
    if routed:             # the accumulator carries its own compensation: no residual plane
        residual = {s.name + "_c" for s in sinks if isinstance(s, AccumWrite) and s.contract in ("compensated", "keyed_auto")}
        sinks = tuple(s for s in sinks if not (isinstance(s, AccumWrite) and s.name in residual))
    if any(isinstance(s, Loop) and any(ordered(b) or isinstance(b, Loop) for b in s.body_sinks) for s in sinks):
        raise RaptorError(f"{name!r}: hawk.scatter(contract='ordered') inside a lowered loop is not built "
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
        out += [ScatterLog(keys, key, at, sink.name), ScatterLog(values, sink.value, at, sink.name)]
    return tuple(out), width


def stages(name: str, sinks, *, spec, wide: bool = False, real_bytes: int = 8, taken=(), inplace: bool = False) -> tuple | None:
    """The stages of the kernel ``name`` with sink set ``sinks``, or ``None``
    when the body reads no reduction over samples (one kernel, unchanged).

    With ``wide`` the placement pass (:func:`lowering.wide.place`) runs on the sinks of every stage that is
    not a team stage, for scalars of ``real_bytes`` bytes, and a kernel that has a wide value placed in a
    scratch plane is returned as the stages it needs, one at least: that stage names the plane in its
    ``planes``. A caller that cannot make the planes leaves ``wide`` off and gets the kernel as it was.
    ``taken`` are plane names the placement must not reuse (those of another body of the same kernel). With ``inplace`` a
    reduction input that is a bare read of an input plane is read by its team stage in place, not copied to scratch (an index-set
    build compacts its inputs, so it copies)."""
    found = _stages(name, sinks, spec, inplace)
    if not wide:
        return found
    return _placed(name, sinks, found, real_bytes, spec, taken)


def _placed(name: str, sinks, found, real_bytes: int, spec, taken_names=()) -> tuple | None:
    """``found`` with every per-lane stage's wide values placed (:func:`lowering.wide.place`); the kernel as
    one stage when it had none and has a placed value, ``None`` when nothing is placed."""
    from dataclasses import replace

    from . import wide as _wide

    if found is None:
        sinks, _ = _ordered_scatters(name, log_ordered(sinks), spec.settings.route_to_ordered)
        found = (Stage(name, tuple(sinks), (), (), (), ()),)
        whole = True
    else:
        whole = False
    taken = {n.name for st in found for n in post_order(st.sinks) if hasattr(n, "name") and n.name} | set(taken_names)
    out, changed = [], False
    for st in found:
        if st.team is not None:
            out.append(st)
            continue
        new, planes, record = _wide.place(st.sinks, spec, real_bytes=real_bytes, taken=taken)
        if record:
            changed = True
            taken |= {p.name for p in planes}
            st = replace(st, sinks=new, planes=planes)
        out.append(st)
    return tuple(out) if changed else (None if whole else found)


def refuse_cut_in_loop(order) -> None:
    """Refuse a cut (a reduction over samples) that a node combines with a loop value: a cut is a stage boundary of its own,
    which would be one launch per iteration."""
    stack = list(order)
    while stack:
        node = stack.pop()
        if isinstance(node, Loop):
            stack.extend(node.body_nodes())
        kinds = [getattr(o, "kind", "") for o in node.operands]
        if REDUCE_KIND in kinds and any(k.startswith("loop") for k in kinds):
            raise RaptorError("a cut inside a `for` is not in this release (it would be one launch per iteration); "
                              "hoist the cut out of the loop")


def _stages(name: str, sinks, spec, inplace: bool = False) -> tuple | None:
    """:func:`stages` without the placement."""
    forced = spec.settings.team_threads
    build_spec = spec       # the loops below rebind ``spec`` to a stage's
    sinks, scattered = _ordered_scatters(name, _lane_sums_as_block_sums(log_ordered(sinks), spec.settings.route_to_ordered, spec), spec.settings.route_to_ordered)
    order = post_order(sinks)
    cuts = [n for n in order if has_feather(n, "reduces_samples")]
    refuse_cut_in_loop(order)
    team.refuse_in_loop(order, team.max_dim_of(spec), spec.settings.linalg_form)
    team.refuse_in_dispatch(sinks, order, team.max_dim_of(spec), spec.settings.linalg_form)
    groups = team.members_of(order, sinks, spec)
    logs = {n.target: n for n in (*sinks, *(b for s in sinks if hasattr(s, "body_sinks") for b in s.body_sinks))
            if is_a(n, "log_store")}
    if not cuts and not groups and not logs and not scattered:
        return None
    blocks = [c for c in cuts if is_block(c)]
    stored_to = _block_cells(sinks, blocks, spec.settings.linalg_form)
    sinks = tuple(s for s in sinks if not is_block_write(s))
    for cut in cuts:
        if not is_block(cut) and cut.key is not None and (
                cut.op != "sum" or cut.contract == "ordered"):
            raise RaptorError(
                f"a {cut.op} over some lane axes (contract={cut.contract!r}) read back in "
                "the body is not in this release: read back hawk.sum/mean(x, over=...) "
                "with contract 'auto' or 'compensated'")
    member_group = {id(m): g for g in groups for m in g.members}
    depth = _depths(order, frozenset(member_group))
    # the values a team stage works on are stored by the per-lane stage that computes them
    stores = [(TeamStore(plane, value), g) for g in groups for value, plane in team.stores(g)]
    out_node = {id(m): team.output(g, m) for g in groups for m in g.members}
    bi = {id(b): i for i, b in enumerate(blocks)}
    read_back = {id(o) for n in order if not is_block_write(n) for o in n.operands
                 if is_block(o)}
    sinks = (*sinks, *(s for s, _ in stores),
             *(s for b in blocks for s in team.factor_stores(
                 bi[id(b)], b, tuple(team.plane_reads(b, stored_to.get(id(b)), id(b) in read_back)) if inplace else ())))

    # a cut stored to a Mutable already has a cell: read that one
    # (a vector's columns are components of the reduced value)
    stored = {(id(s.value.operands[0] if s.columns > 1 else s.value), s.op): s.cell
              for s in sinks if isinstance(s, SampleSum)}
    taken = {getattr(n, "name", None) for n in order} - {None}
    cell_of, made = {}, {}
    for cut in cuts:
        if is_block(cut):
            continue            # a block sum is read from its team stage's own plane
        cell = None if cut.key is not None else stored.get((id(cut.operands[0]), cut.op))
        if cell is None:
            cell = _fresh(taken, len(made))
            made[id(cut)] = cell
        cell_of[id(cut)] = cell
    cell_plane = {plane: cell_of[id(c)] for c in cuts if not is_block(c) and c.key is None
                  for plane in planes(cell_of[id(c)], c.ttype.shape[0] if c.ttype.shape else 1)}
    keyed = {cell_of[id(c)]: Keyed(cell_of[id(c)], c.ttype.shape[0] if c.ttype.shape else 1, c.kept, c.ttype.shape)
             for c in cuts if not is_block(c) and c.key is not None}

    # the stage each sink lands in: after every reduction it reads
    place = {id(s): (depth[id(s.value)] if is_a(s, "team_store") else depth[id(s)]) + 1
             for s in sinks}
    count = max([max(place.values(), default=0), *(depth[id(c)] for c in cuts), *(depth[i] for i in member_group)])
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

    from raptor.ir.cells import swap

    out, seen_before, keys, factored = [], set(), {}, set()
    for k in range(1, count + 1):
        own = [s for s in sinks if place[id(s)] == k]
        own += [s for c in cuts if id(c) in made and depth[id(c)] == k
                for s in _with_companions(cells.stored(c, made[id(c)]))]
        mapping = {id(c): (team.block_read(c, bi[id(c)]) if is_block(c)
                           else _cell_read(c, cell_of[id(c)])) for c in cuts if depth[id(c)] < k}
        mapping.update({i: node for i, node in out_node.items() if depth[i] < k})
        new = swap(own, mapping) if mapping else tuple(own)
        original = post_order(own, stop=mapping)
        interior = [n for n in original
                    if not isinstance(n, (Leaf, Const, RoleConst)) and not has_feather(n, "reduces_samples")
                    and not is_param_cell(n) and n.operands
                    and not any(n is s for s in own)]
        structural_keys(original, keys)
        recomputed = tuple(n.kind for n in interior if keys[id(n)] in seen_before
                           and extent_of(n) == "per_sample")
        seen_before |= {keys[id(n)] for n in original}
        cells_out = tuple(dict.fromkeys(s.cell for s in new if isinstance(s, SampleSum)))
        cells_in = tuple(dict.fromkeys(cell_plane[n.plane.name] for n in post_order(new)
                                       if is_param_cell(n) and n.plane.name in cell_plane))
        keyed_out = tuple(dict.fromkeys(keyed[s.name] for s in new
                                        if isinstance(s, AccumWrite) and s.name in keyed))
        copy = None
        if new and all(kind_name(s) == "team_store" and isinstance(s.value, Leaf) for s in new) and sum(
                prod(s.value.ttype.shape) for s in new) >= team.copy_min(build_spec):
            # nothing but operand copies into team scratch: a block per lane, its threads copy the slice together
            new, copy = tuple(TeamCopy(s.name, s.value) for s in new), team.copy_spec(build_spec)
        if new or cells_in or cells_out:        # a stage whose stores were all read in place is not launched
            out.append(Stage(name if not out else f"{name}_{len(out) + 1}", new, cells_in, cells_out,
                             recomputed, keyed_out, copy))
        spec = team.spec_of(groups, k, depth, factored, forced)
        if spec is not None:
            out.append(Stage(name if not out else f"{name}_{len(out) + 1}", team.markers(spec), (), (), (), (), spec))
        for block in (b for b in blocks if depth[id(b)] == k):
            for spec in team.block_spec(bi[id(block)], block, stored_to.get(id(block)),
                                        Keyed(f"{team.PLANE_PREFIX}b{bi[id(block)]}_g", team.size_of(block.ttype), block.kept, block.ttype.shape),
                                        id(block) in read_back, inplace, forced):
                out.append(Stage(name if not out else f"{name}_{len(out) + 1}", team.markers(spec), (), (), (), (), spec))
    max_chunks, threads = (build_spec.need("scatter", "max_chunks"), build_spec.need("scatter", "threads")) if scattered else (0, 0)
    for target, slots in scattered.items():     # an ordered scatter: add each target's logs after the stages that fill them
        for stage_spec in team.scatter_specs(target, slots, max_chunks, threads):
            out.append(Stage(f"{name}_{len(out) + 1}", team.markers(stage_spec), (), (), (), (), stage_spec))
    for log in logs.values():       # an ordered gradient: fold each log after the stages that fill it
        spec = team.fold_spec(log, Keyed(log.target, 1, log.kept), forced)
        out.append(Stage(f"{name}_{len(out) + 1}", team.markers(spec), (), (), (), (), spec))
    return tuple(out)


def compensated_target(kind, sinks) -> tuple[str, bool] | None:
    """The compensated sink's resolved target, out of ``kind.output`` and
    the kernel's own committed ``sinks`` — shared by the tracer and the
    writer so the two never resolve a different target for the same
    kernel.

    ``None`` when this kind's sink contract is not ``compensated``.
    Otherwise ``(name, own_column)``: ``own_column`` is ``True`` for an
    ``Assign`` or an ``AccumWrite`` with ``index=None``, ``False`` for a
    scattered ``AccumWrite``. The companion (``kind.sink.compensation``)
    is excluded from the candidate pool before counting."""
    if getattr(kind.sink, "id", None) != "compensated":
        return None
    comp = kind.sink.compensation
    all_names = tuple(dict.fromkeys(s.name for s in sinks))
    pool = (
        (kind.output.slot,) if kind.output.returned_ttype is not None
        else kind.output.names if kind.output.names else all_names
    )
    candidates = [n for n in pool if n != comp]
    if len(candidates) != 1:
        source = ("Output.named" + repr(tuple(pool)) if kind.output.names
                 else "the committed sinks " + repr(pool))
        raise RaptorError(
            f"Kind({kind.slug!r}): a compensated sink needs exactly ONE output "
            f"target (excluding the companion {comp!r}); resolved {candidates} "
            f"from {source}"
        )
    name = candidates[0]
    matches = [s for s in sinks if s.name == name]
    if any(isinstance(s, WideWrite) for s in matches):
        raise RaptorError(
            f"Kind({kind.slug!r}): compensated target {name!r} is a WideOut "
            "sink; a compensated commit targets a Mutable only"
        )
    own_column = any(isinstance(s, Assign) for s in matches) or any(
        isinstance(s, AccumWrite) and s.index is None for s in matches)
    return name, own_column
