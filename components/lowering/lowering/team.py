# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Team-form linear algebra: what the split cuts at, and the planes between its stages.

A factorisation of a matrix too large for one lane (``n`` above
:data:`~hawk.ir.linalg.MAX_DIM`, or any size under ``linalg_form="team"``) is worked by a
team of threads, one team per lane, on a matrix that lives in scratch the runtime owns
(``aether::linalg::team``). That makes it a launch boundary, like a reduction over samples
that a later stage reads (:mod:`hawk.passes.split`):

* the per-lane stage before it stores the operands (the matrix, each right-hand side) into
  planes of ``lanes x size`` elements, a lane's slice contiguous (:class:`TeamStore`);
* the team stage (``blockIdx.x`` is the lane, a team of :func:`threads` per lane) factors the
  matrix IN PLACE on its operand plane and runs every solve of it in place on its right-hand
  side's plane (:class:`Step`);
* the per-lane stages after it read each result at the lane's own slice (:class:`TeamRead`).

A solve, a determinant or a field of the factor is a ``member`` of the group its factorisation
makes: one factorisation runs once however many results are read. The scratch each group holds
is listed as :class:`Plane` records, which the runtime allocates (once per kernel and lane count)
and the sidecar names.
"""

from __future__ import annotations

from dataclasses import dataclass

from raptor.ir.types import TensorType
from raptor.ir.fold import exceeds
from raptor.ir.linalg import CONSUMERS, FACTORISATIONS, QR_OPS, dimension, is_linalg
from raptor.ir.block import BlockSum, BlockSumWrite, block_shaped, factors, ordered_case
from raptor.ir.nodes import At, Const, Dispatch, Leaf, Node, Op, SampleIndex, WideWrite
from raptor.errors import RaptorError

#: The prefix of a scratch plane the split makes for team stages.
PLANE_PREFIX = "tm"

#: Environment override of the team size (a test door: results are bit-independent of it).


def form(node: Node, max_dim: int | None, asked: str) -> str:
    """How ``node`` is lowered: ``"team"`` when a team of threads works on its matrix (``n`` above
    ``max_dim`` (the variant's ``linalg.max_dim``), or every size under ``linalg_form="team"``), else ``"thread"``, one lane
    holding the matrix. The build's ``linalg_form`` (``asked``: ``spec.settings.linalg_form``) decides; ``"thread"``
    above ``max_dim`` is refused by name."""
    n = dimension(node)
    if node.kind in QR_OPS:
        if asked == "thread":
            raise RaptorError(
                f"op {node.kind!r}: least squares by QR is worked by a team of threads, so "
                "linalg_form='thread' refuses it; build with linalg_form='auto' or 'team'")
        return "team"
    if max_dim is None:
        raise RaptorError("the variant of the target being built has no linalg.max_dim: a backend whose manifest names no variant "
                        "cannot use a pass that reads it")
    if asked == "thread" and n > max_dim:
        raise RaptorError(
            f"op {node.kind!r} at n={n}: linalg_form='thread' keeps the matrix in one lane, "
            f"which holds n <= {max_dim}; build with linalg_form='auto' or 'team'")
    return "team" if asked == "team" or n > max_dim else "thread"


def threads(n: int, forced: int = 0) -> int:
    """The team size for an ``n x n`` matrix: a perf constant, 128 up to ``n = 128`` and 256 above."""
    if forced:
        return int(forced)
    return 128 if n <= 128 else 256


# -- the nodes a stage boundary leaves behind --------------------------------------

#: Threads of the block of a stage that only copies operands into team scratch (the launch's resolved block for it), and the
#: elements per lane below which one thread per lane copies them: each thread of the block must move at least 4, or the block idles.
COPY_THREADS = 256
COPY_MIN = 4 * COPY_THREADS


def copy_spec() -> "TeamSpec":
    """The team record of a copy stage: a block per lane, no scratch of its own, no steps (its sinks are its body)."""
    return TeamSpec(COPY_THREADS, (), ())


def size_of(ttype: TensorType) -> int:
    """Elements of one lane's slice of a value of ``ttype``."""
    out = 1
    for extent in ttype.shape:
        out *= extent
    return out


def read(plane: str, ttype: TensorType) -> Node:
    """The read of a team result: a rank-0 value is a plain ``at`` of the lane's element, a
    vector or matrix is a :class:`TeamRead`."""
    from . import INDEX, TeamRead     # imported here: the core renderer reads this module's constants

    if not ttype.shape:
        return At(Leaf("table_read", "lookup", plane, ttype), SampleIndex(INDEX), ttype)
    return TeamRead(plane, ttype)


# -- the scratch -------------------------------------------------------------------

@dataclass(frozen=True)
class Plane:
    """One scratch plane: ``per_lane`` elements of ``dtype`` for each ``unit`` (a lane, or a key of
    a :class:`BlockSum`)."""

    name: str
    dtype: str          # "real" (the kernel's scalar type) | "int32" | "partial" (a partial sum: "real", or in a cells mode "band_partial")
    per_lane: int
    unit: str = "lane"  # "lane" | "key" | "target" (chunks x the cells of the stage's ``target`` plane) | "once"
    fill: int | None = None     # what the plane holds at the start of every run (None: nothing is promised)
    layout: str | None = None   # "plane_major": made (per_lane, lanes), bound as a caller's plane of that width; None: flat


@dataclass(frozen=True)
class Step:
    """One collective call of a team stage: ``op`` over an ``n x n`` factor, ``k`` right-hand
    columns, on the scratch planes ``planes`` (``(role, name)`` pairs)."""

    op: str
    n: int
    k: int
    planes: tuple
    m: int = 0          # rows of a tall (QR) factor
    max_chunks: int = 0  # an ordered scatter's first stage: the variant's ``scatter.max_chunks`` (0: not a scatter step)

    def plane(self, role: str) -> str:
        return dict(self.planes)[role]


@dataclass(frozen=True)
class TeamSpec:
    """A team stage: its size, the calls in order and the scratch planes they touch. ``grid`` is
    what a block of the launch is: a lane (``"lanes"``), a key of a block sum (``"keys"``, then
    ``keyed`` is the plane's :class:`~hawk.passes.split.Keyed` record and ``out`` the plane the stage
    commits to, which is not scratch of its own), or, in an ordered scatter into the plane ``target``, a
    chunk of lanes (``"chunks"``) or a cell of ``target`` (``"targets"``)."""

    threads: int
    steps: tuple
    planes: tuple
    grid: str = "lanes"
    keyed: object = None
    out: str | None = None
    target: str | None = None
    #: Blocks per key of a split block sum's first stage (``count`` is then ``keys * groups``).
    groups: int = 1
    #: The lanes a block of a warp-form kernel serves: ``threads / 32`` warps, one per lane (:mod:`hawk.passes.lane`).
    lanes: int = 1
    #: The warps a block of a warp-per-lane kernel holds, a lane each (set only by the card kit's ``WARPS``).
    warps: int = 1
    #: The input planes (read leaves) the stage reads in place, a factor that is a bare plane read needing no scratch copy.
    reads: tuple = ()


#: A scratch plane of the kernel's scalar type: the runtime gives it the width of the build's mode.
REAL = "real"
PARTIAL = "partial"


@dataclass
class Group:
    """The factorisation ``factor`` and every node of the body that reads it (``members``)."""

    index: int
    factor: Op
    members: list
    n: int
    dtype: str
    planes: dict            # role -> Plane (the group's own scratch)
    member_planes: dict     # id(member) -> output Plane or None (a field that is a plane of the group)
    rhs_planes: dict        # id(member) -> the Plane its right-hand side is stored in (and solved in place)
    m: int = 0              # rows of the matrix a QR factors
    warp: bool = False      # an eigenvalues-only f32 n = 32 eigh: one warp per lane (aether::linalg::warp::eighValues)


def _plane(group: Group, role: str, dtype: str, per_lane: int) -> str:
    name = f"{PLANE_PREFIX}{group.index}_{role}"
    group.planes[role] = Plane(name, dtype, per_lane)
    return name


def _identity(node: Node) -> bool:
    """Whether ``node`` is the constant identity matrix (an ``inv`` solves against it)."""
    if not (isinstance(node, Op) and node.kind == "vec" and len(node.ttype.shape) == 2):
        return False
    n, m = node.ttype.shape
    return n == m and all(isinstance(c, Const) and c.literal == (1.0 if k // n == k % n else 0.0)
                          for k, c in enumerate(node.operands))


def warp_factors(order, sinks, spec) -> frozenset:
    """``id`` of every factorisation that has a warp entry (the variant's ``linalg.warp_entries``) for its kind, ``n``, the build's scalar
    (``spec.settings.scalar``) and the outputs read, that the build's device runs at that scalar's rate, with ``linalg_form="auto"``,
    and whose stage of its own is possible (not in a loop body, not only in an arm of a dispatch)."""
    from raptor.ir.nodes import Loop

    scalar = spec.settings.scalar
    if spec.settings.linalg_form != "auto" or not spec.settings.device_only or spec.device is None or (
            scalar != "float32" and spec.device_fp64_ratio() > spec.need("linalg", "fast_fp64")):
        return frozenset()
    entries = {tuple(k[:3]) + (tuple(k[3]),): k[4] for k in spec.need("linalg", "warp_entries")}
    vectors = {id(n.operands[0]) for n in order if isinstance(n, Op) and n.kind == "eigh_v"}
    cand = [n for n in order if isinstance(n, Op) and n.kind in FACTORISATIONS
            and (n.kind, dimension(n), scalar, ("v",) if id(n) in vectors else ()) in entries]
    if not cand:
        return frozenset()
    inner: set = set()
    for node in order:
        if isinstance(node, Loop):
            inner.update(id(b) for b in node.body_nodes())
    seen: set = set()
    stack = list(sinks)
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.extend(node.operands[:1] if isinstance(node, Dispatch) else node.operands)
    return frozenset(id(n) for n in cand if id(n) in seen and id(n) not in inner)


def members_of(order, sinks, spec) -> tuple:
    """The groups of team-form linear algebra in ``order`` (every node under the sinks, children
    first): one per factorisation node, its members in the order they appear."""
    by_factor: dict = {}
    warp = warp_factors(order, sinks, spec)
    max_dim = max_dim_of(spec)
    for node in order:
        if not is_linalg(node):
            continue
        if form(node, max_dim, spec.settings.linalg_form) != "team" and not (id(node if node.kind in FACTORISATIONS else node.operands[0]) in warp):
            continue
        factor = node if node.kind in FACTORISATIONS else node.operands[0]
        if factor.kind not in FACTORISATIONS:
            raise RaptorError(f"op {node.kind!r} reads a factor that no factorisation made")
        group = by_factor.get(id(factor))
        if group is None:
            n = dimension(factor)
            limit = team_max_dim_of(spec)
            if n > limit:
                raise RaptorError(f"op {factor.kind!r}: the dimension is static and n <= {limit} (the variant's linalg.team_max_dim), got n={n}")
            group = Group(len(by_factor), factor, [], n, REAL, {}, {}, {},
                          factor.operands[0].ttype.shape[0] if factor.kind == "qr" else 0)
            group.warp = id(factor) in warp
            by_factor[id(factor)] = group
        if node is not factor:
            group.members.append(node)
    for group in by_factor.values():
        _name_planes(group)
    return tuple(by_factor.values())


def _name_planes(group: Group) -> None:
    """The scratch of ``group``: the factor's planes, then one per member that has a result of its
    own (a solve's right-hand side is solved in place, a determinant is one element)."""
    n, dtype, kind = group.n, group.dtype, group.factor.kind
    _plane(group, "a", dtype, group.m * n if kind == "qr" else n * n)   # the operand, factored in place
    if kind == "lu":
        _plane(group, "piv", "int32", n)
    _plane(group, "info", "int32", 1)
    if kind == "eigh":
        _plane(group, "w", dtype, n)
        if not group.warp:
            _plane(group, "v", dtype, n * n)
            _plane(group, "work", dtype, n * n + 4)
    if kind == "qr":
        widest = max([n] + [_columns(mb) for mb in group.members if mb.kind in CONSUMERS])
        _plane(group, "tau", dtype, n)
        _plane(group, "work", dtype, qr_work(group.m, widest))
    for m, member in enumerate(group.members):
        if member.kind == "qr_solve":                     # the right-hand side is overwritten by Q^T b
            rhs = _plane(group, f"r{m}", dtype, size_of(member.operands[1].ttype))
            _plane(group, f"x{m}", dtype, size_of(member.ttype))
            group.member_planes[id(member)] = group.planes[f"x{m}"]
            group.rhs_planes[id(member)] = group.planes[f"r{m}"]
        elif member.kind in CONSUMERS:
            rhs = member.operands[1]
            _plane(group, f"x{m}", dtype, size_of(member.ttype))
            group.member_planes[id(member)] = group.planes[f"x{m}"]
            if not (member.kind == "lu_solve" and _identity(rhs)):
                group.rhs_planes[id(member)] = group.planes[f"x{m}"]
        elif member.kind == "lu_det":
            _plane(group, f"d{m}", dtype, 1)
            group.member_planes[id(member)] = group.planes[f"d{m}"]


def _columns(member: Op) -> int:
    """Right-hand side columns of a consumer."""
    shape = member.operands[1].ttype.shape
    return shape[1] if len(shape) == 2 else 1


def qr_work(m: int, cols: int) -> int:
    """Scratch elements ``aether::linalg::team::qr`` needs as ``work`` for ``cols`` columns of ``m``
    rows (``qrWorkSize``: one partial sum per column per tile of 64 rows, and one)."""
    return cols * ((m + 63) // 64) + 1


def output(group: Group, member: Op) -> Node:
    """The node that stands for ``member``'s value in the stages after the team stage."""
    p = group.planes
    kind = member.kind
    if kind in CONSUMERS or kind == "lu_det":
        return read(group.member_planes[id(member)].name, member.ttype)
    if kind.endswith("_info"):
        return read(p["info"].name, member.ttype)
    if kind == "chol_factor":
        return read(p["a"].name, member.ttype)
    return read(p["w" if kind == "eigh_w" else "v"].name, member.ttype)       # eigh_w, eigh_v


def stores(group: Group) -> tuple:
    """``(node, plane name)`` for every value the per-lane stages store for ``group``: the
    operand and each right-hand side that is not the identity."""
    out = [(group.factor.operands[0], group.planes["a"].name)]
    for member in group.members:
        plane = group.rhs_planes.get(id(member))
        if plane is not None:
            out.append((member.operands[1], plane.name))
    return tuple(out)


def step_list(group: Group, level: int, depth: dict, first: bool) -> list:
    """The calls ``group`` makes at team level ``level``: the factorisation when it is the first
    level the group has a member at, then each member of that level."""
    kind, n, p = group.factor.kind, group.n, group.planes
    qr = kind == "qr"
    out = []
    here = [m for m in group.members if depth[id(m)] == level]
    if not here:
        return out
    if first:
        roles = [("a", p["a"].name), ("info", p["info"].name)]
        if kind == "lu":
            roles.append(("piv", p["piv"].name))
        if kind == "eigh":
            roles.append(("w", p["w"].name))
            if not group.warp:
                roles += [("v", p["v"].name), ("work", p["work"].name)]
        if qr:
            roles += [("tau", p["tau"].name), ("work", p["work"].name)]
        out.append(Step("eigh_warp" if group.warp else kind, n, 1, tuple(roles), group.m))
    for member in here:
        k = member.operands[1].ttype.shape[1] if member.kind in CONSUMERS and len(
            member.operands[1].ttype.shape) == 2 else 1
        roles = [("a", p["a"].name), ("info", p["info"].name)]
        if kind == "lu":
            roles.append(("piv", p["piv"].name))
        if qr:
            roles += [("tau", p["tau"].name), ("work", p["work"].name)]
        if member.kind == "qr_solve":
            roles += [("r", group.rhs_planes[id(member)].name), ("x", group.member_planes[id(member)].name)]
            out.append(Step("qr_solve", n, k, tuple(roles), group.m))
            continue
        if member.kind in CONSUMERS:
            if member.kind == "lu_solve" and id(member) not in group.rhs_planes:
                out.append(Step("lu_inverse", n, 1, (*roles, ("x", group.member_planes[id(member)].name))))
                continue
            roles.append(("x", group.member_planes[id(member)].name))
        elif member.kind == "lu_det":
            roles.append(("x", group.member_planes[id(member)].name))
        else:
            continue                       # a field of the factor: read from its plane
        out.append(Step(member.kind, n, k, tuple(roles)))
    return out


def refuse_in_loop(order, max_dim: int, asked: str) -> None:
    """Refuse a team-form node inside a ``Loop`` body (a stage boundary cannot be inside a loop;
    iterate by replaying the plan)."""
    from raptor.ir.nodes import Loop

    for node in order:
        if isinstance(node, Loop):
            stack = list(node.body_nodes())
            while stack:
                inner = stack.pop()
                if is_linalg(inner) and form(inner, max_dim, asked) == "team":
                    raise RaptorError(
                        f"op {inner.kind!r} at n={dimension(inner)} is worked by a team of threads, "
                        "which is a launch of its own and cannot sit inside a loop of the body; "
                        "iterate by replaying the plan (one launch per iteration)")
                if isinstance(inner, Loop):
                    stack.extend(inner.body_nodes())


def refuse_in_dispatch(sinks, order, max_dim: int, asked: str) -> None:
    """Refuse a team-form node (or a block sum) that only an arm of a ``Dispatch`` reads: a team
    stage is a launch of its own that every lane takes part in, so it cannot compute for the lanes
    of one arm alone. A node the body also reads outside every arm is the whole lane set's."""
    seen: set = set()
    stack = list(sinks)
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        stack.extend(node.operands[:1] if isinstance(node, Dispatch) else node.operands)
    for node in order:
        if id(node) in seen:
            continue
        if isinstance(node, BlockSum):
            raise RaptorError(
                "a sum over the lanes of a key of per-row products into a dense block is a team "
                "stage, which every lane takes part in, so it cannot be inside an arm of a dispatch; "
                "sum outside the arm and select its result")
        if is_linalg(node) and form(node, max_dim, asked) == "team":
            raise RaptorError(
                f"op {node.kind!r} at n={dimension(node)} is worked by a team of threads, which is a "
                "launch of its own that every lane takes part in, so it cannot be inside an arm of a "
                "dispatch; compute it outside the arm and select its result")


def spec_of(groups, level: int, depth: dict, seen: set, forced: int = 0) -> TeamSpec | None:
    """The team stage at ``level``: every group's calls, or ``None`` when no group has a member
    at that level. ``seen`` holds the indices of the groups already factored."""
    steps, planes, size = [], {}, 0
    for group in groups:
        calls = step_list(group, level, depth, group.index not in seen)
        if not calls:
            continue
        seen.add(group.index)
        steps += calls
        size = max(size, group.n)
        used = {name for step in calls for _, name in step.planes}
        planes.update({pl.name: pl for pl in group.planes.values() if pl.name in used})
    if not steps:
        return None
    size = 32 if all(st.op == "eigh_warp" for st in steps) else threads(size, forced)
    return TeamSpec(size, tuple(steps), tuple(planes.values()))


def markers(spec: TeamSpec) -> tuple:
    """The sinks of a team stage: one ``wide_out`` marker per plane it works on (the scratch and a
    block sum's ``out``), which bind the plane as an argument; the stage's body is
    :mod:`hawk.emit.team`'s."""
    names = [(p.name, p.dtype) for p in spec.planes] + ([(spec.out, REAL)] if spec.out else [])
    return tuple(WideWrite(name, Const(0, TensorType((), "int32")) if dtype == "int32"
                           else Const(0.0, TensorType((), "f64")))
                 for name, dtype in names)


# -- the sum over the rows of a key into a dense block ------------------------------



def block_size(payload: Node) -> int:
    """Elements of the block ``payload`` sums into."""
    return size_of(payload.ttype)


def block_chosen(payload: Node, build_form: str, spec, policy: str = "auto") -> bool:
    """Whether a block-shaped sum of ``payload`` is kept one :class:`BlockSum` under ``build_form`` (the build's
    ``linalg_form``): a block larger than ``rolled_fold_min`` (the variant's ``form.rolled_fold_min``) under ``"auto"``, of any size under
    ``"team"``, never under ``"thread"``; the ordered scalar or vector case always. A block with an extent above the variant's
    ``linalg.team_max_dim`` (the ordered vector's length, or either extent of a rank-1 product) is never kept: the sum is
    worked per entry (per sample), as it was before the limit left the trace."""
    limit = None if spec is None else team_max_dim_of(spec)
    if limit is not None and block_shaped(payload, policy) and not block_shaped(payload, policy, limit):
        return False
    return ordered_case(payload, policy, None if spec is None else team_max_dim_of(spec)) or (
        build_form != "thread" and (build_form == "team" or exceeds(block_size(payload), spec.primary.get("form", {}).get("rolled_fold_min"))))


def ordered_wanted(payload: Node, spec=None) -> bool:
    """Whether the ORDERED keyed sum of ``payload`` is one :class:`BlockSum`: block-shaped, and the ordered case always chooses it
    (within the variant's ``linalg.team_max_dim`` when ``spec`` is given)."""
    return block_shaped(payload, "ordered") and block_chosen(payload, "auto", spec, "ordered")


def plain_blocks(sinks, build_form: str, spec) -> tuple:
    """``sinks`` with every :class:`BlockSum` that ``build_form`` does not keep (:func:`block_chosen`) made its plain
    sum: a stored one as the cell or per-entry sinks of that sum, one read back in the body as a plain
    :class:`~hawk.ir.nodes.ReduceSamples`."""
    from raptor.ir.walk import resink as _resink
    from raptor.ir.walk import canonical_nodes, substitute

    sinks = tuple(sinks)
    mapping = {id(n): n.plain() for n in canonical_nodes(sinks)[0]
               if isinstance(n, BlockSum) and not block_chosen(n.operands[0], build_form, spec, n.policy if n.kept else "auto")}
    if not mapping:
        return sinks
    out: list = []
    for sink in sinks:
        if isinstance(sink, BlockSumWrite) and id(sink.block) in mapping:
            out.extend(mapping[id(sink.block)].stored(sink.name))
        elif isinstance(sink, BlockSumWrite) or not hasattr(sink, "value"):
            out.append(sink)
        else:
            new = substitute(tuple(sink.operands), mapping)
            out.append(_resink(sink, {id(o): n for o, n in zip(sink.operands, new)}))
    return tuple(out)


def team_max_dim_of(spec) -> int:
    """The largest dimension a team of threads works on, from the variant the build lowers for (``linalg.team_max_dim``), refused by
    name when it has none."""
    return spec.need("linalg", "team_max_dim")


def max_dim_of(spec, required: bool = False) -> int | None:
    """The largest dimension one lane factors alone, from the variant the build lowers for (``linalg.max_dim``); ``None`` when it
    states none, or refused by name under ``required``."""
    value = spec.primary.get("linalg", {}).get("max_dim")
    if value is None and required:
        raise RaptorError("a least-squares solve needs the target variant's linalg.max_dim, and this target's variant has none")
    return value


def widths(block: BlockSum) -> tuple:
    """``(p, q, rows)`` of ``block``: the width of ``u``, of ``v`` and the rows a lane holds."""
    u, v, rows = block.factors
    if v is None:
        return (u.ttype.shape[-1] if u.ttype.shape else 1), 1, rows
    return u.ttype.shape[-1], (v.ttype.shape[-1] if len(v.ttype.shape) == 2 or (
        len(v.ttype.shape) == 1 and rows == 1) else 1), rows


def plane_reads(block: BlockSum, out: str | None, read: bool) -> dict:
    """``{role: leaf}`` of the factors of ``block`` that are a bare scalar read of an input plane, which its team stage reads
    in place instead of from a copy in scratch: only the stored ordered sum of scalars, whose lane slice is one element."""
    p, q, rows = widths(block)
    if out is None or read or p * q != 1 or rows > 1:
        return {}
    u, v, _ = block.factors
    bare = lambda f: isinstance(f, Leaf) and f.role == "per_sample" and not f.ttype.shape
    return {role: f for role, f in (("u", u), ("v", v)) if f is not None and bare(f)}


def block_planes(index: int, block: BlockSum, with_g: bool = True, inplace: tuple = ()) -> tuple:
    """The scratch of block sum ``index``: the factors ``u`` (``rows * p`` per lane), ``v`` (``rows * q``
    per lane, absent when it is ``u``) and, when the body reads the sum back, the block ``g``
    (``p * q`` per key). A sum that is only stored goes straight into its output plane."""
    p, q, rows = widths(block)
    u, v, _ = block.factors
    base = f"{PLANE_PREFIX}b{index}"
    out = [] if "u" in inplace else [Plane(f"{base}_u", REAL, rows * p)]
    if v is not None and u is not v and "v" not in inplace:
        out.append(Plane(f"{base}_v", REAL, rows * q))
    if with_g:
        out.append(Plane(f"{base}_g", REAL, p * q, "key"))
    return tuple(out)


def block_read(block: BlockSum, index: int) -> Node:
    """The value of ``block`` at the lane's own key: the block of the key in its team stage's
    plane ``g`` (a stage that follows the one that computes the factors reads it)."""
    from . import TeamRead

    return TeamRead(f"{PLANE_PREFIX}b{index}_g", block.ttype, block.key)


def factor_stores(index: int, block: BlockSum, inplace: tuple = ()) -> tuple:
    """The stores of the factors of ``block`` into their scratch planes, by the per-lane stage (none for a factor read in place)."""
    from . import TeamStore

    u, v, _ = block.factors
    base = f"{PLANE_PREFIX}b{index}"
    stores = [("u", u)] if (u is v or v is None) else [("u", u), ("v", v)]
    return tuple(TeamStore(f"{base}_{role}", f) for role, f in stores if role not in inplace)


#: Blocks per key of a plain ordered sum's first stage; each owns 1024 of the 65536 virtual lanes (constants, never the device).
GROUPS = 64


def block_spec(index: int, block: BlockSum, out: str | None, keyed, read: bool = True, inplace: bool = False, forced: int = 0) -> TeamSpec:
    """The team stage of ``block``: one call that sums the rows of each key into its output plane
    ``out`` (``(*payload, *kept)``, payload first) when the sum is stored and not read back, else
    into the block ``g``, which is then copied to ``out`` when there is one. ``keyed`` is the plane's :class:`~hawk.passes.split.Keyed`: its key count is the
    stage's grid. The step's ``m`` is the rows a lane holds."""
    p, q, rows = widths(block)
    direct = out is not None and not read
    inplace = plane_reads(block, out, read) if inplace else {}
    planes = block_planes(index, block, not direct, tuple(inplace))
    names = {pl.name.rsplit("_", 1)[1]: pl.name for pl in planes}
    names.update({role: leaf.name for role, leaf in inplace.items()})
    ones = block.factors[1] is None         # an ordered plain sum: no v, the constant 1
    roles = (("u", names["u"]), *(() if ones else (("v", names.get("v", names["u"])),)),
             *((("out", out),) if out else ()), *((("g", names["g"]),) if not direct else ()))
    if direct and p * q == 1 and rows <= 1:      # a plain stored sum: GROUPS blocks per key add fixed partials, one block folds them
        part = Plane(f"{PLANE_PREFIX}b{index}_gp", PARTIAL, GROUPS, "key")
        first = TeamSpec(threads(1, forced), (Step("block_part", p, q, (*roles, ("gp", part.name)), rows),), (*planes, part),
                         "keys", keyed, None, groups=GROUPS, reads=tuple(inplace.values()))
        kept = tuple(r for r in roles if r[0] not in inplace)       # the last stage folds partials, it reads no factor in place
        last = TeamSpec(threads(1, forced), (Step("block_sum", p, q, (*kept, ("gp", part.name)), rows),), (*planes, part),
                        "keys", keyed, out)
        return first, last
    return (TeamSpec(threads(max(p, q), forced), (Step("block_sum", p, q, roles, rows),), planes, "keys", keyed, out),)


# -- the ordered gradient: a log of per-lane contributions, folded per key in lane order -------------

def log_names(target: str) -> tuple:
    """``(values, cells)``: the log planes of the ordered accumulate into ``target``."""
    return f"{PLANE_PREFIX}o_{target}_v", f"{PLANE_PREFIX}o_{target}_i"


def fold_spec(log: LogStore, keyed, forced: int = 0) -> TeamSpec:
    """The team stage that folds the log of ``log.target``: each key's lanes are contiguous, and
    for each of the ``log.width`` slots the key's contributions are added in lane order into the
    cell the slot names (``out[cell] = sum``; the cells are held per key, as reals): no atomics, the same bits on every launch. The
    gradient plane is stored, not accumulated into."""
    values, cells = log_names(log.target)
    planes = (Plane(values, REAL, log.width), Plane(cells, REAL, log.width, "key"))
    roles = (("v", values), ("i", cells), ("out", log.target))
    return TeamSpec(threads(log.width, forced), (Step("ordered_fold", log.width, 0, roles, 0),), planes,
                    "keys", keyed, log.target)


# -- the ordered scatter: per-lane logs, added per chunk of lanes, then per cell in chunk order ------

def scatter_names(target: str) -> tuple:
    """``(keys, values, parts, status)``: the scratch planes of the ordered scatter into ``target``."""
    return tuple(f"{PLANE_PREFIX}s_{target}_{role}" for role in ("k", "v", "p", "m"))


#: Threads of the team of an ordered scatter's stages: one does the adds, the rest of a warp idles.
SCATTER_THREADS = 32

#: The key a lane's log slot holds when the lane added nothing there; a chunk pass skips a negative key.
NO_KEY = -1


#: Chunks a status plane has a flag for: the most an ordered scatter cuts its lanes into.
STATUS_FLAGS = 256


def scatter_specs(target: str, width: int, max_chunks: int) -> tuple:
    """The three team stages that add the logs of the ordered scatter into ``target``; a lane holds
    ``width`` log slots (one per add of its body, in program order), a slot a lane did not write
    holds :data:`NO_KEY`. Lanes are cut into ``chunks`` runs of consecutive lanes
    (:func:`hawk._lane_bind.scatter_chunks`). One block per cell zeroes the cell's partial sums
    (block 0 also writes the cell count into the status plane, which is zero at the start of a run); one per chunk adds its lanes'
    slots in order into its own partial sum of each key (cell ``key * chunks + chunk`` of the parts
    plane), and puts :data:`NO_KEY` back in each slot it read; a key outside the cells that is not
    :data:`NO_KEY` adds nothing and sets the chunk's flag in the status plane, which the runtime
    reads after the run; one block per cell adds the cell's partial sums in chunk order into
    ``target``. No atomics and no sort: the same bits on every launch. ``max_chunks`` is the variant's ``scatter.max_chunks``."""
    keys, values, parts, status = scatter_names(target)
    logs = (Plane(keys, "int32", width, fill=NO_KEY), Plane(values, REAL, width))
    part = Plane(parts, PARTIAL, 1, "target")
    flags = Plane(status, "int32", 1 + STATUS_FLAGS, "once", fill=0)    # [cells, flag of chunk 0, flag of chunk 1, ...]

    def stage(op, roles, grid, planes, out=None):
        return TeamSpec(SCATTER_THREADS, (Step(op, width, 0, roles, max_chunks=max_chunks),), planes, grid, out=out, target=target)

    return (stage("scatter_zero", (("part", parts), ("status", status)), "targets", (part, flags)),
            stage("scatter_chunks", (("k", keys), ("v", values), ("part", parts), ("status", status)), "chunks",
                  (*logs, part, flags)),
            stage("scatter_cells", (("part", parts), ("out", target)), "targets", (part,), target))
