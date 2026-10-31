# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Team-form linear algebra: what the split cuts at, and the planes between its stages.

A factorisation of a matrix too large for one lane (``n`` above
:data:`~hawk.ir.linalg.MAX_DIM`, or any size under ``linalg_form="team"``) is worked by a
team of threads, one team per lane, on a matrix that lives in scratch the runtime owns
(``aether::linalg::team``). That makes it a launch boundary, like a reduction over samples
that a later stage reads (:mod:`hawk.ir.split`):

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

import os
from dataclasses import dataclass

from ..types import TensorType
from .linalg import CONSUMERS, FACTORISATIONS, TEAM_MAX_DIM, dimension, form, is_linalg
from .nodes import At, Const, Dispatch, HawkError, Leaf, Node, Op, ReduceSamples, SampleIndex, WideWrite

#: The prefix of a scratch plane the split makes for team stages.
PLANE_PREFIX = "tm"

#: Environment override of the team size (a test door: results are bit-independent of it).
THREADS_ENV = "HAWK_TEAM_THREADS"

#: The index type of a scratch plane: ``lanes x n^2`` passes 2^31.
_INDEX = TensorType((), "i64")


def threads(n: int) -> int:
    """The team size for an ``n x n`` matrix: a perf constant, 128 up to ``n = 128`` and 256 above."""
    forced = os.environ.get(THREADS_ENV)
    if forced:
        return int(forced)
    return 128 if n <= 128 else 256


# -- the nodes a stage boundary leaves behind --------------------------------------

class TeamStore(WideWrite):
    """A per-lane stage's store of ``value`` (a scalar, vector or matrix, row-major) into its
    own slice of the scratch plane ``name``: elements ``[lane * size, (lane + 1) * size)``."""

    __slots__ = ()

    def __init__(self, name: str, value: Node) -> None:
        super().__init__(name, value, SampleIndex(_INDEX), TensorType((), value.ttype.dtype))


class LogStore(TeamStore):
    """The store of one adjoint contribution of lane ``i`` into slot ``index`` of an ordered log plane
    (``lanes * width`` elements): the contribution ``value`` of the accumulate into ``target``, or
    the cell it adds to. :func:`fold_spec` sums each key's contributions per cell in lane order."""

    __slots__ = ("target", "width", "kept")

    def __init__(self, name: str, value: Node, index: Node, target: str, width: int, kept: tuple) -> None:
        WideWrite.__init__(self, name, value, index, TensorType((), value.ttype.dtype))
        self.target, self.width, self.kept = target, width, kept

    @property
    def dedup_extra(self) -> object:
        return (self.role, self.name, self.target, self.width)


class ScatterLog(TeamStore):
    """The store of lane ``i``'s key (an ``int32`` plane) or value, into slot ``index`` of a log plane of an
    ordered ``hawk.scatter`` into ``target`` (:func:`scatter_specs` adds the logs up)."""

    __slots__ = ("target",)

    def __init__(self, name: str, value: Node, index: Node, target: str) -> None:
        WideWrite.__init__(self, name, value, index, TensorType((), value.ttype.dtype))
        self.target = target

    @property
    def dedup_extra(self) -> object:
        return (self.role, self.name, self.target)


class TeamRead(At):
    """The value of ``ttype`` held in slice ``index`` (the lane's own by default) of the scratch
    plane ``plane``: a result of a team stage read by a later per-lane stage."""

    __slots__ = ()

    def __init__(self, plane: str, ttype: TensorType, index: Node | None = None) -> None:
        scalar = TensorType((), ttype.dtype)
        super().__init__(Leaf("table_read", "lookup", plane, scalar),
                         SampleIndex(_INDEX) if index is None else index, ttype)


def size_of(ttype: TensorType) -> int:
    """Elements of one lane's slice of a value of ``ttype``."""
    out = 1
    for extent in ttype.shape:
        out *= extent
    return out


def read(plane: str, ttype: TensorType) -> Node:
    """The read of a team result: a rank-0 value is a plain ``at`` of the lane's element, a
    vector or matrix is a :class:`TeamRead`."""
    if not ttype.shape:
        return At(Leaf("table_read", "lookup", plane, ttype), SampleIndex(_INDEX), ttype)
    return TeamRead(plane, ttype)


# -- the scratch -------------------------------------------------------------------

@dataclass(frozen=True)
class Plane:
    """One scratch plane: ``per_lane`` elements of ``dtype`` for each ``unit`` (a lane, or a key of
    a :class:`BlockSum`)."""

    name: str
    dtype: str          # "real" (the kernel's scalar type) | "int32"
    per_lane: int
    unit: str = "lane"  # "lane" | "key" | "target" (chunks x the cells of the stage's ``target`` plane) | "once"
    fill: int | None = None     # what the plane holds at the start of every run (None: nothing is promised)


@dataclass(frozen=True)
class Step:
    """One collective call of a team stage: ``op`` over an ``n x n`` factor, ``k`` right-hand
    columns, on the scratch planes ``planes`` (``(role, name)`` pairs)."""

    op: str
    n: int
    k: int
    planes: tuple
    m: int = 0          # rows of a tall (QR) factor

    def plane(self, role: str) -> str:
        return dict(self.planes)[role]


@dataclass(frozen=True)
class TeamSpec:
    """A team stage: its size, the calls in order and the scratch planes they touch. ``grid`` is
    what a block of the launch is: a lane (``"lanes"``), a key of a block sum (``"keys"``, then
    ``keyed`` is the plane's :class:`~hawk.ir.split.Keyed` record and ``out`` the plane the stage
    commits to, which is not scratch of its own), or, in an ordered scatter into the plane ``target``, a
    chunk of lanes (``"chunks"``) or a cell of ``target`` (``"targets"``)."""

    threads: int
    steps: tuple
    planes: tuple
    grid: str = "lanes"
    keyed: object = None
    out: str | None = None
    target: str | None = None
    #: The lanes a block of a warp-form kernel serves: ``threads / 32`` warps, one per lane (:mod:`hawk.ir.lane`).
    lanes: int = 1


#: A scratch plane of the kernel's scalar type: the runtime gives it the width of the build's mode.
REAL = "real"


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


def members_of(order) -> tuple:
    """The groups of team-form linear algebra in ``order`` (every node under the sinks, children
    first): one per factorisation node, its members in the order they appear."""
    by_factor: dict = {}
    for node in order:
        if not is_linalg(node) or form(node) != "team":
            continue
        factor = node if node.kind in FACTORISATIONS else node.operands[0]
        if factor.kind not in FACTORISATIONS:
            raise HawkError(f"op {node.kind!r} reads a factor that no factorisation made")
        group = by_factor.get(id(factor))
        if group is None:
            n = dimension(factor)
            group = Group(len(by_factor), factor, [], n, REAL, {}, {}, {},
                          factor.operands[0].ttype.shape[0] if factor.kind == "qr" else 0)
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
            roles += [("w", p["w"].name), ("v", p["v"].name), ("work", p["work"].name)]
        if qr:
            roles += [("tau", p["tau"].name), ("work", p["work"].name)]
        out.append(Step(kind, n, 1, tuple(roles), group.m))
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


def refuse_in_loop(order) -> None:
    """Refuse a team-form node inside a ``Loop`` body (a stage boundary cannot be inside a loop;
    iterate by replaying the plan)."""
    from .loop_nodes import Loop

    for node in order:
        if isinstance(node, Loop):
            stack = list(node.body_nodes())
            while stack:
                inner = stack.pop()
                if is_linalg(inner) and form(inner) == "team":
                    raise HawkError(
                        f"op {inner.kind!r} at n={dimension(inner)} is worked by a team of threads, "
                        "which is a launch of its own and cannot sit inside a loop of the body; "
                        "iterate by replaying the plan (one launch per iteration)")
                if isinstance(inner, Loop):
                    stack.extend(inner.body_nodes())


def refuse_in_dispatch(sinks, order) -> None:
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
            raise HawkError(
                "a sum over the lanes of a key of per-row products into a dense block is a team "
                "stage, which every lane takes part in, so it cannot be inside an arm of a dispatch; "
                "sum outside the arm and select its result")
        if is_linalg(node) and form(node) == "team":
            raise HawkError(
                f"op {node.kind!r} at n={dimension(node)} is worked by a team of threads, which is a "
                "launch of its own that every lane takes part in, so it cannot be inside an arm of a "
                "dispatch; compute it outside the arm and select its result")


def spec_of(groups, level: int, depth: dict, seen: set) -> TeamSpec | None:
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
    return TeamSpec(threads(size), tuple(steps), tuple(planes.values()))


def markers(spec: TeamSpec) -> tuple:
    """The sinks of a team stage: one ``wide_out`` marker per plane it works on (the scratch and a
    block sum's ``out``), which bind the plane as an argument; the stage's body is
    :mod:`hawk.emit.team`'s."""
    names = [(p.name, p.dtype) for p in spec.planes] + ([(spec.out, REAL)] if spec.out else [])
    return tuple(WideWrite(name, Const(0, TensorType((), "int32")) if dtype == "int32"
                           else Const(0.0, TensorType((), "f64")))
                 for name, dtype in names)


# -- the sum over the rows of a key into a dense block ------------------------------

def factors(payload: Node) -> tuple | None:
    """``(u, v, rows)`` when ``payload`` is a sum over per-lane rows of rank-1 products, whose sum
    over the lanes of a key is ``sum_r u_r v_r^T``: ``outer(u, v)`` (a ``p x q`` block), ``u * v`` of
    a rank-1 ``u`` and a rank-0 ``v`` (a ``p`` vector: the ``q = 1`` block), and, for a per-lane
    ``rows x p`` matrix ``M``, ``transpose(M) @ M`` (``u = v = M``, a ``p x p`` block) and
    ``transpose(M) @ r`` (``v = r``, a ``p`` vector). ``rows`` is how many rows a lane holds (1 for
    the first two spellings); else ``None``."""
    if not isinstance(payload, Op):
        return None
    if payload.kind == "outer":
        return (*payload.operands, 1)
    if payload.kind == "mul":
        a, b = payload.operands
        if len(a.ttype.shape) == 1 and not b.ttype.shape:
            return a, b, 1
        if len(b.ttype.shape) == 1 and not a.ttype.shape:
            return b, a, 1
    if payload.kind in ("mm", "mv"):
        t, other = payload.operands
        if isinstance(t, Op) and t.kind == "transpose" and len(t.operands[0].ttype.shape) == 2:
            m = t.operands[0]
            if payload.kind == "mm" and other is m:
                return m, m, m.ttype.shape[0]
            if payload.kind == "mv" and other.ttype.shape == (m.ttype.shape[0],):
                return m, other, m.ttype.shape[0]
    return None


def block_size(payload: Node) -> int:
    """Elements of the block ``payload`` sums into."""
    return size_of(payload.ttype)


def wanted(payload: Node, build_form: str, policy: str = "auto") -> bool:
    """Whether the keyed sum of ``payload`` is one :class:`BlockSum` under ``build_form`` (the
    ``linalg_form`` of the trace): a product of per-row rank-1 values whose block is larger than
    :data:`~hawk.emit.policy.ROLLED_FOLD_MIN` under ``"auto"``, of any size under ``"team"``,
    never under ``"thread"``. Under ``policy="ordered"`` any floating scalar or vector payload is
    one (rows of a key added in lane order: no atomics, the same bits on every launch)."""
    from ..emit.policy import ROLLED_FOLD_MIN

    if policy == "ordered" and (payload.ttype.dtype in ("f64", "f32") and len(payload.ttype.shape) <= 1
                                and all(extent <= TEAM_MAX_DIM for extent in payload.ttype.shape)):
        return True
    pair = factors(payload)
    if pair is None or build_form == "thread" or payload.ttype.dtype not in ("f64", "f32"):
        return False
    if any(extent > TEAM_MAX_DIM for extent in payload.ttype.shape):
        return False
    return build_form == "team" or block_size(payload) > ROLLED_FOLD_MIN


class BlockSum(ReduceSamples):
    """The sum over the lanes of a key of a product of per-row rank-1 values into a dense
    ``p x q`` block (or a ``p`` vector): ONE node where the per-entry form would be ``p * q``
    accumulates. It is a keyed :class:`~hawk.ir.nodes.ReduceSamples` to everything that reads it
    (the payload is its operand, so a derivative and a rebuild see the per-entry sum), and the
    split lowers it to a stage of per-row stores of the factors and a team stage that owns each
    key's block (:func:`hawk.ir.split.stages`). The sum is ordered: the rows of a key are added in
    lane order, whatever the policy."""

    __slots__ = ()

    @property
    def factors(self) -> tuple:
        """``(u, v, rows)`` of the payload (:func:`factors`); an ordered sum of a plain scalar or
        vector is ``(payload, None, 1)``: ``v`` is the constant 1."""
        payload = self.operands[0]
        return factors(payload) or (payload, None, 1)

    def recount(self, count: Node) -> BlockSum:
        return BlockSum("sum", self.operands[0], key=self.key, count=count, policy=self.policy,
                        kept=self.kept)

    def stored(self, name: str, ttype: TensorType | None = None) -> tuple:
        return (BlockSumWrite(name, self),)

    def expanded(self, name: str) -> tuple:
        """The per-entry accumulates that store this sum to ``name`` (what a derivative sees)."""
        return ReduceSamples.stored(self, name)


class BlockSumWrite(WideWrite):
    """The store of a :class:`BlockSum` to the output plane ``name`` (``(*payload, *kept)``,
    payload first). The split turns it into the team stage that writes the plane."""

    __slots__ = ()

    def __init__(self, name: str, block: BlockSum) -> None:
        super().__init__(name, block, None, TensorType((), block.ttype.dtype))

    @property
    def block(self) -> BlockSum:
        return self.operands[0]

    def expanded(self) -> tuple:
        return self.block.expanded(self.name)


def expand_stores(sinks) -> tuple:
    """``sinks`` with every :class:`BlockSumWrite` replaced by its per-entry accumulates."""
    out: list = []
    for sink in sinks:
        out.extend(sink.expanded() if isinstance(sink, BlockSumWrite) else (sink,))
    return tuple(out)


def widths(block: BlockSum) -> tuple:
    """``(p, q, rows)`` of ``block``: the width of ``u``, of ``v`` and the rows a lane holds."""
    u, v, rows = block.factors
    if v is None:
        return (u.ttype.shape[-1] if u.ttype.shape else 1), 1, rows
    return u.ttype.shape[-1], (v.ttype.shape[-1] if len(v.ttype.shape) == 2 or (
        len(v.ttype.shape) == 1 and rows == 1) else 1), rows


def block_planes(index: int, block: BlockSum, with_g: bool = True) -> tuple:
    """The scratch of block sum ``index``: the factors ``u`` (``rows * p`` per lane), ``v`` (``rows * q``
    per lane, absent when it is ``u``) and, when the body reads the sum back, the block ``g``
    (``p * q`` per key). A sum that is only stored goes straight into its output plane."""
    p, q, rows = widths(block)
    u, v, _ = block.factors
    base = f"{PLANE_PREFIX}b{index}"
    out = [Plane(f"{base}_u", REAL, rows * p)]
    if v is not None and u is not v:
        out.append(Plane(f"{base}_v", REAL, rows * q))
    if with_g:
        out.append(Plane(f"{base}_g", REAL, p * q, "key"))
    return tuple(out)


def block_read(block: BlockSum, index: int) -> Node:
    """The value of ``block`` at the lane's own key: the block of the key in its team stage's
    plane ``g`` (a stage that follows the one that computes the factors reads it)."""
    return TeamRead(f"{PLANE_PREFIX}b{index}_g", block.ttype, block.key)


def factor_stores(index: int, block: BlockSum) -> tuple:
    """The stores of the factors of ``block`` into their scratch planes, by the per-lane stage."""
    u, v, _ = block.factors
    base = f"{PLANE_PREFIX}b{index}"
    return (TeamStore(f"{base}_u", u),) if (u is v or v is None) else (TeamStore(f"{base}_u", u), TeamStore(f"{base}_v", v))


def block_spec(index: int, block: BlockSum, out: str | None, keyed, read: bool = True) -> TeamSpec:
    """The team stage of ``block``: one call that sums the rows of each key into its output plane
    ``out`` (``(*payload, *kept)``, payload first) when the sum is stored and not read back, else
    into the block ``g``, which is then copied to ``out`` when there is one. ``keyed`` is the plane's :class:`~hawk.ir.split.Keyed`: its key count is the
    stage's grid. The step's ``m`` is the rows a lane holds."""
    p, q, rows = widths(block)
    direct = out is not None and not read
    planes = block_planes(index, block, not direct)
    names = {pl.name.rsplit("_", 1)[1]: pl.name for pl in planes}
    ones = block.factors[1] is None         # an ordered plain sum: no v, the constant 1
    roles = (("u", names["u"]), *(() if ones else (("v", names.get("v", names["u"])),)),
             *((("out", out),) if out else ()), *((("g", names["g"]),) if not direct else ()))
    return TeamSpec(threads(max(p, q)), (Step("block_sum", p, q, roles, rows),), planes, "keys", keyed, out)


# -- the ordered gradient: a log of per-lane contributions, folded per key in lane order -------------

def log_names(target: str) -> tuple:
    """``(values, cells)``: the log planes of the ordered accumulate into ``target``."""
    return f"{PLANE_PREFIX}o_{target}_v", f"{PLANE_PREFIX}o_{target}_i"


def fold_spec(log: LogStore, keyed) -> TeamSpec:
    """The team stage that folds the log of ``log.target``: each key's lanes are contiguous, and
    for each of the ``log.width`` slots the key's contributions are added in lane order into the
    cell the slot names (``out[cell] = sum``; the cells are held per key, as reals): no atomics, the same bits on every launch. The
    gradient plane is stored, not accumulated into."""
    values, cells = log_names(log.target)
    planes = (Plane(values, REAL, log.width), Plane(cells, REAL, log.width, "key"))
    roles = (("v", values), ("i", cells), ("out", log.target))
    return TeamSpec(threads(log.width), (Step("ordered_fold", log.width, 0, roles, 0),), planes,
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


def scatter_specs(target: str, width: int) -> tuple:
    """The three team stages that add the logs of the ordered scatter into ``target``; a lane holds
    ``width`` log slots (one per add of its body, in program order), a slot a lane did not write
    holds :data:`NO_KEY`. Lanes are cut into ``chunks`` runs of consecutive lanes
    (:func:`hawk._lane_bind.scatter_chunks`). One block per cell zeroes the cell's partial sums
    (block 0 also writes the cell count into the status plane, which is zero at the start of a run); one per chunk adds its lanes'
    slots in order into its own partial sum of each key (cell ``key * chunks + chunk`` of the parts
    plane), and puts :data:`NO_KEY` back in each slot it read; a key outside the cells that is not
    :data:`NO_KEY` adds nothing and sets the chunk's flag in the status plane, which the runtime
    reads after the run; one block per cell adds the cell's partial sums in chunk order into
    ``target``. No atomics and no sort: the same bits on every launch."""
    keys, values, parts, status = scatter_names(target)
    logs = (Plane(keys, "int32", width, fill=NO_KEY), Plane(values, REAL, width))
    part = Plane(parts, REAL, 1, "target")
    flags = Plane(status, "int32", 1 + STATUS_FLAGS, "once", fill=0)    # [cells, flag of chunk 0, flag of chunk 1, ...]

    def stage(op, roles, grid, planes, out=None):
        return TeamSpec(SCATTER_THREADS, (Step(op, width, 0, roles),), planes, grid, out=out, target=target)

    return (stage("scatter_zero", (("part", parts), ("status", status)), "targets", (part, flags)),
            stage("scatter_chunks", (("k", keys), ("v", values), ("part", parts), ("status", status)), "chunks",
                  (*logs, part, flags)),
            stage("scatter_cells", (("part", parts), ("out", target)), "targets", (part,), target))
