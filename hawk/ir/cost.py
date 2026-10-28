# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""What each branch of a ``dispatch`` group costs, counted from the IR.

A launching runtime that must choose between running a dispatch fused (every
sample runs its own branch inline) and split (one kernel per branch) needs two
inputs: how the samples spread over the branches, which only it can measure,
and how heavy each branch is, which the IR already says. This module
supplies the second: one :class:`BranchCost` per branch, a count of the
operations a sample runs when it takes that branch, by class.

Counting rules, per sample:

- an elementwise op counts once per element of its result; vector and matrix
  ops count their usual operation total (a ``dot`` of width ``n`` is ``2 n``)
- ``arith`` ops are adds, multiplies, compares, selects, ``sqrt``, ``div``;
  ``special`` ops are the transcendental and library calls (``exp``, ``sin``,
  ``pow``, ``atan2`` ...) and the random draws
- an op on a 64-bit float lands in the ``64`` field, everything else in ``32``;
  an integer op (an ``int32``/``uint32`` result, or ``//``, ``%``, a bit
  operator or shift on an index) is ``arith32``, a float -> integer cast too
- a plane leaf is one load per element; an ``at`` is one gather per element
- ``stores`` counts the plane elements the branch writes: for a whole kernel
  every sink's elements, for a dispatch branch the elements of the value it
  hands to the sink (``None`` when the branch feeds no sink directly, which
  leaves its traffic to be estimated as one write per load)
- a ``Loop`` counts its body once per trip
- whatever the selector reads is left out (every sample evaluates it, whichever
  branch it takes), as is any value an ``at`` indexes into (the plane itself
  is not read, only the element)
- the chain fields are the branch's longest dependent path of those ops by
  class (the critical path of the DAG, not the count): independent work adds
  to the counts but not to the path, and a ``Loop`` is its trip count times
  the deepest path through its carries
- ``vectorised`` says whether the host compiler can run the branch as a
  vector loop over samples (:func:`hawk.ir.vectorise.vectorises`):
  sample-local, no library call that stays scalar under an exact profile,
  and every ``Loop`` either run interchanged by the host (its trips outside,
  the samples' lanes inside: any trip count) or short enough for the
  compiler to unroll
- a value two branches share is counted in each: a split unit runs only its
  own branch, so that is what it pays
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from . import linalg, rays
from .loop_nodes import Loop, LoopCarry, LoopIndex
from .nodes import SWITCH_RENDERED, At, Const, Dispatch, Leaf, Node, Op, Select, Sink
from ..types import INT_PLANE_DTYPES
from .ops import INT_BINARY, INT_UNARY, OP_KINDS, RANDOM_OPS, UNARY_MATH
from .walk import Walk, dispatch_groups

__all__ = ["BranchCost", "DispatchCost", "dispatch_costs", "fp64_ops", "kernel_cost",
           "linalg_bytes"]

_FIELDS = ("arith32", "arith64", "special32", "special64", "loads", "gathers")
_CHAINS = ("chain32", "chain64", "chainSpecial32", "chainSpecial64")

#: Elementwise ops that are not transcendental: one hardware instruction.
_CHEAP_UNARY = ("neg", "abs", "sqrt", "rsqrt", "floor", "ceil", "trunc", "round", "rint",
                "sign", "spacing")
#: Elementwise binaries that call into the math library.
_LIBRARY_BINARY = ("pow", "atan2", "hypot", "fmod", "remainder")
SPECIAL_KINDS = (frozenset(UNARY_MATH) - frozenset(_CHEAP_UNARY)) | frozenset(_LIBRARY_BINARY) \
    | frozenset(RANDOM_OPS)
#: Ops that only move or re-label values.
_FREE = frozenset(("component", "component_at", "set_component_at", "add_at", "vec", "splat", "transpose",
                   "as_pure", "as_vec3", "quat_conj", *linalg.PROJECTIONS))
#: Role of a leaf that is a scalar parameter or a count, not a plane read.
_NOT_A_PLANE = ("uniform", "nsamples")
#: Operation totals of the quaternion ops (a Hamilton product is 16 multiplies and 12 adds).
_QUAT_OPS = {"quat_mul": 28, "quat_rotate": 56, "quat_recip": 8}


@dataclass(frozen=True)
class BranchCost:
    """Per-sample operation counts of one branch, by class."""

    arith32: float = 0.0
    arith64: float = 0.0
    special32: float = 0.0
    special64: float = 0.0
    loads: float = 0.0
    gathers: float = 0.0
    #: Longest dependent path, by class (``arith``/``special`` x 32/64).
    chain32: float = 0.0
    chain64: float = 0.0
    chainSpecial32: float = 0.0
    chainSpecial64: float = 0.0
    #: The host compiler can run this branch as a vector loop over samples.
    vectorised: bool = False
    #: Plane elements written; ``None`` when not counted.
    stores: float | None = None
    #: Bytes of storage the branch's linear-algebra nodes hold (:func:`linalg_bytes`).
    held_bytes: int = 0

    def as_tuple(self) -> tuple[float, ...]:
        """The six counts in field order (the form an older eagle takes)."""
        return tuple(getattr(self, f) for f in _FIELDS)

    def chain_tuple(self) -> tuple[float, ...]:
        """The six counts followed by the four chain lengths (ten values)."""
        return self.as_tuple() + tuple(getattr(self, f) for f in _CHAINS)

    def row_tuple(self) -> tuple[float, ...]:
        """The ten values of :meth:`chain_tuple`, then ``stores`` when it is counted."""
        return self.chain_tuple() + (() if self.stores is None else (self.stores,))


@dataclass(frozen=True)
class DispatchCost:
    """One dispatch group: a cost per branch, and the planes its selector reads."""

    branches: tuple[BranchCost, ...]
    #: Sorted ``(role, name)`` slots of the selector's leaf closure.
    selector_reads: tuple[tuple[str, str], ...]
    static_selector: bool
    #: The fused host loop is vectorised (every branch is), so it evaluates
    #: every branch and selects: its cost is the sum over branches.
    fused_vectorised: bool = False


def _elements(node: Node) -> int:
    return prod(node.ttype.shape)


def _wide(node: Node) -> bool:
    if node.ttype.dtype in INT_PLANE_DTYPES or getattr(node, "kind", None) in _INT_KINDS:
        return False
    return node.ttype.dtype == "f64" or any(o.ttype.dtype == "f64" for o in node.operands)


_INT_KINDS = INT_BINARY + INT_UNARY


def _op_count(op: Op) -> tuple[str, float]:
    """``(class, amount)`` of an op: ``class`` is ``arith`` or ``special``."""
    kind, ops = op.kind, op.operands
    if kind in linalg.LINALG_OPS:
        return "arith", linalg.op_count(op)
    if kind in rays.RAY_OPS:
        return "arith", rays.op_count(op)
    if kind in SPECIAL_KINDS:
        return "special", _elements(op)
    if kind in _QUAT_OPS:
        return "arith", _QUAT_OPS[kind]
    if kind in ("dot", "norm"):
        return "arith", 2 * _elements(ops[0]) + (kind == "norm")
    if kind == "sum":
        return "arith", _elements(ops[0])
    if kind == "cross":
        return "arith", 9
    if kind == "outer":
        return "arith", _elements(op)
    if kind in ("mv", "mm"):
        return "arith", 2 * prod(ops[0].ttype.shape) * (ops[1].ttype.shape[-1] if kind == "mm" else 1)
    return "arith", max(_elements(op), 1)


def _trips(loop) -> int:
    """The iterations a loop is costed at: its trip, and one for a loop whose
    bound is a run-time value with no ``max=`` (its static trip is the int range)."""
    return 1 if loop.dynamic == "open" else loop.trip


def _own_cost(node: Node) -> list[float]:
    """The counts ``node`` itself adds (not its operands'), in :data:`_FIELDS` order."""
    out = [0.0] * len(_FIELDS)
    if isinstance(node, Leaf):
        if node.role not in _NOT_A_PLANE:
            out[4] = _elements(node)
    elif isinstance(node, At):
        out[5] = max(_elements(node), 1)
    elif isinstance(node, Loop):
        out = [_trips(node) * c for c in _work(node.body_nodes())]
    elif isinstance(node, (Select, Dispatch)):
        out[1 if _wide(node) else 0] = max(_elements(node), 1)
    elif isinstance(node, Op) and node.kind in OP_KINDS and node.kind not in _FREE:
        cls, amount = _op_count(node)
        out[(2 if cls == "special" else 0) + (1 if _wide(node) else 0)] = amount
    return out


def _work(nodes) -> list[float]:
    total = [0.0] * len(_FIELDS)
    for node in nodes:
        if isinstance(node, Const):
            continue
        total = [a + b for a, b in zip(total, _own_cost(node))]
    return total


def _chain_class(node: Node) -> int | None:
    """Index into the four chain classes ``node`` adds one step to, else ``None``."""
    if isinstance(node, (Select, Dispatch)):
        return 1 if _wide(node) else 0
    if isinstance(node, Op) and node.kind in OP_KINDS and node.kind not in _FREE:
        cls, _ = _op_count(node)
        return (2 if cls == "special" else 0) + (1 if _wide(node) else 0)
    return None


def _paths(roots, zero: frozenset, carried: bool = False) -> tuple:
    """The critical path (per class) to the deepest of ``roots``; the nodes in
    ``zero`` and the loop placeholders cost nothing. An ``at`` depends on its index.

    With ``carried`` only the paths that start at a loop-carried read count (what
    one iteration adds to the recurrence); work that does not depend on a carry
    runs alongside it and adds nothing to the chain."""
    none = (0.0, 0.0, 0.0, 0.0)
    memo: dict[int, tuple | None] = {}
    stack = [(r, False) for r in roots]
    while stack:
        node, expanded = stack.pop()
        key = id(node)
        if key in memo:
            continue
        if isinstance(node, LoopCarry) and carried:
            memo[key] = none
            continue
        if key in zero or isinstance(node, (Leaf, Const, LoopIndex, LoopCarry)):
            memo[key] = None if carried else none
            continue
        deps = (node.index,) if isinstance(node, At) else node.operands
        if not expanded:
            stack.append((node, True))
            stack.extend((d, False) for d in deps if id(d) not in memo)
            continue
        best = None if carried else none
        for d in deps:
            reached = memo[id(d)]
            if reached is not None and (best is None or sum(reached) > sum(best)):
                best = reached
        if best is None:
            memo[key] = None
            continue
        own = [0.0] * 4
        if isinstance(node, Loop):
            body = _paths(node.nexts, frozenset(id(o) for o in node.operands), carried=True)
            own = [_trips(node) * c for c in body]
        elif (cls := _chain_class(node)) is not None:
            own[cls] = 1.0
        memo[key] = tuple(a + b for a, b in zip(best, own))
    reached = [memo[id(r)] for r in roots if memo[id(r)] is not None]
    return max(reached, key=sum, default=none)


def is_library_call(node: Node) -> bool:
    """Whether ``node`` is a library call (``exp``, ``sin``, ``pow``, a random draw ...)."""
    return isinstance(node, Op) and node.kind in SPECIAL_KINDS


def _branch_cost(walk: Walk, nodes: list[Node], roots, zero: frozenset, profile: str | None,
                 stores: float | None = None) -> BranchCost:
    from .vectorise import vectorises

    return BranchCost(*_work(nodes), *_paths(roots, zero), vectorises(walk, profile, nodes=nodes), stores,
                      linalg_bytes(nodes))


def _closure(root: Node, skip: set[int]) -> list[Node]:
    """Nodes ``root`` evaluates, each once, minus ``skip`` and what only ``skip`` reaches.
    An ``at`` reaches its index, never its plane."""
    seen = set(skip)
    out: list[Node] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        out.append(node)
        stack.extend((node.index,) if isinstance(node, At) else node.operands)
    return out


def fp64_ops(nodes) -> int:
    """The 64-bit floating-point operations ``nodes`` run once, by this
    module's counting rules: the ``arith64`` and ``special64`` counts (a
    transcendental or library call counts once per element)."""
    work = _work(nodes)
    return int(work[1] + work[3])


def linalg_bytes(nodes) -> int:
    """Bytes of storage the linear-algebra nodes of ``nodes`` hold, each counted
    once (a loop body once, however many trips): what a factorisation, a solve
    and the fields read from it occupy (:func:`hawk.ir.linalg.op_bytes`)."""
    total = 0
    for node in nodes:
        if isinstance(node, Loop):
            total += linalg_bytes(node.body_nodes())
        elif linalg.is_linalg(node):
            total += linalg.op_bytes(node)
    return total


def kernel_cost(walk: Walk, host_profile: str | None = None) -> BranchCost:
    """The cost of one whole sample of a kernel with no dispatch group:
    counts, chains and ``vectorised`` under the same rules as a branch.
    ``host_profile`` as in :func:`dispatch_costs`."""
    nodes = list(walk.order)
    stores = float(sum(_elements(n) for n in nodes if isinstance(n, Sink)))
    return _branch_cost(walk, nodes, nodes, frozenset(), host_profile, stores)


def dispatch_costs(walk: Walk, host_profile: str | None = None) -> dict[str, DispatchCost]:
    """``{group name: DispatchCost}`` for every dispatch group of ``walk``,
    under the names ``walk.dispatches`` publishes.

    Each branch's ``vectorised`` is true for a sample-local branch that calls
    no library math (``exp``, ``sin``, ``pow`` ...) or whose host profile
    vectorises it (``native-vector-math``), and whose every ``Loop`` the host
    runs interchanged (:func:`hawk.ir.vectorise.interchanged_loops`) or has at
    most :data:`hawk.ir.vectorise.UNROLL_LIMIT` trips. ``fused_vectorised`` is
    the group's flag: the fused loop runs every branch in lockstep only when
    every branch vectorises. ``host_profile`` names
    the profile (default: the one in effect, see
    :func:`hawk.compile.toolchain.host_profile`)."""
    sample_local = walk.access.cls == "sample_local"
    infos = {d.name: d for d in walk.dispatches}
    out = {}
    for name, node in dispatch_groups(walk).items():
        selector, *branches = node.operands
        evaluated = {id(n) for n in _closure(selector, set())}
        zero = frozenset(evaluated)
        written = any(isinstance(n, Sink) and n.value is node for n in walk.order)
        costs = tuple(_branch_cost(walk, _closure(b, evaluated), (b,), zero, host_profile,
                                   float(_elements(b)) if written else None)
                      for b in branches)
        out[name] = DispatchCost(
            costs, infos[name].selector_reads, infos[name].static_selector,
            sample_local and node.policy not in SWITCH_RENDERED   # a rendered switch keeps the loop scalar
            and all(c.vectorised for c in costs))
    return out


def working_set_bytes(sinks) -> int:
    """Bytes of per-lane storage the emitted text declares for its library results: every
    linear-algebra node at :func:`hawk.ir.linalg.op_bytes`, and each product or built matrix
    past :data:`hawk.emit.policy.ROLLED_FOLD_MIN` as the ``Item`` it is written into.
    Values the kernel may keep in registers are not counted. A figure to read a kernel's local
    memory against: a kernel whose frame is much larger is copying a matrix it need not."""
    from ..emit import policy
    from .linalg import LINALG_OPS, op_bytes
    from .walk import canonical_nodes

    width = {"f64": 8, "f32": 4}
    total = 0
    for node in canonical_nodes(tuple(sinks))[0]:
        if not isinstance(node, Op):
            continue
        if node.kind in LINALG_OPS and node.ttype.shape:
            total += op_bytes(node)
        elif node.ttype.shape and (policy.rolled(node) or (
                node.kind == "vec" and len(node.ttype.shape) == 2
                and prod(node.ttype.shape) > policy.ROLLED_FOLD_MIN)):
            total += prod(node.ttype.shape) * width.get(node.ttype.dtype, 8)
    return total
