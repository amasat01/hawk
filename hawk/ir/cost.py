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
- a ``Loop`` counts its body once per trip
- whatever the selector reads is left out (every sample evaluates it, whichever
  branch it takes), as is any value an ``at`` indexes into (the plane itself
  is not read, only the element)
- a value two branches share is counted in each: a split unit runs only its
  own branch, so that is what it pays
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from .loop_nodes import Loop
from .nodes import At, Const, Dispatch, Leaf, Node, Op, Select
from ..types import INT_PLANE_DTYPES
from .ops import INT_BINARY, INT_UNARY, OP_KINDS, RANDOM_OPS, UNARY_MATH
from .walk import Walk, dispatch_groups

__all__ = ["BranchCost", "DispatchCost", "dispatch_costs", "fp64_ops"]

_FIELDS = ("arith32", "arith64", "special32", "special64", "loads", "gathers")

#: Elementwise ops that are not transcendental: one hardware instruction.
_CHEAP_UNARY = ("neg", "abs", "sqrt", "rsqrt", "floor", "ceil", "trunc", "round", "rint",
                "sign", "spacing")
#: Elementwise binaries that call into the math library.
_LIBRARY_BINARY = ("pow", "atan2", "hypot", "fmod", "remainder")
_SPECIAL = (frozenset(UNARY_MATH) - frozenset(_CHEAP_UNARY)) | frozenset(_LIBRARY_BINARY) \
    | frozenset(RANDOM_OPS)
#: Ops that only move or re-label values.
_FREE = frozenset(("component", "component_at", "set_component_at", "vec", "splat", "transpose",
                   "as_pure", "as_vec3", "quat_conj"))
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

    def as_tuple(self) -> tuple[float, ...]:
        """The six counts in field order (the form eagle's rule takes)."""
        return tuple(getattr(self, f) for f in _FIELDS)


@dataclass(frozen=True)
class DispatchCost:
    """One dispatch group: a cost per branch, and the planes its selector reads."""

    branches: tuple[BranchCost, ...]
    #: Sorted ``(role, name)`` slots of the selector's leaf closure.
    selector_reads: tuple[tuple[str, str], ...]
    static_selector: bool
    #: The fused host loop is vectorised (the kernel is sample-local and any
    #: library call in a branch has a vector form under the host profile), so
    #: it evaluates every branch and selects: its cost is the sum over branches.
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
    if kind in _SPECIAL:
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


def _own_cost(node: Node) -> list[float]:
    """The counts ``node`` itself adds (not its operands'), in :data:`_FIELDS` order."""
    out = [0.0] * len(_FIELDS)
    if isinstance(node, Leaf):
        if node.role not in _NOT_A_PLANE:
            out[4] = _elements(node)
    elif isinstance(node, At):
        out[5] = max(_elements(node), 1)
    elif isinstance(node, Loop):
        out = [node.trip * c for c in _work(node.body_nodes())]
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


def dispatch_costs(walk: Walk, host_profile: str | None = None) -> dict[str, DispatchCost]:
    """``{group name: DispatchCost}`` for every dispatch group of ``walk``,
    under the names ``walk.dispatches`` publishes.

    ``fused_vectorised`` is set for a sample-local kernel whose branches call
    no library math (``exp``, ``sin``, ``pow`` ...), or whose host profile
    vectorises it (``native-vector-math``); under an exact profile such a call
    stays scalar and the fused loop does not run all branches in lockstep.
    ``host_profile`` names the profile (default: the one in effect, see
    :func:`hawk.compile.toolchain.host_profile`)."""
    from ..compile.toolchain import EXACT_HOST_PROFILES, host_profile as _in_effect
    exact = _in_effect(host_profile) in EXACT_HOST_PROFILES
    infos = {d.name: d for d in walk.dispatches}
    out = {}
    for name, node in dispatch_groups(walk).items():
        selector, *branches = node.operands
        evaluated = {id(n) for n in _closure(selector, set())}
        costs = tuple(BranchCost(*_work(_closure(b, evaluated))) for b in branches)
        library_call = any(c.special32 or c.special64 for c in costs)
        out[name] = DispatchCost(
            costs, infos[name].selector_reads, infos[name].static_selector,
            walk.access.cls == "sample_local" and not (exact and library_call))
    return out
