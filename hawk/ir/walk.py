# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""THE canonical walk.

``canonical(sinks)`` is the ONLY ordered traversal of a HAWK DAG:
depth-first post-order from the sink set, each node emitted once at its
first completion. Interior nodes dedup structurally on
``(kind, dtype, shape, tag, child_ids, literal)``; LEAVES and sinks dedup
on ``(role, name)`` with a TOTAL merge — a collision is refused, never
silently collapsed. Every consumer derives from :attr:`Walk.arg_spec`;
nothing re-walks. A second walk of the same DAG returns a byte-identical
:attr:`Walk.digest`, since every key is structural.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..types import Slot, TensorType
from . import access as _access
from .access import Access
from .compound import Quantity, QuantitySpan
from .loop_nodes import Loop, LoopValue, TapeRead
from .nodes import (
    FINISHED_PLANE,
    FINISHED_TTYPE,
    NAMESPACE_SEP,
    ROLE_ORDER,
    At,
    Dispatch,
    Finish,
    HawkError,
    Leaf,
    MapreducePartial,
    Node,
    Op,
    Primitive,
    Select,
    Sink,
)


@dataclass(frozen=True)
class DispatchInfo:
    """One :class:`~hawk.ir.nodes.Dispatch` node's sidecar record.

    ``name`` is POSITIONAL — ``dispatch0``, ``dispatch1``, ... in canonical
    order — because a dispatch node binds no slot and carries no
    author-given name; positional-in-canonical-order is the one identity
    stable across two renders of the same DAG."""

    name: str
    K: int
    policy: str
    #: The K per-unit names a ``segmented`` group split into
    #: (``hawk/ir/segment.py``) — empty here (not yet split) and for every
    #: other dispatch. A build-time caller overrides a published sidecar's
    #: record with one that DOES carry them, reusing this same shape.
    units: tuple[str, ...] = ()


@dataclass(frozen=True)
class Walk:
    """The frozen product of :func:`canonical`, computed once.

    ``order`` is the emission-order node list; ``leaves`` the ordered
    ``(role, name, ttype)`` records; ``arg_spec`` their canonical-order
    ``(role, name)`` projection; ``slot_of``/``slot_types`` its inverse and
    type projection; ``access`` the inferred class + op; ``quantities`` the
    reached compound quantities' contiguous wire spans; ``digest`` a
    content hash of all of the above (sidecar only); ``dispatches`` the
    per-kernel dispatch listing, positionally named; ``prior_reads`` the
    ``mutable``-role names read before their first store; ``finish`` is
    ``(mask, steps)`` for a self-finishing kernel, else ``None``."""

    order: tuple[Node, ...]
    leaves: tuple[Slot, ...]
    arg_spec: tuple[tuple[str, str], ...]
    slot_of: Mapping[tuple[str, str], int]
    slot_types: Mapping[tuple[str, str], TensorType]
    access: Access
    quantities: Mapping[str, QuantitySpan]
    digest: str
    dispatches: tuple[DispatchInfo, ...] = ()
    prior_reads: frozenset[str] = frozenset()
    finish: tuple[str, int | str] | None = None


#: The ranking, restated so :func:`canonical` can FLOOR access without
#: importing :mod:`hawk.ir.access`'s private table.
_ACCESS_RANK = {"sample_local": 0, "cross_sample_read": 1, "cross_sample_write": 2}


def _floored(acc: Access, floor: str) -> Access:
    """``acc``, raised to at least ``floor`` under the ranking — UNLESS
    ``acc`` is ``mapreduce`` (disjoint from the other three: an un-folded
    partial is a wrong answer, not a placement cost, so nothing floors it)."""
    if acc.cls == "mapreduce" or _ACCESS_RANK[acc.cls] >= _ACCESS_RANK[floor]:
        return acc
    return Access(floor, acc.op)


def canonical(sinks: Sequence[Sink], declared: Sequence[Slot] = (), *,
             force_access: str | None = None) -> Walk:
    """The ONE traversal. Refuses a mixed sink set and a
    total-merge collision before it returns anything.

    ``declared`` are slots a DECLARATION binds whether or not the body
    reaches them (the guard's ``terminated`` mask, like a compound
    quantity's unread wires); they join the slot order and digest like a
    reached leaf, merged on ``(role, name)`` by the same total merge.

    ``force_access`` FLOORS the inferred access class before the digest is
    computed — a ``segmented`` UNIT is ``cross_sample_write`` even though
    its offset read is invisible to node-level classification. Computed
    HERE, inside the traversal, so the digest stays a function of the
    whole Walk, never stale from an ``.access`` edited afterward."""
    sinks = tuple(sinks)
    _check_sink_set(sinks)
    finish = _finish_of(sinks)
    if finish is not None:
        # the finish epilogue's counter: declared, so it binds unconditionally.
        declared = (*declared, Slot("lookup", FINISHED_PLANE, FINISHED_TTYPE))
    order, bindings, position = _post_order(sinks)
    for slot in declared:
        held = bindings.get((slot.role, slot.name))
        if held is None:
            bindings[(slot.role, slot.name)] = Leaf(
                "terminated" if slot.role == "terminated" else "vocab_read",
                slot.role, slot.name, slot.ttype)
        elif held.ttype != slot.ttype:
            raise HawkError(
                f"declared slot (role={slot.role!r}, name={slot.name!r}) is typed "
                f"{slot.ttype!r} but the body binds it as {held.ttype!r}; a slot is "
                "keyed on (role, name) ALONE"
            )
    leaves, quantities = _order_bindings(order, bindings)
    arg_spec = tuple((s.role, s.name) for s in leaves)
    slot_of = {key: i for i, key in enumerate(arg_spec)}
    slot_types = {(s.role, s.name): s.ttype for s in leaves}
    quantities = _spans(quantities, slot_of)
    acc = _access.infer_over(order)
    if force_access is not None:
        acc = _floored(acc, force_access)
    digest = _digest(order, position, leaves, arg_spec, acc, quantities)
    dispatches = _dispatches(order)
    prior_reads = frozenset(n.name for n in order
                            if isinstance(n, Leaf) and n.kind == "prior_read")
    return Walk(order, leaves, arg_spec, MappingProxyType(slot_of),
                MappingProxyType(slot_types), acc, MappingProxyType(quantities),
                digest, dispatches, prior_reads, finish)


def _finish_of(sinks: tuple[Sink, ...]) -> tuple[str, int | str] | None:
    """``(mask, steps)`` of the ONE :class:`Finish` in ``sinks``, or ``None``.
    A second one is refused: the counter counts one mask's newly finished
    samples, and the sidecar's ``finish`` names one mask."""
    found = [s for s in sinks if isinstance(s, Finish)]
    if not found:
        return None
    if len(found) > 1:
        raise HawkError(
            f"canonical(): {len(found)} finish statements ("
            f"{[f.name for f in found]}); a kernel finishes its sample through "
            "ONE mask")
    return (found[0].name, found[0].steps)


def canonical_nodes(sinks: Sequence[Sink]) -> tuple[tuple[Node, ...],
                                                    Mapping[int, int]]:
    """THE traversal's node sequence and its ``id(node) -> position`` map.

    Same traversal as :func:`canonical`, handed to an IR->IR transform
    (:mod:`hawk.diff`) so a derivative is a function of the primal's
    STRUCTURE: structurally equal nodes share ONE position, so a duplicated
    subexpression accumulates one adjoint like a shared one. Derives no
    slots; ``arg_spec``/``slot_of`` stay :func:`canonical`'s alone."""
    nodes, _, position = _post_order(tuple(sinks))
    return nodes, MappingProxyType(position)


# --------------------------------------------------------------------------
# sink set


def _check_sink_set(sinks: tuple[Sink, ...]) -> None:
    if not sinks:
        raise HawkError(
            "canonical(): the sink set is empty — the sinks are the walk's root "
            "set"
        )
    for s in sinks:
        if isinstance(s, Loop):
            # A lowered loop that COMMITS once per iteration is itself a
            # root (its body sinks scatter at that iteration's row); one
            # with nothing to commit is not — read it via LoopValue instead.
            if not s.body_sinks:
                raise HawkError(
                    f"canonical(): the loop over {s.index.name!r} is in the root "
                    "set but commits nothing; a loop is a root only when its body "
                    "commits a sink once per iteration")
            continue
        if not isinstance(s, Sink):
            raise HawkError(
                f"canonical(): {s!r} is not a sink; the root set is sinks only"
            )
    reduces = [s for s in sinks if isinstance(s, MapreducePartial)]
    if reduces and len(sinks) > 1:
        other = next(s for s in sinks if s is not reduces[0])
        raise HawkError(
            f"mapreduce sink {reduces[0].name!r} (op={reduces[0].op!r}) may not "
            f"coexist with sink {other.name!r} (role {other.role!r}) in one kernel "
            "exec_access is ONE scalar and exec_op ONE value per manifest, so "
            "eagle would never fold the partials and Plan.run would return un-combined "
            "per-partition partials — a wrong answer, not a placement cost. A kernel "
            "that needs both is two kernels, sequenced by the plan."
        )


# --------------------------------------------------------------------------
# the traversal


def _post_order(sinks: tuple[Sink, ...]) -> tuple[
        tuple[Node, ...], dict[tuple[str, str], Node], dict[int, int]]:
    order: list[Node] = []
    position: dict[int, int] = {}          # id(node) -> its canonical position
    by_key: dict[Any, int] = {}            # structural key -> canonical position
    bindings: dict[tuple[str, str], Node] = {}   # (role,name) -> the binding node
    stack: list[tuple[Node, bool]] = [(s, False) for s in reversed(sinks)]
    while stack:
        node, expanded = stack.pop()
        if id(node) in position:
            continue
        if not expanded:
            stack.append((node, True))
            for child in reversed(node.operands):
                if id(child) not in position:
                    stack.append((child, False))
            continue
        key = _key(node, position)
        pos = by_key.get(key)
        if pos is None:
            pos = len(order)
            order.append(node)
            by_key[key] = pos
        position[id(node)] = pos
        _merge_binding(node, order[pos], bindings)
        if isinstance(node, Loop):
            # a body sink binds its slot exactly as a top-level one does: the
            # body is not in the flat order (it is a scope), but the PLANE it
            # commits to is a slot of this kernel's arg_spec.
            for sink in node.body_sinks:
                _merge_binding(sink, sink, bindings)
    return tuple(order), bindings, position


def _key(node: Node, position: Mapping[int, int]) -> Any:
    """The dedup key. Leaves key on ``(role, name)`` ALONE — the type is
    compared by :func:`_merge_binding`, so a disagreement is a refusal rather
    than two slots for one name."""
    if isinstance(node, Leaf):
        return ("leaf", node.role, node.name)
    t = node.ttype
    return (node.kind, t.dtype, t.shape, t.tag,
            tuple(position[id(c)] for c in node.operands), node.dedup_extra)


def _merge_binding(node: Node, canonical_node: Node,
                   bindings: dict[tuple[str, str], Node]) -> None:
    """The TOTAL merge."""
    slot = node.binding
    if slot is None:
        return
    key = (slot.role, slot.name)
    held = bindings.get(key)
    if held is None:
        bindings[key] = canonical_node
        return
    if held.ttype != node.ttype:
        raise HawkError(
            f"leaf identity collision on (role={slot.role!r}, name={slot.name!r}): "
            f"{type(held).__name__} is typed {held.ttype!r} but "
            f"{type(node).__name__} is typed {node.ttype!r}. A slot is keyed on "
            "(role, name) ALONE, so these two would share one slot; the "
            "merge is TOTAL and the disagreement is refused, never silently collapsed."
        )


# --------------------------------------------------------------------------
# slot ordering and the quantity spans


def _order_bindings(order: tuple[Node, ...], bindings: dict[tuple[str, str], Node]
                    ) -> tuple[tuple[Slot, ...], dict[str, Quantity]]:
    """The role order, plus the all-or-nothing quantity expansion."""
    reached: dict[str, Quantity] = {}
    q_index: dict[str, int] = {}
    for node in order:
        q = getattr(node, "quantity", None)
        if q is None:
            continue
        held = reached.get(q.prefix)
        if held is None:
            q_index[q.prefix] = len(reached)
            reached[q.prefix] = q
        elif held.wires != q.wires:
            raise HawkError(
                f"two quantities share the namespace {q.prefix!r} but declare "
                f"different wires ({[w.name for w in held.wires]} vs "
                f"{[w.name for w in q.wires]}) — a namespace identifies a quantity "
                ""
            )

    #: (role,name) -> (quantity prefix, wire index); reaching ANY wire binds
    #: EVERY wire of the quantity, unread ones included.
    owner: dict[tuple[str, str], tuple[str, int]] = {}
    slots: dict[tuple[str, str], Slot] = {}
    for prefix, q in reached.items():
        for i, slot in enumerate(q.expand()):
            key = (slot.role, slot.name)
            owner[key] = (prefix, i)
            slots[key] = slot
    for key, node in bindings.items():
        slots.setdefault(key, node.binding)

    for key in slots:
        if NAMESPACE_SEP in key[1] and key not in owner:
            raise HawkError(
                f"wire {key[1]!r} (role {key[0]!r}) is bound with no owning "
                f"quantity: the {NAMESPACE_SEP!r} namespace is reserved for "
                "compound-quantity wires, which only a Quantity declaration may mint "
                ""
            )

    ordered: list[Slot] = []
    for role in ROLE_ORDER:
        in_role = [s for k, s in slots.items() if k[0] == role]
        owned = sorted((s for s in in_role if (s.role, s.name) in owner),
                       key=lambda s: (q_index[owner[(s.role, s.name)][0]],
                                      owner[(s.role, s.name)][1]))
        free = sorted((s for s in in_role if (s.role, s.name) not in owner),
                      key=lambda s: s.name)
        ordered.extend(owned)
        ordered.extend(free)
    unknown = [s for k, s in slots.items() if k[0] not in ROLE_ORDER]
    if unknown:  # pragma: no cover - node constructors already close the role sets
        raise HawkError(
            f"binding with a role outside the canonical order: {unknown!r}"
        )
    return tuple(ordered), reached


def _spans(reached: dict[str, Quantity], slot_of: Mapping[tuple[str, str], int]
           ) -> dict[str, QuantitySpan]:
    """Per role, a reached quantity's slots form ONE consecutive,
    declaration-ordered run. The error fires only on the impossible state."""
    spans: dict[str, QuantitySpan] = {}
    for prefix, q in reached.items():
        expanded = q.expand()
        indices = tuple(slot_of[(s.role, s.name)] for s in expanded)
        per_role: dict[str, list[int]] = {}
        for slot, idx in zip(expanded, indices):
            per_role.setdefault(slot.role, []).append(idx)
        for role, run in per_role.items():
            if run != sorted(run) or run != list(range(run[0], run[0] + len(run))):
                raise HawkError(
                    f"quantity {prefix!r} occupies a non-contiguous or mis-ordered "
                    f"span in role {role!r}: arg_spec indices {run} (require "
                    "one consecutive, declaration-ordered run per role)"
                )
        spans[prefix] = QuantitySpan(prefix, tuple(s.name for s in expanded), indices)
    return spans


# --------------------------------------------------------------------------
# the per-kernel dispatch listing


def _dispatches(order: tuple[Node, ...]) -> tuple[DispatchInfo, ...]:
    """The ``{name, K, policy}`` record per :class:`Dispatch` node reached,
    POSITIONALLY named in the canonical order the digest hashes over —
    computed HERE since a consumer outside ``hawk/ir/`` may not re-derive
    one from ``order`` (see :class:`Walk`'s docstring)."""
    return tuple(DispatchInfo(f"dispatch{i}", node.K, node.policy)
                 for i, node in enumerate(n for n in order if isinstance(n, Dispatch)))


# --------------------------------------------------------------------------
# the digest


def _digest(order: tuple[Node, ...], position: Mapping[int, int],
            leaves: tuple[Slot, ...], arg_spec: tuple[tuple[str, str], ...],
            acc: Access, quantities: Mapping[str, QuantitySpan]) -> str:
    """Content hash of the whole product. Keys are STRUCTURAL — child
    references are positions in the emitted order — so two structurally equal
    DAGs built in any construction order hash identically."""
    h = hashlib.sha256()
    h.update(b"hawk.ir.walk/1\n")
    for i, node in enumerate(order):
        h.update(f"{i}|{_key(node, position)!r}\n".encode())
    for slot in leaves:
        h.update(f"leaf|{slot.role}|{slot.name}|{slot.ttype!r}\n".encode())
    h.update(f"arg_spec|{arg_spec!r}\n".encode())
    h.update(f"access|{acc.cls}|{acc.op}\n".encode())
    for prefix in sorted(quantities):
        h.update(f"quantity|{quantities[prefix]!r}\n".encode())
    return h.hexdigest()


def _rebuild(node: Node, operands: Sequence[Node]) -> Node:
    """``node`` with new operands, same kind, type and literal.

    Enumerated rather than generic because a node's constructor carries its
    own invariants (``At`` checks its plane's role, ``Select`` its branches
    agree); a family with no arm here refuses by name rather than silently
    passing through."""
    if isinstance(node, At):
        return At(operands[0], operands[1], node.ttype)
    if isinstance(node, Select):
        return Select(operands[0], operands[1], operands[2], node.ttype)
    if isinstance(node, TapeRead):
        return TapeRead(operands[0], node.slot, operands[1])
    if isinstance(node, LoopValue):
        return LoopValue(operands[0], node.slot)
    if isinstance(node, Primitive):
        return Primitive(node.name, node.rules, operands[0], operands[1:],
                         node.ttype)
    if isinstance(node, Op):
        return Op(node.kind, tuple(operands), node.ttype, node.literal)
    raise HawkError(                              # pragma: no cover - closed set
        f"no substitution rule for node {node!r} inside a loop body")


def substitute(roots: Sequence[Node], mapping: dict) -> tuple:
    """``roots`` with every node in ``mapping`` (by ``id``) replaced.

    Bottom-up with a memo; an unchanged subtree is REUSED, not copied — it
    must stay structurally shared with the primal wherever it didn't change,
    or the emitter's fan-out rule would see two copies of one expression."""
    memo: dict = dict(mapping)
    order: list = []
    seen: set = set()
    stack = [(r, False) for r in reversed(tuple(roots))]
    while stack:
        node, expanded = stack.pop()
        if id(node) in seen or id(node) in mapping:
            continue
        if not expanded:
            stack.append((node, True))
            for child in reversed(node.operands):
                if id(child) not in seen and id(child) not in mapping:
                    stack.append((child, False))
            continue
        seen.add(id(node))
        order.append(node)
    for node in order:
        new = [memo.get(id(c), c) for c in node.operands]
        memo[id(node)] = (node if all(a is b for a, b in zip(new, node.operands))
                          else _rebuild(node, new))
    return tuple(memo.get(id(r), r) for r in roots)
