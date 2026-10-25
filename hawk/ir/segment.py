# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The per-unit split for the ``segmented`` dispatch policy.

``predicated`` and ``switch`` are per-NODE text choices the renderer makes;
``segmented`` differs in KIND: by the time a renderer sees a body the
dispatch node must already be gone. This module is where it goes:
:func:`split_segmented` takes one kernel's sinks and, if they carry a
``segmented`` :class:`~hawk.ir.nodes.Dispatch`, returns K UNITS — one per
branch, each an ordinary walk with the dispatch node replaced by its own
branch and ONE added ``lookup`` plane (the run-bounds ``offsets``) declared
but read by no IR node (its only reader is the backend's own index
prologue). :mod:`hawk.artifact.bundle` expands a segmented kernel into
these K units BEFORE compiling, so downstream (``device_plan``, eagle's
loader) never has to know segmentation exists.

A GENERIC NODE SUBSTITUTION, NOT A DEDICATED REBUILD PER KIND: every
:class:`~hawk.ir.nodes.Node` stores its children in ``operands``, a plain
attribute no subclass overrides, so "the same DAG with node X replaced by
node Y everywhere" is ONE generic pass — walk forward, keep an unchanged
node's object, ``copy.copy`` one whose children moved (preserving every
other slot) with the new operand tuple.

NOT SUPPORTED: a ``segmented`` dispatch reachable only from INSIDE a
:class:`~hawk.ir.loop_nodes.Loop` body refuses by name rather than silently
missing the split — a loop body is a SCOPE, not part of ``operands``, so
neither the ordinary walk nor this module's substitution reaches it."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

from ..types import Slot, TensorType
from .loop_nodes import Loop
from .nodes import (
    AccumWrite,
    Assign,
    Const,
    Dispatch,
    Finish,
    HawkError,
    Leaf,
    Node,
    Select,
    Sink,
)
from .walk import Walk, canonical

#: The offsets plane rides the SAME role every other run-time-addressed
#: plane does (``lookup``), read like any other lookup table.
OFFSETS_ROLE = "lookup"

#: K+1 exclusive-prefix entries, the language index type's width (HAWK types an
#: integer leaf ``i64`` regardless of the emitted wire width).
OFFSETS_DTYPE = "i64"

_I32 = TensorType((), "i32")
_BOOL = TensorType((), "bool")


def offsets_name(dispatch_name: str) -> str:
    """The offsets plane's slot name for dispatch group ``dispatch_name``."""
    return f"{dispatch_name}_offsets"


def mix_name(dispatch_name: str) -> str:
    """The K-counter ``lookup`` plane an ``auto`` dispatch counts the classes
    it picks into (one uint32 per branch, integer-atomic): the branch mix a
    launching runtime's chooser reads."""
    return f"{dispatch_name}_mix"


def class_name(dispatch_name: str) -> str:
    """The per-sample class plane a segmented unit writes (the clamped
    selector: the next sort's key; its bincount is the branch mix)."""
    return f"{dispatch_name}_class"


def stale_name(dispatch_name: str) -> str:
    """The per-sample mask a segmented unit sets where the selector no
    longer picks the unit's own branch (the sample sits in a stale run)."""
    return f"{dispatch_name}_stale"


class SegmentInfo(NamedTuple):
    """What a segmented UNIT needs beyond an ordinary kernel's
    ``sinks``/``walk``: its run ``j``, the branch count, the offsets
    plane's name, the dispatch group's name, and the sibling list."""

    j: int
    K: int
    offsets: str
    dispatch_name: str
    units: tuple[str, ...]
    #: The group's selector classification, carried from the primal walk
    #: (a unit's own walk holds no Dispatch node to classify).
    static_selector: bool = False
    selector_reads: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ClassifyUnit:
    """The kernel a static-selector split adds beside its K units: it writes
    each sample's clamped selector to the class plane and nothing else, so
    the runtime can sort (and count the mix) without running a branch."""

    name: str
    sinks: tuple
    walk: Walk


def classify_name(kernel_name: str) -> str:
    """The classify kernel's name for segmented kernel ``kernel_name``."""
    return f"{kernel_name}__cls"


@dataclass(frozen=True)
class SegmentUnit:
    """One of the K kernels :func:`split_segmented` returns — an ordinary
    kernel (``.name``/``.sinks``/``.walk``) plus the :class:`SegmentInfo` a
    segment-aware build step reads via ``getattr(kernel, "segment", None)``."""

    name: str
    sinks: tuple
    walk: Walk
    segment: SegmentInfo


def _loop_has_segmented(loop: Loop) -> bool:
    """Does ANY node reachable from ``loop``'s body carry a ``segmented``
    dispatch? The ordinary walk never reaches a loop body, so this exists
    to REFUSE loudly instead of silently mis-splitting or doing nothing."""
    seen: set = set()

    def visit(node: Node) -> bool:
        if id(node) in seen:
            return False
        seen.add(id(node))
        if isinstance(node, Dispatch) and node.policy == "segmented":
            return True
        if any(visit(c) for c in node.operands):
            return True
        if isinstance(node, Loop):
            return any(visit(s) for s in node.body_sinks)
        return False

    return any(visit(s) for s in loop.body_sinks)


def _substitute(order: Sequence[Node], swap) -> dict:
    """One forward pass over the canonical ``order``: a node ``swap`` maps to
    a replacement (``swap(node, rebuilt)``, given the rebuilt-so-far table)
    becomes it; a node whose children moved is cloned with the new operand
    tuple, else reused as-is."""
    rebuilt: dict = {}
    for node in order:
        replacement = swap(node, rebuilt)
        if replacement is not None:
            rebuilt[id(node)] = replacement
            continue
        new_ops = tuple(rebuilt.get(id(c), c) for c in node.operands)
        if new_ops == node.operands:
            rebuilt[id(node)] = node
        else:
            clone = copy.copy(node)
            clone.operands = new_ops
            rebuilt[id(node)] = clone
    return rebuilt


def with_policies(sinks: Sequence[Sink], chosen: dict) -> tuple:
    """``sinks`` with each Dispatch node ``id`` in ``chosen`` lowered under
    the policy ``chosen`` maps it to (same selector, branches and type)."""
    order = canonical(tuple(sinks)).order

    def swap(node, rebuilt):
        if id(node) not in chosen:
            return None
        ops = [rebuilt.get(id(c), c) for c in node.operands]
        return Dispatch(ops[0], ops[1:], chosen[id(node)], node.ttype)

    rebuilt = _substitute(order, swap)
    return tuple(rebuilt[id(s)] for s in sinks)


def clamped(selector: Node, K: int) -> Node:
    """``selector`` clamped to ``[0, K-1]`` — the branch every lowering picks."""
    from .ops import make

    zero, last = Const(0, _I32), Const(K - 1, _I32)
    upper = Select(make("gt", (selector, last)), last, selector, _I32)
    return Select(make("lt", (selector, zero)), zero, upper, _I32)


def _guarded(name: str, sinks: tuple, selector: Node, K: int, j: int,
             group: str) -> tuple:
    """Unit ``j``'s sinks under the SELECTOR GUARD: every commit keeps its
    old value, and the finish does not fire, where the selector no longer
    picks branch ``j`` — the sample is marked stale instead — plus the class
    and stale planes. A stale sample left by a sort that has gone out of date
    is then never committed with another branch's arithmetic."""
    from .ops import make

    sel = clamped(selector, K)
    mine = make("eq", (sel, Const(j, _I32)))
    out = []
    for sink in sinks:
        if isinstance(sink, Finish):
            out.append(Finish(sink.name, make("land", (mine, sink.value)),
                              steps=sink.steps))
        elif isinstance(sink, Assign):
            old = Leaf("prior_read", "mutable", sink.name, sink.ttype)
            out.append(Assign(sink.name, Select(mine, sink.value, old, sink.ttype),
                              sink.ttype))
        elif isinstance(sink, AccumWrite):
            zero = Const(0.0, TensorType((), sink.ttype.dtype))
            if sink.ttype.shape:
                zero = make("splat", (zero,), sink.ttype.shape)
            out.append(AccumWrite(sink.name, Select(mine, sink.value, zero, sink.ttype),
                                  sink.index, sink.ttype, sink.policy))
        else:
            raise HawkError(
                f"{name!r}: a segmented unit guards each commit by the selector, "
                f"and a {sink.kind} {getattr(sink, 'name', '?')!r} has no old value "
                "to keep where it does not apply; commit through a Mutable or a "
                "by-key store, or use policy='switch'")
    out.append(Assign(class_name(group), sel, _I32))
    out.append(Assign(stale_name(group), make("ne", (sel, Const(j, _I32))), _BOOL))
    return tuple(out)


def split_segmented(name: str, sinks: Sequence[Sink]) -> list | None:
    """``None`` when ``sinks`` carries no ``segmented`` dispatch; otherwise
    the K :class:`SegmentUnit` records the split produces. Refuses MORE
    THAN ONE segmented dispatch, and one reachable only from a loop body.

    A static selector (no plane it reads is written by the kernel) cannot
    change between sorts, so its units commit unguarded and write no class or
    stale plane: they stay sample-local, which keeps the host loop
    vectorisable. The classes then come from one extra :class:`ClassifyUnit`,
    appended last. A state-dependent selector gets the guarded units, which
    mark the samples whose branch moved."""
    sinks = tuple(sinks)
    walk0 = canonical(sinks)
    for node in walk0.order:
        if isinstance(node, Loop) and _loop_has_segmented(node):
            raise HawkError(
                f"{name!r}: a segmented dispatch inside a loop body has no "
                "per-unit split yet (the split runs over the top-level walk "
                "only; hawk/ir/segment.py)"
            )
    dispatch_nodes = [n for n in walk0.order if isinstance(n, Dispatch)]
    segmented = [(i, n) for i, n in enumerate(dispatch_nodes)
                 if n.policy == "segmented"]
    if not segmented:
        return None
    if len(segmented) > 1:
        raise HawkError(
            f"{name!r} carries {len(segmented)} segmented dispatch nodes; a "
            "kernel may declare at most one segmentation ('one "
            "segmentation per unit')"
        )
    idx, dnode = segmented[0]
    K = dnode.K
    info = walk0.dispatches[idx]
    dispatch_name = info.name
    offs_name = offsets_name(dispatch_name)
    offsets_slot = Slot(OFFSETS_ROLE, offs_name, TensorType((), OFFSETS_DTYPE))
    group_ids = {id(dnode)}
    unit_names = tuple(f"{name}__k{j}" for j in range(K))

    units = []
    for j in range(K):
        rebuilt = _substitute(
            walk0.order, lambda node, done: (
                done.get(id(node.branches[j]), node.branches[j])
                if id(node) in group_ids else None))
        selector = rebuilt.get(id(dnode.selector), dnode.selector)
        sinks_j = tuple(rebuilt[id(s)] for s in sinks)
        if info.static_selector:
            walk_j = canonical(sinks_j, declared=(offsets_slot,))
        else:
            sinks_j = _guarded(name, sinks_j, selector, K, j, dispatch_name)
            walk_j = canonical(sinks_j, declared=(offsets_slot,),
                               force_access="cross_sample_write")
        units.append(SegmentUnit(
            unit_names[j], sinks_j, walk_j,
            SegmentInfo(j, K, offs_name, dispatch_name, unit_names,
                        info.static_selector, info.selector_reads)))
    if info.static_selector:
        cls = (Assign(class_name(dispatch_name), clamped(dnode.selector, K), _I32),)
        units.append(ClassifyUnit(classify_name(name), cls, canonical(cls)))
    return units
