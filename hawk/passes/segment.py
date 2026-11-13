# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The per-unit split for the ``segmented`` dispatch strategy.

``predicated`` and ``switch`` are per-NODE text choices the renderer makes;
``segmented`` differs in KIND: by the time a renderer sees a body the
dispatch node must already be gone. This module is where it goes:
:func:`split_segmented` takes one kernel's sinks and, if they carry a
``segmented`` :class:`~raptor.ir.nodes.Dispatch`, returns K UNITS — one per
branch, each an ordinary walk with the dispatch node replaced by its own
branch and ONE added ``lookup`` plane (the run-bounds ``offsets``) declared
but read by no IR node (its only reader is the backend's own index
prologue). :mod:`hawk.artifact.bundle` expands a segmented kernel into
these K units BEFORE compiling, so downstream (``device_plan``, eagle's
loader) never has to know segmentation exists.

A GENERIC NODE SUBSTITUTION, NOT A DEDICATED REBUILD PER KIND: every
:class:`~raptor.ir.nodes.Node` stores its children in ``operands``, a plain
attribute no subclass overrides, so "the same DAG with node X replaced by
node Y everywhere" is ONE generic pass — walk forward, keep an unchanged
node's object, ``copy.copy`` one whose children moved (preserving every
other slot) with the new operand tuple.

NOT SUPPORTED: a ``segmented`` dispatch reachable only from INSIDE a
:class:`~raptor.ir.nodes.Loop` body refuses by name rather than silently
missing the split — a loop body is a SCOPE, not part of ``operands``, so
neither the ordinary walk nor this module's substitution reaches it."""

from __future__ import annotations

import copy

from raptor.ir.nodes import has_feather, replace
from collections.abc import Sequence

from raptor.ir.nodes import Loop
from raptor.ir.nodes import AccumWrite, Assign, Const, Dispatch, Finish, Leaf, Node, Select, Sink
from .._errors import HawkError
from lowering.cells import OFFSETS_DTYPE, OFFSETS_ROLE, class_name, classify_name, offsets_name, stale_name
from raptor.ir.segment import ClassifyUnit, SegmentInfo, SegmentUnit
from raptor.ir.types import Slot, TensorType
from .. import components as _components
_dispatch_kind = _components.view("lowering", "dispatch")

_I32 = TensorType((), "i32")
_BOOL = TensorType((), "bool")


def _loop_has_segmented(loop: Loop) -> bool:
    """Does ANY node reachable from ``loop``'s body carry a ``segmented``
    dispatch? The ordinary walk never reaches a loop body, so this exists
    to REFUSE loudly instead of silently mis-splitting or doing nothing."""
    seen: set = set()

    def visit(node: Node) -> bool:
        if id(node) in seen:
            return False
        seen.add(id(node))
        if isinstance(node, Dispatch) and has_feather(node, "splits_by_branch"):
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
            clone = replace(node, operands=new_ops)
            rebuilt[id(node)] = clone
    return rebuilt


def with_strategies(sinks: Sequence[Sink], chosen: dict) -> tuple:
    """``sinks`` with each Dispatch node ``id`` in ``chosen`` lowered under
    the strategy ``chosen`` maps it to (same selector, branches and type)."""
    order = _walk(tuple(sinks)).order

    def swap(node, rebuilt):
        if id(node) not in chosen:
            return None
        ops = [rebuilt.get(id(c), c) for c in node.operands]
        return _dispatch_kind.dispatch(ops[0], ops[1:], chosen[id(node)], node.ttype)

    rebuilt = _substitute(order, swap)
    return tuple(rebuilt[id(s)] for s in sinks)


def clamped(selector: Node, K: int) -> Node:
    """``selector`` clamped to ``[0, K-1]`` — the branch every lowering picks."""
    from raptor.ir.ops import make

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
    from raptor.ir.ops import make

    sel = clamped(selector, K)
    mine = make("eq", (sel, Const(j, _I32)))
    out = []
    for sink in sinks:
        if isinstance(sink, Finish):
            out.append(Finish(sink.name, make("land", (mine, sink.value))))
        elif isinstance(sink, Assign):
            old = Leaf("prior_read", "mutable", sink.name, sink.ttype)
            out.append(Assign(sink.name, Select(mine, sink.value, old, sink.ttype),
                              sink.ttype))
        elif isinstance(sink, AccumWrite):
            zero = Const(0.0, TensorType((), sink.ttype.dtype))
            if sink.ttype.shape:
                zero = make("splat", (zero,), sink.ttype.shape)
            out.append(AccumWrite(sink.name, Select(mine, sink.value, zero, sink.ttype),
                                  sink.index, sink.ttype, sink.contract))
        else:
            raise HawkError(
                f"{name!r}: a segmented unit guards each commit by the selector, "
                f"and a {sink.kind} {getattr(sink, 'name', '?')!r} has no old value "
                "to keep where it does not apply; commit through a Mutable or a "
                "by-key store, or use strategy='switch'")
    out.append(Assign(class_name(group), sel, _I32))
    out.append(Assign(stale_name(group), make("ne", (sel, Const(j, _I32))), _BOOL))
    return tuple(out)


def _walk(sinks, declared=(), **kw):
    """The canonical form of ``sinks`` through the lowering wing's door."""
    return _components.view("lowering", "cells").walk(sinks, declared, **kw)


def rewalk(unit: SegmentUnit, sinks: Sequence[Sink]) -> SegmentUnit:
    """``unit`` over ``sinks`` (its own, rewritten by a pass that keeps what they compute): the walk is rebuilt as
    :func:`split_segmented` built it, the offsets declared and, for a guarded unit, the access class floored."""
    from dataclasses import replace

    slot = Slot(OFFSETS_ROLE, unit.segment.offsets, TensorType((), OFFSETS_DTYPE))
    floor = None if unit.segment.static_selector else "cross_sample_write"
    sinks = tuple(sinks)
    return replace(unit, sinks=sinks, walk=_walk(sinks, declared=(slot,), force_access=floor))


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
    walk0 = _walk(sinks)
    for node in walk0.order:
        if isinstance(node, Loop) and _loop_has_segmented(node):
            raise HawkError(
                f"{name!r}: a segmented dispatch inside a loop body has no "
                "per-unit split yet (the split runs over the top-level walk "
                "only; hawk/ir/segment.py)"
            )
    dispatch_nodes = [n for n in walk0.order if isinstance(n, Dispatch)]
    segmented = [(i, n) for i, n in enumerate(dispatch_nodes)
                 if has_feather(n, "splits_by_branch")]
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
            walk_j = _walk(sinks_j, declared=(offsets_slot,))
        else:
            sinks_j = _guarded(name, sinks_j, selector, K, j, dispatch_name)
            walk_j = _walk(sinks_j, declared=(offsets_slot,),
                               force_access="cross_sample_write")
        units.append(SegmentUnit(
            unit_names[j], sinks_j, walk_j,
            SegmentInfo(j, K, offs_name, dispatch_name, unit_names,
                        info.static_selector, info.selector_reads)))
    if info.static_selector:
        cls = (Assign(class_name(dispatch_name), clamped(dnode.selector, K), _I32),)
        units.append(ClassifyUnit(classify_name(name), cls, _walk(cls)))
    return units
