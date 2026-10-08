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
from .nodes import Dispatch, HawkError, Node, Sink
from .walk import Walk, canonical

#: The offsets plane rides the SAME role every other run-time-addressed
#: plane does (``lookup``), read like any other lookup table.
OFFSETS_ROLE = "lookup"

#: K+1 exclusive-prefix entries, ``aether::idx_t``-wide (HAWK types an
#: integer leaf ``i64`` regardless of the emitted wire width).
OFFSETS_DTYPE = "i64"


def offsets_name(dispatch_name: str) -> str:
    """The offsets plane's slot name for dispatch group ``dispatch_name``."""
    return f"{dispatch_name}_offsets"


class SegmentInfo(NamedTuple):
    """What a segmented UNIT needs beyond an ordinary kernel's
    ``sinks``/``walk``: its run ``j``, the branch count, the offsets
    plane's name, the dispatch group's name, and the sibling list."""

    j: int
    K: int
    offsets: str
    dispatch_name: str
    units: tuple[str, ...]


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


def _substitute(order: Sequence[Node], group_ids: set, j: int) -> dict:
    """One forward pass over the canonical ``order``: every node in
    ``group_ids`` becomes its branch ``j``; a node whose children moved is
    cloned with the new operand tuple, else reused as-is."""
    rebuilt: dict = {}
    for node in order:
        if id(node) in group_ids:
            branch = node.branches[j]                 # type: ignore[attr-defined]
            rebuilt[id(node)] = rebuilt.get(id(branch), branch)
            continue
        new_ops = tuple(rebuilt.get(id(c), c) for c in node.operands)
        if new_ops == node.operands:
            rebuilt[id(node)] = node
        else:
            clone = copy.copy(node)
            clone.operands = new_ops
            rebuilt[id(node)] = clone
    return rebuilt


def split_segmented(name: str, sinks: Sequence[Sink]) -> list | None:
    """``None`` when ``sinks`` carries no ``segmented`` dispatch; otherwise
    the K :class:`SegmentUnit` records the split produces. Refuses MORE
    THAN ONE segmented dispatch, and one reachable only from a loop body."""
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
    dispatch_name = walk0.dispatches[idx].name
    offs_name = offsets_name(dispatch_name)
    offsets_slot = Slot(OFFSETS_ROLE, offs_name, TensorType((), OFFSETS_DTYPE))
    group_ids = {id(dnode)}
    unit_names = tuple(f"{name}__k{j}" for j in range(K))

    units = []
    for j in range(K):
        rebuilt = _substitute(walk0.order, group_ids, j)
        sinks_j = tuple(rebuilt[id(s)] for s in sinks)
        walk_j = canonical(sinks_j, declared=(offsets_slot,),
                           force_access="cross_sample_write")
        units.append(SegmentUnit(
            unit_names[j], sinks_j, walk_j,
            SegmentInfo(j, K, offs_name, dispatch_name, unit_names)))
    return units
