# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The strategy a ``dispatch`` node is lowered under (:func:`with_strategies`).

A GENERIC NODE SUBSTITUTION: every node reads its children as ``operands``, so "the same DAG with node X replaced by
node Y everywhere" is ONE generic pass: walk forward, keep an unchanged node, rebuild one whose children moved
(preserving every other field) over the new operand tuple. The
split of a ``segmented`` dispatch into one unit per branch is the lowering library's."""

from __future__ import annotations

from collections.abc import Sequence

from .. import _ir
from ..trace.value import dispatch_node


def _substitute(order: Sequence, swap) -> dict:
    """One forward pass over the canonical ``order``: a node ``swap`` maps to
    a replacement (``swap(node, rebuilt)``, given the rebuilt-so-far table)
    becomes it; a node whose children moved is cloned with the new operand
    tuple, else reused as-is."""
    rebuilt: dict = {}
    for node in order:
        replacement = swap(node, rebuilt)
        if replacement is not None:
            rebuilt[node] = replacement
            continue
        new_ops = tuple(rebuilt.get(c, c) for c in node.operands)
        if new_ops == tuple(node.operands):
            rebuilt[node] = node
        else:
            rebuilt[node] = _ir.replace(node, operands=new_ops)
    return rebuilt


def with_strategies(sinks: Sequence, chosen: dict) -> tuple:
    """``sinks`` with each Dispatch node in ``chosen`` lowered under
    the strategy ``chosen`` maps it to (same selector, branches and type)."""
    order, _ = _ir.canonical_nodes(tuple(sinks))

    def swap(node, rebuilt):
        if node not in chosen:
            return None
        ops = [rebuilt.get(c, c) for c in node.operands]
        return dispatch_node(ops[0], ops[1:], chosen[node], _ir.type_of(node))

    rebuilt = _substitute(order, swap)
    return tuple(rebuilt[s] for s in sinks)
