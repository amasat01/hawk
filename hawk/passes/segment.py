# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The strategy a ``dispatch`` node is lowered under (:func:`with_strategies`).

A GENERIC NODE SUBSTITUTION: every :class:`~raptor.ir.nodes.Node` stores its children in ``operands``, a plain attribute no
subclass overrides, so "the same DAG with node X replaced by node Y everywhere" is ONE generic pass: walk forward, keep an
unchanged node's object, clone one whose children moved (preserving every other slot) with the new operand tuple. The
split of a ``segmented`` dispatch into one unit per branch is the lowering library's."""

from __future__ import annotations

from collections.abc import Sequence

from raptor.ir.nodes import Node, Sink, replace
from .. import wings as _wings
_dispatch_kind = _wings.view("raptor", "base", "dispatch")


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


def _walk(sinks, declared=(), **kw):
    """The canonical form of ``sinks`` through the lowering wing's door."""
    return _wings.view("raptor", "base", "cells").walk(sinks, declared, **kw)
