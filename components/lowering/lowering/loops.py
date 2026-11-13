# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Graph queries over the ``Loop`` nodes of a kernel: every node inside the loop bodies (:func:`flat`) and whether a trip of a
loop is safe to run and discard for a stopped lane (:func:`speculative_step`)."""

from __future__ import annotations

from collections.abc import Iterable

from raptor.ir.nodes import RAW_KIND, At, Dispatch, Loop, Node, Op, TapeRead

from .dispatch import switch_form


def flat(nodes: Iterable[Node]):
    """``nodes`` and everything inside their loop bodies."""
    for node in nodes:
        yield node
        if isinstance(node, Loop):
            yield from flat(node.body_nodes())


def speculative_step(loop: Loop) -> bool:
    """Whether one trip of ``loop`` is safe to run and discard for a stopped lane: no
    plane read at a computed index, no ``switch``, no raw block, no integer division
    anywhere in it. The interchanged lane loop then runs every lane and keeps a
    stopped lane's values by select; otherwise each lane's trip is a branch."""
    for node in (*flat(loop.body_nodes()), *loop.nexts, loop.exit_cond):
        if isinstance(node, (At, TapeRead)):
            return False
        if isinstance(node, Dispatch) and switch_form(node):
            return False
        if isinstance(node, Op) and (node.kind == RAW_KIND or (
                node.kind in ("div", "fmod", "remainder", "floordiv", "mod")
                and node.ttype.dtype in ("i32", "i64"))):
            return False
    return True
