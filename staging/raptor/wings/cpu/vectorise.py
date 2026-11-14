# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Graph analysis of the loop over samples of a host kernel: which top-level ``Loop`` the host runs interchanged
(:func:`interchanged_loops`) and whether the lanes of a tile may run side by side (:func:`lane_form_access`). What the host compiler then does with the
emitted text is :mod:`aether.wings.cpu.host_vector`.

A top-level ``Loop`` of a sample-local kernel is run INTERCHANGED on the host: its trips outside, a tile of samples
inside, so its trip count no longer matters; a ``Loop`` nested in it stays each lane's own ``for``.
"""

from __future__ import annotations

from collections.abc import Iterable

from raptor.ir import linalg
from raptor.ir.nodes import Loop, LoopCount, TapeRead
from raptor.ir.nodes import RAW_KIND, Dispatch, Node, Op
from raptor.wings.base import loops as _loops, wide as _wide
from raptor.wings.base.dispatch import switch_form
from raptor.ir.ops import RANDOM_OPS
from raptor.ir.walk import Walk

__all__ = ["UNROLL_LIMIT", "interchanged_loops", "kernel_nodes", "lane_form_access"]

#: The most trips of a ``Loop`` the host compiler fully unrolls (GCC's complete-peel
#: limit, ``--param max-completely-peel-times``, whose default is 16); a longer one
#: stays a rolled loop and the sample loop around it is scalar.
UNROLL_LIMIT = 16


def kernel_nodes(walk: Walk, nodes: Iterable[Node] | None = None):
    """``nodes`` (default: the whole kernel) and everything inside their loop bodies."""
    return _loops.flat(walk.order if nodes is None else nodes)


def lane_form_access(walk: Walk) -> bool:
    """Whether the lanes of a tile may run side by side over ``walk``: the kernel is sample-local, or its
    only cross-sample access is a read of a lookup plane, which is read-only and the same for every lane (the
    class itself stays ``cross_sample_read``: a rank partition replicates a lookup plane, so it is legal there).
    A raw block declaring a cross-sample read is a table of unknown shape and keeps its sample loop."""
    cls = walk.access.cls
    if cls == "sample_local":
        return True
    return cls == "cross_sample_read" and not any(
        isinstance(n, Op) and n.kind == RAW_KIND for n in _loops.flat(walk.order))


def interchanged_loops(walk: Walk) -> tuple:
    """The top-level ``Loop`` nodes of ``walk`` the host runs interchanged (``for
    trip { for lane in tile { body } }``), in walk order; ``()`` when the kernel keeps
    its per-sample loop.

    Every top-level ``Loop`` of a sample-local kernel, unless the kernel has one the
    lane form cannot hold: a loop with a per-iteration commit, a tape or a count
    another loop reads (a derived loop pair), a loop only a
    ``switch`` case reaches, or a random draw anywhere (its stream is per sample)."""
    if not lane_form_access(walk):
        return ()
    loops = [n for n in walk.order if isinstance(n, Loop)]
    if not loops:
        return ()
    for node in _loops.flat(walk.order):
        if isinstance(node, (TapeRead, LoopCount)) or linalg.is_linalg(node):
            return ()
        if isinstance(node, Op) and node.kind in RANDOM_OPS:
            return ()
        if isinstance(node, Loop) and node.body_sinks:
            return ()
    # a loop only a switch case reaches renders inside that case
    used = {id(c) for n in walk.order for c in n.operands}
    seen: set = set()
    stack = [n for n in walk.order if id(n) not in used]
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, Dispatch) and switch_form(n):
            stack.append(n.selector)
        else:
            stack.extend(n.operands)
    if any(id(loop) not in seen for loop in loops):
        return ()
    return tuple(loop for loop in loops if not _folds_wide_vector(loop))


def _folds_wide_vector(loop: Loop) -> bool:
    """Whether a trip of ``loop`` folds a vector longer than the compiler unrolls (a row of a matrix product): the fold
    is the vector work already, one lane at a time, and a tile of lanes beside it only adds the traffic of the vector
    through the tile's rows."""
    return any(isinstance(n, Op) and n.kind in _wide.REDUCTIONS and n.operands and len(n.operands[0].ttype.shape) == 1
               and n.operands[0].ttype.shape[0] > UNROLL_LIMIT for n in _loops.flat(loop.body_nodes()))
