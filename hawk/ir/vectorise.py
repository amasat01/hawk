# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Whether the host compiler vectorises the loop over samples of an emitted kernel.

ONE predicate, :func:`vectorises`, answers it for the translation unit the emitter
writes (:func:`hawk.emit.backend.render_source` records it on the
:class:`~hawk.emit.backend.Source`) and for the cost counts a launching runtime
reads (:mod:`hawk.ir.cost`, per dispatch branch), so the two cannot drift apart.

It holds when the kernel is sample-local, no library call stays scalar under an exact
host profile, and no ``Loop`` the lanes run as a ``for`` of their own has more trips
than the compiler fully unrolls. A top-level ``Loop`` of a sample-local kernel is run
INTERCHANGED on the host (:func:`interchanged_loops`): its trips outside, a tile of
samples inside, so its trip count no longer matters; a ``Loop`` nested in it stays
each lane's own ``for``. The scalar type does not matter: a float32 loop's guards and
tallies are float32 too.
"""

from __future__ import annotations

from collections.abc import Iterable

from .loop_nodes import Loop, LoopCount, TapeRead
from .nodes import RAW_KIND, SWITCH_RENDERED, At, Dispatch, Node, Op
from .ops import RANDOM_OPS
from .walk import Walk

__all__ = ["UNROLL_LIMIT", "interchanged_loops", "speculative_step", "vectorises"]

#: The most trips of a ``Loop`` the host compiler fully unrolls (GCC's complete-peel
#: limit); a longer one stays a rolled loop and the sample loop around it is scalar.
UNROLL_LIMIT = 16


def _flat(nodes: Iterable[Node]):
    """``nodes`` and everything inside their loop bodies."""
    for node in nodes:
        yield node
        if isinstance(node, Loop):
            yield from _flat(node.body_nodes())


def interchanged_loops(walk: Walk) -> tuple:
    """The top-level ``Loop`` nodes of ``walk`` the host runs interchanged (``for
    trip { for lane in tile { body } }``), in walk order; ``()`` when the kernel keeps
    its per-sample loop.

    Every top-level ``Loop`` of a sample-local kernel, unless the kernel has one the
    lane form cannot hold: a loop with a per-iteration commit, a tape or a count
    another loop reads (a derived loop pair), a loop only a
    ``switch`` case reaches, or a random draw anywhere (its stream is per sample)."""
    if walk.access.cls != "sample_local":
        return ()
    loops = [n for n in walk.order if isinstance(n, Loop)]
    if not loops:
        return ()
    for node in _flat(walk.order):
        if isinstance(node, (TapeRead, LoopCount)):
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
        if isinstance(n, Dispatch) and n.policy in SWITCH_RENDERED:
            stack.append(n.selector)
        else:
            stack.extend(n.operands)
    if any(id(loop) not in seen for loop in loops):
        return ()
    return tuple(loops)


def speculative_step(loop: Loop) -> bool:
    """Whether one trip of ``loop`` is safe to run and discard for a stopped lane: no
    plane read at a computed index, no ``switch``, no raw block, no integer division
    anywhere in it. The interchanged lane loop then runs every lane and keeps a
    stopped lane's values by select; otherwise each lane's trip is a branch."""
    for node in (*_flat(loop.body_nodes()), *loop.nexts, loop.exit_cond):
        if isinstance(node, (At, TapeRead)):
            return False
        if isinstance(node, Dispatch) and node.policy in SWITCH_RENDERED:
            return False
        if isinstance(node, Op) and (node.kind == RAW_KIND or (
                node.kind in ("div", "fmod", "remainder", "floordiv", "mod")
                and node.ttype.dtype in ("i32", "i64"))):
            return False
    return True


def vectorises(walk: Walk, profile: str | None = None, scalar_type: str | None = None,
               nodes: Iterable[Node] | None = None) -> bool:
    """Whether the host loop over samples running ``nodes`` (default: the whole
    kernel) is vectorised under host ``profile`` (default: the one in effect) and
    ``scalar_type`` (accepted for the callers' sake; the verdict is the same for every type)."""
    from ..compile.toolchain import EXACT_HOST_PROFILES, host_profile
    from .cost import is_library_call

    if walk.access.cls != "sample_local":
        return False
    exact = host_profile(profile) in EXACT_HOST_PROFILES
    lanes = {id(loop) for loop in interchanged_loops(walk)}
    for node in _flat(walk.order if nodes is None else nodes):
        if exact and is_library_call(node):
            return False
        if isinstance(node, Loop):
            if id(node) in lanes:
                if not speculative_step(node):
                    return False
            elif node.trip > UNROLL_LIMIT:
                return False
    return True
