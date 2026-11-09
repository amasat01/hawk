# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Whether the host compiler vectorises the loop over samples of an emitted kernel.

ONE predicate, :func:`vectorises`, answers it for the translation unit the emitter
writes (:func:`hawk.emit.wrapper.render_source` records it on the
:class:`~hawk.emit.wrapper.Source`) and for the cost counts a launching runtime
reads (:mod:`hawk.ir.cost`, per dispatch branch), so the two cannot drift apart.

A host float select is a bitwise blend (:data:`hawk.emit.host.BLEND_HELPERS`), so an interchanged trip
loop calls its math unconditionally and needs no masked vector variant of it, on any compiler. A math call
inside a loop nested in it still stays scalar under ``native-vector-math``.

It holds when the kernel is sample-local (or only reads lookup planes: a read-only table is the same
for every lane, so an interchanged kernel may read one), no library call stays scalar under an exact
host profile, and no ``Loop`` the lanes run as a ``for`` of their own has a run-time trip count or, unless it is nested in an
interchanged loop (GCC vectorises that outer loop whatever the inner count), more trips than the compiler fully unrolls. A top-level ``Loop`` of a sample-local kernel is run
INTERCHANGED on the host (:func:`interchanged_loops`): its trips outside, a tile of
samples inside, so its trip count no longer matters; a ``Loop`` nested in it stays
each lane's own ``for`` (GCC vectorises the lanes around it). The scalar type does not matter: a float32 loop's guards and
tallies are float32 too.
"""

from __future__ import annotations

from collections.abc import Iterable

from . import linalg
from .loop_nodes import Loop, LoopCount, TapeRead
from .nodes import RAW_KIND, SWITCH_RENDERED, At, Dispatch, Node, Op
from .wide import REDUCTIONS
from .ops import RANDOM_OPS
from .walk import Walk

__all__ = ["UNROLL_LIMIT", "interchanged_loops", "lane_form_access", "speculative_step", "vectorise_blocker", "vectorises"]

#: The most trips of a ``Loop`` the host compiler fully unrolls (GCC's complete-peel
#: limit, ``--param max-completely-peel-times``, whose default is 16); a longer one
#: stays a rolled loop and the sample loop around it is scalar.
UNROLL_LIMIT = 16


def _flat(nodes: Iterable[Node]):
    """``nodes`` and everything inside their loop bodies."""
    for node in nodes:
        yield node
        if isinstance(node, Loop):
            yield from _flat(node.body_nodes())


def lane_form_access(walk: Walk) -> bool:
    """Whether the lanes of a tile may run side by side over ``walk``: the kernel is sample-local, or its
    only cross-sample access is a read of a lookup plane, which is read-only and the same for every lane (the
    class itself stays ``cross_sample_read``: a rank partition replicates a lookup plane, so it is legal there).
    A raw block declaring a cross-sample read is a table of unknown shape and keeps its sample loop."""
    cls = walk.access.cls
    if cls == "sample_local":
        return True
    return cls == "cross_sample_read" and not any(
        isinstance(n, Op) and n.kind == RAW_KIND for n in _flat(walk.order))


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
    for node in _flat(walk.order):
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
        if isinstance(n, Dispatch) and n.policy in SWITCH_RENDERED:
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
    return any(isinstance(n, Op) and n.kind in REDUCTIONS and n.operands and len(n.operands[0].ttype.shape) == 1
               and n.operands[0].ttype.shape[0] > UNROLL_LIMIT for n in _flat(loop.body_nodes()))


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


def vectorise_blocker(walk: Walk, profile: str | None = None, scalar_type: str | None = None,
                      nodes: Iterable[Node] | None = None) -> str:
    """Why the host loop over samples running ``nodes`` (default: the whole kernel) is not vectorised under
    host ``profile`` (default: the one in effect), else ``""``. ``scalar_type`` is accepted for the callers' sake;
    the verdict is the same for every type."""
    from ..compile.toolchain import EXACT_HOST_PROFILES, _arm64_gcc, _is_x86_64, host_profile
    from .cost import is_library_call

    lanes = {id(loop) for loop in interchanged_loops(walk)}
    cls = walk.access.cls
    if cls != "sample_local" and not (cls == "cross_sample_read" and lanes):
        return f"the access class is {cls}" + (" and no loop runs interchanged" if cls == "cross_sample_read" else "")
    name = host_profile(profile)
    exact = name in EXACT_HOST_PROFILES or not _is_x86_64()
    # GCC on arm64 widens pow alone (aether's vector variant); every other library call stays scalar there
    routed = ("pow",) if name not in EXACT_HOST_PROFILES and _arm64_gcc() else ()
    # a fixed-count loop nested in an interchanged one is vectorised across the lanes whatever its length
    inner_of_lanes = {id(n) for loop in interchanged_loops(walk) for n in _flat(loop.body_nodes()) if isinstance(n, Loop)}
    for node in _flat(walk.order if nodes is None else nodes):
        if exact and is_library_call(node) and node.kind not in routed:
            return f"a library call ({node.kind}) stays scalar under the exact host profile"
        if linalg.is_linalg(node) and node.kind in linalg.FACTORISATIONS + linalg.CONSUMERS:
            return f"{node.kind} branches per lane (a pivot search or a convergence skip)"
        if isinstance(node, Dispatch) and node.policy in SWITCH_RENDERED:
            return "a rendered switch is control flow the compiler does not vectorise"
        if isinstance(node, Loop):
            if id(node) in lanes:
                if not speculative_step(node):
                    return "an interchanged loop reads at a computed index, switches, holds a raw block or divides integers"
                for inner in _flat((*node.body_nodes(), *node.nexts, node.exit_cond)):
                    if isinstance(inner, Loop) and any(is_library_call(n) for n in _flat(inner.body_nodes())):
                        return "a library call inside a nested loop stays scalar"
            elif node.dynamic is not None or (node.trip > UNROLL_LIMIT and id(node) not in inner_of_lanes):
                return f"a loop of {'a run-time number of' if node.dynamic is not None else node.trip} trips stays its lane's own for"
    return ""


def vectorises(walk: Walk, profile: str | None = None, scalar_type: str | None = None,
               nodes: Iterable[Node] | None = None) -> bool:
    """Whether the host loop over samples running ``nodes`` (default: the whole kernel) is vectorised
    (:func:`vectorise_blocker` gives the reason when it is not)."""
    return not vectorise_blocker(walk, profile, scalar_type, nodes)
