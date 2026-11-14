# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Whether the host compiler vectorises the loop over samples of an emitted kernel.

ONE predicate, :func:`vectorises`, answers it for the translation unit the emitter writes
(:func:`aether.wings.base.wrapper.render_source` records it on the :class:`~aether.wings.base.wrapper.Source`) and for the cost counts a
launching runtime reads (:func:`dispatch_costs`, :func:`kernel_cost`: the counts of :mod:`raptor.wings.base.cost` with the verdict
added per dispatch branch), so the two cannot drift apart.

A host float select is a bitwise blend (:data:`aether.wings.cpu.host.BLEND_HELPERS`), so an interchanged trip
loop calls its math unconditionally and needs no masked vector form of it, on any compiler. A math call
inside a loop nested in it still stays scalar under ``native-vector-math``.

It holds when the kernel is sample-local (or only reads lookup planes: a read-only table is the same
for every lane, so an interchanged kernel may read one), no library call stays scalar under an exact
host profile, and no ``Loop`` the lanes run as a ``for`` of their own has a run-time trip count or, unless it is nested in an
interchanged loop (GCC vectorises that outer loop whatever the inner count), more trips than the compiler fully unrolls
(:data:`raptor.wings.cpu.vectorise.UNROLL_LIMIT`). The scalar type does not matter: a float32 loop's guards and
tallies are float32 too.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, fields

from raptor.ir import linalg
from raptor.ir.nodes import Dispatch, Loop, Node
from raptor.ir.walk import Walk, dispatch_groups

from raptor.wings.base import cost, loops as _loops
from raptor.wings.base.dispatch import switch_form
BranchCost, DispatchCost, is_library_call = cost.BranchCost, cost.DispatchCost, cost.is_library_call
from raptor.wings.cpu.vectorise import UNROLL_LIMIT, interchanged_loops, kernel_nodes


@dataclass(frozen=True)
class HostBranchCost(BranchCost):
    """A branch's counts and whether the host compiler can run it as a vector loop over samples."""

    vectorised: bool = False


@dataclass(frozen=True)
class HostDispatchCost(DispatchCost):
    """A group's costs and whether the fused host loop is vectorised (every branch is), so it evaluates every branch and
    selects: its cost is the sum over branches."""

    fused_vectorised: bool = False


def vectorise_blocker(walk: Walk, spec, profile: str | None = None, scalar_type: str | None = None,
                      nodes: Iterable[Node] | None = None) -> str:
    """Why the host loop over samples running ``nodes`` (default: the whole kernel) is not vectorised under
    host ``profile``, a resolved name (default: the one in effect, a fact), else ``""``. ``scalar_type`` is accepted for the callers' sake;
    the verdict is the same for every type."""
    lanes = {id(loop) for loop in interchanged_loops(walk)}
    cls = walk.access.cls
    if cls != "sample_local" and not (cls == "cross_sample_read" and lanes):
        return f"the access class is {cls}" + (" and no loop runs interchanged" if cls == "cross_sample_read" else "")
    facts = spec.machine.facts
    name = profile or facts["host_profile"]        # a resolved profile name, else the one in effect (a fact)
    exact = name in facts["exact_host_profiles"] or spec.machine.isa != "x86_64"
    # GCC on arm64 widens pow alone (aether's vector form); every other library call stays scalar there
    routed = ("pow",) if name not in facts["exact_host_profiles"] and facts["arm64_gcc"] else ()
    # a fixed-count loop nested in an interchanged one is vectorised across the lanes whatever its length
    inner_of_lanes = {id(n) for loop in interchanged_loops(walk) for n in _loops.flat(loop.body_nodes()) if isinstance(n, Loop)}
    for node in kernel_nodes(walk, nodes):
        if exact and is_library_call(node) and node.kind not in routed:
            return f"a library call ({node.kind}) stays scalar under the exact host profile"
        if linalg.is_linalg(node) and node.kind in linalg.FACTORISATIONS + linalg.CONSUMERS:
            return f"{node.kind} branches per lane (a pivot search or a convergence skip)"
        if isinstance(node, Dispatch) and switch_form(node):
            return "a rendered switch is control flow the compiler does not vectorise"
        if isinstance(node, Loop):
            if id(node) in lanes:
                if not _loops.speculative_step(node):
                    return "an interchanged loop reads at a computed index, switches, holds a raw block or divides integers"
                for inner in _loops.flat((*node.body_nodes(), *node.nexts, node.exit_cond)):
                    if isinstance(inner, Loop) and any(is_library_call(n) for n in _loops.flat(inner.body_nodes())):
                        return "a library call inside a nested loop stays scalar"
            elif node.dynamic is not None or (node.trip > UNROLL_LIMIT and id(node) not in inner_of_lanes):
                return f"a loop of {'a run-time number of' if node.dynamic is not None else node.trip} trips stays its lane's own for"
    return ""


def vectorises(walk: Walk, spec, profile: str | None = None, scalar_type: str | None = None,
               nodes: Iterable[Node] | None = None) -> bool:
    """Whether the host loop over samples running ``nodes`` (default: the whole kernel) is vectorised
    (:func:`vectorise_blocker` gives the reason when it is not)."""
    return not vectorise_blocker(walk, spec, profile, scalar_type, nodes)


def _host(branch: BranchCost, vectorised: bool) -> HostBranchCost:
    return HostBranchCost(**{f.name: getattr(branch, f.name) for f in fields(branch)}, vectorised=vectorised)


def kernel_cost(walk: Walk, spec, host_profile: str | None = None) -> HostBranchCost:
    """:func:`raptor.wings.base.cost.kernel_cost` with ``vectorised`` filled under host ``host_profile``."""
    return _host(cost.kernel_cost(walk), vectorises(walk, spec, host_profile))


def dispatch_costs(walk: Walk, spec, host_profile: str | None = None) -> dict[str, HostDispatchCost]:
    """:func:`raptor.wings.base.cost.dispatch_costs` with every branch's ``vectorised`` filled under host ``host_profile``
    (default: the one in effect, see ``host_profile`` of the cpu backend's toolchain): true for a sample-local branch that calls
    no library math or whose host profile vectorises it (``native-vector-math``), and whose every ``Loop`` the host runs
    interchanged or has at most :data:`raptor.wings.cpu.vectorise.UNROLL_LIMIT` trips. ``fused_vectorised`` is the group's flag:
    the fused loop runs every branch in lockstep only when every branch vectorises."""
    sample_local = walk.access.cls == "sample_local"
    groups = dispatch_groups(walk)
    out = {}
    for name, counted in cost.dispatch_costs(walk).items():
        closures, _ = cost.branch_closures(groups[name])
        branches = tuple(_host(b, vectorises(walk, spec, host_profile, nodes=nodes))
                         for b, nodes in zip(counted.branches, closures))
        out[name] = HostDispatchCost(
            branches, counted.selector_reads, counted.static_selector,
            sample_local and not switch_form(groups[name])   # a rendered switch keeps the loop scalar
            and all(b.vectorised for b in branches))
    return out
