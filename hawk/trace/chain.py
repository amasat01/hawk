# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.chain``: several kernels run in sequence as one kernel."""

from __future__ import annotations

from ..ir import HawkError
from ..ir.chain import bind
from .kernel import Kernel


class Chain(Kernel):
    """A kernel composed of ``members`` in order (:func:`chain`)."""

    __slots__ = ()


def chain(*kernels: Kernel, name: str) -> Chain:
    """Run ``kernels`` in sequence as ONE kernel named ``name``.

    A member's input binds to the value an earlier member committed under the same name; any other input is the
    chain's. A name no later member reads is an output, and a name several members update in turn keeps its last
    value. The values passed between members stay in the kernel: they are neither inputs nor outputs of the chain.
    A chain of chains is the chain of their members. The members must agree on their kind, launch grid, compaction
    group and jump width, and none may be stepped (chain the single step, then run :func:`hawk.steps` over the chain).
    """
    members = tuple(m for k in kernels for m in (k.members if isinstance(k, Chain) else (k,)))
    if not members:
        raise HawkError("chain: give at least one kernel")
    for k in members:
        if not isinstance(k, Kernel):
            raise HawkError(f"chain: {k!r} is not a traced kernel")
    composed = bind(members)
    out = Chain(name, composed.sinks, composed.planes, kind=members[0].kind)
    first, last = members[0], members[-1]
    out.lane_space, out.group, out.jump = first.lane_space, first.group, first.jump
    out.time, out.time_step = first.time, first.time_step
    out.returns = last.returns if last.returns is not None and set(_flat(last.returns)) <= set(composed.outputs) else None
    out.members = members
    return out


def _flat(tree):
    """The slot names in a ``returns`` tree."""
    if isinstance(tree, str):
        yield tree
    else:
        for v in (tree.values() if isinstance(tree, dict) else tree):
            yield from _flat(v)
