# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ordered gradient: ``vjp`` of a kernel whose keyed sum is ``policy="ordered"``.

The reverse of an ordered sum accumulates the gradient of each table it reads by key, a scatter-add
whose atomic order varies between launches. Here every such accumulate (at the top level or in the
body of a loop) is stated as an :class:`~raptor.ir.nodes.OrderedAccumWrite`: the lane's contribution into the
cell, applied in lane order. That is the math; how a backend keeps the order is its own lowering of the node."""

from __future__ import annotations

from raptor.ir import build_loop
from .._errors import HawkError
from raptor.ir.block import BlockSumWrite
from raptor.ir.loops import exit_kwargs
from raptor.ir.nodes import Loop
from raptor.ir.nodes import AccumWrite, OrderedAccumWrite


def kept_of(primal_sinks) -> tuple | None:
    """``(kept lane axes, key)`` of the ordered keyed sum the primal stores, else ``None``."""
    found = [(s.block.kept, s.block.key) for s in primal_sinks
             if isinstance(s, BlockSumWrite) and s.block.policy == "ordered"]
    return found[0] if found else None


def order(sinks, held) -> tuple:
    """``sinks`` with each accumulate stated as an ordered accumulate of the keyed sum ``held`` (``(kept, key)``)."""
    kept, key = held
    for sink in sinks:
        if isinstance(sink, AccumWrite) and (sink.ttype.shape or sink.policy == "or"):
            raise HawkError(f"an ordered gradient accumulates scalars; {sink.name!r} is not one")

    def stated(sink):
        return OrderedAccumWrite(sink.name, sink.value, sink.index, key, kept)

    out: list = []
    for sink in sinks:
        if isinstance(sink, AccumWrite):
            out.append(stated(sink))
        elif isinstance(sink, Loop):
            body: list = []
            for inner in sink.body_sinks:
                if isinstance(inner, Loop):
                    raise HawkError("an ordered gradient inside nested loops is not built")
                if not isinstance(inner, AccumWrite):
                    body.append(inner)
                    continue
                if abs(sink.step) != 1:
                    raise HawkError("an ordered gradient inside a loop of step other than 1 is not built")
                body.append(stated(inner))
            out.append(build_loop(sink.index, sink.carries, sink.inits, sink.nexts, start=sink.start,
                                  stop=sink.stop, step=sink.step, body_sinks=tuple(body),
                                  **exit_kwargs(sink)))
        else:
            out.append(sink)
    return tuple(out)
