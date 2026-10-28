# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ordered gradient: ``vjp`` of a kernel whose keyed sum is ``policy="ordered"``.

The reverse of an ordered sum accumulates the gradient of each table it reads by key, a scatter-add
whose atomic order varies between launches. Here every such accumulate (at the top level or in the
body of a loop) is instead a store of the lane's own contribution, and of the cell it adds to, into a
log plane (``lanes * E`` elements for ``E`` contributions per lane); a stage of its own then adds
each key's contributions per cell in lane order (:func:`hawk.ir.team.fold_spec`). The result is the
same bits on every launch and for any number of samples per key. The log is ``lanes * E`` values
and ``lanes * E`` cell numbers (``int32``) of scratch; the gradient plane is stored, not
accumulated into."""

from __future__ import annotations

from ..ir import Const, HawkError, SampleIndex, build_loop
from ..ir import make as _mk
from ..ir.loops import exit_kwargs
from ..ir.loop_nodes import Loop
from ..ir.nodes import AccumWrite
from ..ir.team import BlockSumWrite, LogStore, log_names
from ..types import TensorType

_INT = TensorType((), "i32")


def kept_of(primal_sinks) -> tuple | None:
    """``(kept lane axes, key)`` of the ordered keyed sum the primal stores, else ``None``."""
    found = [(s.block.kept, s.block.key) for s in primal_sinks
             if isinstance(s, BlockSumWrite) and s.block.policy == "ordered"]
    return found[0] if found else None


def _sites(sinks) -> dict:
    """``{target: entries per lane}``: each accumulate counts once, a loop body's once per iteration."""
    count: dict = {}
    for sink in sinks:
        if isinstance(sink, AccumWrite):
            count[sink.name] = count.get(sink.name, 0) + 1
        elif isinstance(sink, Loop):
            for body in sink.body_sinks:
                if isinstance(body, Loop):
                    raise HawkError("an ordered gradient inside nested loops is not built")
                if isinstance(body, AccumWrite):
                    count[body.name] = count.get(body.name, 0) + sink.trip
    return count


def order(sinks, held) -> tuple:
    """``sinks`` with each accumulate replaced by the stores of its log (see the module text)."""
    kept, key = held
    width = _sites(sinks)
    if not width:
        return tuple(sinks)
    for sink in sinks:
        if isinstance(sink, AccumWrite) and (sink.ttype.shape or sink.policy == "or"):
            raise HawkError(f"an ordered gradient accumulates scalars; {sink.name!r} is not one")
    used: dict = {}

    def stores(sink, slot):
        values, cells = log_names(sink.name)
        w = width[sink.name]
        at = _mk("add", (_mk("mul", (_mk("cast", (SampleIndex(TensorType((), "i64")),), literal="i32"), Const(w, _INT))), slot))
        where = _mk("add", (_mk("mul", (key if key.ttype.dtype == "i32" else _mk("cast", (key,), literal="i32"),
                                        Const(w, _INT))), slot))
        return (LogStore(values, sink.value, at, sink.name, w, kept),
                LogStore(cells, _mk("cast", (sink.index,), literal="f64"), where, sink.name, w, kept))

    out: list = []
    for sink in sinks:
        if isinstance(sink, AccumWrite):
            slot = used.get(sink.name, 0)
            used[sink.name] = slot + 1
            out.extend(stores(sink, Const(slot, _INT)))
        elif isinstance(sink, Loop):
            body: list = []
            for inner in sink.body_sinks:
                if not isinstance(inner, AccumWrite):
                    body.append(inner)
                    continue
                if abs(sink.step) != 1:
                    raise HawkError("an ordered gradient inside a loop of step other than 1 is not built")
                base = used.get(inner.name, 0)
                used[inner.name] = base + sink.trip
                turn = _mk("mul", (_mk("sub", (sink.index, Const(sink.start, sink.index.ttype))),
                                   Const(sink.step, sink.index.ttype)))
                slot = _mk("add", (Const(base, _INT), turn if turn.ttype.dtype == "i32"
                                   else _mk("cast", (turn,), literal="i32")))
                body.extend(stores(inner, slot))
            out.append(build_loop(sink.index, sink.carries, sink.inits, sink.nexts, start=sink.start,
                                  stop=sink.stop, step=sink.step, body_sinks=tuple(body),
                                  **exit_kwargs(sink)))
        else:
            out.append(sink)
    return tuple(out)
