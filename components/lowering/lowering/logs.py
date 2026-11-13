# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The log lowering of an ordered accumulate: the store of each adjoint contribution of lane ``i`` into a slot of a log plane
(``lanes * E`` elements for ``E`` contributions per lane); a stage of its own then adds each key's contributions per cell in
lane order (:func:`lowering.team.fold_spec`). The result is the same bits on every launch and for any number of samples per
key. The log is ``lanes * E`` values and ``lanes * E`` cell numbers (``int32``) of scratch; the gradient plane is stored, not
accumulated into."""

from __future__ import annotations

from raptor.ir import Const, SampleIndex, build_loop
from raptor.errors import RaptorError
from raptor.ir import make as _mk
from raptor.ir.loops import exit_kwargs
from raptor.ir.nodes import Loop
from raptor.ir.kinds import is_a
from raptor.ir.types import TensorType
from . import LogStore, make_node
from .team import log_names

_INT = TensorType((), "i32")


def OrderedAccumWrite(name, value, index, key, kept, ttype=None):
    """The lane's contribution ``value`` into cell ``index`` of the plane ``name``, applied in lane order; ``key`` is the key of the
    keyed sum it is the gradient of and ``kept`` that sum's lane axes."""
    return make_node("ordered_accum_write", ttype or value.ttype, (value, index, key), name=name, role="accum_out", kept=tuple(kept))


def is_ordered_write(node) -> bool:
    return is_a(node, "ordered_accum_write")


def _sites(sinks) -> dict:
    """``{target: entries per lane}``: each accumulate counts once, a loop body's once per iteration."""
    count: dict = {}
    for sink in sinks:
        if is_ordered_write(sink):
            count[sink.name] = count.get(sink.name, 0) + 1
        elif isinstance(sink, Loop):
            for body in sink.body_sinks:
                if is_ordered_write(body):
                    count[body.name] = count.get(body.name, 0) + sink.trip
    return count


def log_ordered(sinks) -> tuple:
    """``sinks`` with each ordered accumulate replaced by the stores of its log (see the module text)."""
    width = _sites(sinks)
    if not width:
        return tuple(sinks)
    used: dict = {}

    def stores(sink, slot):
        values, cells = log_names(sink.name)
        w = width[sink.name]
        key = sink.operands[2]
        at = _mk("add", (_mk("mul", (_mk("cast", (SampleIndex(TensorType((), "i64")),), literal="i32"), Const(w, _INT))), slot))
        where = _mk("add", (_mk("mul", (key if key.ttype.dtype == "i32" else _mk("cast", (key,), literal="i32"),
                                        Const(w, _INT))), slot))
        return (LogStore(values, sink.value, at, sink.name, w, sink.kept),
                LogStore(cells, _mk("cast", (sink.index,), literal="f64"), where, sink.name, w, sink.kept))

    out: list = []
    for sink in sinks:
        if is_ordered_write(sink):
            slot = used.get(sink.name, 0)
            used[sink.name] = slot + 1
            out.extend(stores(sink, Const(slot, _INT)))
        elif isinstance(sink, Loop):
            body: list = []
            for inner in sink.body_sinks:
                if not is_ordered_write(inner):
                    body.append(inner)
                    continue
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
