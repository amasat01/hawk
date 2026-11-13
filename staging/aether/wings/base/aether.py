# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The aether writer's spellings of the neutral lowering nodes (registered; the writer module names none of them)."""

from __future__ import annotations

from math import prod
from types import MappingProxyType

from .box import BOX_OWN
from raptor.protocols.binding import binding_name
from raptor.ir.kinds import is_a

def _element(r, held: str, shape: tuple) -> str:
    """Element ``hawk_lk`` (row-major) of the vector or matrix ``held``."""
    if held in r.fresh_items or held in r.items.values():
        return f"{held}.data()[hawk_lk]"
    if len(shape) == 1:
        return f"{held}.at(hawk_lk)"
    return f"{held}.at(hawk_lk / {shape[1]}, hawk_lk % {shape[1]})"


def team_store(w, r, sink, target: str, value: str) -> None:
    """A value a team stage works on, stored row-major into the lane's own slice of its scratch
    plane (:class:`~hawk.dialects.simt.TeamStore`): one loop over the elements."""
    shape = sink.value.ttype.shape
    size = prod(shape)
    base = f"static_cast<std::size_t>({r._ref(sink.index)}) * {size}"
    if not shape:
        r._emit(f"{target}.data()[{base}] = {value};")
        return
    held = w._operand(r, sink.value, value)
    if is_a(sink, "team_copy"):      # a block per lane: its threads copy the slice together (a team of one on the host)
        r._emit("{ const aether::linalg::team::Team hawk_tm = aether::linalg::team::Team::here(); "
                f"for (Int hawk_lk = hawk_tm.rank; hawk_lk < {size}; hawk_lk += hawk_tm.size) "
                f"{target}.data()[{base} + hawk_lk] = {_element(r, held, shape)}; }}")
        return
    r._emit(f"for (Int hawk_lk = 0; hawk_lk < {size}; ++hawk_lk) "
            f"{target}.data()[{base} + hawk_lk] = {_element(r, held, shape)};")


def team_read(w, r, node) -> str:
    """A vector or matrix a team stage left in the lane's own slice of its scratch plane, copied
    row-major into a fresh ``Item`` by a loop ahead of the line that uses it."""
    plane = binding_name(node.plane.role, node.plane.name)
    shape = node.ttype.shape
    size = prod(shape)
    out = w._fresh_item(r, node.ttype)
    base = f"static_cast<std::size_t>({r._ref(node.index)}) * {size}"
    r._emit(f"for (Int hawk_lk = 0; hawk_lk < {size}; ++hawk_lk) "
            f"{out}.data()[hawk_lk] = {plane}.data()[{base} + hawk_lk];")
    return out


def box_store(w, r, target: str, lane: str, value: str) -> None:
    """A thread box lane's own-column store, only for a lane the thread
    owns (:mod:`aether.wings.base.box`)."""
    r._emit(f"if ({lane} >= {BOX_OWN}) {target}[{w._sample_index(lane)}] = {w._packed(None, value)};")


#: The writer's spellings by (dialect, name): ``fn(w, r, *args)``, a read-only table written after the functions it names. The front
#: end reads them through the manifest's spellings arm.
SPELLINGS = MappingProxyType({
    ("lowering", "team_store"): team_store,
    ("lowering", "team_read"): team_read,
    ("lowering", "box_store"): box_store,
})
