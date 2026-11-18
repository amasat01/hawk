# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ordered gradient: ``vjp`` of a kernel whose keyed sum is ``contract="ordered"``.

The reverse of an ordered sum accumulates the gradient of each table it reads by key, a scatter-add
whose atomic order varies between launches. Here every such accumulate (at the top level or in the
body of a loop) is stated as an ``ordered_accum_write`` of the lowering wing: the lane's contribution into the
cell, applied in lane order. That is the math; how a backend keeps the order is its own lowering of the node."""

from __future__ import annotations

from typing import Any

from .. import _ir
from ..errors import HawkError
from .loops import with_body_sinks


def kept_of(primal_sinks) -> tuple | None:
    """``(kept lane axes, key)`` of the ordered keyed sum the primal stores, else ``None``."""
    # a block write commits one block (operand 0); the block's operands are payload, key, count
    blocks = (s.operands[0] for s in primal_sinks if _ir.is_a(s, "block_sum_write"))
    found = [
        (_ir.field_pairs(b, "kept"), b.operands[1])
        for b in blocks
        if b.field_text("contract") == "ordered"
    ]
    return found[0] if found else None


def _is_scatter(sink: Any) -> bool:
    """Whether ``sink`` is an accumulate (a scatter-add into a plane)."""
    return _ir.is_a(sink, "accum_write")


def order(sinks, held) -> tuple:
    """``sinks`` with each accumulate stated as an ordered accumulate of the keyed sum ``held`` (``(kept, key)``)."""
    kept, key = held
    for sink in sinks:
        if _is_scatter(sink) and (_ir.type_of(sink).shape or sink.field_text("contract") == "or"):
            raise HawkError(
                f"an ordered gradient accumulates scalars; {sink.field_text('name')!r} is not one"
            )

    def stated(sink):
        # a scatter commits its value and, when it names a row, that row as the second operand
        value, *row = sink.operands
        index = row[0] if row else None
        return _ir.make_node(
            "OrderedAccumWrite",
            _ir.type_of(value),
            [value, index, key],
            name=sink.field_text("name"),
            role="accum_out",
            kept=tuple(kept),
        )

    out: list = []
    for sink in sinks:
        if _is_scatter(sink):
            out.append(stated(sink))
        elif _ir.is_a(sink, "loop"):
            body: list = []
            for inner in sink.field_nodes("body_sinks"):
                if _ir.is_a(inner, "loop"):
                    raise HawkError("an ordered gradient inside nested loops is not built")
                if not _is_scatter(inner):
                    body.append(inner)
                    continue
                if abs(sink.field_int("step")) != 1:
                    raise HawkError(
                        "an ordered gradient inside a loop of step other than 1 is not built"
                    )
                body.append(stated(inner))
            out.append(with_body_sinks(sink, body))
        else:
            out.append(sink)
    return tuple(out)
