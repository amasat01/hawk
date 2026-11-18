# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Programmatic IR builders shared by the rows.

 has no tracing frontend (that is): every DAG below is built with the
node constructors directly, which is how tracer will build it later.
"""

from __future__ import annotations

from hawk import _ir

F64 = _ir.TensorType((), "f64")
V3 = _ir.TensorType((3,), "f64")
QUAT = _ir.TensorType((4,), "f64", "quaternion")
IDX = _ir.TensorType((), "i32")
Wire = _ir.Wire


def _accumulate(name, value, lane, ttype):
    """A keyed accumulate of ``value`` into ``name`` at ``lane``."""
    return _ir.make_node(
        "AccumWrite", ttype, (value, lane), name=name, contract="auto", reduced=None
    )


def state_quantity() -> _ir.Quantity:
    """The second worked example: ``sc__state__pos`` / ``sc__state__vel``."""
    return _ir.Quantity(
        slug="sc", name="state", wires=[Wire("pos", "vec_in", (3,)), Wire("vel", "vec_in", (3,))]
    )


def orientation_quantity() -> _ir.Quantity:
    """The first worked example, and the two-wire compound-quantity shape: one
    ACCESS lowering to a quaternion wire plus a per-sample scalar wire."""
    return _ir.Quantity(
        slug="earth",
        name="orientation",
        wires=[Wire("q", "vec_in", (4,), tag="quaternion"), Wire("w", "per_sample", ())],
    )


def drag_dag(context, *, permuted: bool = False):
    """A sink set exercising every surface: a two-wire quantity read (one
    wire unread), a free leaf, a uniform, a shared subexpression at fan-out 2,
    a lookup ``at()`` read and a scattered accumulate, all in ``context``.

    ``permuted=True`` builds the SAME DAG in a different construction order and
    duplicates the shared subexpression instead of sharing it — structurally
    equal, so the dedup must fold it back."""
    pos, _vel = state_quantity().read(context)  # vel is bound but never read
    rho = _ir.leaf(context, "uniform", "uniform", "rho", F64)
    table = _ir.leaf(context, "table_read", "lookup", "grav", V3)
    idx = _ir.const(context, 7, IDX)
    lane = _ir.const(context, 3, IDX)

    if not permuted:
        shared = _ir.make("mul", (pos, rho))
        pulled = _ir.make_node("At", V3, (table, idx))
        body = _ir.make("add", (shared, pulled))
        total = _ir.make("add", (body, shared))
        return (
            _ir.make_node("Assign", V3, (total,), name="acc"),
            _accumulate("energy", _ir.make("dot", (shared, shared)), lane, F64),
        )

    scatter_val = _ir.make("dot", (_ir.make("mul", (pos, rho)), _ir.make("mul", (pos, rho))))
    scattered = _accumulate("energy", scatter_val, lane, F64)
    dup_shared = _ir.make("mul", (pos, rho))
    pulled = _ir.make_node("At", V3, (table, _ir.const(context, 7, IDX)))
    total = _ir.make("add", (_ir.make("add", (dup_shared, pulled)), _ir.make("mul", (pos, rho))))
    return (_ir.make_node("Assign", V3, (total,), name="acc"), scattered)
