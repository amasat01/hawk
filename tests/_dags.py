# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Programmatic IR builders shared by the rows.

 has no tracing frontend (that is): every DAG below is built with the
node constructors directly, which is how tracer will build it later.
"""

from __future__ import annotations

from hawk import Quantity
from hawk.ir import AccumWrite, Assign, At, Const, Leaf, Op
from hawk.types import TensorType, Wire

F64 = TensorType((), "f64")
V3 = TensorType((3,), "f64")
QUAT = TensorType((4,), "f64", "quaternion")
IDX = TensorType((), "i32")


def state_quantity() -> Quantity:
    """The second worked example: ``sc__state__pos`` / ``sc__state__vel``."""
    return Quantity(slug="sc", name="state",
                    wires=[Wire("pos", "vec_in", (3,)), Wire("vel", "vec_in", (3,))],
                    reconstruct=lambda p, v: (p, v))


def orientation_quantity() -> Quantity:
    """The first worked example, and the two-wire compound-quantity shape: one
    ACCESS lowering to a quaternion wire plus a per-sample scalar wire."""
    return Quantity(slug="earth", name="orientation",
                    wires=[Wire("q", "vec_in", (4,), tag="quaternion"),
                           Wire("w", "per_sample", ())],
                    reconstruct=lambda q, w: (q, w))


def drag_dag(*, permuted: bool = False):
    """A sink set exercising every surface: a two-wire quantity read (one
    wire unread), a free leaf, a uniform, a shared subexpression at fan-out 2,
    a lookup ``at()`` read and a scattered accumulate.

    ``permuted=True`` builds the SAME DAG in a different construction order and
    duplicates the shared subexpression instead of sharing it — structurally
    equal, so the dedup must fold it back."""
    pos, _vel = state_quantity().read()          # vel is bound but never read
    rho = Leaf("uniform", "uniform", "rho", F64)
    table = Leaf("table_read", "lookup", "grav", V3)
    idx = Const(7, IDX)
    lane = Const(3, IDX)

    if not permuted:
        shared = Op("mul", (pos, rho), V3)
        pulled = At(table, idx, V3)
        body = Op("add", (shared, pulled), V3)
        total = Op("add", (body, shared), V3)
        return (Assign("acc", total, V3),
                AccumWrite("energy", Op("dot", (shared, shared), F64), lane, F64))

    scatter_val = Op("dot", (Op("mul", (pos, rho), V3), Op("mul", (pos, rho), V3)), F64)
    scattered = AccumWrite("energy", scatter_val, lane, F64)
    dup_shared = Op("mul", (pos, rho), V3)
    total = Op("add", (Op("add", (dup_shared, At(table, Const(7, IDX), V3)), V3),
                       Op("mul", (pos, rho), V3)), V3)
    return (Assign("acc", total, V3), scattered)
