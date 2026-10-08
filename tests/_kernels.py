# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The kernel corpus the emitter rows and the compile smoke share.

One kernel per shape the emitter must lower: a rank-0 chain, a rank-1 chain, a
compound-quantity (quaternion) read, a ``lookup`` gather, a mapreduce partial,
a scatter, a six-op chain with a fan-out-2 node (the subject) and a VJP of
the vector kernel. Traced through the real frontend, so what the rows observe
is what an author's kernel actually produces.
"""

from __future__ import annotations

from hawk import (
    Accum,
    Index,
    Mutable,
    Param,
    Quantity,
    Quat,
    Reduce,
    Scalar,
    Table,
    Vector,
    kernel,
)
from hawk.diff import vjp
from hawk.math import as_vec3, dot, exp, norm, quat_rotate, sqrt
from hawk.types import Wire


@kernel
def scale(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


@kernel
def drag(state: Vector[3], rho: Param, out: Mutable[Vector[3]]):
    out = -rho * state * norm(state)


#: The first worked example: one ACCESS lowering to a quaternion wire plus a
#: per-sample scalar wire, namespaced by the DECLARATION (``earth__orientation__q``).
ORIENTATION = Quantity(
    slug="earth", name="orientation",
    wires=[Wire("q", "vec_in", (4,), tag="quaternion"), Wire("w", "per_sample", ())],
    reconstruct=lambda q, w: (q, w),
)


@kernel
def spin(orientation: ORIENTATION, v: Vector[3], out: Mutable[Vector[3]]):
    q, w = orientation
    out = w * quat_rotate(q, v)


@kernel
def rotate_quat(orientation: ORIENTATION, out: Mutable[Quat]):
    q, w = orientation
    out = q * w


@kernel
def as_body(orientation: ORIENTATION, out: Mutable[Vector[3]]):
    q, _w = orientation
    out = as_vec3(q)


@kernel
def gather(table: Table[Scalar], where: Index, y: Mutable[Scalar]):
    y = table.at(where)


@kernel
def energy(v: Vector[3], total: Reduce("sum")):
    total.contribute(dot(v, v))


@kernel
def scatter(x: Scalar, lane: Index, acc: Accum[Scalar]):
    acc.add(x * x, at=lane)


@kernel
def chain6(state: Vector[3], k: Param, out: Mutable[Vector[3]]):
    # six ops deep, with `shared` at fan-out 2 -- the subject: the chain must
    # stay ONE right-hand side and the fan-out-2 node must be named once.
    shared = state * k
    y = shared + shared
    y = y * sqrt(k)
    y = y - state
    out = y * exp(k)


def drag_vjp():
    """The reverse-mode derivative IR of :data:`drag`."""
    return vjp(drag)


#: Every traced kernel, by name -- the corpus the rows sweep.
CORPUS = {
    "scale": scale, "drag": drag, "spin": spin, "rotate_quat": rotate_quat,
    "as_body": as_body, "gather": gather, "energy": energy, "scatter": scatter,
    "chain6": chain6,
}
