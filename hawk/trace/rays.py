# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The authoring surface of ray intersection, as ``hawk.math`` names it.

Each name is ONE op (:mod:`hawk.ir.rays`) returning the rank-1 result
``[hit, a, b, c]`` of the aether primitive, read with ``r[k]``. ``hit`` is 1.0 or
0.0 and has zero derivative; the other entries are the computed values also on
a miss, so a program selects between candidates by ``hit`` and differentiates
through the live one. A degenerate input (a ray parallel to the triangle, a
zero-area triangle, a zero direction component, a negative discriminant) gives
finite values and a derivative of exactly zero. There is no silhouette term: the
jump of ``t`` at a triangle edge or a sphere's rim is not modelled.

The vector arguments are length-3 values or lists of three; ``tmax`` defaults to
no limit."""

from __future__ import annotations

from typing import Any

from ..ir import Const, HawkError, Node
from ..ir import make as _make
from ..types import TensorType
from .value import Value, node_of, vec

_INF = Const(float("inf"), TensorType((), "f64"))


def _v3(name: str, x: Any) -> Node:
    if isinstance(x, (list, tuple)):
        x = vec(*x)
    node = node_of(x)
    if node.ttype.shape != (3,):
        raise HawkError(f"{name}: a length-3 vector is wanted, got {node.ttype!r}")
    return node


def _s(x: Any) -> Node:
    return _INF if x is None else node_of(x)


def ray_tri(o: Any, d: Any, a: Any, b: Any, c: Any, tmax: Any = None) -> Value:
    """Ray ``o + t d`` against the triangle ``(a, b, c)``, Moller-Trumbore, two-sided:
    ``[hit, t, u, v]`` with ``(u, v)`` the barycentric weights of ``b`` and ``c``."""
    ops = [_v3("ray_tri", x) for x in (o, d, a, b, c)]
    return Value(_make("ray_tri", (*ops, _s(tmax))))


def ray_aabb(o: Any, d: Any, lo: Any, hi: Any, tmax: Any = None) -> Value:
    """Ray against the box ``[lo, hi]``, slab method: ``[hit, t_enter, t_exit, 0]``;
    ``t_enter`` is negative when the origin is inside."""
    ops = [_v3("ray_aabb", x) for x in (o, d, lo, hi)]
    return Value(_make("ray_aabb", (*ops, _s(tmax))))


def ray_sphere(o: Any, d: Any, c: Any, r: Any, tmax: Any = None) -> Value:
    """Ray against the sphere of centre ``c`` and radius ``r``: ``[hit, t0, t1, 0]``,
    the two roots ascending; a tangent ray is a hit with ``t0 == t1``."""
    ops = [_v3("ray_sphere", x) for x in (o, d, c)]
    return Value(_make("ray_sphere", (*ops, node_of(r), _s(tmax))))
