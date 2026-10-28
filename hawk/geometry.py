# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A bounding-volume hierarchy over triangles, and the closest hit of a ray against it.

:func:`build_bvh` builds the tree on the host (NumPy, median split on the longest
axis of the centroids) into flat tables. :func:`closest_hit` is the traversal as
a kernel helper: a stack of node indices in a vector, the box test
(:func:`hawk.math.ray_aabb`) against the nearest hit so far, and the triangle test
(:func:`hawk.math.ray_tri`) on each leaf, so it runs on the host and the device
alike and nothing in it is a library call. The value is correct, not tuned.

A kernel declares the tables as planes and gets the tree's static facts from the
host object::

    bvh = build_bvh(triangles)                    # (n, 3, 3) vertices

    @hawk.kernel
    def shadow(o: Vector[3], d: Vector[3], tmax: Scalar, y: Mutable[Vector[5]],
               lo: Table[Scalar], hi: Table[Scalar], left: Table[Index],
               right: Table[Index], first: Table[Index], size: Table[Index],
               tri: Table[Scalar]):
        y = closest_hit(bvh.tables(lo, hi, left, right, first, size, tri), o, d, tmax)

``bvh.planes()`` gives the arrays to bind, under the same names. The result is
``[hit, t, u, v, leaf_position]``; ``bvh.order[leaf_position]`` is the triangle's
index in the input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import math as m
from .ir import HawkError
from .trace.helpers import call, traced

__all__ = ["BVH", "BVHTables", "PLANES", "build_bvh", "closest_hit"]

#: The planes of a tree, in the order :meth:`BVH.tables` takes them.
PLANES = ("lo", "hi", "left", "right", "first", "size", "tri")


@dataclass(frozen=True)
class BVH:
    """A built tree. Node ``i`` has bounds ``lo[i]``, ``hi[i]``; an internal node
    has children ``left[i]``, ``right[i]`` (and ``size[i] == 0``), a leaf holds the
    ``size[i]`` triangles from ``first[i]`` of ``tri`` (nine floats each, in leaf
    order). ``depth`` is the number of levels below the root, ``order[k]`` the
    input index of the triangle at leaf position ``k``."""

    lo: np.ndarray
    hi: np.ndarray
    left: np.ndarray
    right: np.ndarray
    first: np.ndarray
    size: np.ndarray
    tri: np.ndarray
    order: np.ndarray
    depth: int
    leaf: int

    @property
    def n_nodes(self) -> int:
        return int(self.left.shape[0])

    def planes(self, dtype=np.float64) -> dict:
        """The arrays a launch binds, by plane name (the vectors flattened)."""
        return {"lo": self.lo.astype(dtype).ravel(), "hi": self.hi.astype(dtype).ravel(),
                "left": self.left, "right": self.right, "first": self.first,
                "size": self.size, "tri": self.tri.astype(dtype).ravel()}

    def tables(self, *refs: Any) -> BVHTables:
        """This tree's static facts with the kernel's table parameters, in
        :data:`PLANES` order."""
        if len(refs) != len(PLANES):
            raise HawkError(f"BVH.tables: the {len(PLANES)} planes {PLANES} are wanted, "
                            f"got {len(refs)}")
        return BVHTables(self, dict(zip(PLANES, refs)))


@dataclass(frozen=True)
class BVHTables:
    """A tree's static facts and the kernel's tables that hold its planes."""

    bvh: BVH
    ref: dict


def build_bvh(tris, *, leaf: int = 4) -> BVH:
    """The tree over ``tris``, an ``(n, 3, 3)`` array of vertices, with at most
    ``leaf`` triangles in a leaf. Deterministic."""
    tris = np.asarray(tris, dtype=np.float64)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3) or tris.shape[0] < 1:
        raise HawkError(f"build_bvh: an (n, 3, 3) array of triangles is wanted, got shape {tris.shape}")
    if leaf < 1:
        raise HawkError(f"build_bvh: leaf must be at least 1, got {leaf}")
    centre = tris.mean(axis=1)
    lo_t, hi_t = tris.min(axis=1), tris.max(axis=1)
    nodes: list = []
    order: list = []

    def build(ids: np.ndarray, level: int) -> tuple:
        k = len(nodes)
        nodes.append(None)
        lo, hi = lo_t[ids].min(axis=0), hi_t[ids].max(axis=0)
        if len(ids) <= leaf:
            nodes[k] = (lo, hi, 0, 0, len(order), len(ids))
            order.extend(ids.tolist())
            return k, level
        axis = int(np.argmax(centre[ids].max(axis=0) - centre[ids].min(axis=0)))
        ranked = ids[np.argsort(centre[ids, axis], kind="stable")]
        half = len(ids) // 2
        left, dl = build(ranked[:half], level + 1)
        right, dr = build(ranked[half:], level + 1)
        nodes[k] = (lo, hi, left, right, 0, 0)
        return k, max(dl, dr)

    _, depth = build(np.arange(len(tris)), 0)
    order_a = np.asarray(order, dtype=np.int64)
    return BVH(lo=np.array([n[0] for n in nodes]), hi=np.array([n[1] for n in nodes]),
               left=np.array([n[2] for n in nodes], dtype=np.int32),
               right=np.array([n[3] for n in nodes], dtype=np.int32),
               first=np.array([n[4] for n in nodes], dtype=np.int32),
               size=np.array([n[5] for n in nodes], dtype=np.int32),
               tri=tris[order_a].reshape(-1, 9), order=order_a, depth=depth, leaf=leaf)


def closest_hit(tables: BVHTables, o: Any, d: Any, tmax: Any = None, *, depth: int = 32):
    """The nearest hit of the ray ``o + t d``, ``0 <= t <= tmax``, against the tree:
    ``[hit, t, u, v, leaf_position]``, with ``t = 0`` and zeros elsewhere when nothing
    is hit. The stack holds ``depth`` levels; a tree deeper than that is refused,
    never traversed with a truncated stack."""
    bvh, ref = tables.bvh, tables.ref
    if depth < bvh.depth:
        raise HawkError(
            f"closest_hit: the tree is {bvh.depth} levels deep, past depth={depth}; pass "
            f"depth={bvh.depth} or more (a shorter stack would drop nodes without saying)")
    zero = 0.0 * m.real(o[0])
    limit = (m.inf if tmax is None else tmax) + zero
    out = call(_traverse,
        o, d, limit, m.vec(*([0.0] * (depth + 2))), 1.0 + zero, zero, bvh.n_nodes, depth + 2,
        bvh.leaf, ref["lo"], ref["hi"], ref["left"], ref["right"], ref["first"],
        ref["size"], ref["tri"])
    best, bu, bv, where = out[0], out[1], out[2], out[3]
    found = m.where(best < limit, 1.0, 0.0)
    return m.vec(found, m.where(found > 0.5, best, 0.0), bu, bv, where)


@traced
def _traverse(o, d, best, stack, top, zero, n, slots, leaf, lo, hi, left, right, first, size, tri):
    """The stack traversal: the nearest hit ``(t, u, v, leaf position)`` found, ``best``
    unchanged when none. Every statement is the kernel body's own form (it is
    rewritten like one), so a node's work is a mask, never a branch."""
    bu = zero
    bv = zero
    where = zero
    for _step in range(n):
        if top < 0.5:
            break
        top = top - 1.0
        node = m.index(stack[m.index(top, slots)], n)
        at = 3 * node
        box = m.ray_aabb(o, d, m.vec(lo.at(at), lo.at(at + 1), lo.at(at + 2)),
                         m.vec(hi.at(at), hi.at(at + 1), hi.at(at + 2)), best)
        inside = box[0] > 0.5
        size_leaf = size.at(node)
        for j in range(leaf):
            pos = first.at(node) + j
            at9 = 9 * pos
            hit = m.ray_tri(o, d,
                            m.vec(tri.at(at9), tri.at(at9 + 1), tri.at(at9 + 2)),
                            m.vec(tri.at(at9 + 3), tri.at(at9 + 4), tri.at(at9 + 5)),
                            m.vec(tri.at(at9 + 6), tri.at(at9 + 7), tri.at(at9 + 8)), best)
            better = m.land(m.land(inside, j < size_leaf), m.land(hit[0] > 0.5, hit[1] < best))
            best = m.where(better, hit[1], best)
            bu = m.where(better, hit[2], bu)
            bv = m.where(better, hit[3], bv)
            where = m.where(better, m.real(pos), where)
        push = m.land(inside, size_leaf < 1)
        stack[m.index(top, slots)] = m.real(right.at(node))
        stack[m.index(top + 1.0, slots)] = m.real(left.at(node))
        top = top + m.where(push, 2.0, 0.0)
    return m.vec(best, bu, bv, where)
