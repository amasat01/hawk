# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The host finish of a kernel's cells: rows to the cell, in eagle's ``ordered`` tree.

A kernel that stores ``hawk.sum(x)`` (or ``hawk.mean``, ``hawk.max``,
``hawk.min``) writes one row per ``cell_row`` samples (the sidecar's number, from the variant). :func:`hawk.runtime.run` owns that
row scratch (made once per kernel and sample count), launches, and finishes the
rows into the cell the caller bound. The finish replays the ``ordered`` tree
of ``eagle/reduce/canonical.h`` operation for operation (leaves of 256 slots of
16 terms, the in-leaf halving tree, then pairwise levels; a sum's partial is a
compensated (hi, comp) pair), so the cell is bitwise the one eagle's
``ordered`` contract gives.
"""

from __future__ import annotations

from raptor.ir.cells import planes

import numpy as np

from lowering.cells import SAMPLE_SUM_COMP

from ._errors import HawkError

SLOTS, TERMS = 256, 16
LEAF = SLOTS * TERMS


def comp_name(name: str) -> str:
    return f"{name}{SAMPLE_SUM_COMP}"


def _finite(x):
    with np.errstate(invalid="ignore"):
        return x - x == 0


def _two_sum_err(a, b, s):
    """aether's ``twoSumErr``: the exact residual of ``s == a + b``, the larger magnitude first, a zero residual ``+0``."""
    big_a = np.abs(a) >= np.abs(b)
    big, small = np.where(big_a, a, b), np.where(big_a, b, a)
    with np.errstate(invalid="ignore"):
        return np.where(_finite(s), (small - (s - big)) + 0.0, np.nan)


def _neumaier(h, c, x, xc):
    s = h + x
    c = np.where(_finite(s), c + _two_sum_err(h, x, s), 0)
    return s, np.where(_finite(s), c + xc, 0)


def _pair(ah, ac, bh, bc):
    s = ah + bh
    return s, np.where(_finite(s), (ac + bc) + _two_sum_err(ah, bh, s), 0)


def _max(ah, ac, bh, bc):
    return np.where(ah > bh, ah, bh), ac


def _min(ah, ac, bh, bc):
    return np.where(ah < bh, ah, bh), ac


def _leaves(h, c, add, join):
    """The partial of each leaf of ``h``, ``c`` (shape ``(leaves, cnt)``, one
    row per leaf, ``cnt <= LEAF``): the slot walk, then the halving tree."""
    cnt = h.shape[1]
    live = min(cnt, SLOTS)
    sh, sc = h[:, :live].copy(), c[:, :live].copy()
    sc = np.where(_finite(sh), sc, 0)
    for k in range(1, TERMS):
        m = min(live, cnt - k * SLOTS)
        if m <= 0:
            break
        lo = k * SLOTS
        sh[:, :m], sc[:, :m] = add(sh[:, :m], sc[:, :m], h[:, lo:lo + m], c[:, lo:lo + m])
    off = SLOTS // 2
    while off > 0:
        lim = min(off, live - off) if live > off else 0
        if lim:
            sh[:, :lim], sc[:, :lim] = join(sh[:, :lim], sc[:, :lim],
                                            sh[:, off:off + lim], sc[:, off:off + lim])
        off >>= 1
    return sh[:, 0], sc[:, 0]


def finish(hi, comp, op: str):
    """The cell of the rows ``hi`` (and ``comp``, for a sum or mean) under
    ``op``: a numpy scalar of the rows' dtype."""
    hi = np.ascontiguousarray(hi)
    c = np.zeros_like(hi) if comp is None or op in ("max", "min") else np.ascontiguousarray(comp)
    add, join = {"sum": (_neumaier, _pair), "mean": (_neumaier, _pair),
                 "max": (_max, _max), "min": (_min, _min)}[op]
    n = hi.shape[0]
    full = n // LEAF
    parts = []
    if full:
        parts.append(_leaves(hi[:full * LEAF].reshape(full, LEAF),
                             c[:full * LEAF].reshape(full, LEAF), add, join))
    if n > full * LEAF:
        parts.append(_leaves(hi[None, full * LEAF:], c[None, full * LEAF:], add, join))
    vh = np.concatenate([p[0] for p in parts])
    vc = np.concatenate([p[1] for p in parts])
    while vh.shape[0] > 1:
        m, pairs = vh.shape[0], vh.shape[0] // 2
        nh, nc = join(vh[0:2 * pairs:2], vc[0:2 * pairs:2], vh[1:2 * pairs:2], vc[1:2 * pairs:2])
        if m & 1:
            nh, nc = np.append(nh, vh[-1]), np.append(nc, vc[-1])
        vh, vc = nh, nc
    if op in ("sum", "mean"):
        return np.where(_finite(vh[0]), vh[0] + vc[0], vh[0])
    return vh[0]


def row_of(sidecar: dict) -> int:
    """The samples per row of the unit ``sidecar`` describes (``cell_row``); a sidecar without it is refused by name."""
    if "cell_row" not in sidecar:
        raise HawkError("the sidecar states no cell_row (a unit built before the row width became a variant fact): rebuild the unit")
    return int(sidecar["cell_row"])


class Scratch:
    """The rows of one kernel at one sample count: ``rows[plane]`` for every
    cell plane and its ``_comp`` companion."""

    def __init__(self, cells: dict, n: int, dtype, row: int) -> None:
        count = -(-int(n) // int(row))
        self.cells = cells
        self.rows = {}
        for cell, spec in cells.items():
            for plane in planes(cell, spec["columns"]):
                self.rows[plane] = np.zeros(count, dtype=dtype)
                if spec["op"] in ("sum", "mean"):
                    self.rows[comp_name(plane)] = np.zeros(count, dtype=dtype)

    def finish(self, outputs: dict, lo: int = 0, hi: int | None = None) -> None:
        """Fold rows ``[lo, hi)`` of every cell into ``outputs[cell]``."""
        for cell, spec in self.cells.items():
            op = spec["op"]
            out = np.asarray(outputs[cell]).reshape(-1)
            for k, plane in enumerate(planes(cell, spec["columns"])):
                comp = self.rows.get(comp_name(plane))
                out[k] = finish(self.rows[plane][lo:hi], None if comp is None else comp[lo:hi], op)
