# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The wide-value corpus: the per-lane shapes whose values are wider than a thread's registers.

Each entry is a kernel, the planes it runs on and a NumPy reference. The data are small integers
and halves, so every sum is exact in float32 and float64: any spelling of the same arithmetic
(FMA or not, any order) returns the reference's bits, and a difference is a wrong answer.

* ``fill_mask``       a 512-wide value written at a runtime index in a loop, then masked;
* ``matvec_in``       a 24 x 512 matrix-vector product, the vector a plane;
* ``matvec_computed`` the same product over a value the kernel computed;
* ``masked_sum``      a 384-term masked, dispatched sum;
* ``wide51``          a 51-wide output written at a runtime index under a condition;
* ``index_set``       a gather-heavy kernel on a caller-built lane set (the integer chain is 32-bit).

No ``from __future__ import annotations``: the plane annotations are read when the kernel is
defined."""

from dataclasses import dataclass

import os
import sys

import numpy as np

import hawk
import hawk.math as m
from hawk import Index, Mutable, Scalar, Table, Terminated, Vector
from hawk.ext import Guard, IndexSet, Kind

W = 512                         # the wide value
ROWS = 24                       # rows of the matrix
TERMS = 384                     # terms of the masked sum
OUT = 51                        # width of the wide output
TABLE = 64                      # entries of the index-set kernel's tables


def zeros(n: int):
    """A ``Vector[n]`` of zeros."""
    return m.vec([0.0 for _ in range(n)])


def mask_vec(tbl, n: int):
    """``tbl`` as a ``Vector[n]``."""
    return m.vec([tbl.at(i) for i in range(n)])


def row_vec(tbl, r, n: int):
    """Row ``r`` of the matrix ``tbl`` as a ``Vector[n]``."""
    return m.vec([tbl.at(o=r, i=i) for i in range(n)])


@hawk.kernel
def fill_mask(a: Scalar, swg: Table["i":W], bf: Mutable[Vector[W]]):
    r = zeros(W)
    for j in range(W):
        r[j] = a * (j + 1.5)
    bf = r * mask_vec(swg, W)


@hawk.kernel
def matvec_in(x: Vector[W], w: Table["o":ROWS, "i":W], y: Mutable[Vector[ROWS]]):
    acc = zeros(ROWS)
    for r in range(ROWS):
        acc[r] = m.sum(x * row_vec(w, r, W))
    y = acc


@hawk.kernel
def matvec_computed(a: Scalar, swg: Table["i":W], w: Table["o":ROWS, "i":W], y: Mutable[Vector[ROWS]]):
    bf = zeros(W)
    for j in range(W):
        bf[j] = a * (j + 1.5)
    bf = bf * mask_vec(swg, W)
    acc = zeros(ROWS)
    for r in range(ROWS):
        acc[r] = m.sum(bf * row_vec(w, r, W))
    y = acc


@hawk.kernel
def masked_sum(f1: Scalar, f2: Scalar, f3: Scalar, gf: Vector[TERMS], sfx: Table["j":TERMS],
               beta: Table["j":TERMS], y: Mutable[Scalar]):
    total = 0.0
    for j in range(TERMS):
        mi = m.int32(sfx.at(j))
        g = gf[j]
        b = beta.at(j)
        total = total + m.dispatch(mi, [0.0, b * g * f1, b * g * f2, b * g * f3], policy="predicated")
    y = total


@hawk.kernel
def wide51(a: Scalar, c: Vector[OUT], y: Mutable[Vector[OUT]]):
    r = zeros(OUT)
    for j in range(OUT):
        if c[j] > 0.5:
            r[j] = a * (j + 1.0)
    y = r



def doubled(x, n: int):
    """``x`` doubled, as a value built from parts."""
    return m.vec([x[j] * 2.0 for j in range(n)])


# -- the narrow shapes: a vector of a handful of elements, written in a loop and read at a runtime index ---------

def _narrow_body(a, s, swg, n: int):
    r = zeros(n)
    for j in range(n):
        r[j] = a * (j + 1.5)
    k = m.index(s, n)
    q = zeros(n)
    for j in range(n):
        q[j] = r[(j + k) % n]
    return q * mask_vec(swg, n)


@hawk.kernel
def narrow4(a: Scalar, s: Scalar, swg: Table["i":4], y: Mutable[Vector[4]]):
    y = _narrow_body(a, s, swg, 4)


@hawk.kernel
def narrow8(a: Scalar, s: Scalar, swg: Table["i":8], y: Mutable[Vector[8]]):
    y = _narrow_body(a, s, swg, 8)


@hawk.kernel
def narrow16(a: Scalar, s: Scalar, swg: Table["i":16], y: Mutable[Vector[16]]):
    y = _narrow_body(a, s, swg, 16)


@hawk.kernel
def rotate16(x: Vector[16], s: Scalar, y: Mutable[Vector[16]]):
    """A width-16 value rotated in place by a run-time count: element ``j`` takes its own and element ``j + s``'s sum, in order."""
    k = m.index(s, 16)
    q = doubled(x, 16)
    for j in range(16):
        q[j] = q[j] + q[(j + k) % 16]
    y = q


# -- the same shapes as a stepping kernel: ``hawk.steps`` runs ``K`` steps of the body per launch, a sample finishing at step 8 ---

STEPS_TO_FINISH = 8


def steps_fill_body(a: Scalar, swg: Table["i":W], terminated: Terminated, c: Mutable[Scalar], bf: Mutable[Vector[W]]):
    r = zeros(W)
    for j in range(W):
        r[j] = a * (j + 1.5)
    bf = r * mask_vec(swg, W) + c
    cn = c + 1.0
    c = cn
    terminated = cn >= 8.0


def steps_early_body(a: Scalar, lim: Scalar, swg: Table["i":W], terminated: Terminated, c: Mutable[Scalar],
                     bf: Mutable[Vector[W]]):
    """Lanes finish at different steps of one launch: each must leave the plane the step it finished on wrote."""
    r = zeros(W)
    for j in range(W):
        r[j] = a * (j + 1.5)
    bf = r * mask_vec(swg, W) + c
    cn = c + 1.0
    c = cn
    terminated = cn >= lim


def steps_matvec_body(x: Vector[W], w: Table["o":ROWS, "i":W], terminated: Terminated, c: Mutable[Scalar],
                      y: Mutable[Vector[ROWS]]):
    acc = zeros(ROWS)
    for r in range(ROWS):
        acc[r] = m.sum(x * row_vec(w, r, W))
    y = acc + c
    cn = c + 1.0
    c = cn
    terminated = cn >= 8.0


steps_fill_k1 = hawk.kernel(steps_fill_body, steps=1)
steps_fill_k8 = hawk.kernel(steps_fill_body, steps=8)
steps_early_k8 = hawk.kernel(steps_early_body, steps=8)
steps_matvec_k1 = hawk.kernel(steps_matvec_body, steps=1)
steps_matvec_k8 = hawk.kernel(steps_matvec_body, steps=8)



# -- the derivatives (vjp, jvp) of the same shapes ---------------------------------------------------------------
# A derivative kernel is built beside its primal's unit in a bundle: here alone, naming a primal unit that is not built.

PRIMAL_UNIT = "0" * 64


def _derived(name, primal, transform, wrt):
    from hawk.trace.kernel import Kernel

    return Kernel(name, transform(primal, wrt=wrt, primal_unit=PRIMAL_UNIT))


def fill_mask_vjp_kernel():
    from hawk.diff import vjp

    return _derived("fill_mask_vjp", fill_mask, vjp, ("a",))


def fill_mask_jvp_kernel():
    from hawk.diff import jvp

    return _derived("fill_mask_jvp", fill_mask, jvp, ("a",))


def matvec_in_vjp_kernel():
    from hawk.diff import vjp

    return _derived("matvec_in_vjp", matvec_in, vjp, ("x",))


def matvec_in_jvp_kernel():
    from hawk.diff import jvp

    return _derived("matvec_in_jvp", matvec_in, jvp, ("x",))


fill_mask_vjp = fill_mask_vjp_kernel()
fill_mask_jvp = fill_mask_jvp_kernel()
matvec_in_vjp = matvec_in_vjp_kernel()
matvec_in_jvp = matvec_in_jvp_kernel()



def steps_acc_body(a: Scalar, swg: Table["i":W], terminated: Terminated, c: Mutable[Scalar], bf: Mutable[Vector[W]]):
    """A step that adds to a wide state plane it reads."""
    r = zeros(W)
    for j in range(W):
        r[j] = a * (j + 1.5)
    bf = bf + r * mask_vec(swg, W)
    cn = c + 1.0
    c = cn
    terminated = cn >= 8.0


steps_acc_k1 = hawk.kernel(steps_acc_body, steps=1)


def steps_acc_jvp_k8_kernel():
    """The tangent of ``steps_acc_body`` with respect to ``a``, carried: ``hawk.steps(step_jvp(step), 8)``."""
    from hawk.diff import step_jvp

    return hawk.steps(step_jvp(steps_acc_k1, wrt=("a",)), 8)


steps_acc_jvp_k8 = steps_acc_jvp_k8_kernel()


# -- compaction groups and segmented dispatches --------------------------------------------------------------------
# A grouped kernel runs over an active map of groups of GROUP lanes (a lane's slot is its sample's, wherever the map sends it);
# a segmented one is split by hawk into one kernel per branch (here a selector the step rewrites, so each unit guards its
# commit; no lane changes branch, so no sweep runs), launched over runs of samples sorted by branch. The segmented rows are written ``policy='auto'`` and driven by
# ``eagle.simulate(..., dispatch='segmented')``; the build checks (A1, text) read the units that run builds.

GROUP = 32
GROUP_KIND = Kind("wide_group", guard=Guard(active_set=True))
BRANCHES = 3
SEG_STEPS = 8.0         # a segmented row runs this many steps, so the launches of its units outweigh the run's set-up


def fill_mask_grp(a: Scalar, swg: Table["i":W], terminated: Terminated, bf: Mutable[Vector[W]]):
    r = zeros(W)
    for j in range(W):
        r[j] = a * (j + 1.5)
    bf = r * mask_vec(swg, W)


def matvec_in_grp(x: Vector[W], w: Table["o":ROWS, "i":W], terminated: Terminated, y: Mutable[Vector[ROWS]]):
    acc = zeros(ROWS)
    for r in range(ROWS):
        acc[r] = m.sum(x * row_vec(w, r, W))
    y = acc


def fill_mask_seg(k: Mutable[Index], a: Scalar, swg: Table["i":W], terminated: Terminated, c: Mutable[Scalar],
                  bf: Mutable[Vector[W]]):
    r = zeros(W)
    for j in range(W):
        r[j] = a * (j + 1.5)
    bf = r * mask_vec(swg, W) + m.dispatch(k, [0.0, 1.0, 2.0], policy="auto")
    k = m.select(k > 99, k - 1, k)
    cn = c + 1.0
    c = cn
    terminated = cn >= SEG_STEPS


def matvec_in_seg(k: Mutable[Index], x: Vector[W], w: Table["o":ROWS, "i":W], terminated: Terminated, c: Mutable[Scalar],
                  y: Mutable[Vector[ROWS]]):
    acc = zeros(ROWS)
    for r in range(ROWS):
        acc[r] = m.sum(x * row_vec(w, r, W))
    y = acc + m.dispatch(k, [0.0, 1.0, 2.0], policy="auto")
    k = m.select(k > 99, k - 1, k)
    cn = c + 1.0
    c = cn
    terminated = cn >= SEG_STEPS


fill_mask_grp = hawk.kernel(fill_mask_grp, kind=GROUP_KIND, group=GROUP)
matvec_in_grp = hawk.kernel(matvec_in_grp, kind=GROUP_KIND, group=GROUP)
fill_mask_seg = hawk.kernel(fill_mask_seg)
matvec_in_seg = hawk.kernel(matvec_in_seg)


def segmented_units(kernel):
    """``(the kernel with its one dispatch group segmented, the names of its branch units)``."""
    seg = hawk.resolve_dispatch(kernel, {"dispatch0": "segmented"})
    return seg, tuple(f"{seg.name}__k{j}" for j in range(BRANCHES))


LANE_SET = Kind("wide_lane_set", lanes=IndexSet(width=32))


def _live_offsets(seg, perm, kinds, k: int):
    """``k`` integers derived from the lane and the tables, every one needed in the loop that follows: the chain of
    integer indices a lane set's gather-heavy kernel keeps live."""
    out = []
    row, col = m.split_index(TABLE)
    at = m.index(seg.at(0), TABLE) + row
    for j in range(k):
        at = m.index(perm.at(at % TABLE), TABLE) * (j + 3) + col
        out = out + [at + m.index(kinds.at(at % TABLE), 8) * 5]
    return out


def index_set(lane: hawk.Index, seg: Table[hawk.Scalar], perm: Table[hawk.Scalar], kinds: Table[hawk.Scalar],
              th: Table[hawk.Scalar], y: Mutable[Vector[4]]):
    o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15, o16, o17, o18, o19, o20, o21, o22, o23, o24, o25, o26, o27, o28, o29, o30, o31, o32, o33, o34, o35, o36, o37, o38, o39 = _live_offsets(
        seg, perm, kinds, 40)
    acc = m.vec(0.0, 0.0, 0.0, 0.0)
    for t in range(96):
        v = (th.at((o0 + t) % TABLE) + th.at((o1 + t) % TABLE) + th.at((o2 + t) % TABLE) + th.at((o3 + t) % TABLE) +
             th.at((o4 + t) % TABLE) + th.at((o5 + t) % TABLE) + th.at((o6 + t) % TABLE) + th.at((o7 + t) % TABLE) +
             th.at((o8 + t) % TABLE) + th.at((o9 + t) % TABLE) + th.at((o10 + t) % TABLE) + th.at((o11 + t) % TABLE) +
             th.at((o12 + t) % TABLE) + th.at((o13 + t) % TABLE) + th.at((o14 + t) % TABLE) + th.at((o15 + t) % TABLE) +
             th.at((o16 + t) % TABLE) + th.at((o17 + t) % TABLE) + th.at((o18 + t) % TABLE) + th.at((o19 + t) % TABLE) +
             th.at((o20 + t) % TABLE) + th.at((o21 + t) % TABLE) + th.at((o22 + t) % TABLE) + th.at((o23 + t) % TABLE) +
             th.at((o24 + t) % TABLE) + th.at((o25 + t) % TABLE) + th.at((o26 + t) % TABLE) + th.at((o27 + t) % TABLE) +
             th.at((o28 + t) % TABLE) + th.at((o29 + t) % TABLE) + th.at((o30 + t) % TABLE) + th.at((o31 + t) % TABLE) +
             th.at((o32 + t) % TABLE) + th.at((o33 + t) % TABLE) + th.at((o34 + t) % TABLE) + th.at((o35 + t) % TABLE) +
             th.at((o36 + t) % TABLE) + th.at((o37 + t) % TABLE) + th.at((o38 + t) % TABLE) + th.at((o39 + t) % TABLE))
        acc = m.vec(acc[0] + v, acc[1] + v * 2.0, acc[2] + th.at((o0 + t * 3) % TABLE), acc[3] - v)
    y = acc


def lane_set_kernel(kind):
    """The gather-heavy kernel on a lane set of ``kind``."""
    return hawk.kernel(index_set, kind=kind)


index_set_kernel = lane_set_kernel(LANE_SET)


#: ``{(entry, scalar mode): local bytes}`` the register form may keep: the rows where the probe measured it faster than the
#: resident form (USER, 10-29: speed is the proof; a small spill is healthy where it is faster). Everything else has none.
LOCAL_OK = {("narrow4", "float64"): 32,
            ("matvec_in", "float64"): 192,     # a warp per lane: the 24 row sums are indexed by the row loop, below the width the pass places
            ("matvec_in_jvp", "float32"): 96, ("matvec_in_jvp", "float64"): 192,
            ("steps_matvec_k1", "float64"): 192, ("steps_matvec_k8", "float64"): 192, ("matvec_in_seg", "float64"): 192}     # (the same, for the tangent's 24 sums)


@dataclass(frozen=True)
class Entry:
    """One corpus kernel: ``planes(rng, dtype, n, exact)`` its inputs (``{name: ndarray}``: a plane of width ``w`` shaped
    ``(w, n)``, a table 1-D or 2-D), ``out`` the output plane's name and ``shape(n)`` its shape, ``want(dtype, planes, n)``
    its reference, and ``kind`` the build kind of a kernel on a lane set."""

    kernel: object
    out: str
    shape: object
    planes: object
    want: object
    kind: object = None
    segmented: bool = False        # run through ``eagle.simulate(..., dispatch='segmented')``; built as its branch units
    modes: tuple = ("float32", "float64")      # the scalar modes it runs in (the dispatch door builds the units in float64 alone)


def _ints(rng, shape, hi, dtype):
    return rng.integers(0, hi, size=shape).astype(dtype)


def _vals(rng, shape, hi, dtype, exact):
    """A data plane: small integers (every sum exact), or, for a twin run that compares one build with another, reals
    in ``[0, hi)`` (a sum then rounds, and the two builds must round alike)."""
    return _ints(rng, shape, hi, dtype) if exact else (rng.random(size=shape) * hi).astype(dtype)


def _fill_mask_planes(rng, dtype, n, exact=True):
    return {"a": _vals(rng, (n,), 5, dtype, exact), "swg": _ints(rng, (W,), 2, dtype)}


def _fill_mask_want(dtype, p, n):
    r = p["a"][None, :] * (np.arange(W, dtype=dtype)[:, None] + dtype(1.5))
    return (r * p["swg"][:, None]).astype(dtype)


def _matvec_in_planes(rng, dtype, n, exact=True):
    return {"x": _vals(rng, (W, n), 8, dtype, exact), "w": _vals(rng, (ROWS, W), 3, dtype, exact)}


def _matvec_in_want(dtype, p, n):
    return (p["w"].astype(np.float64) @ p["x"].astype(np.float64)).astype(dtype)


def _matvec_computed_planes(rng, dtype, n, exact=True):
    return {**_fill_mask_planes(rng, dtype, n, exact), "w": _vals(rng, (ROWS, W), 3, dtype, exact)}


def _matvec_computed_want(dtype, p, n):
    return (p["w"].astype(np.float64) @ _fill_mask_want(np.float64, p, n)).astype(dtype)


def _masked_sum_planes(rng, dtype, n, exact=True):
    return {"f1": _vals(rng, (n,), 4, dtype, exact), "f2": _vals(rng, (n,), 4, dtype, exact),
            "f3": _vals(rng, (n,), 4, dtype, exact), "gf": _vals(rng, (TERMS, n), 4, dtype, exact),
            "sfx": _ints(rng, (TERMS,), 4, dtype), "beta": _vals(rng, (TERMS,), 4, dtype, exact)}


def _masked_sum_want(dtype, p, n):
    pick = p["sfx"].astype(int)
    total = np.zeros(p["f1"].shape, dtype=np.float64)
    for j in range(TERMS):
        f = {1: p["f1"], 2: p["f2"], 3: p["f3"]}.get(pick[j])
        if f is not None:
            total += p["beta"][j].astype(np.float64) * p["gf"][j] * f
    return total.astype(dtype)


def _wide51_planes(rng, dtype, n, exact=True):
    return {"a": _vals(rng, (n,), 5, dtype, exact), "c": _ints(rng, (OUT, n), 2, dtype)}


def _wide51_want(dtype, p, n):
    r = p["a"][None, :] * (np.arange(OUT, dtype=dtype)[:, None] + dtype(1.0))
    return np.where(p["c"] > 0.5, r, dtype(0.0)).astype(dtype)



def _narrow_planes(n):
    def planes(rng, dtype, count, exact=True):
        return {"a": _vals(rng, (count,), 5, dtype, exact), "s": _ints(rng, (count,), n, dtype), "swg": _ints(rng, (n,), 2, dtype)}
    return planes


def _narrow_want(n):
    def want(dtype, p, count):
        r = p["a"][None, :] * (np.arange(n, dtype=dtype)[:, None] + dtype(1.5))
        k = p["s"].astype(int)
        q = np.stack([r[(j + k) % n, np.arange(count)] for j in range(n)])
        return (q * p["swg"][:, None]).astype(dtype)
    return want


def _rotate16_planes(rng, dtype, n, exact=True):
    return {"x": _vals(rng, (16, n), 8, dtype, exact), "s": _ints(rng, (n,), 16, dtype)}


def _rotate16_want(dtype, p, n):
    k, lanes = p["s"].astype(int), np.arange(n)
    q = (p["x"] * dtype(2.0)).astype(dtype)
    for j in range(16):
        q[j] = q[j] + q[(j + k) % 16, lanes]
    return q.astype(dtype)


def _stepping(planes):
    """``planes`` with the state a stepping kernel needs: its step counter, finish mask and finished count."""
    def make(rng, dtype, n, exact=True):
        return {**planes(rng, dtype, n, exact), "c": np.zeros(n, dtype), "terminated": np.zeros(n, bool),
                "finished_count": np.zeros(1, np.uint32)}
    return make


def _steps_fill_want(steps):
    def want(dtype, p, n):
        r = p["a"][None, :] * (np.arange(W, dtype=dtype)[:, None] + dtype(1.5))
        return (r * p["swg"][:, None] + dtype(steps - 1)).astype(dtype)
    return want


def _early_planes(rng, dtype, n, exact=True):
    return {**_fill_mask_planes(rng, dtype, n, exact), "lim": _ints(rng, (n,), 11, dtype)}


def _early_want(dtype, p, n):
    stopped = np.clip(p["lim"], 1, STEPS_TO_FINISH)             # a lane finishes at step ``lim`` (at the first, if it asks for none)
    r = p["a"][None, :] * (np.arange(W, dtype=dtype)[:, None] + dtype(1.5))
    return (r * p["swg"][:, None] + (stopped[None, :] - dtype(1))).astype(dtype)


def _steps_matvec_want(steps):
    def want(dtype, p, n):
        return (p["w"].astype(np.float64) @ p["x"].astype(np.float64) + (steps - 1)).astype(dtype)
    return want



def _fill_mask_vjp_planes(rng, dtype, n, exact=True):
    return {"swg": _ints(rng, (W,), 2, dtype), "bar_bf": _vals(rng, (W, n), 3, dtype, exact)}


def _fill_mask_vjp_want(dtype, p, n):
    scale = (np.arange(W, dtype=np.float64) + 1.5) * p["swg"].astype(np.float64)
    return (scale[:, None] * p["bar_bf"].astype(np.float64)).sum(axis=0).astype(dtype)


def _fill_mask_jvp_planes(rng, dtype, n, exact=True):
    return {"swg": _ints(rng, (W,), 2, dtype), "a": _vals(rng, (n,), 5, dtype, exact), "dot_a": _vals(rng, (n,), 4, dtype, exact)}


def _fill_mask_jvp_want(dtype, p, n):
    scale = (np.arange(W, dtype=dtype) + dtype(1.5)) * p["swg"]
    return (scale[:, None] * p["dot_a"][None, :]).astype(dtype)


def _matvec_in_vjp_planes(rng, dtype, n, exact=True):
    return {"w": _vals(rng, (ROWS, W), 3, dtype, exact), "bar_y": _vals(rng, (ROWS, n), 4, dtype, exact)}


def _matvec_in_vjp_want(dtype, p, n):
    return (p["w"].astype(np.float64).T @ p["bar_y"].astype(np.float64)).astype(dtype)


def _matvec_in_jvp_planes(rng, dtype, n, exact=True):
    return {"w": _vals(rng, (ROWS, W), 3, dtype, exact), "x": _vals(rng, (W, n), 8, dtype, exact),
            "dot_x": _vals(rng, (W, n), 4, dtype, exact)}


def _matvec_in_jvp_want(dtype, p, n):
    return (p["w"].astype(np.float64) @ p["dot_x"].astype(np.float64)).astype(dtype)



def _steps_acc_jvp_planes(rng, dtype, n, exact=True):
    return {"a": _vals(rng, (n,), 5, dtype, exact), "dot_a": _vals(rng, (n,), 4, dtype, exact), "swg": _ints(rng, (W,), 2, dtype),
            "bf": np.zeros((W, n), dtype), "c": np.zeros(n, dtype), "dot_c": np.zeros(n, dtype),
            "terminated": np.zeros(n, bool), "finished_count": np.zeros(1, np.uint32)}


def _steps_acc_jvp_want(dtype, p, n):
    scale = (np.arange(W, dtype=dtype) + dtype(1.5)) * p["swg"]
    return (STEPS_TO_FINISH * scale[:, None] * p["dot_a"][None, :]).astype(dtype)


def _index_set_planes(rng, dtype, n, exact=True):
    return {"seg": _ints(rng, (1,), TABLE, dtype), "perm": _ints(rng, (TABLE,), TABLE, dtype),
            "kinds": _ints(rng, (TABLE,), 4, dtype), "th": _vals(rng, (TABLE,), 5, dtype, exact),
            "active_map": np.arange(0, n, 3, dtype=np.int32),
            "active_count": np.array([len(range(0, n, 3))], dtype=np.uint32)}


def _index_set_want(dtype, p, n):
    """``(4, n)``: the kernel on each lane of the map, zero elsewhere."""
    seg, perm, kinds, th = (p[k].astype(np.int64) for k in ("seg", "perm", "kinds", "th"))
    lanes = p["active_map"].astype(np.int64)
    at, offs = int(seg[0]) + lanes // TABLE, []
    for j in range(40):
        at = perm[at % TABLE] * (j + 3) + lanes % TABLE
        offs.append(at + kinds[at % TABLE] * 5)
    acc = np.zeros((4, len(lanes)), dtype=np.float64)
    for t in range(96):
        v = sum(th[(o + t) % TABLE].astype(np.float64) for o in offs)
        acc[0] += v
        acc[1] += v * 2.0
        acc[2] += th[(offs[0] + t * 3) % TABLE]
        acc[3] -= v
    out = np.zeros((4, n), dtype=np.float64)
    out[:, lanes] = acc
    return out.astype(dtype)


def _grouped(planes, want):
    """``planes`` and ``want`` for a kernel on the groups of GROUP lanes whose number is not 1 mod 3 (``terminated`` marks the
    others; the run compacts them into the active map, :func:`run_device`); the lanes of the other groups keep the zeros of the
    output plane."""
    def make(rng, dtype, n, exact=True):
        dead = np.repeat(np.arange(n // GROUP) % 3 == 1, GROUP)
        return {**planes(rng, dtype, n, exact), "terminated": dead}

    def reference(dtype, p, n):
        out = want(dtype, p, n)
        dead = np.repeat(np.arange(n // GROUP) % 3 == 1, GROUP)
        return np.where(dead, dtype(0), out).astype(dtype)
    return make, reference


def _selected(planes, want):
    """``planes`` and ``want`` for a segmented row: ``k`` the branch of each lane (0..2), ``c`` its step count; each step puts
    ``k`` into the output, and ``k`` is rewritten (to itself, in effect: that makes the selector state-dependent, so each unit guards its commit, with no lane changing branch)."""
    def make(rng, dtype, n, exact=True):
        return {"k": rng.integers(0, BRANCHES, size=(n,)).astype(np.int64), **planes(rng, dtype, n, exact), "c": np.zeros(n, dtype)}

    def reference(dtype, p, n):
        return (want(dtype, p, n) + p["k"].astype(dtype)[None, :]).astype(dtype)
    return make, reference


CORPUS = {
    "fill_mask": Entry(fill_mask, "bf", lambda n: (W, n), _fill_mask_planes, _fill_mask_want),
    "matvec_in": Entry(matvec_in, "y", lambda n: (ROWS, n), _matvec_in_planes, _matvec_in_want),
    "matvec_computed": Entry(matvec_computed, "y", lambda n: (ROWS, n), _matvec_computed_planes, _matvec_computed_want),
    "masked_sum": Entry(masked_sum, "y", lambda n: (n,), _masked_sum_planes, _masked_sum_want),
    "wide51": Entry(wide51, "y", lambda n: (OUT, n), _wide51_planes, _wide51_want),
    "steps_fill_k1": Entry(steps_fill_k1, "bf", lambda n: (W, n), _stepping(_fill_mask_planes), _steps_fill_want(1)),
    "steps_fill_k8": Entry(steps_fill_k8, "bf", lambda n: (W, n), _stepping(_fill_mask_planes), _steps_fill_want(8)),
    "steps_early_k8": Entry(steps_early_k8, "bf", lambda n: (W, n), _stepping(_early_planes), _early_want),
    "steps_matvec_k1": Entry(steps_matvec_k1, "y", lambda n: (ROWS, n), _stepping(_matvec_in_planes), _steps_matvec_want(1)),
    "steps_matvec_k8": Entry(steps_matvec_k8, "y", lambda n: (ROWS, n), _stepping(_matvec_in_planes), _steps_matvec_want(8)),
    "narrow4": Entry(narrow4, "y", lambda n: (4, n), _narrow_planes(4), _narrow_want(4)),
    "narrow8": Entry(narrow8, "y", lambda n: (8, n), _narrow_planes(8), _narrow_want(8)),
    "narrow16": Entry(narrow16, "y", lambda n: (16, n), _narrow_planes(16), _narrow_want(16)),
    "rotate16": Entry(rotate16, "y", lambda n: (16, n), _rotate16_planes, _rotate16_want),
    "fill_mask_vjp": Entry(fill_mask_vjp, "bar_a", lambda n: (n,), _fill_mask_vjp_planes, _fill_mask_vjp_want),
    "fill_mask_jvp": Entry(fill_mask_jvp, "dot_bf", lambda n: (W, n), _fill_mask_jvp_planes, _fill_mask_jvp_want),
    "matvec_in_vjp": Entry(matvec_in_vjp, "bar_x", lambda n: (W, n), _matvec_in_vjp_planes, _matvec_in_vjp_want),
    "matvec_in_jvp": Entry(matvec_in_jvp, "dot_y", lambda n: (ROWS, n), _matvec_in_jvp_planes, _matvec_in_jvp_want),
    "steps_acc_jvp_k8": Entry(steps_acc_jvp_k8, "dot_bf", lambda n: (W, n), _steps_acc_jvp_planes, _steps_acc_jvp_want),
    "index_set": Entry(index_set_kernel, "y", lambda n: (4, n), _index_set_planes, _index_set_want, LANE_SET),
    "fill_mask_grp": Entry(fill_mask_grp, "bf", lambda n: (W, n), *_grouped(_fill_mask_planes, _fill_mask_want), kind=GROUP_KIND),
    "matvec_in_grp": Entry(matvec_in_grp, "y", lambda n: (ROWS, n), *_grouped(_matvec_in_planes, _matvec_in_want), kind=GROUP_KIND),
    "fill_mask_seg": Entry(fill_mask_seg, "bf", lambda n: (W, n), *_selected(_fill_mask_planes, _fill_mask_want), segmented=True, modes=("float64",)),
    "matvec_in_seg": Entry(matvec_in_seg, "y", lambda n: (ROWS, n), *_selected(_matvec_in_planes, _matvec_in_want), segmented=True, modes=("float64",)),
}

#: The trivial wide kernel of the memory check (:func:`hawk.ir.wide`): a 512-wide plane scaled.
@hawk.kernel
def trivial(x: Vector[W], y: Mutable[Vector[W]]):
    y = x * 2.0


def finite_difference(name: str, planes: dict, n: int, h: float = 0.5) -> np.ndarray:
    """The derivative row ``name`` by central differences of its primal's NumPy reference, in float64 (the primals are linear in
    the differentiated input, so the steps are exact): a ``jvp`` row is ``(F(x + h t) - F(x - h t)) / 2h``, a ``vjp`` row the
    same of ``<bar, F(x)>`` along each input component."""
    f64 = np.float64
    p = {k: np.asarray(v, f64) if np.asarray(v).dtype.kind == "f" else v for k, v in planes.items()}
    if name == "fill_mask_jvp":
        up, dn = {**p, "a": p["a"] + h * p["dot_a"]}, {**p, "a": p["a"] - h * p["dot_a"]}
        return (_fill_mask_want(f64, up, n) - _fill_mask_want(f64, dn, n)) / (2 * h)
    if name == "matvec_in_jvp":
        up, dn = {**p, "x": p["x"] + h * p["dot_x"]}, {**p, "x": p["x"] - h * p["dot_x"]}
        return (_matvec_in_want(f64, up, n) - _matvec_in_want(f64, dn, n)) / (2 * h)
    if name == "fill_mask_vjp":
        base = {"a": np.ones(n), "swg": p["swg"]}
        up, dn = {**base, "a": base["a"] + h}, {**base, "a": base["a"] - h}
        return (((_fill_mask_want(f64, up, n) - _fill_mask_want(f64, dn, n)) / (2 * h)) * p["bar_bf"]).sum(axis=0)
    if name == "matvec_in_vjp":
        out = np.zeros((W, n))
        for i in range(W):
            e = np.zeros((W, n))
            e[i] = h
            up, dn = {"w": p["w"], "x": e}, {"w": p["w"], "x": -e}
            out[i] = (((_matvec_in_want(f64, up, n) - _matvec_in_want(f64, dn, n)) / (2 * h)) * p["bar_y"]).sum(axis=0)
        return out
    raise KeyError(name)


# -- building and running an entry ----------------------------------------------------------------

def build(name: str, dtype: str, arch: str, directory, cache_dir, **options):
    """The bundle of corpus entry ``name`` for the device in scalar mode ``dtype`` (``float32``/``float64``) at ``arch``."""
    from hawk.artifact import build_bundle

    entry = CORPUS[name]
    kind = {} if entry.kind is None else {"kind": entry.kind}
    kernel = segmented_units(entry.kernel)[0] if entry.segmented else entry.kernel
    return build_bundle([kernel], directory, mode=dtype, targets=("cuda",), device_arch=arch,
                        cache_dir=cache_dir, **kind, **options)


def unit_names(name: str) -> tuple:
    """The kernels a build of corpus entry ``name`` makes: itself, or a segmented row's branch units."""
    entry = CORPUS[name]
    return segmented_units(entry.kernel)[1] if entry.segmented else (entry.kernel.name,)


def unit_text(bundle, name: str) -> str:
    """The device text of corpus entry ``name`` as built: its units' texts, one after another."""
    import pathlib

    return "".join((pathlib.Path(bundle.directory) / f"{u}.cu").read_text() for u in unit_names(name))


def resources(bundle, name: str) -> dict:
    """``{regs, stack_B, spill_store_B, spill_load_B, local_bytes}`` of corpus entry ``name`` as the sidecar records them
    (the ptxas frame itself, not the ``spills`` flag, which a frame with no spill store leaves false); the largest of its units."""
    import json
    import pathlib

    rows = []
    for unit in unit_names(name):
        side = json.loads((pathlib.Path(bundle.directory) / f"{unit}.json").read_text())
        row = side["ptxas"][unit]
        rows.append({"regs": row["regs"], "stack_B": row["stack_B"], "spill_store_B": row["spill_store_B"],
                     "spill_load_B": row["spill_load_B"], "local_bytes": side["launch"]["local_bytes"]})
    return {k: max(r[k] for r in rows) for k in rows[0]}


def _run_segmented(entry, dtype, planes, n, cache_dir, repeats, up) -> tuple:
    """The steps of a segmented row through ``eagle.simulate``: ``(output plane, median ns per lane)`` of ``repeats`` runs
    (each starts from fresh planes; the run sorts the lanes by branch, launches the units and sorts them back)."""
    import statistics

    import eagle
    from eagle._alloc import synchronize

    os.environ["HAWK_CACHE_DIR"] = str(cache_dir)
    real = np.float32 if dtype == "float32" else np.float64
    times, out = [], None
    for _ in range(repeats + 1):
        kw = {k: up(v) for k, v in planes.items()}
        kw[entry.out] = up(np.zeros(entry.shape(n), real))
        synchronize()
        res = eagle.simulate(entry.kernel, max_steps=int(SEG_STEPS), dispatch="segmented", scalar_type=dtype, **kw)
        synchronize()
        times.append(res.wall_s)        # the run itself: the build and the binding are not the kernels' time
        out = np.asarray(res.state[entry.out].get())
    return out, statistics.median(times[1:]) * 1e9 / n


def reference_digest(name: str, dtype: str, n: int, planes: dict, want_dir=None) -> str:
    """The digest of corpus entry ``name``'s NumPy reference over ``planes`` (:func:`array_digest`): computed, or read from
    ``want_dir`` when an earlier run kept it. The reference is a function of the corpus source, NumPy and the seeded data
    alone, so it is kept under the digest of those."""
    import hashlib
    import pathlib

    real = np.float32 if dtype == "float32" else np.float64
    if want_dir is None:
        return array_digest(CORPUS[name].want(real, planes, n))
    key = hashlib.sha1(pathlib.Path(__file__).read_bytes() + np.__version__.encode()).hexdigest()[:16]
    kept = pathlib.Path(want_dir) / key / f"{name}_{dtype}_{n}.sha"
    if not kept.exists():
        kept.parent.mkdir(parents=True, exist_ok=True)
        kept.write_text(array_digest(CORPUS[name].want(real, planes, n)))
    return kept.read_text()


def run_device(name: str, dtype: str, n: int, *, cache_dir, repeats: int = 5, seed: int = 7, exact: bool = True,
               want_dir=None) -> dict:
    """Run corpus entry ``name`` over ``n`` lanes on the device through eagle: ``{"out", "want", "ns"}``, the plane the
    kernel wrote, the reference (``None`` unless ``exact``: the data are then reals no reference rounds as the device does)
    and the median nanoseconds a lane takes to launch."""
    import statistics
    import time

    import eagle.plan
    from eagle import arrays
    from eagle._alloc import synchronize

    entry, real = CORPUS[name], np.float32 if dtype == "float32" else np.float64
    planes = entry.planes(np.random.default_rng(seed), real, n, exact)
    want = want_sha = None
    if exact and want_dir is None:
        want = entry.want(real, planes, n)
        want_sha = array_digest(want)
    elif exact:
        want_sha = reference_digest(name, dtype, n, planes, want_dir)

    def up(a):
        return arrays.asarray(np.ascontiguousarray(a), device="cuda:0")

    def bind():
        kw = {k: up(v) for k, v in planes.items()}
        kw[entry.out] = up(np.zeros(entry.shape(n), real))
        if entry.kernel.group > 1:      # the group map comes from the runtime's ActiveSet, which checks the group it was made for
            from eagle import ActiveSet

            live = ActiveSet(kw["terminated"], group=entry.kernel.group)
            live.compact()
            kw["active_map"], kw["active_count"] = live.map, live.count
        return kw, plan.bind(**kw)

    if entry.segmented:
        out, ns = _run_segmented(entry, dtype, planes, n, cache_dir, repeats, up)
        return {"out": out, "want": want, "want_sha": want_sha, "ns": ns}
    plan = eagle.plan.auto(entry.kernel, targets=("cuda",), scalar_type=dtype, cache_dir=str(cache_dir))
    stepping = "terminated" in planes        # a stepping kernel leaves its lanes finished: every timed launch starts afresh
    kw, bound = bind()
    bound.launch()
    synchronize()
    out = np.asarray(kw[entry.out].get())
    times = []
    for _ in range(repeats):
        if stepping:
            kw, bound = bind()
        synchronize()
        start = time.perf_counter()
        bound.launch()
        synchronize()
        times.append(time.perf_counter() - start)
    return {"out": out, "want": want, "want_sha": want_sha, "ns": statistics.median(times) * 1e9 / n}


# -- the same corpus in another environment (the gate runs it in the 0.7.2 wheel's, which has none of the pass) ----------

LANES = {"fill_mask": 8192, "matvec_in": 16384, "matvec_computed": 16384, "masked_sum": 16384, "wide51": 16384,
         "steps_fill_k1": 8192, "steps_fill_k8": 8192, "steps_early_k8": 8192, "steps_matvec_k1": 16384,
         "steps_matvec_k8": 16384, "narrow4": 1048576, "narrow8": 1048576, "narrow16": 1048576, "rotate16": 1048576, "fill_mask_vjp": 8192, "fill_mask_jvp": 8192, "matvec_in_vjp": 8192, "matvec_in_jvp": 8192,
         "steps_acc_jvp_k8": 8192, "index_set": 131072,
         "fill_mask_grp": 8192, "matvec_in_grp": 16384, "fill_mask_seg": 32768, "matvec_in_seg": 32768}


#: Lanes of a host row (a short run: the host tests compare two builds bit for bit, not time them).
LANES_HOST = 301

#: Timed launches per row where the default seven leave the median within the timing noise of the 0.98x line.
REPEATS = {"fill_mask_seg": 3, "matvec_in_seg": 3, "index_set": 25, "narrow4": 15, "narrow8": 15, "narrow16": 15, "rotate16": 15}


def device_arch() -> str:
    """The device's own ``sm_<N>`` (what the gate builds for and times on)."""
    if os.environ.get("HAWK_GATE_ARCH"):      # the gate hands its device's arch to the other release's environment
        return os.environ["HAWK_GATE_ARCH"]
    from _toolchain import real_arch

    return real_arch()


def array_digest(a) -> str:
    """A digest of an output plane (its bytes, shape and dtype): what two environments' values are compared by, so that
    nothing the size of the planes is kept on disk."""
    import hashlib

    a = np.ascontiguousarray(a)
    return hashlib.sha1(a.tobytes() + str((a.shape, a.dtype)).encode()).hexdigest()


def _prebuild(cache) -> None:
    """Compile every (entry, mode) of the corpus on all cores at once, so the launches that follow find their kernels built
    (a build is nvcc, a subprocess: threads overlap them, and the compile of the 0.7.2 wheel is not served from a disk cache)."""
    from concurrent.futures import ThreadPoolExecutor

    import eagle.plan

    def build_one(job):
        entry, mode = job
        eagle.plan.auto(CORPUS[entry].kernel, targets=("cuda",), scalar_type=mode, cache_dir=str(cache))
        if CORPUS[entry].segmented:     # the door builds the branch units on its first run: a short one builds them here, beside the rest
            run_device(entry, mode, 256, cache_dir=cache, repeats=1)

    with ThreadPoolExecutor(int(os.environ.get("HAWK_GATE_JOBS", "6"))) as pool:
        list(pool.map(build_one, [(e, m) for e in CORPUS for m in CORPUS[e].modes]))


def twin_run(root, cache=None) -> None:
    """Write to ``root`` the corpus values (``<entry>_<mode>_<exact>.sha``, digests (:func:`array_digest`)), their times (``times.json``) and the device text
    of each kernel (``texts.json``): what the gate compares two environments by. ``cache`` is the compile cache (``root / "cache"``).

    The values of rounding data and the texts are functions of the corpus source and the 0.7.2 wheel alone, so they are kept beside
    the compile cache (``results/<digest of this file and the wheel>``) and reused; the times, which depend on the card's state, are
    measured on every run (the exact-data launches, seven repeats)."""
    import hashlib
    import json
    import pathlib
    import shutil
    import time

    import hawk

    root = pathlib.Path(root)
    cache = root / "cache" if cache is None else pathlib.Path(cache)
    digest = hashlib.sha1(pathlib.Path(__file__).read_bytes() + str(getattr(hawk, "__version__", "")).encode()).hexdigest()[:16]
    kept = cache / "results" / digest
    kept.mkdir(parents=True, exist_ok=True)
    kept_texts = json.loads((kept / "texts.json").read_text()) if (kept / "texts.json").exists() else {}
    times, texts = {}, {}
    _prebuild(cache)
    for entry in CORPUS:
        started = time.perf_counter()
        for mode in CORPUS[entry].modes:
            got = run_device(entry, mode, LANES[entry], cache_dir=cache, exact=True, repeats=REPEATS.get(entry, 7))
            (root / f"{entry}_{mode}_1.sha").write_text(array_digest(got["out"]))
            times[f"{entry}_{mode}_1"] = got["ns"]
            rounding = kept / f"{entry}_{mode}_0.sha"
            if not rounding.exists():
                rounding.write_text(array_digest(run_device(entry, mode, LANES[entry], cache_dir=cache, exact=False, repeats=1)["out"]))
            shutil.copy(rounding, root / f"{entry}_{mode}_0.sha")
            key = f"{entry}_{mode}"
            if key not in kept_texts:
                bundle = build(entry, mode, device_arch(), root / key, cache)
                kept_texts[key] = unit_text(bundle, entry)
                (kept / "texts.json").write_text(json.dumps(kept_texts))
            texts[key] = kept_texts[key]
        if os.environ.get("HAWK_GATE_TRACE"):
            print(f"twin {entry} {time.perf_counter() - started:.1f}s", file=sys.stderr, flush=True)
    (root / "times.json").write_text(json.dumps(times))
    (root / "texts.json").write_text(json.dumps(texts))


def _suite_kernels() -> dict:
    """``{module.attr: kernel}`` of the module-level kernels of the suite's helper modules."""
    import importlib

    from hawk.trace.kernel import Kernel

    out = {}
    for module in ("_kernels", "_deployable", "_emitted", "_block_kernels", "_net_axis_kernels", "_spline_forms",
                   "_autodiff_kernels", "_linalg_kernels"):
        try:
            mod = importlib.import_module(module)
        except Exception:
            continue
        for attr, kernel in sorted(vars(mod).items()):
            if isinstance(kernel, Kernel):
                out[f"{module}.{attr}"] = kernel
    return out


def runtime_indexed(kernel) -> bool:
    """Whether ``kernel`` reads or writes a vector at an index that is not a compile-time constant (or a placed one): the
    only kernels whose generated text may differ from the 0.7.2 wheel's."""
    from hawk.ir.loop_nodes import Loop
    from hawk.ir.nodes import Const
    from hawk.ir.walk import canonical_nodes

    nodes, seen = list(canonical_nodes(kernel.sinks)[0]), set()
    while nodes:
        n = nodes.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        if isinstance(n, Loop):
            nodes.extend(n.body_nodes())
        if getattr(n, "kind", None) in ("component_at", "set_component_at", "add_at") and not isinstance(n.operands[1], Const):
            return True
    return False


def suite_texts() -> dict:
    """``{kernel/mode: {target: text}}`` for the module-level kernels of the suite's helper modules, both targets and both
    scalar modes, as a build makes them (the placement pass included), or the refusal a build gave: the kernels with no
    run-time indexed value, which the pass must leave as they were."""
    from hawk.artifact import bundle as B
    from hawk.emit import scalar_mode

    out = {}
    for key, kernel in _suite_kernels().items():
        for mode in ("float64", "float32"):
            texts = {}
            for target, device in (("cuda", True), ("host", False)):
                try:
                    try:
                        stages = B._expand_split([kernel], [kernel.name], [kernel.kind], smode=scalar_mode(mode), device=device)
                    except TypeError:        # the 0.7.2 wheel has no placement
                        stages = B._expand_split([kernel], [kernel.name], [kernel.kind])
                    text = ""
                    for k, n, kd in zip(*stages[:3]):
                        unit = B._active_unit(k, n, kd)
                        src = B._sources(unit, name=n, smode=scalar_mode(mode), kind=kd, targets=(target,),
                                         layout_sizes_override=None)
                        text += src[target].text
                    texts[target] = text
                except Exception as exc:        # a refusal is part of what must not change
                    texts[target] = f"{type(exc).__name__}: {str(exc)[:200]}"
            out[f"{key}/{mode}"] = texts
    return out


def memory_delta(mode: str) -> float:
    """In this process, which must be fresh (the runtime reserves local memory for the largest kernel a process ever launched):
    the device memory in MB that the first launch of the trivial wide kernel takes."""
    import eagle.plan
    from eagle import _core, arrays
    from eagle._alloc import synchronize

    def used():
        synchronize()
        total, free = _core.device_mem_info(-1)
        return (total - free) / 2 ** 20

    real, n = (np.float32 if mode == "float32" else np.float64), 65536
    plan = eagle.plan.auto(trivial, targets=("cuda",), scalar_type=mode)
    bound = plan.bind(x=arrays.asarray(np.ones((W, n), real), device="cuda:0"),
                      y=arrays.asarray(np.zeros((W, n), real), device="cuda:0"))
    before = used()
    bound.launch()
    return used() - before


if __name__ == "__main__":
    import json
    import pathlib
    import sys

    flag, where = sys.argv[1:3]
    if flag == "--memory":
        print(f"{memory_delta(where):.2f}")
    elif flag == "--twin":
        twin_run(where, sys.argv[3] if len(sys.argv) > 3 else None)
    elif flag == "--texts":
        pathlib.Path(where, "texts_072.json").write_text(json.dumps(suite_texts()))
