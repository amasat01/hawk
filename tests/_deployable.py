# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The deployment fixture set: one kernel per SHAPE a v2 artifact must carry.

Kept out of ``_kernels.py`` on purpose — that corpus is emitter sweep and
its rows count it. These are the kernels rows /////////
 BUILD, COMPILE and RUN, and between them they cover every role shape eagle's
packer resolves: a ``Vector[3]`` output plane (the 40-byte GRef mirror), a
``vec_in``, a ``mat_in``, a ``float32`` arm, TWO output planes, a ``lookup``
gather (``cross_sample_read``), a mapreduce partial, a scatter
(``cross_sample_write``), the compensated-accum kind and the mask-free
guard kind, plus a vocabulary sweep over the aether spellings the rank>=1
alignment added, and a two-wire compound read whose wires are
written out SEPARATELY.

:func:`cases` at the bottom is the ONE table of "kernel + the arguments it is
run with + what it should produce". It lives here rather than inside any single
row because rows sweep the SAME set deploys: the serial oracle, the partition-soundness sweep and the deployment rows must not be
able to disagree about WHICH kernels the fixture set is.
"""

from __future__ import annotations

import numpy as np

from hawk import (
    Accum,
    Index,
    Matrix,
    Mutable,
    Param,
    Quantity,
    Quat,
    Reduce,
    Scalar,
    Table,
    Terminated,
    Vector,
    kernel,
)
from hawk.ext import DATA_ONLY, Guard, Kind, sink_policy
from hawk.math import dot, maximum, minimum, n_samples, norm, quat_rotate, tanh
from hawk.types import Wire

ORIENTATION = Quantity(
    slug="earth", name="orientation",
    wires=[Wire("q", "vec_in", (4,), tag="quaternion"), Wire("w", "per_sample", ())],
    reconstruct=lambda q, w: (q, w),
)


@kernel
def axpb(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


@kernel
def vec3_scale(x: Vector[3], a: Param, y: Mutable[Vector[3]]):
    y = a * x


@kernel
def mat_apply(m: Matrix[3, 3], v: Vector[3], y: Mutable[Vector[3]]):
    y = m @ v


@kernel
def two_outputs(x: Vector[3], y: Mutable[Vector[3]], s: Mutable[Scalar]):
    y = x + x
    s = dot(x, x)


@kernel
def gather(table: Table[Scalar], where: Index, y: Mutable[Scalar]):
    y = table.at(where)


@kernel
def energy(v: Vector[3], total: Reduce("sum")):
    total.contribute(dot(v, v))


@kernel
def speed(v: Vector[3], y: Mutable[Scalar]):
    """The shape the whole fixture set was missing: a rank>=1 leaf whose
    FAN-OUT IS 1, read by a rank-collapsing op.

    Every other rank>=1 leaf here is read at least twice (``energy`` spells
    ``dot(v, v)``, ``vocab`` reads ``v`` three times), and a leaf read twice was
    already being materialised into an ``Item`` for address-computation
    reason -- which is exactly what made it sample-BOUND. At fan-out 1 the
    renderer used to hand aether the whole plane, and aether's folds evaluate at
    ``SampleIndex::make(0)``, so every sample received sample 0's norm. The
    fixture set now carries the shape, so / deploy and run it."""
    y = norm(v)


@kernel
def scatter(x: Scalar, lane: Index, acc: Accum[Scalar]):
    acc.add(x, at=lane)


@kernel
def scatter_c(x: Scalar, lane: Index, acc: Accum[Scalar], acc_c: Accum[Scalar]):
    """ fixture: the SAME body as :func:`scatter`, but built under the
    compensated kind, which commits its Neumaier correction to ``acc_c``. The
    second plane is the kernel's own declaration; the SEAM decides the
    arithmetic."""
    acc.add(x, at=lane)
    acc_c.add(0.0, at=lane)


@kernel
def diagnostic(x: Scalar, terminated: Terminated, y: Mutable[Scalar]):
    """ fixture: a per-sample diagnostic that must run for EVERY sample.
    Under the default guard the terminated samples keep their prior value;
    under the mask-free (data-only) kind every sample is written."""
    y = x + 1.0


@kernel
def diagnostic_two(x: Scalar, terminated: Terminated, rejected: Terminated,
                   y: Mutable[Scalar]):
    """ fixture: the same diagnostic with a SECOND mask. The default guard gates
    on ``terminated`` alone; a two-mask kind skips a sample flagged by either."""
    y = x + 1.0


@kernel
def fraction(x: Scalar, y: Mutable[Scalar]):
    """The subject: a body that READS the sample count, so the ``nsamples``
    ROLE is a real parameter of the emitted entry — the one whose width must
    track ``EAGLE_ABI_INDEX_T`` and NOT the triple's int64 ``nSamples``."""
    y = x / n_samples()


@kernel
def spin(orientation: ORIENTATION, v: Vector[3], out: Mutable[Vector[3]]):
    """ fixture: a NAMESPACED compound quantity, read and rotated
    with no text surgery and no raw IR."""
    q, w = orientation
    out = w * quat_rotate(q, v)


@kernel
def two_wire(orientation: ORIENTATION, out_q: Mutable[Quat],
             out_w: Mutable[Scalar]):
    """The fixture (the row). ONE quantity read mints TWO wires —
    a ``vec_in`` quaternion and a ``per_sample`` scalar — and this body writes
    each wire's VALUE to its own output plane, scaled by a different exact
    constant. That is what makes the row POSITIVE rather than a refusal: if the
    second wire read the first mirror's bytes, ``out_w`` would carry a
    pointer-as-double (the 4.65e-310 signature) or the quaternion's first
    component, and either is visible in the answer."""
    q, w = orientation
    out_q = 2.0 * q
    out_w = 3.0 * w


@kernel
def vocab(v: Vector[3], k: Param, y: Mutable[Vector[3]]):
    """The rank>=1 spellings aether added: ``cwiseTanh``, ``cwiseMin`` /
    ``cwiseMax`` with an expression right operand, ``cwiseDiv`` and
    ``cwisePow``."""
    a = tanh(v)
    b = minimum(a, v / (k + 2.0))
    y = maximum(b, a ** v)


#: The compensated kind and the mask-free kind.
COMPENSATED = Kind("compensated", sink=sink_policy("compensated", into="acc_c"))
MASK_FREE = Kind("data_only", guard=DATA_ONLY)
#: The two-mask kind: a sample commits only when neither mask is set.
TWO_MASKS = Kind("two_masks", guard=Guard(masks=("terminated", "rejected")))


# --------------------------------------------------------------------------- #
# Numpy references. One per kernel, taking the same kwargs eagle's plan does.
# --------------------------------------------------------------------------- #
def ref_axpb(x, a, b):
    return a * np.asarray(x) + b


def ref_vec3_scale(x, a):
    return a * np.asarray(x)


def ref_mat_apply(m, v):
    return np.einsum("rcn,cn->rn", np.asarray(m), np.asarray(v))


def ref_two_outputs(x):
    x = np.asarray(x)
    return x + x, (x * x).sum(axis=0)


def ref_gather(table, where):
    return np.asarray(table)[np.asarray(where).astype(int)]


def ref_energy(v):
    return (np.asarray(v) ** 2).sum(axis=0)


def ref_speed(v):
    return np.linalg.norm(np.asarray(v), axis=0)


def ref_scatter(x, lane, lanes):
    out = np.zeros(lanes, dtype=np.asarray(x).dtype)
    np.add.at(out, np.asarray(lane).astype(int), np.asarray(x))
    return out


def ref_fraction(x, n):
    return np.asarray(x) / float(n)


def ref_spin(q, w, v):
    q, w, v = np.asarray(q), np.asarray(w), np.asarray(v)
    out = np.empty_like(v)
    for i in range(v.shape[-1]):
        qi, vi = q[:, i], v[:, i]
        s, u = qi[0], qi[1:]
        rot = (2.0 * np.dot(u, vi) * u + (s * s - np.dot(u, u)) * vi
               + 2.0 * s * np.cross(u, vi))
        out[:, i] = w[i] * rot
    return out


def ref_vocab(v, k):
    v = np.asarray(v)
    a = np.tanh(v)
    b = np.minimum(a, v / (k + 2.0))
    return np.maximum(b, a ** v)


def ref_two_wire(q, w):
    """The reference. The two scalings are EXACT in binary floating point for
    the first and reproduce IEEE ``3.0 * w`` for the second, so the comparison
    is bit-for-bit and a wrong-mirror read cannot hide inside a tolerance."""
    return 2.0 * np.asarray(q), 3.0 * np.asarray(w)


# --------------------------------------------------------------------------- #
# THE fixture table: kernel + arguments + reference, one entry per deployed shape.
# --------------------------------------------------------------------------- #
def quat(n: int) -> np.ndarray:
    """A per-sample unit quaternion plane, no two columns alike."""
    q = np.stack([np.linspace(1.0, 2.0, n), np.linspace(0.1, 0.5, n),
                  np.linspace(-0.2, 0.3, n), np.linspace(0.05, 0.4, n)])
    return q / np.linalg.norm(q, axis=0)


def plane(n: int, w: int = 3, dtype=np.float64) -> np.ndarray:
    """A ``(w, n)`` input plane whose every entry differs, so a row comparing
    the wrong samples cannot pass by coincidence."""
    return np.arange(w * n, dtype=dtype).reshape(w, n) * 0.25 + 1.0


def cases(n: int) -> list:
    """``(bundle, kernel, kwargs, reference)`` for every deployed shape."""
    x3, xs = plane(n), plane(n, w=1)[0]
    m = np.arange(9 * n, dtype=float).reshape(3, 3, n) * 0.1
    table = np.arange(n, dtype=float) * 3.0
    # An INDEX plane binds at the emitted wire width: HAWK spells every integer
    # element `Int` (`long long`, hawk/emit/aether.element_spelling), and eagle's
    # packer passes an integer array through AT ITS OWN dtype rather than
    # re-casting it -- so an int32 array would be decoded 8 bytes at a time.
    where = ((np.arange(n) * 7 + 3) % n).astype(np.int64)
    # a PERMUTATION of [0, n): the scatter targets a lane that is not the
    # sample's own column (so the class is cross_sample_write) while every lane
    # has exactly ONE writer -- a multi-writer scatter's DEVICE commit is the 
    # seam's `atomic` slot, which is named and not built (hawk/ext), and racing
    # a plain read-modify-write here would be measuring that gap, not the row.
    lane = ((np.arange(n) * 7 + 3) % n).astype(np.int64)
    q, w = quat(n), np.linspace(0.5, 2.0, n)
    return [
        ("axpb", "axpb", {"x": xs, "a": 2.0, "b": -1.0}, ref_axpb(xs, 2.0, -1.0)),
        ("vec3", "vec3_scale", {"x": x3, "a": 2.0}, ref_vec3_scale(x3, 2.0)),
        ("vec3_f32", "vec3_scale", {"x": x3.astype(np.float32), "a": 2.0},
         ref_vec3_scale(x3.astype(np.float32), np.float32(2.0))),
        # a `mat_in` binds as eagle packs it: the FLAT (R*C, n) plane whose
        # component pitch is the mirror's `compStride_` (eagle.roles' own note --
        # "a matrix GRef is a width-R*C vector GRef").
        ("mat", "mat_apply", {"m": m.reshape(9, n), "v": x3}, ref_mat_apply(m, x3)),
        # the outputs come back in, which is HAWK's canonical
        # role order -- `s` before `y`, both `mutable`.
        ("multi", "two_outputs", {"x": x3}, tuple(reversed(ref_two_outputs(x3)))),
        ("gather", "gather", {"table": table, "where": where},
         ref_gather(table, where)),
        ("energy", "energy", {"v": x3}, ref_energy(x3)),
        ("speed", "speed", {"v": x3}, ref_speed(x3)),
        ("scatter", "scatter", {"x": xs, "lane": lane}, ref_scatter(xs, lane, n)),
        ("fraction", "fraction", {"x": xs}, ref_fraction(xs, n)),
        ("spin", "spin", {"earth__orientation__q": q, "v": x3,
                          "earth__orientation__w": w}, ref_spin(q, w, x3)),
        ("two_wire", "two_wire", {"earth__orientation__q": q,
                                  "earth__orientation__w": w}, ref_two_wire(q, w)),
    ]
