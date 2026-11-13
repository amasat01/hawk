# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A wide value placed in its plane on the host loop interchange is an aether expression: every whole-value op reads it.

A wide value that lives in a row of the tile is referenced through a lane proxy. The proxy used to answer only ``.at(k)``,
so a product, sum, unary function, dot or any other method over the whole value failed to compile on the interchanged host
TU (``no member named 'cwiseMul'``). Each kernel below writes one element of a 512-wide value at a run-time index (which
places it in its plane), and combines it with a product of the kernel's own in ONE whole-value op; each is built with the static spelling (``emit_form="static"``, the unrolled expression nodes the proxy
failed under) and must return the bits of NumPy on exact data. Every build pins the EXACT ``native`` host profile.
"""

import _oracle as O
import numpy as np
import pytest
from _wide_kernels import zeros
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import Mutable, Scalar, Vector
from hawk.artifact import build_bundle
from raptor.wings.cpu.vectorise import interchanged_loops

W = 512
N = 70


def _scattered(a, s):
    q = zeros(W)
    q[m.index(s, W)] = a
    return q


def _both(a, s, t):
    w = zeros(W)
    for j in range(W):
        w[j] = a * (j + 1.5)
    return w * t, _scattered(a, s)


@hawk.kernel
def placed_mul(a: Scalar, s: Scalar, t: Vector[W], y: Mutable[Vector[W]]):
    w, q = _both(a, s, t)
    y = w * q


@hawk.kernel
def placed_add(a: Scalar, s: Scalar, t: Vector[W], y: Mutable[Vector[W]]):
    w, q = _both(a, s, t)
    y = w + q


@hawk.kernel
def placed_unary(a: Scalar, s: Scalar, t: Vector[W], y: Mutable[Vector[W]]):
    w, q = _both(a, s, t)
    y = w + m.sqrt(q)


@hawk.kernel
def placed_dot(a: Scalar, s: Scalar, t: Vector[W], y: Mutable[Scalar]):
    w, q = _both(a, s, t)
    y = m.dot(w, q)


def _q(a, s):
    q = np.zeros((W, N))
    q[s.astype(int), np.arange(N)] = a
    return q


def _w(a, t):
    return a[None, :] * (np.arange(W)[:, None] + 1.5) * t


CASES = {
    "mul": (placed_mul, lambda a, s, t: _w(a, t) * _q(a, s)),
    "add": (placed_add, lambda a, s, t: _w(a, t) + _q(a, s)),
    "unary": (placed_unary, lambda a, s, t: _w(a, t) + np.sqrt(_q(a, s))),
    "dot": (placed_dot, lambda a, s, t: (_w(a, t) * _q(a, s)).sum(axis=0)),
}


@pytest.mark.parametrize("name", CASES)
def test_a_placed_wide_value_feeds_a_whole_value_op(name, tmp_path, cache_dir):
    k, want = CASES[name]
    assert interchanged_loops(k.walk)
    rng = np.random.default_rng(5)
    a = rng.integers(1, 6, N).astype(np.float64) / 2
    s = rng.integers(0, W, N).astype(np.float64)
    t = rng.choice([1.0, 4.0, 9.0, 16.0], (W, N))          # perfect squares: the square roots, and so every sum, are exact
    b = build_bundle([k], tmp_path / name, targets=("host",), cache_dir=cache_dir, host_profile="native",
                     emit_form="static")
    kernel = O.load(b.directory, k.name, sidecar_of(b, k.name))
    planes = {"a": a, "s": s, "t": t, "y": np.zeros((W, N) if name != "dot" else N)}
    for _role, nm in kernel.arg_spec:
        if nm.startswith("hwd"):
            planes[nm] = np.zeros(1)
    got = np.asarray(O.run_kernel(kernel, N, **planes))
    assert np.array_equal(got, want(a, s, t)), f"{name}: the result differs from NumPy's"
