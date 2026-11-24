# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A plain ``def`` called in a kernel body is an authoring helper: no decorator, no types, traced inline.

Complements test_helper_defs.py (branches, loops, recursion, rewrite once) with the cases it lacks: vectors, a helper that
returns several values, a helper that calls a helper, a helper called from an ``@hawk.function``, and the derivatives of
those, each against NumPy. Every kernel is compiled for the host and run."""

from __future__ import annotations

import numpy as np

import hawk
import hawk.diff
import hawk.math as m
from hawk.trace.kernel import Kernel

LANES = 32
EPS = np.finfo(np.float64).eps


def rhs(x, k):
    return m.vec(x[1], -k * x[0])


def euler(x, k, dt):
    """Two helper levels: this one calls ``rhs``."""
    return x + rhs(x, k) * dt


def split(v):
    return v[0] * v[0], v[1] + v[2]


def unit(v):
    return v / m.sqrt(m.dot(v, v))


def cube(x):
    return x * x * x


@hawk.function
def cubed_plus(x: hawk.Scalar, y: hawk.Scalar) -> hawk.Scalar:
    return cube(x) + y


@hawk.kernel
def steps(x: hawk.Vector[2], k: hawk.Scalar, out: hawk.Mutable[hawk.Vector[2]]):
    s = x
    for _ in range(3):
        s = euler(s, k, 0.1)
    out = s


@hawk.kernel
def pair(v: hawk.Vector[3], out: hawk.Mutable[hawk.Scalar]):
    a, b = split(v)
    out = a * b


@hawk.kernel
def along(v: hawk.Vector[3], w: hawk.Vector[3], out: hawk.Mutable[hawk.Scalar]):
    out = m.dot(unit(v), w)


@hawk.kernel
def from_function(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    out = cubed_plus(x, y)


def _random(*shape):
    return np.random.default_rng(7).uniform(0.5, 1.5, shape)


def _run(kernel, cache_dir, **planes):
    hawk.deploy(kernel, targets=("host",), cache_dir=cache_dir).run(**planes)


def _gradient(kernel, cache_dir, names, **planes):
    derived = Kernel(f"{kernel.name}_vjp", hawk.diff.vjp(kernel))
    for name in names:
        planes[f"bar_{name}"] = np.zeros_like(planes[name])
    _run(derived, cache_dir, **planes)
    return [planes[f"bar_{name}"] for name in names]


def _close(got, want):
    return np.all(np.abs(got - want) <= 16.0 * EPS * np.maximum(1.0, np.abs(want)))


def test_a_helper_calling_a_helper_on_vectors_equals_numpy(cache_dir):
    x, k = _random(2, LANES), _random(LANES)
    out = np.zeros_like(x)
    _run(steps, cache_dir, x=x, k=k, out=out)
    want = x.copy()
    for _ in range(3):
        want = want + np.stack([want[1], -k * want[0]]) * 0.1
    assert _close(out, want)


def test_a_helper_returning_several_values_is_unpacked_and_equals_numpy(cache_dir):
    v = _random(3, LANES)
    out = np.zeros(LANES)
    _run(pair, cache_dir, v=v, out=out)
    assert _close(out, v[0] * v[0] * (v[1] + v[2]))


def test_the_gradient_through_a_vector_helper_equals_the_closed_form(cache_dir):
    v, w = _random(3, LANES), _random(3, LANES)
    (gv,) = _gradient(along, cache_dir, ("v",), v=v, w=w, bar_out=np.ones(LANES))
    norm = np.sqrt((v * v).sum(axis=0))
    want = w / norm - v * (v * w).sum(axis=0) / norm**3
    assert _close(gv, want)


def test_a_plain_helper_called_from_a_function_has_the_derivative_of_its_body(
    cache_dir,
):
    x, y = _random(LANES), _random(LANES)
    out = np.zeros(LANES)
    _run(from_function, cache_dir, x=x, y=y, out=out)
    assert _close(out, x**3 + y)
    gx, gy = _gradient(
        from_function, cache_dir, ("x", "y"), x=x, y=y, bar_out=np.ones(LANES)
    )
    assert _close(gx, 3.0 * x * x) and _close(gy, np.ones(LANES))
    tangent = Kernel("from_function_jvp", hawk.diff.jvp(from_function))
    planes = {
        "x": x,
        "y": y,
        "dot_x": np.ones(LANES),
        "dot_y": np.zeros(LANES),
        "dot_out": np.zeros(LANES),
    }
    _run(tangent, cache_dir, **planes)
    assert _close(planes["dot_out"], 3.0 * x * x)
