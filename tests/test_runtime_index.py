# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The gradient of a runtime-indexed read (``add_at``): the emitted text of an
accumulate on the own column, and the compiled HOST kernel against the scratch
interpreter."""

from __future__ import annotations


import _spline_forms as S
import numpy as np
from conftest import declared
import pytest
from _eval import evaluate

import hawk
import hawk.math as m
from hawk import Index, Kernel, Mutable, Scalar, Vector
from hawk.diff import vjp
from hawk import _ir


# --------------------------------------------------------------------------
# the compiled host kernel against the interpreter


@pytest.fixture(scope="module")
def host(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("runtime_index")
    cache = {}

    def get(name, primal, sinks):
        if name not in cache:
            d = root / name
            d.mkdir()
            hawk.build([primal, Kernel(name, sinks)], d, targets=("host",),
                       cache_dir=cache_dir)
            cache[name] = hawk.load(d, name)
        return cache[name]

    return get


@hawk.kernel
def _double_hit(theta: Vector[4], j: Index, y: Mutable[Scalar]):
    """One lane read BOTH by constant and by runtime index."""
    y = theta[2] * 3.0 + theta[j]


def test_a_lane_hit_by_a_constant_and_a_runtime_index_sums_both(host):
    """The accumulate ADDS to the slot a plain store wrote (an ``=`` here would
    overwrite the static contribution): compiled host against the interpreter."""
    sinks = vjp(_double_hit, wrt=("theta",))
    n = 6
    rng = np.random.default_rng(3)
    theta = rng.normal(size=(4, n))
    j = np.array([2, 2, 0, 1, 3, 2], dtype=np.int32)
    seed = rng.normal(size=n)
    bar = np.zeros((4, n))
    kernel = host("double_hit_vjp", _double_hit, sinks)
    hawk.run(kernel, **declared(kernel, {"theta": theta, "j": j, "bar_y": seed}), bar_theta=bar)
    for s in range(n):
        want = evaluate(sinks, {"theta": theta[:, s], "j": int(j[s]), "bar_y": seed[s]})
        assert np.array_equal(bar[:, s], want["bar_theta"]), s


@pytest.mark.parametrize("n_in", [8])
def test_the_local_spline_vjp_host_kernel_equals_the_interpreter(host, n_in):
    """Every lane of the compiled local-spline vjp against the interpreter. The
    host profile contracts multiply-adds (the interpreter does not), so the lanes
    agree to rounding of the largest term of the sample, not bit for bit."""
    cell = S.spline_layer("local", n_in)
    sinks = vjp(cell, wrt=("x", "theta"))
    n = 5
    rng = np.random.default_rng(8)
    x = rng.uniform(-0.9, 0.9, size=(n_in * S.ROWS, n))
    theta = rng.normal(0.0, 0.3, size=(n_in * S.W, n))
    theta[S.W - 2::S.W] = 0.0
    theta[S.W - 1::S.W] = 1.0
    seed = rng.normal(size=(2, n))
    bar_x, bar_theta = np.zeros_like(x), np.zeros_like(theta)
    hawk.run(host(f"spline_local_vjp_{n_in}", cell, sinks), x=x, theta=theta,
             terminated=np.zeros(n, dtype=bool), bar_y=seed, bar_x=bar_x,
             bar_theta=bar_theta)
    for s in range(n):
        want = evaluate(sinks, {"x": x[:, s], "theta": theta[:, s], "terminated": False,
                                "bar_y": seed[:, s]})
        for got, ref in ((bar_theta[:, s], want["bar_theta"]), (bar_x[:, s], want["bar_x"])):
            np.testing.assert_allclose(got, ref, rtol=1e-12,
                                       atol=1e-13 * np.abs(ref).max())
        # the slots no edge reads stay exactly zero
        assert np.array_equal(bar_theta[:, s] == 0.0, np.asarray(want["bar_theta"]) == 0.0)


# --------------------------------------------------------------------------
# a store at a runtime index of a short vector is one select per component


def _stores(width):
    def build(v, j, e, o):
        acc = v
        for c in range(width):
            acc[c] = acc[c] + e
        acc[m.index(j, width)] = e * 2.0
        o = acc
    build.__annotations__ = {"v": Vector[width], "j": Index, "e": Scalar,
                             "o": Mutable[Vector[width]]}
    return hawk.kernel(build)


def _zeros():
    return m.vec(*([0.0] * 15))


def _scatter(trips):
    """The spline scatter: one static slot and four slots at a runtime base
    plus static offsets, in a rolled loop."""
    def build(u, e, o):
        acc = _zeros()
        for t in range(trips):
            i = m.index(u + t, 10)
            w = e * (t + 1.0)
            acc[0] = acc[0] + w
            acc[i + 2] = acc[i + 2] + w
            acc[i + 3] = acc[i + 3] + w * 2.0
            acc[i + 4] = acc[i + 4] + w * 3.0
            acc[i + 5] = acc[i + 5] + w * 4.0
        o = acc
    build.__annotations__ = {"u": Scalar, "e": Scalar, "o": Mutable[Vector[15]]}
    return hawk.kernel(build)


def _zeros_uniform():
    return m.vec(*([0.0] * 15))


@hawk.kernel
def _clamped(u: Scalar, e: Scalar, o: Mutable[Vector[15]]):
    """Three accumulates whose indices are each clamped on their own."""
    acc = _zeros()
    for t in range(6):
        p = m.index(u + t, 16)
        acc[2 + m.index(m.real(p - 3), 13)] = acc[2 + m.index(m.real(p - 3), 13)] + e
        acc[2 + m.index(m.real(p - 2), 13)] = acc[2 + m.index(m.real(p - 2), 13)] + e * 2.0
        acc[2 + m.index(m.real(p - 1), 13)] = acc[2 + m.index(m.real(p - 1), 13)] + e * 3.0
    o = acc


def _scatter_inputs(n, trips):
    rng = np.random.default_rng(trips)
    return rng.integers(-2, 12, size=n).astype(float), rng.integers(1, 5, size=n).astype(float)


def test_the_fused_scatter_host_run_equals_the_interpreter_bitwise(host):
    """Integer-valued data: every product and sum is exact, so the fused pass
    and the interpreter agree bit for bit (slot 1 stays the zero it started as)."""
    kernel = _scatter(6)
    n = 9
    u, e = _scatter_inputs(n, 6)
    out = np.full((15, n), 7.0)
    hawk.run(host("scatter_6", kernel, kernel.sinks), u=u, e=e, o=out)
    for s in range(n):
        want = evaluate(kernel.sinks, {"u": u[s], "e": e[s]})
        assert np.array_equal(out[:, s], np.asarray(want["o"])), s
        assert out[1, s] == 0.0


@pytest.mark.parametrize("width", [15, 17, 256])
def test_a_runtime_store_host_run_equals_the_interpreter(host, width):
    kernel = _stores(width)
    n = 7
    rng = np.random.default_rng(width)
    v, e = rng.normal(size=(width, n)), rng.normal(size=n)
    j = rng.integers(0, width, size=n).astype(np.int32)
    out = np.zeros((width, n))
    hawk.run(host(f"store_{width}", kernel, kernel.sinks), v=v, j=j, e=e, o=out)
    for s in range(n):
        want = evaluate(kernel.sinks, {"v": v[:, s], "j": int(j[s]), "e": e[s]})
        assert np.array_equal(out[:, s], np.asarray(want["o"])), s


def _on_device(kernel, cache_dir, **planes):
    """The kernel on the GPU: the arrays go over as given, the output comes back."""
    import cupy as cp

    plan = hawk.deploy([kernel], targets=("host", "cuda"), cache_dir=cache_dir, scalar_type="float64")[0]
    arrays = {k: cp.asarray(v) for k, v in planes.items()}
    graph = hawk.Graph()
    graph.add(plan.bind(**arrays), name="n")
    graph.build().launch()
    hawk.synchronize()
    return {k: cp.asnumpy(v) for k, v in arrays.items()}


@pytest.mark.gpu
def test_the_fused_scatter_on_the_device_equals_the_host(host, cache_dir):
    kernel, n = _scatter(6), 33
    u, e = _scatter_inputs(n, 6)
    want = np.zeros((15, n))
    hawk.run(host("scatter_6", kernel, kernel.sinks), u=u, e=e, o=want)
    got = _on_device(kernel, cache_dir, u=u, e=e, o=np.zeros((15, n)))["o"]
    assert np.array_equal(got, want)


@pytest.mark.gpu
def test_an_in_place_store_on_the_device_equals_the_host(host, cache_dir):
    kernel, n, width = _stores(256), 33, 256
    rng = np.random.default_rng(3)
    v, e = rng.normal(size=(width, n)), rng.normal(size=n)
    j = rng.integers(0, width, size=n).astype(np.int32)
    want = np.zeros((width, n))
    hawk.run(host("store_256", kernel, kernel.sinks), v=v, j=j, e=e, o=want)
    got = _on_device(kernel, cache_dir, v=v, j=j, e=e, o=np.zeros((width, n)))["o"]
    assert np.array_equal(got, want)


def test_independent_clamped_stores_host_run_equals_the_interpreter(host):
    n = 9
    rng = np.random.default_rng(4)
    u, e = rng.integers(-2, 14, size=n).astype(float), rng.integers(1, 5, size=n).astype(float)
    out = np.zeros((15, n))
    hawk.run(host("clamped", _clamped, _clamped.sinks), u=u, e=e, o=out)
    for s in range(n):
        want = evaluate(_clamped.sinks, {"u": u[s], "e": e[s]})
        assert np.array_equal(out[:, s], np.asarray(want["o"])), s


@hawk.kernel
def _int32_pick(theta: Vector[4], x: Scalar, y: Mutable[Scalar]):
    """A Vector read at an index computed as a 32-bit integer value (m.int32)."""
    y = theta[m.int32(x * 3.0)]


def test_an_int32_value_indexes_a_vector(host):
    """``m.int32`` makes an int plane value; it addresses a Vector component as an ``i32`` index does."""
    n = 5
    rng = np.random.default_rng(7)
    theta = rng.normal(size=(4, n))
    x = np.array([0.0, 0.4, 0.7, 1.0, 0.2])
    y = np.zeros(n)
    kernel = host("int32_pick", _int32_pick, _int32_pick.sinks)
    hawk.run(kernel, **declared(kernel, {"theta": theta, "x": x}), y=y)
    for s in range(n):
        assert y[s] == theta[int(x[s] * 3.0), s], s


def _read_of_the_accumulated_plane():
    """``vjp(_double_hit)`` plus a runtime-index read of the accumulated ``bar_theta`` value."""
    sinks = vjp(_double_hit, wrt=("theta",))
    read = next(
        n for n in _ir.post_order(_double_hit.sinks)
        if _ir.is_a(n, "op") and n.field_text("kind") == "component_at"
    )
    picked = _ir.make("component_at", (sinks[0].operands[0], read.operands[1]))
    return (*sinks, _ir.make_node("Assign", _ir.TensorType((), "f64"), (picked,), name="z"))


def test_a_runtime_read_of_an_accumulated_plane_equals_the_interpreter(host):
    sinks = _read_of_the_accumulated_plane()
    n = 6
    rng = np.random.default_rng(3)
    theta, seed = rng.normal(size=(4, n)), rng.normal(size=n)
    j = np.array([2, 2, 0, 1, 3, 2], dtype=np.int32)
    kernel = host("double_hit_read", _double_hit, sinks)
    bar, z = np.zeros((4, n)), np.zeros(n)
    hawk.run(kernel, **declared(kernel, {"theta": theta, "j": j, "bar_y": seed}), bar_theta=bar, z=z)
    for s in range(n):
        want = evaluate(sinks, {"theta": theta[:, s], "j": int(j[s]), "bar_y": seed[s]})
        assert z[s] == want["z"] and np.array_equal(bar[:, s], want["bar_theta"]), s
