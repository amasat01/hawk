# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A runtime-index read does not copy a whole vector into an ``aether::Item``.

A read of a value that is already a named ``Item`` (a mutable plane's row) reads it in place; a read of an
element-wise value (``(v * s)[k]``) computes that one element from the operands' own elements. The host values
are bitwise the interpreter's under the exact ``native`` profile; the device text loses the local ``Item`` too."""

from __future__ import annotations

import time

import numpy as np
import pytest
from _eval import evaluate
from conftest import sidecar_of

import hawk
import hawk.math as hm
from hawk import Int32, Mutable, Scalar, Vector
from hawk.artifact import build_bundle, emit_source
from hawk import runtime as rt

N = 128


@hawk.kernel
def _held(m: Mutable[Vector[N]], k: Int32, y: Mutable[Scalar]):
    y = m[k]
    m = m * 2.0


@hawk.kernel
def _held_free(m: Mutable[Vector[N]], k: Int32, y: Mutable[Scalar]):
    y = hm.real(k)
    m = m * 2.0


@hawk.kernel
def _scaled(v: Vector[N], s: Scalar, k: Int32, y: Mutable[Scalar]):
    u = v * s
    y = u[k]


@hawk.kernel
def _chain(v: Vector[N], w: Vector[N], s: Scalar, k: Int32, y: Mutable[Scalar]):
    u = v * s + w
    y = u[k]


@hawk.kernel
def _unary(v: Vector[N], k: Int32, y: Mutable[Scalar]):
    u = hm.sin(v)
    y = u[k]


@hawk.kernel
def _plain(v: Vector[N], k: Int32, y: Mutable[Scalar]):
    y = v[k]


def _text(kernel, target):
    return emit_source(kernel, mode="float64", targets=(target,)).sources[target].text


@pytest.mark.parametrize("target", ["host", "cuda"])
def test_a_named_item_is_read_in_place(target):
    text = _text(_held, target)
    assert "mut_m_i(" in text and "hawk_it0 = mut_m_i" not in text, text


@pytest.mark.parametrize("kernel", [_scaled, _chain, _unary], ids=lambda k: k.name)
@pytest.mark.parametrize("target", ["host", "cuda"])
def test_an_elementwise_read_builds_no_vector(kernel, target):
    text = _text(kernel, target)
    assert "hawk_it" not in text and "vin_v[i].at(" in text, text


@pytest.fixture(scope="module")
def host(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("itemcopies")
    kernels = (_held, _held_free, _scaled, _chain, _unary, _plain)
    b = build_bundle(list(kernels), root, targets=("host",), cache_dir=cache_dir,
                     host_profile="native", mode="float64")
    return lambda kernel: rt.load(b.directory, kernel.name, sidecar_of(b, kernel.name))


def _inputs(n, seed=5):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(N, n)), rng.normal(size=(N, n)), rng.normal(size=n),
            rng.integers(0, N, size=n).astype(np.int32))


def test_the_host_values_are_the_interpreters_bitwise(host):
    n = 9
    v, w, s, k = _inputs(n)
    cases = {_scaled: dict(v=v, s=s, k=k), _chain: dict(v=v, w=w, s=s, k=k), _unary: dict(v=v, k=k)}
    for kernel, planes in cases.items():
        y = np.zeros(n)
        rt.run(host(kernel), **planes, y=y)
        for j in range(n):
            env = {name: (a[:, j] if a.ndim == 2 else a[j]) for name, a in planes.items()}
            assert y[j] == evaluate(kernel.sinks, env)["y"], (kernel.name, j)
    y, mm = np.zeros(n), v.copy()
    rt.run(host(_held), m=mm, k=k, y=y)
    for j in range(n):
        want = evaluate(_held.sinks, {"m": v[:, j], "k": int(k[j])})
        assert y[j] == want["y"] and np.array_equal(mm[:, j], want["m"]), j


@pytest.mark.gpu
def test_the_device_value_equals_the_host(host):
    import cupy as cp
    import eagle
    import eagle.plan

    n = 33
    v, w, s, k = _inputs(n)
    want = np.zeros(n)
    rt.run(host(_chain), v=v, w=w, s=s, k=k, y=want)
    plan = eagle.plan.auto([_chain], targets=("cuda",), scalar_type="float64")[0]
    arrays = {name: cp.asarray(a) for name, a in dict(v=v, w=w, s=s, k=k, y=np.zeros(n)).items()}
    graph = eagle.Graph()
    graph.add(plan.bind(**arrays), name="n")
    graph.build().launch()
    cp.cuda.runtime.deviceSynchronize()
    np.testing.assert_allclose(cp.asnumpy(arrays["y"]), want, rtol=1e-14)


def _best(kernel, load, planes, reps=40):
    fn = load(kernel)
    out = np.zeros(planes["k"].size)
    fresh = planes.get("m")
    best = np.inf
    for _ in range(reps + 1):
        if fresh is not None:
            planes["m"] = fresh.copy()      # the kernel doubles it: keep every run on finite values
        t0 = time.perf_counter()
        rt.run(fn, **planes, y=out)
        best = min(best, time.perf_counter() - t0)
    return best


@pytest.mark.wall_clock
def test_an_elementwise_read_costs_what_an_input_read_costs(host):
    n = 20000
    v, w, s, k = _inputs(n)
    scaled = _best(_scaled, host, dict(v=v, s=s, k=k))
    plain = _best(_plain, host, dict(v=v, k=k))
    print(f"scaled {scaled / n * 1e9:.1f} ns, input read {plain / n * 1e9:.1f} ns")
    assert scaled < 2 * plain, (scaled, plain)


@pytest.mark.wall_clock
def test_a_read_of_a_held_item_costs_under_twice_the_kernel_without_it(host):
    n = 20000
    v, w, s, k = _inputs(n)
    with_read = _best(_held, host, dict(m=v.copy(), k=k))
    without = _best(_held_free, host, dict(m=v.copy(), k=k))
    print(f"with read {with_read / n * 1e9:.1f} ns, without {without / n * 1e9:.1f} ns")
    # the row lives on the stack for a runtime index: 1.7x here (1.8x with the second copy)
    assert with_read < 2.0 * without, (with_read, without)
