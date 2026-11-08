# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A runtime-index read does not copy a whole vector into an ``aether::Item``.

A read of a mutable plane's row reads the plane in place when no store to it comes first (the stores are the
final phase, after every node); a read of an element-wise value (``(v * s)[k]``) computes that one element from the operands' own elements. The host values
are bitwise the interpreter's under the exact ``native`` profile; the device text loses the local ``Item`` too."""

from __future__ import annotations

import time

import numpy as np
import pytest
from _eval import evaluate
from conftest import sidecar_of

import hawk
import hawk.math as hm
from hawk import Int32, Matrix, Mutable, Scalar, Vector
from hawk.diff import jvp, vjp
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
def _held512(m: Mutable[Vector[512]], k: Int32, y: Mutable[Scalar]):
    y = m[k]
    m = m * 2.0


@hawk.kernel
def _free512(m: Mutable[Vector[512]], k: Int32, y: Mutable[Scalar]):
    y = hm.real(k)
    m = m * 2.0


@hawk.kernel
def _after(m: Mutable[Vector[N]], k: Int32, y: Mutable[Scalar]):
    m = m * 2.0
    y = m[k]


@hawk.kernel
def _static(v: Vector[N], m: Mutable[Vector[N]], k: Int32, j: Int32, y: Mutable[Scalar]):
    y = m[k] + m[j] + m[3]
    m = v


@hawk.kernel
def _looped(m: Mutable[Vector[N]], y: Mutable[Scalar]):
    t = 0.0
    for c in range(N):
        t = t + m[c] * (c + 1.0)
    y = t
    m = m * 2.0


@hawk.kernel
def _matrix(m: Mutable[Matrix[3, 3]], k: Int32, y: Mutable[Scalar]):
    y = m[1, 2]
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
def _derived(v: Vector[N], k: Int32, y: Mutable[Scalar], m: Mutable[Vector[N]]):
    m = v * 2.0
    y = m[k]


_JVP = hawk.Kernel("derived_jvp", jvp(_derived, wrt=("v",)))
_VJP = hawk.Kernel("derived_vjp", vjp(_derived, wrt=("v",)))


@hawk.kernel
def _plain(v: Vector[N], k: Int32, y: Mutable[Scalar]):
    y = v[k]


def _text(kernel, target):
    return emit_source(kernel, mode="float64", targets=(target,)).sources[target].text


@pytest.mark.parametrize("target", ["host", "cuda"])
def test_a_mutable_read_before_its_store_reads_the_plane_in_place(target):
    lines = _text(_held, target).splitlines()
    reads = [n for n, ln in enumerate(lines) if "mut_m[i].at(" in ln]
    store = [n for n, ln in enumerate(lines) if ln.strip().startswith("mut_m[i] =")]
    text = "\n".join(lines)
    assert len(reads) == 1 and len(store) == 1 and reads[0] < store[0], text
    assert text.count("mut_m_i = mut_m[i].get()") == 1 and "hawk_it" not in text, text


@pytest.mark.parametrize("target", ["host", "cuda"])
def test_a_row_only_read_by_element_is_never_copied(target):
    text = _text(_static, target)
    assert "mut_m[i].get()" not in text and text.count("mut_m[i].at(") == 2, text


@pytest.mark.parametrize("target", ["host", "cuda"])
def test_a_rank_two_mutable_keeps_its_item(target):
    assert "mut_m[i].at(" not in _text(_matrix, target)


@pytest.fixture(scope="module")
def host(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("itemcopies")
    kernels = (_held, _held_free, _after, _static, _looped, _matrix, _scaled, _chain, _unary, _plain, _derived,
               _JVP, _VJP, _held512, _free512)
    b = build_bundle(list(kernels), root, targets=("host",), cache_dir=cache_dir,
                     host_profile="native", mode="float64")
    return lambda kernel: rt.load(b.directory, kernel.name, sidecar_of(b, kernel.name))


def _inputs(n, seed=5):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(N, n)), rng.normal(size=(N, n)), rng.normal(size=n),
            rng.integers(0, N, size=n).astype(np.int32))


def _mutable_cases(n):
    v, w, _s, k = _inputs(n)
    return {_held: dict(m=v, k=k), _after: dict(m=v, k=k), _static: dict(v=w, m=v, k=k, j=np.roll(k, 1)),
            _looped: dict(m=v)}


@pytest.mark.parametrize("kernel", [_held, _after, _static, _looped], ids=lambda k: k.name)
def test_a_mutable_read_stores_what_the_interpreter_stores_bitwise(host, kernel):
    n = 9
    cases = _mutable_cases(n)[kernel]
    planes = {name: a.copy() for name, a in cases.items()}
    y = np.zeros(n)
    rt.run(host(kernel), **planes, y=y)
    for j in range(n):
        env = {name: (a[:, j] if a.ndim == 2 else a[j]) for name, a in cases.items()}
        want = evaluate(kernel.sinks, env)
        assert y[j] == want["y"] and np.array_equal(planes["m"][:, j], want["m"]), (kernel.name, j)


def test_the_derivatives_equal_the_interpreters_bitwise(host):
    n = 9
    v, dv, s, k = _inputs(n)
    dm, dy, bar = np.zeros((N, n)), np.zeros(n), np.zeros((N, n))
    rt.run(host(_JVP), dot_m=dm, dot_y=dy, dot_v=dv, k=k)
    rt.run(host(_VJP), k=k, bar_y=s, bar_m=np.zeros((N, n)), bar_v=bar)
    for j in range(n):
        want = evaluate(_JVP.sinks, {"dot_v": dv[:, j], "k": int(k[j])})
        assert dy[j] == want["dot_y"] and np.array_equal(dm[:, j], want["dot_m"]), j
        want = evaluate(_VJP.sinks, {"k": int(k[j]), "bar_y": s[j], "bar_m": np.zeros(N)})
        assert np.array_equal(bar[:, j], want["bar_v"]), j


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
    zeros = np.zeros((N, n))
    cases = {_chain: (dict(v=v, w=w, s=s, k=k), dict(y=np.zeros(n))),
             _held: (dict(k=k), dict(m=v, y=np.zeros(n))),
             _after: (dict(k=k), dict(m=v, y=np.zeros(n))),
             _static: (dict(v=w, k=k, j=np.roll(k, 1)), dict(m=v, y=np.zeros(n))),
             _looped: ({}, dict(m=v, y=np.zeros(n))),
             _JVP: (dict(dot_v=w, k=k), dict(dot_y=np.zeros(n), dot_m=zeros)),
             _VJP: (dict(k=k, bar_y=s, bar_m=zeros), dict(bar_v=zeros))}
    for kernel, (ins, outs) in cases.items():
        want = {name: a.copy() for name, a in outs.items()}
        rt.run(host(kernel), **ins, **want)
        built = [_derived, kernel] if kernel in (_JVP, _VJP) else [kernel]      # a derivative ships with its primal
        plan = eagle.plan.auto(built, targets=("cuda",), scalar_type="float64")[-1]
        arrays = {name: cp.asarray(a) for name, a in (ins | outs).items()}
        plan.bind(**arrays).launch()
        cp.cuda.runtime.deviceSynchronize()
        for name, a in want.items():
            np.testing.assert_allclose(cp.asnumpy(arrays[name]), a, rtol=1e-14, err_msg=f"{kernel.name}.{name}")


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
@pytest.mark.parametrize("width, read, free", [(128, _held, _held_free), (512, _held512, _free512)])
def test_a_read_of_a_mutable_plane_costs_what_the_kernel_without_it_costs(host, width, read, free):
    n = 20000
    rng = np.random.default_rng(5)
    k = rng.integers(0, width, size=n).astype(np.int32)
    m = rng.normal(size=(width, n))
    with_read = _best(read, host, dict(m=m.copy(), k=k))
    without = _best(free, host, dict(m=m.copy(), k=k))
    print(f"N={width}: with read {with_read / n * 1e9:.1f} ns, without {without / n * 1e9:.1f} ns, "
          f"ratio {with_read / without:.2f}")
    assert with_read < 1.1 * without, (with_read, without)
