# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``k(x, a, b)``: a kernel called as a function runs one launch through the plan
eagle deploys for it — the same artifact ``eagle.deploy(k).run`` uses."""

from __future__ import annotations

import sys
import threading

import numpy as np
import pytest

eagle = pytest.importorskip("eagle")

import hawk  # noqa: E402
from hawk import Grid, Index, Mutable, Param, Scalar, Table, Vector  # noqa: E402
from hawk.compile import drivers  # noqa: E402


@pytest.fixture
def private_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HAWK_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


@pytest.fixture
def compiler_calls(monkeypatch):
    calls = []
    real = drivers.run_compiler
    monkeypatch.setattr(drivers, "run_compiler",
                        lambda argv, what: calls.append(what) or real(argv, what))
    return calls


def _scale():
    @hawk.kernel
    def scale(x: Scalar, a: Param, b: Param) -> Scalar:
        return a * x + b

    return scale


def _bundles(cache):
    return sorted(p.name for p in cache.rglob("*") if p.suffix == ".so")


def test_kernel_call_cpu(private_cache, compiler_calls):
    k = _scale()
    x = np.arange(8.0)
    y = k(x, 2.0, 1.0)
    np.testing.assert_array_equal(y, 2.0 * x + 1.0)
    built, first = _bundles(private_cache), len(compiler_calls)
    assert built and first >= 1
    twin = eagle.deploy(k).run(x=x, a=2.0, b=1.0)
    np.testing.assert_array_equal(twin, y)
    assert _bundles(private_cache) == built and len(compiler_calls) == first
    assert k(x, 3.0, 0.0).tolist() == (3.0 * x).tolist()
    assert len(compiler_calls) == first


def test_kernel_call_positional_and_keyword(private_cache):
    k = _scale()
    x = np.arange(4.0)
    expect = 2.0 * x + 1.0
    np.testing.assert_array_equal(k(x, 2.0, 1.0), expect)
    np.testing.assert_array_equal(k(x, a=2.0, b=1.0), expect)
    np.testing.assert_array_equal(k(x=x, a=2.0, b=1.0), expect)
    with pytest.raises(ValueError, match="scale"):
        k(x, 2.0)


def test_kernel_call_returns_the_tree(private_cache):
    @hawk.kernel
    def kick(p: Vector[3], v: Vector[3], dt: Param):
        return {"pos": p + dt * v, "vel": v - dt * p}

    p, v = np.ones((3, 8)), np.full((3, 8), 2.0)
    got = kick(p, v, 0.5)
    assert sorted(got) == ["pos", "vel"]
    np.testing.assert_allclose(got["pos"], p + 0.5 * v)
    np.testing.assert_allclose(got["vel"], v - 0.5 * p)


def test_kernel_call_missing_eagle(monkeypatch):
    monkeypatch.setitem(sys.modules, "eagle", None)
    with pytest.raises(ImportError, match=r"scale\(\.\.\.\) runs through eagle: pip install raptor-eagle"):
        _scale()(np.arange(4.0), 2.0, 1.0)


def test_kernel_call_threads_first_call_builds_once(private_cache, monkeypatch):
    k = _scale()
    deploys, real = [], eagle.deploy
    monkeypatch.setattr(eagle, "deploy", lambda *a, **kw: deploys.append(1) or real(*a, **kw))
    x, out, gate = np.arange(16.0), [], threading.Barrier(6)

    def work():
        gate.wait()
        out.append(k(x, 2.0, 1.0))

    threads = [threading.Thread(target=work) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(deploys) == 1 and len(out) == 6
    assert all(np.array_equal(o, 2.0 * x + 1.0) for o in out)


def _binned(n):
    """One body, ``n`` bins: every size has the same walk and name, and differs
    only in the declared extent of ``b``."""
    @hawk.kernel
    def binned(lane: Grid["e", "a":3], conn: Table[Index], b: Mutable["node":n]):
        hawk.scatter(b)[conn.at(lane.flat)] += 1.0

    return binned


def _check_binned(xp):
    kernels = {n: _binned(n) for n in (4, 9)}
    for n in (4, 9, 4):
        conn = np.arange(6, dtype=np.int32) % n
        out = eagle.deploy(kernels[n]).run(conn=xp.asarray(conn), e=2)
        want = np.zeros(n)
        np.add.at(want, conn, 1.0)
        got = out.get() if hasattr(out, "get") else out
        assert got.shape == (n,), f"size {n} was served another size's build"
        np.testing.assert_array_equal(got, want)


def test_a_factory_builds_each_size_cpu(private_cache):
    _check_binned(np)


def test_a_redefined_kernel_runs_its_new_body(private_cache):
    @hawk.kernel
    def scale(x: Scalar, a: Param, b: Param) -> Scalar:
        return a * x + b

    x = np.arange(4.0)
    np.testing.assert_array_equal(scale(x, 2.0, 1.0), 2.0 * x + 1.0)

    @hawk.kernel
    def scale(x: Scalar, a: Param, b: Param) -> Scalar:  # noqa: F811
        return a * x - b

    np.testing.assert_array_equal(scale(x, 2.0, 1.0), 2.0 * x - 1.0)


def _device():
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not _device(), reason="needs a CUDA device")
def test_kernel_call_gpu(private_cache, compiler_calls):
    import cupy as cp

    k = _scale()
    x = cp.arange(8.0)
    y = k(x, 2.0, 1.0)
    assert isinstance(y, (cp.ndarray, eagle.Array))     # eagle hands the result back as its own array unless returns= asks for cupy's
    y = cp.asarray(y)
    cp.testing.assert_array_equal(y, 2.0 * x + 1.0)
    first = len(compiler_calls)
    twin = cp.asarray(eagle.deploy(k).run(x=x, a=2.0, b=1.0))
    cp.testing.assert_array_equal(twin, y)
    assert len(compiler_calls) == first


@pytest.mark.gpu
@pytest.mark.skipif(not _device(), reason="needs a CUDA device")
def test_a_factory_builds_each_size_gpu(private_cache):
    import cupy as cp

    _check_binned(cp)
