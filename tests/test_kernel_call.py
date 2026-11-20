# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``k(x, a, b)``: a kernel called as a function runs one launch through the plan
``hawk.deploy`` makes for it, kept on the kernel: the same artifact ``hawk.deploy(k).run`` uses."""

from __future__ import annotations

import sys
import threading

import numpy as np
import pytest

import hawk
from hawk import Grid, HawkError, Index, Mutable, Param, Scalar, Table, Vector
from hawk import plans as plans_module
from hawk.artifact import cache_stats


@pytest.fixture
def private_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HAWK_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


def _tool_runs() -> int:
    """How many times the OSPREYs ran a tool for the store under ``$HAWK_CACHE_DIR``, as their counters state."""
    return sum(figures["tool_runs"] for figures in cache_stats().values())


def _scale():
    @hawk.kernel
    def scale(x: Scalar, a: Param, b: Param) -> Scalar:
        return a * x + b

    return scale


def _bundles(cache):
    return sorted(p.name for p in cache.rglob("*") if p.suffix == ".so")


def test_kernel_call_cpu(private_cache):
    k = _scale()
    x = np.arange(8.0)
    y = k(x, 2.0, 1.0)
    np.testing.assert_array_equal(y, 2.0 * x + 1.0)
    built, first = _bundles(private_cache), _tool_runs()
    assert built and first >= 1
    twin = hawk.deploy(k).run(x=x, a=2.0, b=1.0)
    np.testing.assert_array_equal(twin, y)
    assert _bundles(private_cache) == built and _tool_runs() == first
    assert k(x, 3.0, 0.0).tolist() == (3.0 * x).tolist()
    assert _tool_runs() == first


def test_kernel_call_positional_and_keyword(private_cache):
    k = _scale()
    x = np.arange(4.0)
    expect = 2.0 * x + 1.0
    np.testing.assert_array_equal(k(x, 2.0, 1.0), expect)
    np.testing.assert_array_equal(k(x, a=2.0, b=1.0), expect)
    np.testing.assert_array_equal(k(x=x, a=2.0, b=1.0), expect)
    with pytest.raises(HawkError, match="scale"):
        k(x, 2.0)
    with pytest.raises(HawkError, match=r"scale\(\): argument 'a' is given both by position and by name"):
        k(x, 2.0, 1.0, a=2.0)
    with pytest.raises(HawkError, match=r"scale\(\): 5 positional arguments, but the kernel declares 4 planes"):
        k(x, 2.0, 1.0, np.zeros(4), 5.0)


def test_kernel_call_returns_the_tree(private_cache):
    @hawk.kernel
    def kick(p: Vector[3], v: Vector[3], dt: Param):
        return {"pos": p + dt * v, "vel": v - dt * p}

    p, v = np.ones((3, 8)), np.full((3, 8), 2.0)
    got = kick(p, v, 0.5)
    assert sorted(got) == ["pos", "vel"]
    np.testing.assert_allclose(got["pos"], p + 0.5 * v)
    np.testing.assert_allclose(got["vel"], v - 0.5 * p)


def test_a_kernel_call_needs_no_eagle(private_cache, monkeypatch):
    monkeypatch.setitem(sys.modules, "eagle", None)
    np.testing.assert_array_equal(_scale()(np.arange(4.0), 2.0, 1.0), 2.0 * np.arange(4.0) + 1.0)


def test_kernel_call_threads_first_call_builds_once(private_cache, monkeypatch):
    k = _scale()
    deploys, real = [], plans_module.deploy
    monkeypatch.setattr(plans_module, "deploy", lambda *a, **kw: deploys.append(1) or real(*a, **kw))
    x, out, gate = np.arange(16.0), [], threading.Barrier(6)

    def work():
        gate.wait()
        out.append(k(x, 2.0, 1.0))

    threads = [threading.Thread(target=work) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(deploys) == 1 and len(out) == 6
    assert all(np.array_equal(o, 2.0 * x + 1.0) for o in out)


def test_every_run_form_ends_in_one_plan_function(private_cache, monkeypatch, tmp_path):
    """A call of the kernel, of its plan, ``Plan.run``, ``Plan.bind(...).launch()`` and ``hawk.run`` bind through the one
    ``Plan.bound``, each once, whatever the rest of the form does."""
    k = _scale()
    plan = hawk.deploy(k)
    loaded = hawk.load(plan.kernel.path.parent, "scale")
    seen = []
    real = plans_module.Plan.bound
    monkeypatch.setattr(plans_module.Plan, "bound", lambda self, *a, **kw: seen.append(self) or real(self, *a, **kw))
    x = np.arange(4.0)
    forms = {
        "kernel call": lambda: k(x, 2.0, 1.0),
        "plan call": lambda: plan(x, 2.0, 1.0),
        "Plan.run": lambda: plan.run(x=x, a=2.0, b=1.0),
        "bind and launch": lambda: plan.bind(x=x, a=2.0, b=1.0, out=np.zeros(4)).launch(),
        "hawk.run": lambda: hawk.run(loaded, x=x, a=2.0, b=1.0, out=np.zeros(4)),
    }
    for name, form in forms.items():
        before = len(seen)
        form()
        assert len(seen) == before + 1, name


def test_a_call_returns_the_one_output_or_a_dict_of_them(private_cache):
    @hawk.kernel
    def pair(x: Scalar, y: Mutable[Scalar], z: Mutable[Scalar]):
        y = 2.0 * x
        z = x + 1.0

    x = np.arange(4.0)
    got = pair(x)
    assert sorted(got) == ["y", "z"]
    np.testing.assert_array_equal(got["y"], 2.0 * x)
    np.testing.assert_array_equal(got["z"], x + 1.0)
    keep = np.zeros(4)
    got = pair(x, y=keep)
    assert got["y"] is keep and keep.tolist() == (2.0 * x).tolist()


def test_a_value_a_kernel_reads_before_it_writes_must_be_given(private_cache):
    @hawk.kernel
    def accumulate(x: Scalar, total: Mutable[Scalar]):
        total = total + x

    x = np.arange(4.0)
    with pytest.raises(HawkError, match=r"accumulate\(\): 'total' is updated from its previous value.*pass total=\.\.\."):
        accumulate(x)
    np.testing.assert_array_equal(accumulate(x, total=np.ones(4)), x + 1.0)


def _binned(n):
    """One body, ``n`` bins: every size has the same walk and name, and differs
    only in the declared extent of ``b``."""
    @hawk.kernel
    def binned(lane: Grid["e", "a":3], conn: Table[Index], b: Mutable["node":n]):
        hawk.scatter(b)[conn.at(lane.flat)] += 1.0

    return binned


def _check_binned(xp, deploy):
    kernels = {n: _binned(n) for n in (4, 9)}
    for n in (4, 9, 4):
        conn = np.arange(6, dtype=np.int32) % n
        out = deploy(kernels[n]).run(conn=xp.asarray(conn), e=2)
        want = np.zeros(n)
        np.add.at(want, conn, 1.0)
        got = out.get() if hasattr(out, "get") else out
        assert got.shape == (n,), f"size {n} was served another size's build"
        np.testing.assert_array_equal(got, want)


def test_a_factory_builds_each_size_cpu(private_cache):
    _check_binned(np, hawk.deploy)


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
def test_kernel_call_gpu(private_cache):
    import cupy as cp

    k = _scale()
    x = cp.arange(8.0)
    y = k(x, 2.0, 1.0)
    assert isinstance(y, cp.ndarray)        # an output made for device inputs is read back by the library the inputs come from
    cp.testing.assert_array_equal(y, 2.0 * x + 1.0)
    first = _tool_runs()
    twin = hawk.deploy(k).run(x=x, a=2.0, b=1.0)
    cp.testing.assert_array_equal(twin, y)
    assert _tool_runs() == first


@pytest.mark.gpu
@pytest.mark.skipif(not _device(), reason="needs a CUDA device")
def test_a_factory_builds_each_size_gpu(private_cache):
    import cupy as cp

    _check_binned(cp, hawk.deploy)
