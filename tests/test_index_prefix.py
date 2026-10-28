# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``Kind(..., lanes=IndexSet(prefix=True))``: the first ``count`` lanes run, ``count`` read per run.

One build bound at ``n`` lanes runs any count ``m <= n``: lane ``t < m`` runs lane ``t``, no map is
read, and the count is the one-element plane ``active_count``. Lanes at or past the count are not
touched (their outputs keep their values); a ``hawk.sum`` and a keyed sum add the live lanes only;
``vjp`` under the same kind carries the count. Rows run on the host and (``gpu`` mark) the device."""

from __future__ import annotations

import _deploy as L
import numpy as np
import pytest
from conftest import sidecar_of

import hawk
from hawk import HawkError, Index, Kernel, Mutable, Scalar
from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.diff import vjp
from hawk.ext import IndexSet, Kind

PRE = Kind("prefix", lanes=IndexSet(prefix=True))
N = 200
COUNTS = (N, 77, 33, 1, 0)
TARGETS = ["host", pytest.param("cuda", marks=pytest.mark.gpu)]


def _square(x: Scalar, y: Mutable[Scalar]):
    y = x * x + 1.0


def _loss(x: Scalar, total: Mutable[Scalar]):
    total = hawk.sum(x * x)


def _bin(x: Scalar, key: Index, acc: Mutable[Scalar]):
    hawk.scatter(acc)[key] += x


@pytest.fixture(scope="module")
def built(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("prefix")
    out = {}
    for name, fn in (("square", _square), ("loss", _loss), ("bin", _bin)):
        pre = hawk.kernel(fn, kind=PRE)
        dense = hawk.kernel(fn)
        out[name] = (build_bundle([pre], root / f"{name}_pre", targets=("cuda", "host"), cache_dir=cache_dir,
                                  host_profile="native", kind=PRE),
                     build_bundle([dense], root / f"{name}_dense", targets=("cuda", "host"), cache_dir=cache_dir,
                                  host_profile="native"), pre.name)
    loss, plain = hawk.kernel(_loss, kind=PRE), hawk.kernel(_loss)
    grad = Kernel("loss_grad", vjp(loss, wrt=("x",)), {}, kind=PRE)
    dense = Kernel("loss_grad", vjp(plain, wrt=("x",)), {})
    out["grad"] = (build_bundle([loss, grad], root / "grad_pre", targets=("cuda", "host"), cache_dir=cache_dir,
                                host_profile="native", kind=PRE),
                   build_bundle([plain, dense], root / "grad_dense", targets=("cuda", "host"), cache_dir=cache_dir,
                                host_profile="native"), "loss_grad")
    return out


def _go(bundle, name, target, planes, outs):
    """One launch; returns the planes named ``outs`` afterwards."""
    planes = {k: (v.copy() if k in outs else v) for k, v in planes.items()}
    if target == "host":
        rt.run(rt.load(bundle.directory, name, sidecar_of(bundle, name)), **planes)
        return {k: planes[k] for k in outs}
    import cupy as cp
    import eagle.exec as eexec
    from eagle import plan as eplan
    from hawk import _cells

    cells = sidecar_of(bundle, name).get("cells") or {}
    rows = -(-N // 32)
    rows = -(-N // 32)
    dev = {k: cp.asarray(v) for k, v in planes.items() if k not in cells}
    for c in cells:                     # a cell is rows on the device (stale: 9.0 in every row), the finish is the caller's
        dev[c], dev[c + "_comp"] = cp.full(rows, 9.0), cp.zeros(rows)
    plugin = L.device_plugin(bundle.directory, name, sidecar_of(bundle, name))
    eplan.plan(plugin, structure=eexec.DeviceKernel).bind(**dev).launch()
    cp.cuda.get_current_stream().synchronize()
    out = {k: cp.asnumpy(dev[k]) for k in outs if k not in cells}
    for c in cells:                     # the plain finish over every row: a row past the count holds the identity
        out[c] = np.asarray(_cells.finish(cp.asnumpy(dev[c]), cp.asnumpy(dev[c + "_comp"]), cells[c]["op"])).reshape(1)
    return out


def _count(m):
    return np.array([m], np.uint32)


def test_the_sidecar_declares_the_prefix_form(built):
    bundle, _, name = built["square"]
    assert sidecar_of(bundle, name)["launch"]["index_set"] == {"width": 32, "prefix": True, "count": "active_count"}
    assert "active_map" not in {n for _, n in sidecar_of(bundle, name)["arg_spec"]}
    assert "active_count" in {n for _, n in sidecar_of(bundle, name)["arg_spec"]}


@pytest.mark.parametrize("target", TARGETS)
def test_a_prefix_runs_the_first_lanes_and_leaves_the_rest_alone(built, target):
    pre, dense, name = built["square"]
    x = np.random.default_rng(1).normal(size=N)
    prior = np.linspace(-5, -4, N)
    for m in COUNTS:
        got = _go(pre, name, target, {"x": x, "y": prior, "active_count": _count(m)}, ("y",))["y"]
        want = _go(dense, name, target, {"x": x[:m].copy(), "y": prior[:m].copy()}, ("y",))["y"] if m else prior[:0]
        np.testing.assert_array_equal(got[:m], want)
        np.testing.assert_array_equal(got[m:], prior[m:])         # lanes at or past the count are untouched


@pytest.mark.parametrize("target", TARGETS)
def test_a_sum_adds_the_live_lanes_only(built, target):
    pre, dense, name = built["loss"]
    x = np.random.default_rng(2).normal(size=N)
    for m in COUNTS:
        got = _go(pre, name, target, {"x": x, "total": np.full(1, 9.0), "active_count": _count(m)}, ("total",))["total"]
        np.testing.assert_allclose(got[0], float(x[:m] @ x[:m]), rtol=1e-13, atol=0)


def test_the_host_reads_the_count_each_run_from_one_loaded_kernel(built):
    pre, _, name = built["loss"]
    kern = rt.load(pre.directory, name, sidecar_of(pre, name))
    x = np.random.default_rng(3).normal(size=N)
    for m in (N, 5, 100, 0, 64, 65, N):      # a large count after a small one: no stale row is summed
        total = np.zeros(1)
        rt.run(kern, x=x, total=total, active_count=_count(m))
        np.testing.assert_allclose(total[0], float(x[:m] @ x[:m]), rtol=1e-13, atol=0)


@pytest.mark.parametrize("target", TARGETS)
def test_a_keyed_sum_adds_the_live_lanes_only(built, target):
    pre, _, name = built["bin"]
    rng = np.random.default_rng(4)
    x, key = rng.normal(size=N), rng.integers(0, 6, N).astype(np.int64)
    for m in COUNTS:
        got = _go(pre, name, target, {"x": x, "key": key, "acc": np.zeros(6), "active_count": _count(m)}, ("acc",))["acc"]
        want = np.zeros(6)
        np.add.at(want, key[:m], x[:m])
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("target", TARGETS)
def test_the_vjp_carries_the_count(built, target):
    pre, _, name = built["grad"]
    x = np.random.default_rng(5).normal(size=N)
    for m in COUNTS:
        got = _go(pre, name, target, {"x": x, "bar_total": np.float64(0.5), "bar_x": np.zeros(N),
                                      "active_count": _count(m)}, ("bar_x",))["bar_x"]
        want = np.zeros(N)
        want[:m] = 2 * x[:m] * 0.5
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=0)


def test_a_count_above_the_bound_lanes_is_refused_by_name(built):
    pre, _, name = built["square"]
    kern = rt.load(pre.directory, name, sidecar_of(pre, name))
    with pytest.raises(HawkError, match=rf"active_count is {N + 1}, but the kernel is bound at {N} lanes"):
        rt.run(kern, x=np.zeros(N), y=np.zeros(N), active_count=_count(N + 1))


def test_a_prefix_has_no_width_and_its_key_is_its_own():
    with pytest.raises(HawkError, match="no width"):
        IndexSet(prefix=True, width=64)
    assert repr(IndexSet(prefix=True)) != repr(IndexSet())
