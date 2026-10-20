# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Reductions: a value's own axes, and ``hawk.all_samples(x)`` cells.

Rows (the ``cuda`` arm is marked ``gpu``):

* **own axes** — ``.sum/.max/.min/.mean(axis)``, ``hawk.math.sum/mean`` and the
  views ``.components / .rows / .columns`` on one sample's value equal numpy;
* **cells** — ``hawk.all_samples(x).sum/max/min/mean()`` stored to a
  ``Mutable`` gives one row per 32 samples: sums within the sum contract,
  max and min exactly, for ragged and aligned ``n``, float32 and float64;
* **bitwise** — repeated device launches give the same bits;
* **identity** — a hand-written ``all_samples(bar_y * x).sum()`` is the
  parameter gradient ``vjp`` derives: the same sidecar entry, the same rows;
* **derivative** — the sum over samples differentiates (a broadcast of its
  adjoint); a max or a min refuses, naming the tie;
* **refusals** — one test each, with the text that migrates the author.
"""

from __future__ import annotations

import _deploy as L
import numpy as np
import pytest
from _eval import evaluate
from conftest import sidecar_of

import hawk
from hawk import HawkError, Kernel, Matrix, Mutable, Param, Scalar, Vector, kernel
from hawk.artifact import build_bundle
from hawk.diff import jvp, param_planes, vjp

SIZES = (1, 31, 32, 33, 1000, 65536)
TOL = {"float32": 1e-6, "float64": 1e-14}


@kernel
def own(v: Vector[3], m: Matrix[2, 3], x: Scalar, o: Mutable[Vector[3]], p: Mutable[Vector[2]],
        q: Mutable[Vector[3]], r: Mutable[Scalar]):
    o = hawk.math.vec(v.sum(), v.max(), v.min())
    p = m.sum(axis="columns")
    q = m.max(axis="rows")
    r = (v.mean() + hawk.math.sum(m) + hawk.math.mean(m, axis=None) + v.components[1]
         + m.rows[1].components[2] + m.columns[2].components[0]
         + v.sum(axis="components") + max(x, 0.5) + min(x, v[0]))


def test_own_axes_equal_numpy():
    rng = np.random.default_rng(3)
    v, m, x = rng.standard_normal(3), rng.standard_normal((2, 3)), 0.3
    got = evaluate(own.sinks, {"v": v, "m": m, "x": x})
    np.testing.assert_allclose(got["o"], [v.sum(), v.max(), v.min()], rtol=1e-14)
    np.testing.assert_allclose(got["p"], m.sum(axis=1), rtol=1e-14)
    np.testing.assert_allclose(got["q"], m.max(axis=0), rtol=1e-14)
    want = (v.mean() + m.sum() + m.mean() + v[1] + m[1, 2] + m[0, 2] + v.sum()
            + max(x, 0.5) + min(x, v[0]))
    np.testing.assert_allclose(got["r"], want, rtol=1e-13)


def test_builtin_max_min_keep_python_numbers():
    @kernel
    def k(x: Scalar, y: Mutable[Scalar]):
        y = x * max(2, 3) + min(1.0, 4.0)

    np.testing.assert_allclose(evaluate(k.sinks, {"x": 2.0})["y"], 7.0)


def _cell(op):
    if op == "mean":
        @kernel
        def cell(x: Scalar, t: Mutable[Scalar]):
            t = hawk.all_samples(x * x).mean()
    elif op == "max":
        @kernel
        def cell(x: Scalar, t: Mutable[Scalar]):
            t = hawk.all_samples(x * 1.5).max()
    elif op == "min":
        @kernel
        def cell(x: Scalar, t: Mutable[Scalar]):
            t = hawk.all_samples(x * 1.5).min()
    else:
        @kernel
        def cell(x: Scalar, t: Mutable[Scalar]):
            t = hawk.all_samples(x * x).sum()
    return cell


OPS = ("sum", "mean", "max", "min")


@kernel
def kx(x: Scalar, k: Param, y: Mutable[Scalar]):
    y = k * x


@kernel
def user_grad(x: Scalar, bar_y: Scalar, bar_k: Mutable[Scalar]):
    bar_k = hawk.all_samples(bar_y * x).sum()


KX = param_planes(kx)


@pytest.fixture(scope="module", params=["float64", "float32"])
def built(request, tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp(f"reduce_{request.param}")
    kernels = [Kernel(f"cell_{op}", _cell(op).sinks) for op in OPS]
    kernels += [KX, Kernel("kx_vjp", vjp(KX, wrt=("k",))), user_grad]
    bundle = build_bundle(kernels, root, mode=request.param, targets=("cuda", "host"),
                          cache_dir=cache_dir, host_profile="native")
    return bundle, request.param


def _launch(bundle, name, target, planes):
    import eagle.exec as eexec
    from eagle import plan as eplan

    sidecar = sidecar_of(bundle, name)
    if target == "host":
        plugin, structure = L.host_plugin(bundle.directory, name, sidecar), eexec.HostTeam
    else:
        plugin, structure = L.device_plugin(bundle.directory, name, sidecar), eexec.DeviceKernel
    return eplan.plan(plugin, structure=structure).bind(**planes)


def _xp(target):
    if target == "host":
        return np
    import cupy

    return cupy


def _x(n, mode):
    return np.random.default_rng(n).uniform(-2.0, 2.0, n).astype(mode)


def _read(planes):
    return {k: np.array(a.get() if hasattr(a, "get") else a) for k, a in planes.items()}


def _run_cell(bundle, op, target, n, mode):
    xp = _xp(target)
    rows = -(-n // 32)
    names = ("t", "t_comp") if op in ("sum", "mean") else ("t",)
    planes = {k: xp.full(rows, np.nan, dtype=mode) for k in names}
    planes["x"] = xp.asarray(_x(n, mode))
    bound = _launch(bundle, f"cell_{op}", target, planes)
    bound.launch()
    return _read(planes), (bound, planes)


def _terms(op, n, mode):
    x = _x(n, mode).astype(np.float64)
    return {"sum": x * x, "mean": x * x / n, "max": x * 1.5, "min": x * 1.5}[op]


def _per_row(t, fill):
    pad = np.full(-(-t.size // 32) * 32, fill)
    pad[:t.size] = t
    return pad.reshape(-1, 32)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_cells_equal_the_float64_reference(built, target, op, n):
    bundle, mode = built
    got, _ = _run_cell(bundle, op, target, n, mode)
    t = _terms(op, n, mode)
    if op in ("sum", "mean"):
        rows, scale = _per_row(t, 0.0), np.abs(_per_row(t, 0.0)).sum(1)
        row = got["t"].astype(np.float64) + got["t_comp"].astype(np.float64)
        assert np.all(np.isfinite(got["t"]))
        assert np.all(np.abs(row - rows.sum(1)) <= TOL[mode] * scale + 1e-300)
    else:
        fill = -np.inf if op == "max" else np.inf
        xs = _x(n, mode)
        # the term is computed in the plane's precision
        ref = _per_row((xs * 1.5).astype(np.float64), fill)
        ref = ref.max(1) if op == "max" else ref.min(1)
        assert np.array_equal(got["t"].astype(np.float64), ref)


def test_sidecar_lists_the_cells(built):
    bundle, _ = built
    assert sidecar_of(bundle, "cell_sum")["cells"] == {"t": {"op": "sum", "columns": 1}}
    assert sidecar_of(bundle, "cell_max")["cells"] == {"t": {"op": "max", "columns": 1}}
    assert all("sample_sums" not in sidecar_of(bundle, f"cell_{op}") for op in OPS)


@pytest.mark.gpu
@pytest.mark.parametrize("op", OPS)
def test_device_cells_are_bitwise_across_launches(built, op):
    bundle, mode = built
    first, (bound, planes) = _run_cell(bundle, op, "cuda", 65536 + 17, mode)
    first.pop("x")
    for _ in range(5):
        for k in first:
            planes[k].fill(np.nan)  # the rows only; ``first`` holds the outputs
        bound.launch()
        again = _read(planes)
        for k in first:
            assert again[k].tobytes() == first[k].tobytes(), k


@pytest.mark.parametrize("target", ["host", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_a_written_sum_is_the_derived_parameter_gradient(built, target):
    bundle, mode = built
    n = 1000
    xp = _xp(target)
    rng = np.random.default_rng(1)
    x, gy = (xp.asarray(rng.standard_normal(n).astype(mode)) for _ in range(2))
    rows = -(-n // 32)

    def planes(name):
        p = {"x": x, "bar_y": gy, "bar_k": xp.full(rows, np.nan, dtype=mode),
             "bar_k_comp": xp.full(rows, np.nan, dtype=mode)}
        _launch(bundle, name, target, p).launch()
        return _read(p)

    a, b = planes("kx_vjp"), planes("user_grad")
    assert a["bar_k"].tobytes() == b["bar_k"].tobytes()
    assert a["bar_k_comp"].tobytes() == b["bar_k_comp"].tobytes()
    assert sidecar_of(bundle, "kx_vjp")["cells"]["bar_k"] == \
        sidecar_of(bundle, "user_grad")["cells"]["bar_k"]


def test_a_sum_over_samples_has_a_broadcast_adjoint():
    d = vjp(_cell("sum"), wrt=("x",))
    sinks = Kernel("d", d).sinks
    assert [s.name for s in sinks] == ["bar_x"]
    got = evaluate(sinks, {"x": 1.5, "bar_t": 2.0})
    np.testing.assert_allclose(got["bar_x"], 2.0 * 2 * 1.5)


@pytest.mark.parametrize("op", ("max", "min"))
def test_max_and_min_over_samples_refuse_to_differentiate(op):
    with pytest.raises(HawkError, match="subgradient is ambiguous at a tie"):
        vjp(_cell(op), wrt=("x",))
    with pytest.raises(HawkError, match="subgradient is ambiguous at a tie"):
        jvp(_cell(op), wrt=("x",))


def test_axis_samples_on_a_locked_value_names_all_samples():
    with pytest.raises(HawkError, match=r"hawk\.all_samples\(x\)\.sum\(\)"):
        @kernel
        def k(v: Vector[3], y: Mutable[Scalar]):
            y = v.sum(axis="samples")


def test_a_scalar_has_no_axis_to_fold():
    with pytest.raises(HawkError, match=r"hawk\.all_samples\(x\)\.sum\(\)"):
        @kernel
        def k(x: Scalar, y: Mutable[Scalar]):
            y = x.sum()


def test_max_of_an_iterable_refuses():
    with pytest.raises(HawkError, match="not a HAWK form"):
        @kernel
        def k(v: Vector[3], y: Mutable[Scalar]):
            y = max(v)


def test_a_reduced_value_read_later_is_a_launch_boundary():
    from hawk.ir.split import stages

    @kernel
    def k(x: Scalar, t: Mutable[Scalar], u: Mutable[Scalar]):
        t = hawk.all_samples(x).sum()
        u = t + 1.0
    first, second = stages("k", k.sinks)
    assert first.cells_out == ("t",) and second.cells_in == ("t",)


def test_arithmetic_on_the_view_and_a_nested_view_refuse():
    with pytest.raises(HawkError, match="compute per sample first"):
        @kernel
        def k2(x: Scalar, t: Mutable[Scalar]):
            t = hawk.all_samples(x) * 2.0
    with pytest.raises(HawkError, match="already every sample"):
        hawk.all_samples(hawk.all_samples(1.0))


def test_one_kernel_reduces_with_one_op():
    @kernel
    def k(x: Scalar, a: Mutable[Scalar], b: Mutable[Scalar]):
        a = hawk.all_samples(x).sum()
        b = hawk.all_samples(x).max()

    from hawk.emit.backend import render_source
    from hawk.emit.cuda import CudaBackend
    with pytest.raises(HawkError, match="one op"):
        render_source(k.name, k.sinks, k.walk, CudaBackend())
