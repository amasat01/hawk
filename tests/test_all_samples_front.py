# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.all_samples(x)`` beyond the reduce: gather, mask, by-key stores.

Each form equals the numpy oracle written on the whole arrays, on the host
and (``gpu`` mark) the device, float32 and float64, for ragged and aligned
``n``; the device rows that are order-free are bitwise across launches; the
refusals each have one test with the text that migrates the author.
"""

from __future__ import annotations

import _deploy as L
import numpy as np
import pytest
from _eval import evaluate
from conftest import sidecar_of

import hawk
from hawk import (HawkError, Index, Kernel, Matrix, Mutable, Scalar,
                  Table, Vector, all_samples, kernel)
from hawk.artifact import build_bundle
from hawk.diff import vjp

SIZES = (1, 31, 32, 33, 1000)
TOL = {"float32": 1e-6, "float64": 1e-13}
NB = 7


@kernel
def gather(x: Scalar, i: Index, d: Mutable[Scalar]):
    d = all_samples(x)[i]


@kernel
def cgather(x: Scalar, i: Index, d: Mutable[Scalar]):
    d = all_samples(x * x + 1.0)[i]


@kernel
def vgather(v: Vector[3], i: Index, d: Mutable[Scalar]):
    V = all_samples(v)
    d = V[i][0] + V[i][2]


@kernel
def positive(x: Scalar, s: Mutable[Scalar], c: Mutable[Scalar]):
    X = all_samples(x)
    s = X[x > 0.0].sum()
    c = all_samples(x > 0.0).sum()


@kernel
def positive_max(x: Scalar, m: Mutable[Scalar]):
    m = all_samples(x)[x > 0.0].max()


@kernel
def hist(lane: Index, u: Vector[3], bins: Mutable[Scalar], acc0: Mutable[Scalar],
         acc1: Mutable[Scalar], acc2: Mutable[Scalar]):
    all_samples(bins)[lane] += 1.0
    all_samples(acc0)[lane] += u[0]
    all_samples(acc1)[lane] += u[1]
    all_samples(acc2)[lane] += u[2]


@kernel
def store(lane: Index, x: Scalar, buf: Mutable[Scalar]):
    all_samples(buf)[lane] = x


@kernel
def mstore(x: Scalar, y: Mutable[Scalar]):
    y = x
    all_samples(y)[x > 0.0] = 2.0 * x


@kernel
def vcell(v: Vector[3], m: Matrix[2, 3], t: Mutable[Vector[3]], u: Mutable[Vector[3]]):
    t = all_samples(v).sum(axis="samples")
    u = all_samples(m).sum(axis=("samples", "rows"))


@kernel
def rows(m: Matrix[2, 3], i: Index, r: Mutable[Vector[3]], c: Mutable[Vector[2]]):
    r = m.rows[i]
    c = m.columns[i]


@kernel
def grad_gather(x: Scalar, i: Index, d: Mutable[Scalar]):
    d = all_samples(x * x)[i] * x


KERNELS = (gather, cgather, vgather, positive, positive_max, hist, store, mstore, vcell, rows)


@pytest.fixture(scope="module", params=["float64", "float32"])
def built(request, tmp_path_factory, cache_dir):
    """``{kernel name: bundle}``: a bundle holds one access class."""
    root = tmp_path_factory.mktemp(f"front_{request.param}")
    groups = ((gather, cgather, vgather),
              (hist, store),
              (positive, positive_max, mstore, vcell, rows))
    out = {}
    for g, group in enumerate(groups):
        bundle = build_bundle(list(group), root / str(g), mode=request.param,
                              targets=("cuda", "host"), cache_dir=cache_dir,
                              host_profile="native")
        out.update({k.name: bundle for k in group})
    return out, request.param


def _launch(bundle, name, target, planes):
    import eagle.exec as eexec
    from eagle import plan as eplan

    bundle = bundle[name]
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


def _run(bundle, name, target, ins, outs):
    xp = _xp(target)
    planes = {k: xp.asarray(v) for k, v in {**ins, **outs}.items()}
    bound = _launch(bundle, name, target, planes)
    bound.launch()
    return ({k: np.array(planes[k].get() if hasattr(planes[k], "get") else planes[k])
             for k in outs}, bound, planes)


def _data(n, mode):
    rng = np.random.default_rng(n)
    return {"x": rng.uniform(-2, 2, n).astype(mode),
            "v": rng.uniform(-2, 2, (3, n)).astype(mode),
            "i": rng.integers(0, n, n).astype(np.int64),
            "lane": rng.integers(0, NB, n).astype(np.int64)}


def _empty(n, mode, **shapes):
    return {k: np.full(s, np.nan, dtype=mode) for k, s in shapes.items()}


TARGETS = ["host", pytest.param("cuda", marks=pytest.mark.gpu)]


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("target", TARGETS)
def test_gather_equals_take(built, target, n):
    bundle, mode = built
    d = _data(n, mode)
    x, i, v = d["x"], d["i"], d["v"]
    got, *_ = _run(bundle, "gather", target, {"x": x, "i": i}, _empty(n, mode, d=n))
    assert np.array_equal(got["d"], np.take(x, i))
    xs = x.astype(np.float64)
    got, *_ = _run(bundle, "cgather", target, {"x": x, "i": i}, _empty(n, mode, d=n))
    np.testing.assert_allclose(got["d"], np.take(xs * xs + 1.0, i), rtol=TOL[mode] * 10)
    got, *_ = _run(bundle, "vgather", target, {"v": v, "i": i}, _empty(n, mode, d=n))
    vt = v.astype(np.float64)
    np.testing.assert_allclose(got["d"], vt[0][i] + vt[2][i], rtol=TOL[mode] * 10)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("target", TARGETS)
def test_mask_reduce_equals_numpy(built, target, n):
    bundle, mode = built
    x = _data(n, mode)["x"]
    rows_n = -(-n // 32)
    outs = _empty(n, mode, s=rows_n, s_comp=rows_n, c=rows_n, c_comp=rows_n)
    got, *_ = _run(bundle, "positive", target, {"x": x}, outs)
    top, *_ = _run(bundle, "positive_max", target, {"x": x}, _empty(n, mode, m=rows_n))
    xs = x.astype(np.float64)
    tol = TOL[mode] * 10 * max(1.0, np.abs(xs).sum())
    s = (got["s"].astype(np.float64) + got["s_comp"]).sum()
    assert abs(s - xs[xs > 0].sum()) <= tol
    assert (got["c"].astype(np.float64) + got["c_comp"]).sum() == (xs > 0).sum()
    want = xs[xs > 0].max() if (xs > 0).any() else -np.inf
    assert top["m"].astype(np.float64).max() == want


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("target", TARGETS)
def test_accumulate_by_key_equals_add_at(built, target, n):
    bundle, mode = built
    d = _data(n, mode)
    outs = {k: np.zeros(NB, mode) for k in ("bins", "acc0", "acc1", "acc2")}
    got, bound, planes = _run(bundle, "hist", target, {"lane": d["lane"], "u": d["v"]}, outs)
    want = np.zeros(NB)
    np.add.at(want, d["lane"], 1.0)
    np.testing.assert_allclose(got["bins"], want, rtol=TOL[mode])
    for k in range(3):
        wa = np.zeros(NB)
        np.add.at(wa, d["lane"], d["v"][k].astype(np.float64))
        np.testing.assert_allclose(got[f"acc{k}"], wa, rtol=TOL[mode] * 100,
                                   atol=TOL[mode] * 100 * n)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("target", TARGETS)
def test_store_by_key_and_masked_store(built, target, n):
    bundle, mode = built
    x = _data(n, mode)["x"]
    perm = np.random.default_rng(5).permutation(n).astype(np.int64)
    got, bound, planes = _run(bundle, "store", target, {"lane": perm, "x": x}, _empty(n, mode, buf=n))
    want = np.empty(n, mode)
    want[perm] = x
    assert np.array_equal(got["buf"], want)
    got, *_ = _run(bundle, "mstore", target, {"x": x}, _empty(n, mode, y=n))
    assert np.array_equal(got["y"], np.where(x > 0, mode_two(mode) * x, x))


def mode_two(mode):
    return np.asarray(2.0, mode)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("target", TARGETS)
def test_vector_and_matrix_cells_equal_columnwise_numpy(built, target, n):
    bundle, mode = built
    rng = np.random.default_rng(n + 1)
    v = rng.uniform(-2, 2, (3, n)).astype(mode)
    m = rng.uniform(-2, 2, (2, 3, n)).astype(mode)
    rows_n = -(-n // 32)
    names = [f"{c}_{k}{s}" for c in "tu" for k in range(3) for s in ("", "_comp")]
    outs = _empty(n, mode, **{k: rows_n for k in names})
    got, *_ = _run(bundle, "vcell", target, {"v": v, "m": m.reshape(6, n)}, outs)
    sidecar = sidecar_of(bundle["vcell"], "vcell")
    assert sidecar["cells"] == {"t": {"op": "sum", "columns": 3},
                                "u": {"op": "sum", "columns": 3}}
    for c, want in (("t", v.astype(np.float64).sum(axis=1)),
                    ("u", m.astype(np.float64).sum(axis=(0, 2)))):
        cell = np.array([(got[f"{c}_{k}"].astype(np.float64) + got[f"{c}_{k}_comp"]).sum()
                         for k in range(3)])
        np.testing.assert_allclose(cell, want, rtol=TOL[mode] * 100, atol=TOL[mode] * 100 * n)


def test_traced_rows_and_columns_equal_numpy():
    rng = np.random.default_rng(2)
    m = rng.standard_normal((2, 3))
    for i in (0, 1):
        got = evaluate(rows.sinks, {"m": m, "i": i})
        np.testing.assert_allclose(got["r"], m[i] if i < 2 else None)
        np.testing.assert_allclose(got["c"], m[:, i])


def _fd(kernel_, wrt, point, seed, eps=1e-6):
    base = dict(point)
    out = {}
    for name in wrt:
        up, dn = dict(base), dict(base)
        up[name] = np.array(base[name], dtype=np.float64) + eps
        dn[name] = np.array(base[name], dtype=np.float64) - eps
        out[name] = (evaluate(kernel_.sinks, up)["d"] - evaluate(kernel_.sinks, dn)["d"]) / (2 * eps)
    return out


def test_gather_adjoint_is_a_scatter_add_of_the_seed():
    """d[s] = x[i[s]]^2 * x[s]: the adjoint of x[j] sums the seeds of every
    sample that gathered j, plus the own term -- against central differences
    of the unrolled reference."""
    n = 6
    rng = np.random.default_rng(4)
    x, i, seed = rng.uniform(1, 2, n), rng.integers(0, n, n), rng.uniform(1, 2, n)
    want = np.zeros(n)
    np.add.at(want, i, seed * 2 * x[i] * x)
    want += seed * x[i] ** 2
    ref = np.zeros(n)
    for j in range(n):
        up, dn = x.copy(), x.copy()
        up[j] += 1e-6
        dn[j] -= 1e-6
        f = lambda a: (seed * a[i] ** 2 * a).sum()
        ref[j] = (f(up) - f(dn)) / 2e-6
    np.testing.assert_allclose(want, ref, rtol=1e-6)
    d = vjp(grad_gather, wrt=("x",))
    names = [type(k).__name__ + ":" + k.name for k in d]
    assert any(nm.startswith("AccumWrite") for nm in names), names


# ---- refusals: one named test each, the text that migrates the author ------


def test_refuses_a_read_of_an_outputs_other_sample():
    with pytest.raises(HawkError, match=r"'y' is written by this launch.*stage it \(Staged\)"):
        @kernel
        def k(x: Scalar, i: Index, y: Mutable[Scalar]):
            y = all_samples(y)[i] + x


def test_refuses_a_plane_at_two_extents():
    with pytest.raises(HawkError, match="one plane has one extent"):
        @kernel
        def k(x: Scalar, lane: Index, y: Mutable[Scalar]):
            y = x
            all_samples(y)[lane] += 1.0
    with pytest.raises(HawkError, match="one plane has one extent"):
        @kernel
        def k2(x: Scalar, lane: Index, y: Mutable[Scalar]):
            all_samples(y)[lane] += 1.0
            y = x


def test_refuses_a_cut_inside_a_for():
    with pytest.raises(HawkError, match=r"cut inside a `for`"):
        @kernel
        def k(x: Scalar, t: Mutable[Scalar]):
            s = 0.0 * x
            for j in range(3):
                s = s + all_samples(x).sum()
            t = s


def test_refuses_a_mask_of_a_mask():
    with pytest.raises(HawkError, match="a mask of a mask"):
        @kernel
        def k(x: Scalar, t: Mutable[Scalar]):
            t = all_samples(x > 0.0)[x > 1.0].sum()


def test_refuses_a_masked_view_outside_reduce_or_store():
    with pytest.raises(HawkError, match=r"a masked view has no fixed length; reduce it"):
        @kernel
        def k(x: Scalar, t: Mutable[Scalar]):
            t = all_samples(x)[x > 0.0]
    with pytest.raises(HawkError, match=r"a masked view has no fixed length"):
        @kernel
        def k2(x: Scalar, t: Mutable[Scalar]):
            t = all_samples(x)[x > 0.0] + 1.0


def test_refuses_an_index_on_a_locked_rank0_output():
    with pytest.raises(HawkError,
                       match=r"'bins'\[key\] indexes a component; samples are indexed on the view"):
        @kernel
        def k(lane: Index, x: Scalar, bins: Mutable[Scalar]):
            bins[lane] = x
    with pytest.raises(HawkError, match=r"indexes a component"):
        @kernel
        def k2(lane: Index, x: Scalar, bins: Mutable[Scalar]):
            bins[lane] += x


@pytest.mark.parametrize("what", ("prior", "random", "loop"))
def test_refuses_a_computed_gather_over_a_per_sample_value(what):
    with pytest.raises(HawkError, match="re-reads expr at sample i, but expr reads"):
        if what == "prior":
            @kernel
            def k(x: Scalar, i: Index, d: Mutable[Scalar]):
                d = all_samples(d * 2.0 + x)[i]
        elif what == "random":
            @kernel
            def k(x: Scalar, i: Index, s: Index, d: Mutable[Scalar]):
                d = all_samples(x + hawk.math.random_uniform(s, 0))[i]
        else:
            @kernel
            def k(x: Scalar, i: Index, d: Mutable[Scalar]):
                a = x
                for j in range(3):
                    a = a + x
                d = all_samples(a)[i]


def test_refuses_arithmetic_and_a_bare_store_on_the_view():
    with pytest.raises(HawkError, match=r"is every sample; compute per sample first"):
        @kernel
        def k(x: Scalar, y: Mutable[Scalar]):
            y = all_samples(x) * 2.0
    with pytest.raises(HawkError, match=r"is every sample; compute per sample first"):
        @kernel
        def k2(x: Scalar, y: Mutable[Scalar]):
            y = all_samples(x)


def test_refuses_a_store_inside_a_branch():
    with pytest.raises(HawkError, match="may not sit inside a branch"):
        @kernel
        def k(x: Scalar, lane: Index, y: Mutable[Scalar]):
            if x > 0.0:
                all_samples(y)[lane] += 1.0


def test_refuses_all_samples_of_all_samples_and_a_vector_by_key():
    with pytest.raises(HawkError, match="already every sample"):
        all_samples(all_samples(hawk.math.vec(1.0, 2.0)[0]))
    with pytest.raises(HawkError, match="per component in this release"):
        @kernel
        def k(lane: Index, u: Vector[3], acc: Mutable[Vector[3]]):
            all_samples(acc)[lane] += u


def test_masked_mean_names_the_count():
    with pytest.raises(HawkError, match="needs the count of the mask"):
        @kernel
        def k(x: Scalar, t: Mutable[Scalar]):
            t = all_samples(x)[x > 0.0].mean()


def test_the_samples_axis_has_no_attribute():
    with pytest.raises(HawkError, match=r"read through hawk\.all_samples\(v\)\[i\]"):
        @kernel
        def k(x: Scalar, t: Mutable[Scalar]):
            t = all_samples(x).batch


def test_table_subscript_is_at():
    @kernel
    def k(t: Table[Scalar], i: Index, y: Mutable[Scalar]):
        y = t[i]

    @kernel
    def k2(t: Table[Scalar], i: Index, y: Mutable[Scalar]):
        y = t.at(i)

    assert k.walk.digest == k2.walk.digest


@pytest.mark.gpu
@pytest.mark.parametrize("name", ("gather", "mstore", "store"))
def test_order_free_device_forms_are_bitwise_across_launches(built, name):
    bundle, mode = built
    n = 1000
    d = _data(n, mode)
    perm = np.random.default_rng(5).permutation(n).astype(np.int64)
    ins = {"gather": {"x": d["x"], "i": d["i"]}, "mstore": {"x": d["x"]},
           "store": {"lane": perm, "x": d["x"]}}[name]
    out = {"gather": "d", "mstore": "y", "store": "buf"}[name]
    first, bound, planes = _run(bundle, name, "cuda", ins, _empty(n, mode, **{out: n}))
    for _ in range(3):
        planes[out].fill(np.nan)
        bound.launch()
        assert planes[out].get().tobytes() == first[out].tobytes()
