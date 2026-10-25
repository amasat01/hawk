# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Across lanes: ``hawk.sum/max/min/mean(x, over=, where=, policy=)``,
``hawk.scatter(sink)[key] += / = v`` and ``u.at(i)``.

Each form equals the numpy oracle written on the whole arrays, on the host and
(``gpu`` mark) the device, float32 and float64; sums are checked against
``math.fsum`` in ULP (the contract: a plain bound that grows with the terms per slot, 8 at these
sizes; 1 compensated); the
order-free device forms are bitwise across launches; every refusal has one test
with the text that migrates the author.
"""

from __future__ import annotations

import math

import _deploy as L
import numpy as np
import pytest
from _eval import evaluate
from conftest import _xp, sidecar_of

import hawk
import hawk.math as m
from hawk import (Grid, HawkError, Index, Kernel, Matrix, Mutable, Param, Scalar,
                  Table, Vector, kernel)
from hawk import _lane_bind as LB
from hawk.artifact import build_bundle
from hawk.diff import vjp
from hawk.ir.lanes import key_run
from hawk.ir.nodes import AccumWrite, ReduceSamples, SampleSum

SIZES = (1, 31, 32, 33, 1000)
TOL = {"float32": 1e-6, "float64": 1e-13}
NB = 7
#: the sum contract: |error| / (eps(dtype) * sum|t_i|), plain and compensated
CONTRACT = {"auto": 8.0, "compensated": 1.0}


@kernel
def gather(x: Scalar, i: Index, d: Mutable[Scalar]):
    d = x.at(i)


@kernel
def vgather(v: Vector[3], i: Index, d: Mutable[Scalar]):
    d = v.at(i)[0] + v.at(i)[2]


@kernel
def positive(x: Scalar, s: Mutable[Scalar], c: Mutable[Scalar]):
    s = hawk.sum(x, where=x > 0.0)
    c = hawk.sum(x > 0.0)


@kernel
def positive_max(x: Scalar, top: Mutable[Scalar]):
    top = hawk.max(x, where=x > 0.0)


@kernel
def hist(lane: Index, u: Vector[3], bins: Mutable[Scalar], acc0: Mutable[Scalar],
         acc1: Mutable[Scalar], acc2: Mutable[Scalar]):
    hawk.scatter(bins)[lane] += 1.0
    hawk.scatter(acc0)[lane] += u[0]
    hawk.scatter(acc1)[lane] += u[1]
    hawk.scatter(acc2)[lane] += u[2]


@kernel
def store(lane: Index, x: Scalar, buf: Mutable[Scalar]):
    hawk.scatter(buf)[lane] = x


@kernel
def mstore(x: Scalar, y: Mutable[Scalar]):
    y = m.where(x > 0.0, 2.0 * x, x)


@kernel
def vcell(v: Vector[3], mm: Matrix[2, 3], t: Mutable[Vector[3]], u: Mutable[Vector[3]]):
    t = hawk.sum(v)
    u = hawk.sum(mm.sum(axis="rows"))


@kernel
def rows(mm: Matrix[2, 3], i: Index, r: Mutable[Vector[3]], c: Mutable[Vector[2]]):
    r = mm.rows[i]
    c = mm.columns[i]


@kernel
def grad_gather(x: Scalar, i: Index, d: Mutable[Scalar]):
    d = x.at(i) * x.at(i) * x


# ---- the worked forms: contraction backward, FEM assembly, histogram, a per-sweep count

NO, NI, B, S = 3, 4, 64, 3


@kernel
def contract_bwd(lane: Grid["channel", "input":NI, "sample":B],
            feat: Table["slot":S, "input":NI, "sample":B],
            yb: Table["channel":NO, "sample":B]) -> Vector[S]:
    g = yb.at(channel=lane.channel, sample=lane.sample)
    f0 = feat.at(slot=0, input=lane.input, sample=lane.sample)
    f1 = feat.at(slot=1, input=lane.input, sample=lane.sample)
    f2 = feat.at(slot=2, input=lane.input, sample=lane.sample)
    return hawk.sum(m.vec(f0 * g, f1 * g, f2 * g), over="sample")


@kernel
def contract_bwd_static(lane: Grid["channel":NO, "input":NI, "sample":B],
                   feat: Table["slot":S, "input":NI, "sample":B],
                   yb: Table["channel":NO, "sample":B]) -> Vector[S]:
    g = yb.at(channel=lane.channel, sample=lane.sample)
    f0 = feat.at(slot=0, input=lane.input, sample=lane.sample)
    f1 = feat.at(slot=1, input=lane.input, sample=lane.sample)
    f2 = feat.at(slot=2, input=lane.input, sample=lane.sample)
    return hawk.sum(m.vec(f0 * g, f1 * g, f2 * g), over="sample")


NE, NN = 6, 9


@kernel
def assemble(lane: Grid["elem":NE, "qp":4], conn: Table[Index], shape: Table["qp":4, "vert":4],
             wdet: Table["elem":NE, "qp":4], f: Mutable["node":NN]):
    w = wdet.at(elem=lane.elem, qp=lane.qp)
    for a in range(4):
        hawk.scatter(f)[conn.at(lane.elem * 4 + a)] += w * shape.at(qp=lane.qp, vert=a)


@kernel
def histogram(lane: Grid["i"], x: Scalar, bins: Mutable["bin":256]):
    hawk.scatter(bins)[m.index(x * 256.0, 256)] += 1.0


MC = 500


@kernel(group="mc")
def phobos(lane: Grid["sweep", "mc":MC], state: Vector[3], gm: Table["sweep":4],
           body: Param) -> dict:
    r = state - body
    d = m.norm(r)
    return {"accel": -gm.at(sweep=lane.sweep) * r / (d * d * d),
            "close": hawk.sum(m.where(d < 20.0, 1.0, 0.0), over="mc")}


@kernel
def kmean(lane: Grid["row":2, "col":4], x: Scalar) -> Scalar:
    return hawk.mean(x, over="col")


KP = 64


def _sum_over(policy):
    @kernel
    def ksum(lane: Grid["key", "sample":8192], x: Scalar) -> Scalar:
        return hawk.sum(x, over="sample", policy=policy)
    return ksum


AUTO, COMP = _sum_over("auto"), _sum_over("compensated")
AUTO_KERNEL = Kernel("ksum_a", AUTO.sinks, AUTO.planes)
COMP_KERNEL = Kernel("ksum_c", COMP.sinks, COMP.planes)

GROUPS = ((gather, vgather),
          (hist, store),
          (positive, positive_max, mstore, vcell, rows),
          (contract_bwd,), (assemble,), (histogram,), (phobos,), (kmean,),
          (AUTO_KERNEL,), (COMP_KERNEL,))


@pytest.fixture(scope="module", params=["float64", "float32"])
def built(request, tmp_path_factory, cache_dir):
    """``{kernel name: bundle}``: a bundle holds one access class."""
    root = tmp_path_factory.mktemp(f"across_{request.param}")
    out = {}
    for g, group in enumerate(GROUPS):
        bundle = build_bundle(list(group), root / str(g), mode=request.param,
                              targets=("cuda", "host"), cache_dir=cache_dir,
                              host_profile="native")
        out.update({k.name: bundle for k in group})
    return out, request.param


def _launch(bundle, name, target, planes, **extent):
    import eagle.exec as eexec
    from eagle import plan as eplan

    bundle = bundle[name]
    sidecar = sidecar_of(bundle, name)
    if target == "host":
        plugin, structure = L.host_plugin(bundle.directory, name, sidecar), eexec.HostTeam
    else:
        plugin, structure = L.device_plugin(bundle.directory, name, sidecar), eexec.DeviceKernel
    return eplan.plan(plugin, structure=structure).bind(**extent, **planes)


def _run(bundle, name, target, ins, outs, **extent):
    xp = _xp(target)
    planes = {k: xp.asarray(v) for k, v in {**ins, **outs}.items()}
    bound = _launch(bundle, name, target, planes, **extent)
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


def _ulp(got, exact, mode):
    """The error in ULP of the exactly rounded sum: a card number, not a gate
    (it grows with the sum's cancellation, not with the commit)."""
    exact = np.asarray(exact, dtype=np.float64)
    return float((np.abs(np.asarray(got, dtype=np.float64) - exact)
                  / np.spacing(exact.astype(mode)).astype(np.float64)).max())


def _scaled(got, exact, magnitude, mode):
    """The error over ``eps(dtype) * sum|t_i|``: the measure the contract bounds."""
    err = np.abs(np.asarray(got, dtype=np.float64) - np.asarray(exact, dtype=np.float64))
    return float((err / (np.finfo(mode).eps * np.asarray(magnitude))).max())


TARGETS = ["host", pytest.param("cuda", marks=pytest.mark.gpu)]


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("target", TARGETS)
def test_at_equals_take(built, target, n):
    bundle, mode = built
    d = _data(n, mode)
    x, i, v = d["x"], d["i"], d["v"]
    got, *_ = _run(bundle, "gather", target, {"x": x, "i": i}, _empty(n, mode, d=n))
    assert np.array_equal(got["d"], np.take(x, i))
    got, *_ = _run(bundle, "vgather", target, {"v": v, "i": i}, _empty(n, mode, d=n))
    vt = v.astype(np.float64)
    np.testing.assert_allclose(got["d"], vt[0][i] + vt[2][i], rtol=TOL[mode] * 10)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("target", TARGETS)
def test_where_reduce_equals_numpy(built, target, n):
    bundle, mode = built
    x = _data(n, mode)["x"]
    rows_n = -(-n // 32)
    outs = _empty(n, mode, s=rows_n, s_comp=rows_n, c=rows_n, c_comp=rows_n)
    got, *_ = _run(bundle, "positive", target, {"x": x}, outs)
    top, *_ = _run(bundle, "positive_max", target, {"x": x}, _empty(n, mode, top=rows_n))
    xs = x.astype(np.float64)
    tol = TOL[mode] * 10 * max(1.0, np.abs(xs).sum())
    s = (got["s"].astype(np.float64) + got["s_comp"]).sum()
    assert abs(s - xs[xs > 0].sum()) <= tol
    assert (got["c"].astype(np.float64) + got["c_comp"]).sum() == (xs > 0).sum()
    want = xs[xs > 0].max() if (xs > 0).any() else -np.inf
    assert top["top"].astype(np.float64).max() == want


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("target", TARGETS)
def test_scatter_add_equals_add_at(built, target, n):
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
def test_scatter_store_and_where(built, target, n):
    bundle, mode = built
    x = _data(n, mode)["x"]
    perm = np.random.default_rng(5).permutation(n).astype(np.int64)
    got, bound, planes = _run(bundle, "store", target, {"lane": perm, "x": x}, _empty(n, mode, buf=n))
    want = np.empty(n, mode)
    want[perm] = x
    assert np.array_equal(got["buf"], want)
    got, *_ = _run(bundle, "mstore", target, {"x": x}, _empty(n, mode, y=n))
    assert np.array_equal(got["y"], np.where(x > 0, np.asarray(2.0, mode) * x, x))


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("target", TARGETS)
def test_vector_and_matrix_cells_equal_columnwise_numpy(built, target, n):
    bundle, mode = built
    rng = np.random.default_rng(n + 1)
    v = rng.uniform(-2, 2, (3, n)).astype(mode)
    mm = rng.uniform(-2, 2, (2, 3, n)).astype(mode)
    rows_n = -(-n // 32)
    names = [f"{c}_{k}{s}" for c in "tu" for k in range(3) for s in ("", "_comp")]
    outs = _empty(n, mode, **{k: rows_n for k in names})
    got, *_ = _run(bundle, "vcell", target, {"v": v, "mm": mm.reshape(6, n)}, outs)
    sidecar = sidecar_of(bundle["vcell"], "vcell")
    assert sidecar["cells"] == {"t": {"op": "sum", "columns": 3},
                                "u": {"op": "sum", "columns": 3}}
    for c, want in (("t", v.astype(np.float64).sum(axis=1)),
                    ("u", mm.astype(np.float64).sum(axis=(0, 2)))):
        cell = np.array([(got[f"{c}_{k}"].astype(np.float64) + got[f"{c}_{k}_comp"]).sum()
                         for k in range(3)])
        np.testing.assert_allclose(cell, want, rtol=TOL[mode] * 100, atol=TOL[mode] * 100 * n)


@pytest.mark.parametrize("target", TARGETS)
def test_contraction_backward_sums_over_the_sample_axis(built, target):
    """(a): one value per (channel, input), payload first: ``(S, n_out, n_in)``."""
    bundle, mode = built
    rng = np.random.default_rng(3)
    feat, yb = rng.random((S, NI, B)).astype(mode), rng.random((NO, B)).astype(mode)
    sidecar = sidecar_of(bundle["contract_bwd"], "contract_bwd")
    shape = LB.output_shape(sidecar, "out", 1, NO * NI * B)
    assert shape == (S, NO, NI)
    got, *_ = _run(bundle, "contract_bwd", target, {"feat": feat.ravel(), "yb": yb.ravel()},
                   {"out": np.zeros(shape, mode)}, channel=NO)
    want = np.einsum("kib,ob->koi", feat.astype(np.float64), yb.astype(np.float64))
    np.testing.assert_allclose(got["out"], want, rtol=TOL[mode] * 100)


def test_contraction_backward_key_runs_along_the_sample_axis():
    """Every component's key is invariant along ``sample`` (64 lanes): one commit
    per warp."""
    for s in contract_bwd_static.sinks:
        assert isinstance(s, AccumWrite)
        run = key_run(contract_bwd_static.sinks, s)
        assert run is not None and (run.form, run.width) == ("proven", 32)


@pytest.mark.parametrize("target", TARGETS)
def test_fem_assembly_scatters_into_the_nodes(built, target):
    """(b): four vertex scatters per (element, quadrature point), one per
    iteration of a ``for`` unrolled while the body is traced."""
    bundle, mode = built
    rng = np.random.default_rng(4)
    conn = rng.integers(0, NN, (NE, 4)).astype(np.int64)
    shape, wdet = rng.random((4, 4)).astype(mode), rng.random((NE, 4)).astype(mode)
    got, *_ = _run(bundle, "assemble", target,
                   {"conn": conn.ravel(), "shape": shape.ravel(), "wdet": wdet.ravel()},
                   {"f": np.zeros(NN, mode)})
    want = np.zeros(NN)
    for e in range(NE):
        for q in range(4):
            for a in range(4):
                want[conn[e, a]] += float(wdet[e, q]) * float(shape[q, a])
    np.testing.assert_allclose(got["f"], want, rtol=TOL[mode] * 100)


@pytest.mark.parametrize("target", TARGETS)
def test_a_histogram_and_a_passed_sink_keeps_its_contents(built, target):
    """(d): the sink you pass keeps its contents; the adds land on top."""
    bundle, mode = built
    x = np.random.default_rng(6).random(5000).astype(mode)
    start = np.arange(256, dtype=mode)
    got, *_ = _run(bundle, "histogram", target, {"x": x}, {"bins": start.copy()}, i=len(x))
    want = np.bincount(np.clip((x.astype(np.float64) * 256).astype(int), 0, 255),
                       minlength=256)
    np.testing.assert_array_equal(got["bins"], start + want)


@pytest.mark.parametrize("target", TARGETS)
def test_a_count_per_sweep_beside_a_per_lane_output(built, target):
    """(e): ``over="mc"`` with ``group="mc"``: one count per sweep."""
    bundle, mode = built
    rng = np.random.default_rng(7)
    sweeps = 4
    state = rng.uniform(-30, 30, (3, sweeps * MC)).astype(mode)
    gm = rng.random(sweeps).astype(mode)
    got, *_ = _run(bundle, "phobos", target, {"state": state, "gm": gm, "body": 1.0},
                   {"accel": np.zeros((3, sweeps * MC), mode), "close": np.zeros(sweeps, mode)},
                   sweep=sweeps)
    d = np.linalg.norm(state.astype(np.float64) - 1.0, axis=0).reshape(sweeps, MC)
    np.testing.assert_array_equal(got["close"], (d < 20.0).sum(axis=1))
    assert phobos.group == MC


@pytest.mark.parametrize("target", TARGETS)
def test_the_mean_over_an_axis_divides_by_its_extent(built, target):
    bundle, mode = built
    x = np.arange(8.0, dtype=mode)
    got, *_ = _run(bundle, "kmean", target, {"x": x}, {"out": np.zeros(2, mode)})
    np.testing.assert_allclose(got["out"], x.reshape(2, 4).mean(axis=1), rtol=TOL[mode])


@pytest.mark.parametrize("policy", ("auto", "compensated"))
@pytest.mark.parametrize("target", TARGETS)
def test_a_sum_over_8192_lanes_is_within_the_contract(built, target, policy):
    """Against ``math.fsum`` per key, scaled by ``eps * sum|t_i|``: at most 8
    plain, 1 compensated (the compensated value is the plane plus its companion
    ``out_c``). The error in ULP is printed as a card line."""
    bundle, mode = built
    name = "ksum_a" if policy == "auto" else "ksum_c"
    x = np.random.default_rng(8).uniform(-1, 1, (KP, 8192)).astype(mode)
    outs = {"out": np.zeros(KP, mode)}
    if policy == "compensated":
        outs["out_c"] = np.zeros(KP, mode)
    got, *_ = _run(bundle, name, target, {"x": x.ravel()}, outs, key=KP)
    total = got["out"].astype(np.float64) + got.get("out_c", 0.0)
    terms = x.astype(np.float64)
    exact = [math.fsum(r) for r in terms]
    scaled = _scaled(total, exact, np.abs(terms).sum(axis=1), mode)
    print(f"card {target} {mode} {policy}: scaled {scaled:.3g}, "
          f"ulp {_ulp(total, exact, mode):.3g}")
    assert scaled <= CONTRACT[policy]


# ---- the IR: one record for both doors ----------------------------------------


def test_a_full_sum_is_the_cell_it_always_was():
    """``hawk.sum(v)`` over every axis is the cell: the same IR, so the same
    text, as the reduction stored directly."""
    @kernel
    def k(x: Scalar, t: Mutable[Scalar]):
        t = hawk.sum(x * x)

    x = k.sinks[0].value
    assert [type(s) for s in k.sinks] == [SampleSum]
    assert Kernel("k", ReduceSamples("sum", x).stored("t")).walk.digest == k.walk.digest


def test_scatter_by_the_kept_coordinates_is_the_sum_over_the_rest():
    """``scatter(tb)[lane.channel, lane.input] += t`` and ``hawk.sum(t, over="sample")``
    commit the same keyed accumulate."""
    @kernel
    def summed(lane: Grid["channel":4, "input":8, "sample":32], x: Scalar,
               tb: Mutable["channel":4, "input":8]):
        tb = hawk.sum(x * x, over="sample")

    @kernel
    def scattered(lane: Grid["channel":4, "input":8, "sample":32], x: Scalar,
                  tb: Mutable["channel":4, "input":8]):
        hawk.scatter(tb, policy="plain")[lane.channel, lane.input] += x * x

    assert summed.walk.digest == scattered.walk.digest
    (sink,) = summed.sinks
    assert isinstance(sink, AccumWrite) and sink.policy == "auto"
    assert key_run(summed.sinks, sink).form == "proven"


@pytest.mark.parametrize("over", ("sample", -1, 2, ("sample",)))
def test_over_by_label_position_and_tuple_agree(over):
    @kernel
    def by(lane: Grid["channel":4, "input":8, "sample":32], x: Scalar) -> Scalar:
        return hawk.sum(x, over=over)

    @kernel
    def ref(lane: Grid["channel":4, "input":8, "sample":32], x: Scalar) -> Scalar:
        return hawk.sum(x, over="sample")

    assert by.walk.digest == ref.walk.digest
    assert [(a, e) for a, e, _s in by.planes["out"].dims] == [("channel", 4), ("input", 8)]


def test_over_two_axes_keeps_one_and_over_every_axis_is_the_cell():
    @kernel
    def two(lane: Grid["channel":4, "input":8, "sample":32], x: Scalar) -> Scalar:
        return hawk.sum(x, over=("input", "sample"))

    @kernel
    def every(lane: Grid["channel":4, "input":8, "sample":32], x: Scalar) -> Scalar:
        return hawk.sum(x, over=(0, 1, 2))

    assert [(a, e) for a, e, _s in two.planes["out"].dims] == [("channel", 4)]
    assert [type(s) for s in every.sinks] == [SampleSum]


@pytest.mark.parametrize("policy", ("auto", "compensated", "ordered"))
def test_every_policy_is_accepted_on_a_full_sum(policy):
    @kernel
    def k(x: Scalar, t: Mutable[Scalar]):
        t = hawk.sum(x, policy=policy)

    assert [type(s) for s in k.sinks] == [SampleSum]


def test_compensated_marks_the_sink_and_binds_its_companion():
    assert [s.policy for s in COMP.sinks if s.name == "out"] == ["compensated"]
    assert ("accum_out", "out_c") in COMP.arg_spec
    assert ("accum_out", "out_c") not in AUTO.arg_spec

    @kernel
    def k(x: Scalar, key: Index, acc: Mutable[Scalar]):
        hawk.scatter(acc, policy="compensated")[key] += x

    assert [(s.name, s.policy) for s in k.sinks] == [("acc", "compensated"), ("acc_c", "auto")]


# ---- refusals: one named test each, the text that migrates the author ------


def test_refuses_an_unknown_policy():
    with pytest.raises(HawkError, match=r"policy='fast' is not one of \('auto', "
                                        r"'compensated', 'ordered'\)"):
        @kernel
        def k(x: Scalar, t: Mutable[Scalar]):
            t = hawk.sum(x, policy="fast")


def test_refuses_ordered_over_some_axes_naming_what_works():
    with pytest.raises(HawkError, match=r"policy='ordered' over some of the lane axes is in "
                                        r"the next release; use policy='auto' or 'compensated'"):
        @kernel
        def k(lane: Grid["a", "b":4], x: Scalar) -> Scalar:
            return hawk.sum(x, over="b", policy="ordered")
    with pytest.raises(HawkError, match=r"'ordered' on a scatter is in the next release; use "
                                        r"policy='auto' or 'compensated'"):
        @kernel
        def k2(x: Scalar, key: Index, acc: Mutable[Scalar]):
            hawk.scatter(acc, policy="ordered")[key] += x


def test_refuses_a_policy_on_a_store():
    with pytest.raises(HawkError, match="a store has no sum contract"):
        @kernel
        def k(x: Scalar, key: Index, acc: Mutable[Scalar]):
            hawk.scatter(acc, policy="compensated")[key] = x


@pytest.mark.parametrize("op", ("max", "min"))
def test_refuses_max_and_min_over_some_axes_naming_what_works(op):
    with pytest.raises(HawkError, match=rf"a {op} over some of the lane axes is in the next "
                                        rf"release; hawk.{op}\(x\) \(one value for the run\)"):
        @kernel
        def k(lane: Grid["a", "b":4], x: Scalar) -> Scalar:
            return getattr(hawk, op)(x, over="b")


def test_refuses_a_reduction_inside_a_branch_or_a_for():
    with pytest.raises(HawkError, match="reduces over the lanes, a collective"):
        @kernel
        def k(x: Scalar, t: Mutable[Scalar]):
            t = x
            if x > 0.0:
                t = hawk.sum(x)
    with pytest.raises(HawkError, match="reduces over the lanes, a collective"):
        @kernel
        def k2(x: Scalar, t: Mutable[Scalar]):
            s = 0.0 * x
            for j in range(3):
                s = s + hawk.sum(x)
            t = s


def test_refuses_an_axis_the_grid_does_not_declare():
    with pytest.raises(HawkError, match=r"over='mc' names no lane axis; this kernel's Grid "
                                        r"declares \['a', 'b'\]"):
        @kernel
        def k(lane: Grid["a", "b":4], x: Scalar) -> Scalar:
            return hawk.sum(x, over="mc")
    with pytest.raises(HawkError, match=r"declares no Grid\[\.\.\.\], so its one axis is over=0"):
        @kernel
        def k2(x: Scalar) -> Scalar:
            return hawk.sum(x, over="sample")


def test_refuses_a_sink_whose_axes_are_not_the_kept_ones():
    with pytest.raises(HawkError, match=r"keeps the lane axes \[\"a\":2\], so 'tb' is declared "
                                        r"Mutable\[\.\.\., \"a\":2\]"):
        @kernel
        def k(lane: Grid["a":2, "b":4], x: Scalar, tb: Mutable["b":4]):
            tb = hawk.sum(x, over="b")


def test_refuses_an_at_of_an_output_or_a_computed_value():
    with pytest.raises(HawkError, match="only a plane that existed before the launch"):
        @kernel
        def k(x: Scalar, i: Index, y: Mutable[Scalar]):
            y = y.at(i) + x
    with pytest.raises(HawkError, match="only a plane that existed before the launch"):
        @kernel
        def k2(x: Scalar, i: Index, d: Mutable[Scalar]):
            d = (x * x).at(i)


def test_refuses_a_plane_at_two_extents():
    with pytest.raises(HawkError, match="one plane has one extent"):
        @kernel
        def k(x: Scalar, lane: Index, y: Mutable[Scalar]):
            y = x
            hawk.scatter(y)[lane] += 1.0
    with pytest.raises(HawkError, match="one plane has one extent"):
        @kernel
        def k2(x: Scalar, lane: Index, y: Mutable[Scalar]):
            hawk.scatter(y)[lane] += 1.0
            y = x


def test_refuses_an_index_on_a_locked_rank0_output():
    with pytest.raises(HawkError,
                       match=r"'bins'\[key\] indexes a component; a store by key is "
                             r"hawk.scatter\(bins\)\[key\]"):
        @kernel
        def k(lane: Index, x: Scalar, bins: Mutable[Scalar]):
            bins[lane] = x


def test_refuses_a_scatter_inside_a_branch_and_a_bare_scatter():
    with pytest.raises(HawkError, match="may not sit inside a branch"):
        @kernel
        def k(x: Scalar, lane: Index, y: Mutable[Scalar]):
            if x > 0.0:
                hawk.scatter(y)[lane] += 1.0
    with pytest.raises(HawkError, match=r"hawk.scatter\(y\) is the target of a store"):
        hawk.scatter(None)


def test_refuses_a_scatter_under_an_if_inside_a_for():
    with pytest.raises(HawkError, match="may not sit inside a branch"):
        @kernel
        def k(x: Scalar, lane: Index, y: Mutable[Scalar]):
            for a in range(3):
                if x > 0.0:
                    hawk.scatter(y)[lane + a] += 1.0


def test_a_scatter_in_a_for_is_one_commit_per_iteration():
    @kernel
    def looped(x: Scalar, lane: Index, y: Mutable[Scalar]):
        for a in range(3):
            hawk.scatter(y, policy="plain")[lane + a] += x * a

    @kernel
    def written(x: Scalar, lane: Index, y: Mutable[Scalar]):
        hawk.scatter(y, policy="plain")[lane + 0] += x * 0
        hawk.scatter(y, policy="plain")[lane + 1] += x * 1
        hawk.scatter(y, policy="plain")[lane + 2] += x * 2

    assert len(looped.sinks) == 3
    assert looped.walk.digest == written.walk.digest


def test_refuses_a_mask_as_a_key_naming_where():
    with pytest.raises(HawkError, match=r"a mask is not a key.*y = m.where\(mask, v, y\)"):
        @kernel
        def k(x: Scalar, y: Mutable[Scalar]):
            hawk.scatter(y)[x > 0.0] += x


def test_refuses_a_vector_by_key_into_a_flat_sink():
    with pytest.raises(HawkError, match="into a sink declared with named axes"):
        @kernel
        def k(lane: Index, u: Vector[3], acc: Mutable[Vector[3]]):
            hawk.scatter(acc)[lane] += u


def test_a_vector_by_key_into_a_named_axis_sink_is_payload_first():
    @kernel
    def k(lane: Index, u: Vector[3], acc: Mutable[Vector[3], "bin":4]):
        hawk.scatter(acc)[lane] += u

    assert len(k.sinks) == 3 and all(isinstance(s, AccumWrite) for s in k.sinks)


def test_masked_mean_names_the_count():
    with pytest.raises(HawkError, match=r"hawk.sum\(x, where=mask\) / hawk.sum\(mask\)"):
        @kernel
        def k(x: Scalar, t: Mutable[Scalar]):
            t = hawk.mean(x, where=x > 0.0)


def test_table_subscript_is_at():
    @kernel
    def k(t: Table[Scalar], i: Index, y: Mutable[Scalar]):
        y = t[i]

    @kernel
    def k2(t: Table[Scalar], i: Index, y: Mutable[Scalar]):
        y = t.at(i)

    assert k.walk.digest == k2.walk.digest


def test_traced_rows_and_columns_equal_numpy():
    rng = np.random.default_rng(2)
    mm = rng.standard_normal((2, 3))
    for i in (0, 1):
        got = evaluate(rows.sinks, {"mm": mm, "i": i})
        np.testing.assert_allclose(got["r"], mm[i])
        np.testing.assert_allclose(got["c"], mm[:, i])


def test_at_adjoint_is_a_scatter_add_of_the_seed():
    """d[s] = x[i[s]]^2 * x[s]: the adjoint of x[j] sums the seeds of every
    lane that read j, plus the own term -- against central differences."""
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
