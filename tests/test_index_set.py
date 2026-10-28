# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A caller-built lane set: ``Kind(..., lanes=IndexSet())``.

Launch lane ``t < count`` runs lane ``map[t]`` of the kernel's own space, with no
mask. Rows: the source (reserved planes, the prologue, the unit key), runs against a
dense launch on the host and the device for a flat, a two-axis and a pair grid and
a plain kernel, in float32 and float64; keyed sums under the map (the ULP
contract, not bit identity); ``lane_position()``; the map word; the checks at bind.
"""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

import hawk
import hawk.math as m
from hawk import Grid, Index, Mutable, Scalar, Table
from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.emit import scalar_mode
from hawk.ext import DEFAULT_KIND, IndexSet, Kind
from hawk.ir import HawkError
from conftest import sidecar_of

B = importlib.import_module("hawk.artifact.bundle")

SET = Kind("lane_set", lanes=IndexSet())
N = 40
MAP = np.array([7, 2, 5, 31, 12], dtype=np.int32)


def _square(x: Scalar, y: Mutable[Scalar]):
    y = x * x + 1.0


def _sources(k, kind, name="sq"):
    unit = B._active_unit(k, name, kind)
    return B._sources(unit, name=name, smode=scalar_mode("float64"), kind=kind,
                      targets=("cuda", "host"), layout_sizes_override=None)


def test_a_lane_set_reads_the_map_prologue_without_a_mask():
    src = _sources(hawk.kernel(_square, kind=SET), SET)
    for target in ("cuda", "host"):
        assert "lut_active_map" in src[target].text and "hawk_t" in src[target].text
    assert src["cuda"].body == _sources(hawk.kernel(_square), DEFAULT_KIND)["cuda"].body


def test_the_unit_key_changes_only_for_the_kinds_that_declare_a_set():
    assert repr(IndexSet()) == repr(IndexSet(width=32)) != repr(IndexSet(width=64))
    assert "lanes" not in repr(DEFAULT_KIND)
    assert repr(SET) != repr(DEFAULT_KIND)
    assert repr(Kind("lane_set", lanes=IndexSet(width=32))) == repr(SET)



# --------------------------------------------------------------------------- #
# -- the dense-launch oracle: the same body under a dense and a set launch.
# --------------------------------------------------------------------------- #
MODES = ("float64", "float32")
PAIRS = 12                       # items of the pair grid
NPAIR = PAIRS * (PAIRS - 1) // 2


def _plain(x: Scalar, y: Mutable[Scalar]):
    y = x * x + 1.0


def _flat(lane: Grid["item"], x: Scalar, y: Mutable[Scalar]):
    y = x * 3.0 + lane.item


def _grid2(lane: Grid["a", "b":8], x: Scalar, y: Mutable[Scalar]):
    y = x + lane.a * 10.0 + lane.b


def _pair(lane: Grid["pair"], x: Table["node"], d: Mutable[Scalar], rs: Mutable["node"]):
    i, j = m.pair_index()
    t = x.at(i) * x.at(j) * x.at(j) * x.at(i) + 0.25 * x.at(j)
    d = t
    hawk.scatter(rs)[j] += t


BODIES = {"plain": _plain, "flat": _flat, "grid2": _grid2, "pair": _pair}
#: kernel -> (lanes in the dense space, launch extent keyword)
SPACE = {"plain": (N, {}), "flat": (N, {"item": N}), "grid2": (32, {"a": 4}), "pair": (NPAIR, {"pair": NPAIR})}
LIVE = {"plain": MAP, "flat": MAP, "grid2": np.array([30, 0, 9, 17], dtype=np.int32),
        "pair": np.array([0, 4, 17, 44, 65], dtype=np.int32)}


def _named(fn, tag):
    out = hawk.kernel(fn) if tag == "dense" else SET(fn)
    return out


@pytest.fixture(scope="module", params=MODES)
def bundles(request, tmp_path_factory, cache_dir):
    """``{(tag, kernel): (bundle, kernel)}`` for one scalar mode. The set kernels
    take the plain names with a suffix so the two bundles can sit side by side."""
    out = {}
    for tag in ("dense", "set"):
        # a bundle has one execution axis: the keyed pair kernel builds apart
        for group in (("plain", "flat", "grid2"), ("pair",)):
            ks = [_named(BODIES[name], tag) for name in group]
            bundle = build_bundle(ks, tmp_path_factory.mktemp(tag), mode=request.param,
                                  targets=("cuda", "host"), cache_dir=cache_dir,
                                  host_profile="native")
            out.update({(tag, name): (bundle, k) for name, k in zip(group, ks)})
    return request.param, out


def _data(name, mode):
    rng = np.random.default_rng(7)
    n, _ = SPACE[name]
    if name == "pair":
        return {"x": rng.uniform(0.5, 2.0, PAIRS).astype(mode)}
    return {"x": rng.uniform(-2.0, 2.0, n).astype(mode)}


def _outputs(name, mode):
    n, _ = SPACE[name]
    if name == "pair":
        return {"d": np.zeros(n, mode), "rs": np.zeros(PAIRS, mode)}
    return {"y": np.zeros(n, mode)}


def _host(entry):
    bundle, k = entry
    return rt.load(bundle.directory, k.name, sidecar_of(bundle, k.name))


def _run_host(bundles, tag, name, live=None):
    mode, built = bundles
    ins, outs = _data(name, mode), _outputs(name, mode)
    _, extent = SPACE[name]
    if live is not None:
        extent = dict(extent, active_map=live, active_count=np.array([len(live)], np.uint32))
    rt.run(_host(built[tag, name]), **ins, **outs, **extent,
           **({} if name != "plain" else {"n_samples": N}))
    return ins, outs


@pytest.mark.parametrize("name", list(BODIES))
def test_a_lane_set_equals_the_dense_launch_on_the_listed_lanes_on_the_host(bundles, name):
    mode = bundles[0]
    live = LIVE[name]
    _, dense = _run_host(bundles, "dense", name)
    _, got = _run_host(bundles, "set", name, live)
    lane_out = "d" if name == "pair" else "y"
    want = np.zeros_like(dense[lane_out])
    want[live] = dense[lane_out][live]
    assert np.array_equal(got[lane_out], want), "lane-local outputs are bit-exact"
    assert np.count_nonzero(got[lane_out]) == len(live)
    if name == "pair":
        # keyed sums under the map: order-rounded, so the ULP contract, not bit identity
        ins = _data(name, mode)["x"].astype(np.float64)
        i, j = np.array([_ij(k) for k in live]).T
        terms = ins[i] * ins[j] * ins[j] * ins[i] + 0.25 * ins[j]
        for node in range(PAIRS):
            sel = terms[j == node]
            err = abs(float(got["rs"][node]) - math.fsum(sel))
            assert err <= 8 * np.finfo(mode).eps * float(np.abs(sel).sum()) + 1e-300


def _ij(k):
    j = int((1 + (8 * k + 1) ** 0.5) / 2)
    while j * (j - 1) // 2 > k:
        j -= 1
    while (j + 1) * j // 2 <= k:
        j += 1
    return k - j * (j - 1) // 2, j


def _run_device(bundles, tag, name, live=None):
    import cupy as cp
    import _deploy as L
    import eagle.exec as eexec
    from eagle import plan as eplan

    mode, built = bundles
    bundle, k = built[tag, name]
    ins, outs = _data(name, mode), _outputs(name, mode)
    _, extent = SPACE[name]
    planes = {key: cp.asarray(v) for key, v in {**ins, **outs}.items()}
    if live is not None:
        planes["active_map"] = cp.asarray(live)
        planes["active_count"] = cp.asarray(np.array([len(live)], np.uint32))
    plugin = L.device_plugin(bundle.directory, k.name, sidecar_of(bundle, k.name))
    eplan.plan(plugin, structure=eexec.DeviceKernel).bind(**extent, **planes).launch()
    cp.cuda.Device().synchronize()
    return {key: planes[key].get() for key in outs}


@pytest.mark.gpu
@pytest.mark.parametrize("name", list(BODIES))
def test_a_lane_set_equals_the_dense_launch_on_the_listed_lanes_on_the_device(bundles, name):
    mode = bundles[0]
    live = LIVE[name]
    dense = _run_device(bundles, "dense", name)
    got = _run_device(bundles, "set", name, live)
    host = _run_host(bundles, "set", name, live)[1]
    lane_out = "d" if name == "pair" else "y"
    want = np.zeros_like(dense[lane_out])
    want[live] = dense[lane_out][live]
    assert np.array_equal(got[lane_out], want), "lane-local outputs match the dense device launch"
    tol = 1e-5 if mode == "float32" else 1e-13
    np.testing.assert_allclose(got[lane_out], host[lane_out], rtol=tol, atol=tol)
    if name == "pair":
        np.testing.assert_allclose(got["rs"], host["rs"], rtol=20 * tol, atol=20 * tol)


# --------------------------------------------------------------------------- #
# -- lane_position(): the launch slot, for a compacted output.
# --------------------------------------------------------------------------- #
def _compact(x: Scalar, out: Mutable[Scalar], slot: Mutable[Index]):
    out = x * 2.0
    slot = hawk.lane_position()


def test_lane_position_is_the_map_slot_under_a_set_and_the_lane_id_without_one(tmp_path, cache_dir):
    plain = hawk.kernel(_compact)
    mapped = SET(_compact)
    src = _sources(mapped, SET, name="compact")
    assert "static_cast<Int>(hawk_t)" in src["cuda"].text
    assert "static_cast<Int>(hawk_t)" in src["host"].text
    assert "hawk_t" not in _sources(plain, DEFAULT_KIND, name="compact")["host"].text
    bundle = build_bundle([mapped], tmp_path, targets=("host",), cache_dir=cache_dir,
                          host_profile="native")
    x = np.linspace(1.0, 2.0, N)
    slot = np.full(N, -1, np.int64)
    out = np.zeros(N)
    rt.run(rt.load(bundle.directory, "_compact", sidecar_of(bundle, "_compact")),
           x=x, out=out, slot=slot, n_samples=N, active_map=MAP,
           active_count=np.array([len(MAP)], np.uint32))
    assert list(slot[MAP]) == list(range(len(MAP))), "the slot a lane ran at, written at its own id"
    assert (np.delete(slot, MAP) == -1).all()


def _pack(x: Scalar, packed: Mutable["slot"]):
    hawk.scatter(packed, policy="plain")[hawk.lane_position()] += x * 2.0


@pytest.fixture(scope="module")
def pack_bundle(tmp_path_factory, cache_dir):
    return build_bundle([SET(_pack)], tmp_path_factory.mktemp("pack"), targets=("cuda", "host"),
                        cache_dir=cache_dir, host_profile="native")


def test_a_compacted_output_is_written_in_launch_order_on_the_host(pack_bundle):
    x = np.linspace(1.0, 2.0, N)
    packed = np.zeros(len(MAP) + 3)
    rt.run(rt.load(pack_bundle.directory, "_pack", sidecar_of(pack_bundle, "_pack")),
           x=x, packed=packed, n_samples=N, active_map=MAP,
           active_count=np.array([len(MAP)], np.uint32))
    np.testing.assert_array_equal(packed[:len(MAP)], x[MAP] * 2.0)
    assert not packed[len(MAP):].any()


@pytest.mark.gpu
def test_a_compacted_output_is_written_in_launch_order_on_the_device(pack_bundle):
    import _deploy as L
    import cupy as cp
    import eagle.exec as eexec
    from eagle import plan as eplan

    x = np.linspace(1.0, 2.0, N)
    packed = cp.zeros(len(MAP) + 3)
    planes = {"x": cp.asarray(x), "packed": packed, "active_map": cp.asarray(MAP),
              "active_count": cp.asarray(np.array([len(MAP)], np.uint32))}
    plugin = L.device_plugin(pack_bundle.directory, "_pack", sidecar_of(pack_bundle, "_pack"))
    eplan.plan(plugin, structure=eexec.DeviceKernel).bind(**planes).launch()
    cp.cuda.Device().synchronize()
    np.testing.assert_array_equal(packed.get()[:len(MAP)], x[MAP] * 2.0)


# --------------------------------------------------------------------------- #
# -- the map word, and the checks at bind.
# --------------------------------------------------------------------------- #
WIDE = 2**31 + 8


def _wide(lane: Grid["pair"], x: Table["node"], rs: Mutable["node"]):
    i, j = m.pair_index()
    hawk.scatter(rs)[j] += x.at(i) * x.at(j)


@pytest.fixture(scope="module")
def wide_bundles(tmp_path_factory, cache_dir):
    out = {}
    for bits in (32, 64):
        kind = Kind("wide", lanes=IndexSet(width=bits))
        out[bits] = build_bundle([kind(_wide)], tmp_path_factory.mktemp(f"w{bits}"),
                                 targets=("host",), cache_dir=cache_dir, host_profile="native")
    return out


def test_the_64_bit_unit_is_a_second_unit_and_the_32_bit_key_is_unchanged(wide_bundles):
    narrow, wide = wide_bundles[32], wide_bundles[64]
    assert narrow.digest != wide.digest
    plain = Kind("wide", lanes=IndexSet())
    assert repr(plain) == repr(Kind("wide", lanes=IndexSet(width=32)))
    mapped = sidecar_of(wide, "_wide")
    assert mapped["launch"]["index_set"] == {"width": 64}
    assert mapped["arg_dtypes"]["active_map"] == "int64"
    assert sidecar_of(narrow, "_wide")["arg_dtypes"]["active_map"] == "int32"
    assert sidecar_of(narrow, "_wide")["launch"]["index_set"] == {"width": 32}


def test_a_four_entry_map_over_2_pow_31_lanes_runs_on_the_64_bit_unit(wide_bundles):
    bundle = wide_bundles[64]
    lanes = np.array([3, 2**31 + 1, 2**31 - 1, WIDE - 1], dtype=np.int64)
    x = np.linspace(0.5, 1.5, 70000)
    rs = np.zeros(70000)
    rt.run(rt.load(bundle.directory, "_wide", sidecar_of(bundle, "_wide")), x=x, rs=rs,
           pair=WIDE, active_map=lanes, active_count=np.array([4], np.uint32))
    want = np.zeros(70000)
    for k in lanes:
        i, j = _ij(int(k))
        want[j] += x[i] * x[j]
    np.testing.assert_allclose(rs, want, rtol=1e-14)


def test_the_width_follows_the_extent_and_refuses_by_name():
    from hawk._lane_bind import index_width

    assert index_width(2**31 - 1) == 32 and index_width(2**31) == 64
    assert index_width(10, forced=64) == 64 and index_width(2**31, forced=64) == 64
    with pytest.raises(HawkError, match="cast the map to int32"):
        index_width(10, map_bits=64)
    with pytest.raises(HawkError, match="float32 kernel cannot hold"):
        index_width(2**31, scalar_type="float32")
    with pytest.raises(HawkError, match="width=32.*cannot address"):
        index_width(2**31, forced=32)
    with pytest.raises(HawkError, match="'auto', 32 or 64"):
        IndexSet(width=16)


@pytest.fixture(scope="module")
def checked(tmp_path_factory, cache_dir):
    bundle = build_bundle([SET(_square)], tmp_path_factory.mktemp("checked"), targets=("host",),
                          cache_dir=cache_dir, host_profile="native")
    return rt.load(bundle.directory, "_square", sidecar_of(bundle, "_square"))


def _bind(kernel, lanes, count=None, dtype=np.int32):
    x, y = np.ones(N), np.zeros(N)
    lanes = np.asarray(lanes, dtype=dtype)
    rt.run(kernel, x=x, y=y, n_samples=N, active_map=lanes,
           active_count=np.array([len(lanes) if count is None else count], np.uint32))
    return y


def test_a_good_map_runs_and_only_the_live_count_is_checked(checked):
    assert _bind(checked, [3, 9, 1]).sum() == 6.0
    assert _bind(checked, [3, 9, 9, 99], count=2).sum() == 4.0     # the tail is dead


def test_an_out_of_range_lane_is_refused_with_its_position(checked):
    with pytest.raises(HawkError, match=r"active_map\[2\] = 40 is outside the dense lanes \[0, 40\)"):
        _bind(checked, [1, 2, 40, 3])
    with pytest.raises(HawkError, match=r"active_map\[0\] = -1"):
        _bind(checked, [-1, 2])


def test_a_repeated_lane_is_refused_with_both_positions(checked):
    with pytest.raises(HawkError, match=r"active_map\[3\] = 5 repeats active_map\[1\]"):
        _bind(checked, [2, 5, 7, 5])


def test_an_int64_map_over_a_32_bit_unit_is_refused_by_name(checked):
    with pytest.raises(HawkError, match="cast the map to int32"):
        _bind(checked, [1, 2], dtype=np.int64)


def test_a_count_past_the_map_is_refused(checked):
    with pytest.raises(HawkError, match="active_count is 5, but active_map holds only 2"):
        _bind(checked, [1, 2], count=5)


def test_a_lane_set_in_a_segmented_unit_is_refused():
    from hawk.ext import Guard  # noqa: F401  (the thinning guard keeps its own refusals)

    @SET
    def seg(x: Scalar, kind: Scalar, y: Mutable[Scalar]):
        y = m.dispatch(m.int32(kind), [x, x * 2.0], policy="segmented")

    with pytest.raises(HawkError, match="segmented"):
        B._active_unit(seg, "seg", SET)


def test_a_lane_set_and_the_thinning_guard_are_not_declared_together():
    with pytest.raises(HawkError, match="declare one of them"):
        Kind("both", guard=hawk.ext.Guard(active_set=True), lanes=IndexSet())


# --------------------------------------------------------------------------- #
# -- derivatives under the map, against central differences of the same launch.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def derived(tmp_path_factory, cache_dir):
    from hawk.diff import jvp, vjp
    from hawk.trace.kernel import Kernel

    primal = SET(_pair)
    tangent = Kernel("pair_jvp", jvp(primal, wrt=("x",)), {}, kind=SET)
    adjoint = Kernel("pair_vjp", vjp(primal, wrt=("x",)), {}, kind=SET)
    bundle = build_bundle([primal, tangent, adjoint], tmp_path_factory.mktemp("derived"),
                          targets=("host",), cache_dir=cache_dir, host_profile="native",
                          derivative={"pair_jvp": "_pair", "pair_vjp": "_pair"})
    return {k: rt.load(bundle.directory, k, sidecar_of(bundle, k))
            for k in ("_pair", "pair_jvp", "pair_vjp")}


def _live():
    lanes = LIVE["pair"]
    return {"active_map": lanes, "active_count": np.array([len(lanes)], np.uint32)}


def _primal(derived, x):
    d, rs = np.zeros(NPAIR), np.zeros(PAIRS)
    rt.run(derived["_pair"], x=x, d=d, rs=rs, pair=NPAIR, **_live())
    return d, rs


def test_the_tangent_of_a_keyed_sum_under_a_set_matches_central_differences(derived):
    x = np.linspace(0.5, 1.5, PAIRS)
    v = np.random.default_rng(3).normal(size=PAIRS)
    dot_d, dot_rs, comp = np.zeros(NPAIR), np.zeros(PAIRS), np.zeros(PAIRS)
    rt.run(derived["pair_jvp"], x=x, dot_x=v, dot_d=dot_d, dot_rs=dot_rs, dot_rs_c=comp,
           pair=NPAIR, **_live())
    h = 1e-6
    (pd, prs), (md, mrs) = _primal(derived, x + h * v), _primal(derived, x - h * v)
    np.testing.assert_allclose(dot_d, (pd - md) / (2 * h), rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(dot_rs, (prs - mrs) / (2 * h), rtol=1e-8, atol=1e-8)
    assert np.count_nonzero(dot_d) == len(LIVE["pair"])


def test_the_adjoint_of_a_keyed_sum_under_a_set_matches_central_differences(derived):
    x = np.linspace(0.5, 1.5, PAIRS)
    rng = np.random.default_rng(4)
    u_d, u_rs = rng.normal(size=NPAIR), rng.normal(size=PAIRS)
    bar_x = np.zeros(PAIRS)
    rt.run(derived["pair_vjp"], x=x, bar_d=u_d, bar_rs=u_rs, bar_x=bar_x, pair=NPAIR, **_live())
    h = 1e-6
    want = np.zeros(PAIRS)
    for e in range(PAIRS):
        step = np.zeros(PAIRS)
        step[e] = h
        (pd, prs), (md, mrs) = _primal(derived, x + step), _primal(derived, x - step)
        want[e] = (u_d @ (pd - md) + u_rs @ (prs - mrs)) / (2 * h)
    np.testing.assert_allclose(bar_x, want, rtol=1e-7, atol=1e-8)


# --------------------------------------------------------------------------- #
# -- the three design patterns, one row each (host and device).
# --------------------------------------------------------------------------- #
KMAX = 5
WIDTH = 8


def _padded(v: Scalar, slot: Index, row: Mutable[hawk.Vector[WIDTH]]):
    r = m.vec(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    r[slot] = v
    row = r


def _segment(lo: Index, hi: Index, t: Table[Scalar], y: Mutable[Scalar]):
    s = 0.0
    for k in range(hi - lo, max=KMAX):
        s = s + t.at(lo + k)
    y = s


def _inverse(inv: Index, t: Table[Scalar], y: Mutable[Scalar]):
    y = t.at(inv) * 2.0


PATTERNS = {"padded": _padded, "segment": _segment, "inverse": _inverse}


def _pattern_inputs(name, mode):
    rng = np.random.default_rng(11)
    n = 24
    if name == "padded":
        return n, {"v": rng.uniform(1, 2, n).astype(mode), "slot": rng.integers(0, WIDTH, n)}, \
            lambda i: {"row": np.zeros((WIDTH, n), mode)}
    table = rng.uniform(1, 2, 64).astype(mode)
    if name == "segment":
        kind = np.sort(rng.integers(0, 6, n))                      # lanes sorted by kind
        lo = np.cumsum(np.r_[0, rng.integers(0, KMAX + 3, 5)])[kind]
        hi = lo + np.minimum(rng.integers(0, KMAX + 3, 6), 7)[kind]
        return n, {"lo": lo, "hi": hi, "t": table}, lambda i: {"y": np.zeros(n, mode)}
    order = rng.permutation(n)
    return n, {"inv": np.argsort(order), "t": table}, lambda i: {"y": np.zeros(n, mode)}


def _pattern_want(name, ins):
    if name == "padded":
        want = np.zeros((WIDTH, len(ins["v"])))
        want[ins["slot"], np.arange(len(ins["v"]))] = ins["v"]
        return want
    if name == "segment":
        return np.array([ins["t"][lo:lo + min(hi - lo, KMAX)].sum()
                         for lo, hi in zip(ins["lo"], ins["hi"])])
    return ins["t"][ins["inv"]] * 2.0


@pytest.fixture(scope="module", params=MODES)
def pattern_bundles(request, tmp_path_factory, cache_dir):
    ks = [hawk.kernel(fn) for fn in PATTERNS.values()]
    out = {}
    for name, k in zip(PATTERNS, ks):
        b = build_bundle([k], tmp_path_factory.mktemp(name), mode=request.param,
                         targets=("cuda", "host"), cache_dir=cache_dir, host_profile="native")
        out[name] = (b, k)
    return request.param, out


@pytest.mark.parametrize("name", list(PATTERNS))
def test_the_design_pattern_matches_numpy_on_the_host(pattern_bundles, name):
    mode, built = pattern_bundles
    bundle, k = built[name]
    n, ins, outs = _pattern_inputs(name, mode)
    outs = outs(0)
    rt.run(rt.load(bundle.directory, k.name, sidecar_of(bundle, k.name)), **ins, **outs, n_samples=n)
    got = next(iter(outs.values()))
    np.testing.assert_allclose(got, _pattern_want(name, ins), rtol=1e-5 if mode == "float32" else 1e-13)


@pytest.mark.gpu
@pytest.mark.parametrize("name", list(PATTERNS))
def test_the_design_pattern_matches_numpy_on_the_device(pattern_bundles, name):
    import _deploy as L
    import cupy as cp
    import eagle.exec as eexec
    from eagle import plan as eplan

    mode, built = pattern_bundles
    bundle, k = built[name]
    n, ins, outs = _pattern_inputs(name, mode)
    outs = outs(0)
    planes = {key: cp.asarray(v) for key, v in {**ins, **outs}.items()}
    plugin = L.device_plugin(bundle.directory, k.name, sidecar_of(bundle, k.name))
    eplan.plan(plugin, structure=eexec.DeviceKernel).bind(**planes).launch()
    cp.cuda.Device().synchronize()
    got = planes[next(iter(outs))].get()
    np.testing.assert_allclose(got, _pattern_want(name, ins), rtol=1e-5 if mode == "float32" else 1e-13)


# --------------------------------------------------------------------------- #
# -- the C target, and lane_position in other units.
# --------------------------------------------------------------------------- #
def test_the_c_target_refuses_a_lane_set_and_spells_lane_position_as_the_loop_index():
    from hawk.embedded import render as render_c

    with pytest.raises(HawkError, match="index map"):
        render_c(SET(_square), mode="float64")
    text = render_c(hawk.kernel(_compact), mode="float64").text
    assert "= hawk_i;" in text or "hawk_i)" in text


def test_lane_position_in_a_segmented_unit_is_refused(tmp_path, cache_dir):
    @hawk.kernel
    def seg(x: Scalar, kind: Scalar, slot: Mutable[Index], y: Mutable[Scalar]):
        y = m.dispatch(m.int32(kind), [x, x * 2.0], policy="segmented")
        slot = hawk.lane_position()

    with pytest.raises(HawkError, match="launch slot.*segmented"):
        build_bundle([seg], tmp_path, targets=("host",), cache_dir=cache_dir)
