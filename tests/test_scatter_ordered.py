# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.scatter(acc, policy="ordered")[key] += x``: a keyed add with no atomics.

Lanes are cut into chunks (``B = clamp(N // T, 1, 256)`` for ``N`` lanes and ``T`` cells), each chunk
adds its lanes in order into its own partial sum of every key, and each cell adds its partials in chunk
order. The same ``N`` and ``T`` give the same bits on every launch; the result is ``np.add.at`` up to
the order of the adds. Rows run on the host and (``gpu`` mark) the device, which the driver of
``_team_stages`` launches stage by stage."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import HawkError, Index, Mutable, Scalar, Terminated, Vector, kernel
from hawk.ext import IndexSet, Kind
from hawk._lane_bind import scatter_chunks
from hawk.artifact import build_bundle
from hawk.diff import vjp

SET = Kind("lane_set", lanes=IndexSet())
TARGETS = ["host", pytest.param("cuda", marks=pytest.mark.gpu)]


@kernel
def add(key: Index, x: Scalar, acc: Mutable[Scalar]):
    hawk.scatter(acc, policy="ordered")[key] += x


@kernel
def add_sq(key: Index, x: Scalar, acc: Mutable[Scalar]):
    hawk.scatter(acc, policy="ordered")[key] += x * x


@pytest.fixture(scope="module")
def bundles(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("scatter_ordered")
    out = {"add": build_bundle([add], root / "add", targets=("cuda", "host"), cache_dir=cache_dir,
                               host_profile="native")}
    sq = build_bundle([add_sq], root / "sq", targets=("cuda", "host"), cache_dir=cache_dir,
                      host_profile="native")
    back = vjp(add_sq, wrt=("x",), primal_unit=sq.digest)
    from hawk import Kernel

    grad = Kernel("add_sq_grad", back)
    out["sq"] = sq
    out["grad"] = build_bundle([grad], root / "grad", targets=("cuda", "host"), cache_dir=cache_dir,
                               host_profile="native")
    return out


def _exec(bundle, name, target, planes, out="acc", state=None):
    """One launch of kernel ``name`` over ``planes`` (numpy); the plane ``out`` as it is afterwards.
    ``state`` (a dict) keeps the loaded kernel and its scratch planes for the next launch."""
    planes = {k: v.copy() if k == out else v for k, v in planes.items()}
    state = {} if state is None else state
    if target == "host":
        from hawk import runtime as rt

        kern = state.setdefault("kern", rt.load(bundle.directory, name, sidecar_of(bundle, name)))
        rt.run(kern, **planes)
        return planes[out]
    import cupy as cp
    from _team_stages import run_device

    dev = {k: cp.asarray(v) for k, v in planes.items()}
    state["scratch"] = run_device(bundle, name, dev, len(planes["key"]), scratch=state.get("scratch"))
    return cp.asnumpy(dev[out])


def _run(bundle, name, target, key, x, cells, acc0=None):
    """One launch: ``acc`` after adding ``x`` at ``key`` into ``cells`` cells (from ``acc0``)."""
    return _exec(bundle, name, target, {"key": key, "x": x, "acc": np.zeros(cells) if acc0 is None else acc0})


def _case(n, cells, seed, heavy=False):
    rng = np.random.default_rng(seed)
    key = rng.integers(0, cells, n).astype(np.int32)
    if heavy:
        key[rng.random(n) < 0.9] = cells // 2       # most lanes on one cell
    x = rng.normal(size=n) * np.exp(rng.normal(size=n))
    return key, x


def _same(got, key, x, cells):
    want = np.zeros(cells)
    np.add.at(want, key, x)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * np.abs(want).max(initial=1.0))


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("n, cells, heavy", [
    (20000, 7, True),        # heavy collisions: 256 chunks
    (3000, 1, False),        # a single target
    (5000, 40, False),       # an empty cell: no key is 13 (set below)
    (50, 200, False),        # T >= N: one chunk
    (1000, 1000, False),     # T = N
])
def test_ordered_scatter_equals_add_at_and_replays_bitwise(bundles, target, n, cells, heavy):
    key, x = _case(n, cells, seed=n, heavy=heavy)
    if cells == 40:
        key[key == 13] = 14
    seen = [_run(bundles["add"], "add", target, key, x, cells) for _ in range(3)]
    assert seen[0].tobytes() == seen[1].tobytes() == seen[2].tobytes()
    _same(seen[0], key, x, cells)
    if cells == 40:
        assert seen[0][13] == 0.0
    # the sum adds onto what the plane held
    base = np.linspace(1.0, 2.0, cells)
    more = _run(bundles["add"], "add", target, key, x, cells, acc0=base)
    np.testing.assert_allclose(more - base, seen[0], rtol=1e-9, atol=1e-9 * np.abs(seen[0]).max())


def test_chunks_are_chosen_from_lanes_and_targets():
    assert [scatter_chunks(n, t) for n, t in ((20000, 7), (50, 200), (1000, 1000), (10**6, 10**4),
                                              (10**6, 1), (10, 0))] == [256, 1, 1, 100, 256, 10]


@pytest.mark.parametrize("target", TARGETS)
def test_the_vjp_is_a_gather_and_matches_finite_differences(bundles, target):
    n, cells = 3000, 11
    key, x = _case(n, cells, seed=5)
    g = np.random.default_rng(9).normal(size=cells)

    def loss(v):
        return float(g @ _run(bundles["sq"], "add_sq", target, key, v, cells))

    if target == "host":
        from hawk import runtime as rt

        bar = np.zeros(n)
        kern = rt.load(bundles["grad"].directory, "add_sq_grad", sidecar_of(bundles["grad"], "add_sq_grad"))
        rt.run(kern, key=key, x=x, bar_acc=g, bar_x=bar)
    else:
        import cupy as cp
        import _deploy as L
        import eagle.exec as eexec
        from eagle import plan as eplan

        planes = {"key": cp.asarray(key), "x": cp.asarray(x), "bar_acc": cp.asarray(g), "bar_x": cp.zeros(n)}
        sidecar = sidecar_of(bundles["grad"], "add_sq_grad")
        plugin = L.device_plugin(bundles["grad"].directory, "add_sq_grad", sidecar)
        eplan.plan(plugin, structure=eexec.DeviceKernel).bind(**planes).launch()
        cp.cuda.get_current_stream().synchronize()
        bar = cp.asnumpy(planes["bar_x"])
    np.testing.assert_allclose(bar, 2 * x * g[key], rtol=1e-12)
    eps = 1e-6
    for i in np.random.default_rng(2).choice(n, 6, replace=False):
        hi, lo = x.copy(), x.copy()
        hi[i] += eps
        lo[i] -= eps
        np.testing.assert_allclose((loss(hi) - loss(lo)) / (2 * eps), bar[i], rtol=1e-5, atol=1e-7)


def test_the_run_records_its_chunks(bundles):
    from hawk import runtime as rt

    key, x = _case(20000, 7, seed=1)
    kern = rt.load(bundles["add"].directory, "add", sidecar_of(bundles["add"], "add"))
    rt.run(kern, key=key, x=x, acc=np.zeros(7))
    assert kern.scatter_chunks == 256
    rt.run(kern, key=key[:50], x=x[:50], acc=np.zeros(200))
    assert kern.scatter_chunks == 1


# ---- lanes that add nothing, poisoned scratch, vector payloads, loops, two scatters ----------------

@kernel
def add_term(key: Index, x: Scalar, terminated: Terminated, acc: Mutable[Scalar]):
    hawk.scatter(acc, policy="ordered")[key] += x


def _add_set(key: Index, x: Scalar, acc: Mutable[Scalar]):
    hawk.scatter(acc, policy="ordered")[key] += x


@kernel
def add_masked(key: Index, x: Scalar, acc: Mutable[Scalar]):
    hawk.scatter(acc, policy="ordered")[key] += m.where(x > 0.0, x, 0.0)


@kernel
def add_vec(key: Index, v: Vector[3], acc: Mutable[Vector[3], "bin":9]):
    hawk.scatter(acc, policy="ordered")[key] += v


@kernel
def add_loop(key: Index, x: Scalar, acc: Mutable[Scalar]):
    for a in range(3):
        hawk.scatter(acc, policy="ordered")[key + a] += x * a


@kernel
def add_two(key: Index, key2: Index, x: Scalar, acc: Mutable[Scalar]):
    hawk.scatter(acc, policy="ordered")[key] += x
    hawk.scatter(acc, policy="ordered")[key2] += x * 2.0


@pytest.fixture(scope="module")
def more(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("scatter_ordered_more")
    kernels = {"add_term": add_term, "add_masked": add_masked, "add_vec": add_vec,
               "add_loop": add_loop, "add_two": add_two, "_add_set": hawk.kernel(_add_set, kind=SET)}
    return {n: build_bundle([k], root / n, targets=("cuda", "host"), cache_dir=cache_dir,
                            host_profile="native") for n, k in kernels.items()}


def _bits(runs):
    assert runs[0].tobytes() == runs[1].tobytes() == runs[2].tobytes()
    return runs[0]


@pytest.mark.parametrize("target", TARGETS)
def test_a_masked_lane_and_a_negative_key_add_nothing(more, target):
    n, cells = 6000, 9
    key, x = _case(n, cells, seed=2)
    live = np.random.default_rng(4).random(n) < 0.6
    skip = np.where(live, key, -1)
    got = _bits([_run(more["add_masked"], "add_masked", target, key, x, cells) for _ in range(3)])
    _same(got, key, np.where(x > 0, x, 0.0), cells)
    got = _bits([_run(more["add_masked"], "add_masked", target, skip, x, cells) for _ in range(3)])
    _same(got, skip[live], np.where(x > 0, x, 0.0)[live], cells)


@pytest.mark.parametrize("target", TARGETS)
def test_a_terminated_lane_adds_nothing_and_leaves_no_trace_in_the_next_run(more, target):
    n, cells = 6000, 9
    key, x = _case(n, cells, seed=3)
    rng = np.random.default_rng(6)
    first, second = rng.random(n) < 0.5, rng.random(n) < 0.3
    state: dict = {}
    for term in (first, second, first):      # a lane live in one run and terminated in the next
        planes = {"key": key, "x": x, "terminated": term, "acc": np.zeros(cells)}
        got = _bits([_exec(more["add_term"], "add_term", target, planes, state=state) for _ in range(3)])
        _same(got, key[~term], x[~term], cells)


@pytest.mark.parametrize("target", TARGETS)
def test_an_index_set_adds_the_listed_lanes_only(more, target):
    n, cells = 6000, 9
    key, x = _case(n, cells, seed=7)
    live = np.sort(np.random.default_rng(8).choice(n, 1700, replace=False)).astype(np.int32)
    other = np.sort(np.random.default_rng(9).choice(n, 900, replace=False)).astype(np.int32)
    state: dict = {}
    for lanes in (live, other, live):
        planes = {"key": key, "x": x, "acc": np.zeros(cells), "active_map": lanes,
                  "active_count": np.array([len(lanes)], np.uint32)}
        got = _bits([_exec(more["_add_set"], "_add_set", target, planes, state=state) for _ in range(3)])
        _same(got, key[lanes], x[lanes], cells)


@pytest.mark.parametrize("target", TARGETS)
def test_poisoned_partial_sums_do_not_change_the_result(bundles, target):
    """A launch that stopped between stages leaves partial sums behind: the first stage zeroes them."""
    key, x = _case(5000, 9, seed=11)
    acc = np.zeros(9)
    if target == "host":
        from hawk import runtime as rt

        kern = rt.load(bundles["add"].directory, "add", sidecar_of(bundles["add"], "add"))
        rt.run(kern, key=key, x=x, acc=acc)
        planes = [p for p in kern.__dict__["_split"]["team"].values() if p.dtype == np.float64]
        again = np.zeros(9)
        for plane in planes:
            plane[:] = 1e9
        rt.run(kern, key=key, x=x, acc=again)
    else:
        import cupy as cp
        from _team_stages import run_device

        dev = {"key": cp.asarray(key), "x": cp.asarray(x), "acc": cp.asarray(acc)}
        scratch = run_device(bundles["add"], "add", dev, len(key))
        planes = [p for n, p in scratch.items() if p.dtype == np.float64 and n.endswith("_p")]
        for plane in planes:
            plane[:] = 1e9
        dev["acc"] = cp.zeros(9)
        run_device(bundles["add"], "add", dev, len(key), scratch=scratch)
        again = cp.asnumpy(dev["acc"])
        acc = _run(bundles["add"], "add", target, key, x, 9)
    assert planes and again.tobytes() == _run(bundles["add"], "add", target, key, x, 9).tobytes()


@pytest.mark.parametrize("target", TARGETS)
def test_a_vector_payload_adds_each_component_into_its_own_plane_of_cells(more, target):
    n = 4000
    rng = np.random.default_rng(13)
    key = rng.integers(0, 9, n).astype(np.int32)
    v = rng.normal(size=(3, n))
    planes = {"key": key, "v": np.ascontiguousarray(v), "acc": np.zeros(27)}
    got = _bits([_exec(more["add_vec"], "add_vec", target, planes) for _ in range(3)])
    want = np.zeros((3, 9))
    for e in range(3):
        np.add.at(want[e], key, v[e])
    np.testing.assert_allclose(got.reshape(3, 9), want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


@pytest.mark.parametrize("target", TARGETS)
def test_a_scatter_in_a_loop_equals_the_unrolled_adds(more, target):
    n, cells = 4000, 12
    key, x = _case(n, cells - 2, seed=17)
    planes = {"key": key, "x": x, "acc": np.zeros(cells)}
    got = _bits([_exec(more["add_loop"], "add_loop", target, planes) for _ in range(3)])
    want = np.zeros(cells)
    for a in range(3):
        np.add.at(want, key + a, x * a)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


@pytest.mark.parametrize("target", TARGETS)
def test_two_ordered_scatters_into_one_plane_add_both(more, target):
    n, cells = 4000, 10
    key, x = _case(n, cells, seed=19)
    key2 = np.roll(key, 7)
    planes = {"key": key, "key2": key2, "x": x, "acc": np.zeros(cells)}
    got = _bits([_exec(more["add_two"], "add_two", target, planes) for _ in range(3)])
    want = np.zeros(cells)
    np.add.at(want, key, x)
    np.add.at(want, key2, x * 2.0)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


# ---- an aborted launch, the refusals by name, a key out of range -------------------------------------

@kernel
def add_nested(key: Index, x: Scalar, acc: Mutable[Scalar]):
    for a in range(2):
        for b in range(3):
            hawk.scatter(acc, policy="ordered")[key + a + b] += x * (a + 1)


@kernel
def add_step(key: Index, x: Scalar, acc: Mutable[Scalar]):
    for a in range(0, 6, 2):
        hawk.scatter(acc, policy="ordered")[key + a] += x


def test_a_run_time_trip_bound_that_scatters_is_refused_by_name():
    with pytest.raises(HawkError, match="`max=` belongs to a run-time `range"):
        @kernel
        def k(n: Index, key: Index, x: Scalar, acc: Mutable[Scalar]):
            for a in range(n, max=4):
                hawk.scatter(acc, policy="ordered")[key + a] += x


@pytest.mark.parametrize("target", TARGETS)
def test_nested_and_stepped_loops_are_unrolled_adds_in_program_order(bundles_loops, target):
    n, cells = 3000, 12
    key, x = _case(n, cells - 6, seed=31)
    got = _bits([_exec(bundles_loops["add_nested"], "add_nested", target,
                       {"key": key, "x": x, "acc": np.zeros(cells)}) for _ in range(3)])
    want = np.zeros(cells)
    for a in range(2):
        for b in range(3):
            np.add.at(want, key + a + b, x * (a + 1))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())
    got = _bits([_exec(bundles_loops["add_step"], "add_step", target,
                       {"key": key, "x": x, "acc": np.zeros(cells)}) for _ in range(3)])
    want = np.zeros(cells)
    for a in range(0, 6, 2):
        np.add.at(want, key + a, x)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


@pytest.fixture(scope="module")
def bundles_loops(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("scatter_ordered_loops")
    return {n: build_bundle([k], root / n, targets=("cuda", "host"), cache_dir=cache_dir, host_profile="native")
            for n, k in (("add_nested", add_nested), ("add_step", add_step))}


@pytest.mark.parametrize("target", TARGETS)
def test_a_launch_that_stopped_after_the_per_lane_stage_does_not_poison_the_next(more, target):
    """The aborted launch leaves live keys in every slot; the next run, whose lanes are half terminated,
    must add the live lanes only."""
    n, cells = 6000, 9
    key, x = _case(n, cells, seed=23)
    term = np.random.default_rng(5).random(n) < 0.5
    every = {"key": key, "x": x, "terminated": np.zeros(n, bool), "acc": np.zeros(cells)}
    halved = {**every, "terminated": term}
    bundle = more["add_term"]
    if target == "host":
        from hawk import runtime as rt

        kern = rt.load(bundle.directory, "add_term", sidecar_of(bundle, "add_term"))
        rt.run(kern, **every)       # makes the scratch
        held = kern.__dict__["_split"]
        scratch = rt._team_planes(kern, held, n, every)
        kern._bind_all({**every, **scratch}, n, None)
        kern._launch(0, n, n)       # the per-lane stage alone
        assert (next(p for k, p in scratch.items() if k.endswith("_k")) >= 0).any()
        halved["acc"] = np.zeros(cells)
        rt.run(kern, **halved)
        got = halved["acc"]
    else:
        import cupy as cp
        from _team_stages import run_device

        dev = {k: cp.asarray(v) for k, v in every.items()}
        scratch = run_device(bundle, "add_term", dev, n, only=1)
        dev = {k: cp.asarray(v) for k, v in halved.items()}
        run_device(bundle, "add_term", dev, n, scratch=scratch)
        got = cp.asnumpy(dev["acc"])
    _same(got, key[~term], x[~term], cells)


@pytest.mark.parametrize("target", TARGETS)
def test_a_key_out_of_range_stays_in_bounds_and_is_reported_by_name(bundles, target):
    n, cells = 4000, 9
    key, x = _case(n, cells, seed=29)
    for bad in (cells, cells + 5, -2, 2**31 - 2):
        wild = key.copy()
        wild[[3, n - 1]] = bad
        with pytest.raises(HawkError, match=r"hawk.scatter\(acc, policy='ordered'\) got a key outside \[0, 9\)"):
            _run(bundles["add"], "add", target, wild, x, cells)
    # the in-range adds are in, the out-of-range ones are dropped, and the next run is clean
    wild = key.copy()
    wild[7] = cells
    good = _run(bundles["add"], "add", target, key, x, cells)
    assert good.tobytes() == _run(bundles["add"], "add", target, key, x, cells).tobytes()


@pytest.mark.parametrize("target", TARGETS)
def test_a_target_of_no_cells_reports_any_key_but_the_skip_key(bundles, target):
    n = 300
    x = np.ones(n)
    _run(bundles["add"], "add", target, np.full(n, -1, np.int32), x, 0)       # all skipped: clean
    key = np.full(n, -1, np.int32)
    key[17] = 0
    with pytest.raises(HawkError, match=r"got a key outside \[0, 0\)"):
        _run(bundles["add"], "add", target, key, x, 0)
    _run(bundles["add"], "add", target, np.full(n, -1, np.int32), x, 0)       # and the next run is clean
