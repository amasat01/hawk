# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.scatter(acc, contract="ordered")[key] += x``: a keyed add with no atomics.

Lanes are cut into chunks (``B = clamp(N // T, 1, 256)`` for ``N`` lanes and ``T`` cells), each chunk
adds its lanes in order into its own partial sum of every key, and each cell adds its partials in chunk
order. The same ``N`` and ``T`` give the same bits on every launch; the result is ``np.add.at`` up to
the order of the adds. Rows run on the host and (``gpu`` mark) the device, which the driver of
``_team_stages`` launches stage by stage."""

from __future__ import annotations

import numpy as np
import pytest
from _team_stages import needs_hawk, run_staged
from conftest import assert_within, fd_bound, sidecar_of, sum_bound

import hawk
import hawk.math as m
from hawk import HawkError, Index, Mutable, Scalar, Terminated, Vector, kernel
from hawk.ext import IndexSet, Kind
from hawk.artifact import build_bundle
from hawk.diff import vjp

SET = Kind("lane_set", lanes=IndexSet())
TARGETS = ["host", pytest.param("cuda", marks=pytest.mark.gpu)]


@kernel
def add(key: Index, x: Scalar, acc: Mutable[Scalar]):
    hawk.scatter(acc, contract="ordered")[key] += x


@kernel
def add_sq(key: Index, x: Scalar, acc: Mutable[Scalar]):
    hawk.scatter(acc, contract="ordered")[key] += x * x


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
    ``state`` (a dict) keeps the loaded host kernel for the next launch."""
    planes = {k: v.copy() if k == out else v for k, v in planes.items()}
    state = {} if state is None else state
    if target == "host":
        from hawk import runtime as rt

        kern = state.setdefault("kern", rt.load(bundle.directory, name, sidecar_of(bundle, name)))
        rt.run(kern, **planes)
        return planes[out]
    import cupy as cp

    dev = {k: cp.asarray(v) for k, v in planes.items()}
    run_staged(bundle, name, dev, target)
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
    assert_within(got, want, [sum_bound(x[key == cell]) for cell in range(cells)])


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


@pytest.mark.gpu
@pytest.mark.parametrize("n, cells, heavy", [(20000, 7, True), (1000, 1000, False)])
def test_the_ordered_scatter_is_bit_identical_on_the_host_and_the_device(bundles, n, cells, heavy):
    key, x = _case(n, cells, seed=n, heavy=heavy)
    host = _run(bundles["add"], "add", "host", key, x, cells)
    device = _run(bundles["add"], "add", "cuda", key, x, cells)
    np.testing.assert_array_equal(host, device)




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

        planes = {"key": cp.asarray(key), "x": cp.asarray(x), "bar_acc": cp.asarray(g), "bar_x": cp.zeros(n)}
        run_staged(bundles["grad"], "add_sq_grad", planes, target)
        bar = cp.asnumpy(planes["bar_x"])
    np.testing.assert_array_equal(bar, 2 * x * g[key])
    eps = 1e-6
    bound = fd_bound(eps, np.sum(np.abs(g[key]) * x * x))
    for i in np.random.default_rng(2).choice(n, 6, replace=False):
        hi, lo = x.copy(), x.copy()
        hi[i] += eps
        lo[i] -= eps
        assert_within((loss(hi) - loss(lo)) / (2 * eps), bar[i], bound)



# ---- lanes that add nothing, poisoned scratch, vector payloads, loops, two scatters ----------------

@kernel
def add_term(key: Index, x: Scalar, terminated: Terminated, acc: Mutable[Scalar]):
    hawk.scatter(acc, contract="ordered")[key] += x


def _add_set(key: Index, x: Scalar, acc: Mutable[Scalar]):
    hawk.scatter(acc, contract="ordered")[key] += x


@kernel
def add_masked(key: Index, x: Scalar, acc: Mutable[Scalar]):
    hawk.scatter(acc, contract="ordered")[key] += m.where(x > 0.0, x, 0.0)


@kernel
def add_vec(key: Index, v: Vector[3], acc: Mutable[Vector[3], "bin":9]):
    hawk.scatter(acc, contract="ordered")[key] += v


@kernel
def add_grid(row: Index, col: Index, x: Scalar, acc: Mutable[Scalar, "row":4, "col":5]):  # noqa: F821 axis labels
    hawk.scatter(acc, contract="ordered")[row, col] += x


@kernel
def add_vec_grid(row: Index, col: Index, v: Vector[3], acc: Mutable[Vector[3], "row":4, "col":5]):  # noqa: F821 axis labels
    hawk.scatter(acc, contract="ordered")[row, col] += v


@kernel
def add_loop(key: Index, x: Scalar, acc: Mutable[Scalar]):
    for a in range(3):
        hawk.scatter(acc, contract="ordered")[key + a] += x * a


@kernel
def add_two(key: Index, key2: Index, x: Scalar, acc: Mutable[Scalar]):
    hawk.scatter(acc, contract="ordered")[key] += x
    hawk.scatter(acc, contract="ordered")[key2] += x * 2.0


@pytest.fixture(scope="module")
def more(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("scatter_ordered_more")
    kernels = {"add_term": add_term, "add_masked": add_masked, "add_vec": add_vec,
               "add_grid": add_grid, "add_vec_grid": add_vec_grid,
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

        # the library makes the scratch planes between the stages and keeps them in the plan: the partial sums of the first
        # launch are what the second launch starts over, and its first stage must zero them
        bound = hawk.plan(rt.load(bundles["add"].directory, "add", sidecar_of(bundles["add"], "add"))).bind(
            key=key, x=x, acc=acc)
        bound.launch()
        again = np.zeros(9)
        bound.rebind(acc=again).launch()
        planes = [again]
    else:
        needs_hawk("read the planes between the stages of a split kernel and stop a run after stage k: hawk.run_stages(loaded, planes, *, stop=None) -> {plane name: array}")
        acc = _run(bundles["add"], "add", target, key, x, 9)
    assert planes and again.tobytes() == _run(bundles["add"], "add", target, key, x, 9).tobytes()


@pytest.mark.parametrize("target", TARGETS)
def test_a_vector_payload_adds_each_component_into_its_own_plane_of_cells(more, target):
    n = 4000
    rng = np.random.default_rng(13)
    key = rng.integers(0, 9, n).astype(np.int32)
    v = rng.normal(size=(3, n))
    planes = {"key": key, "v": np.ascontiguousarray(v), "acc": np.zeros((3, 9))}
    got = _bits([_exec(more["add_vec"], "add_vec", target, planes) for _ in range(3)])
    want = np.zeros((3, 9))
    for e in range(3):
        np.add.at(want[e], key, v[e])
    np.testing.assert_allclose(got.reshape(3, 9), want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


@pytest.mark.parametrize("target", TARGETS)
def test_a_scatter_at_several_key_axes_adds_into_the_cell_the_indices_name(more, target):
    """Reference: ``np.add.at`` with the row and column index arrays, per component for a vector."""
    n = 4000
    rng = np.random.default_rng(23)
    row = rng.integers(0, 4, n).astype(np.int32)
    col = rng.integers(0, 5, n).astype(np.int32)
    x = rng.normal(size=n)
    planes = {"row": row, "col": col, "x": x, "acc": np.zeros((4, 5))}
    got = _bits([_exec(more["add_grid"], "add_grid", target, planes) for _ in range(3)])
    want = np.zeros((4, 5))
    np.add.at(want, (row, col), x)
    np.testing.assert_allclose(got.reshape(4, 5), want, rtol=1e-12, atol=1e-12 * np.abs(want).max())

    v = rng.normal(size=(3, n))
    planes = {"row": row, "col": col, "v": v, "acc": np.zeros((3, 4, 5))}
    got = _bits([_exec(more["add_vec_grid"], "add_vec_grid", target, planes) for _ in range(3)])
    want = np.zeros((3, 4, 5))
    for e in range(3):
        np.add.at(want[e], (row, col), v[e])
    np.testing.assert_allclose(got.reshape(3, 4, 5), want, rtol=1e-12,
                               atol=1e-12 * np.abs(want).max())


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
            hawk.scatter(acc, contract="ordered")[key + a + b] += x * (a + 1)


@kernel
def add_step(key: Index, x: Scalar, acc: Mutable[Scalar]):
    for a in range(0, 6, 2):
        hawk.scatter(acc, contract="ordered")[key + a] += x


def test_a_run_time_trip_bound_that_scatters_is_refused_by_name():
    with pytest.raises(HawkError, match="`max=` belongs to a run-time `range"):
        @kernel
        def k(n: Index, key: Index, x: Scalar, acc: Mutable[Scalar]):
            for a in range(n, max=4):
                hawk.scatter(acc, contract="ordered")[key + a] += x


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
        halved["acc"] = np.zeros(cells)
        rt.run(kern, **halved)
        got = halved["acc"]
    else:
        needs_hawk("read the planes between the stages of a split kernel and stop a run after stage k: hawk.run_stages(loaded, planes, *, stop=None) -> {plane name: array}")
    _same(got, key[~term], x[~term], cells)


@pytest.mark.parametrize("target", TARGETS)
def test_a_key_out_of_range_stays_in_bounds_and_is_reported_by_name(bundles, target):
    n, cells = 4000, 9
    key, x = _case(n, cells, seed=29)
    for bad in (cells, cells + 5, -2, 2**31 - 2):
        wild = key.copy()
        wild[[3, n - 1]] = bad
        with pytest.raises(HawkError, match=r"hawk.scatter\(acc, contract='ordered'\) got a key outside \[0, 9\)"):
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
