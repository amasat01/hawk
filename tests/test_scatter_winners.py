# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.scatter(y, contract="ordered")[key].min(v)`` and ``.max(v)`` record the winner of each element.

The plane holds the value an unordered minimum or maximum gives, bit for bit; the plan also reads which lane gave it
(:meth:`hawk.runtime.BoundPlan.winner`): the lowest lane among those whose value equals the finished element, ``-1`` for an element
no lane reached. The oracle of each test is a NumPy scan: the first index of ``np.flatnonzero(key == k)`` whose value equals the
minimum (maximum) of the lanes on key ``k``.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import _launch

import hawk
from hawk import HawkError, Index, Int32, Mutable, Scalar, Vector, kernel
from hawk.artifact import build_bundle

N = 3000
CELLS = 5


@kernel
def ranked(key: Index, x: Scalar, lo: Mutable[Scalar], hi: Mutable[Scalar]):
    hawk.scatter(lo, contract="ordered")[key].min(x)
    hawk.scatter(hi, contract="ordered")[key].max(x)


@kernel
def unranked(key: Index, x: Scalar, lo: Mutable[Scalar], hi: Mutable[Scalar]):
    hawk.scatter(lo)[key].min(x)
    hawk.scatter(hi)[key].max(x)


@kernel
def ranked_ints(key: Index, v: Int32, lo: Mutable[Int32], hi: Mutable[Int32]):
    hawk.scatter(lo, contract="ordered")[key].min(v)
    hawk.scatter(hi, contract="ordered")[key].max(v)


@kernel
def ranked_two_axes(i: Index, k: Index, x: Scalar, lo: Mutable[Scalar, "row":3, "col":4]):  # noqa: F821
    hawk.scatter(lo, contract="ordered")[i, k].min(x)


@kernel
def ranked_by_an_odd_name(key: Index, x: Scalar, best_winner: Mutable[Scalar]):
    hawk.scatter(best_winner, contract="ordered")[key].min(x)


@pytest.fixture(scope="module", params=["float64", "float32"])
def mode(request):
    return request.param


@pytest.fixture(scope="module")
def bundle(mode, tmp_path_factory, cache_dir):
    return build_bundle([ranked, unranked, ranked_ints, ranked_two_axes, ranked_by_an_odd_name],
                        tmp_path_factory.mktemp("winners"), mode=mode, targets=("host",), cache_dir=cache_dir)


def _real(mode):
    return np.float32 if mode == "float32" else np.float64


def _plan(bundle, name, planes, structure):
    import hawk as h

    return h.plan(h.load(bundle.directory, name), structure).bind(**planes)


def _winners_by_scan(key, value, cells, pick):
    """The lowest lane among those whose value equals the element's finished value, per element of a flat plane; -1 for none."""
    want = np.full(cells, -1, np.int32)
    for cell in range(cells):
        lanes = np.flatnonzero(key == cell)
        if lanes.size:
            want[cell] = lanes[np.flatnonzero(value[lanes] == pick(value[lanes]))[0]]
    return want


def _tied(rng, real):
    """Lanes crowded on few cells whose values are drawn from three numbers, so that most cells have many tied lanes."""
    key = np.where(rng.random(N) < 0.5, 0, rng.integers(0, CELLS, N)).astype(np.int32)
    return key, rng.choice(np.array([-1.5, 0.25, 7.0]), N).astype(real)


@pytest.mark.parametrize("structure", [hawk.HostSerial, hawk.HostTeam], ids=["one-thread", "team"])
def test_the_winner_of_a_minimum_and_a_maximum_is_the_lowest_tied_lane(bundle, mode, structure):
    key, x = _tied(np.random.default_rng(1), _real(mode))
    lo, hi = np.full(CELLS + 1, np.inf, x.dtype), np.full(CELLS + 1, -np.inf, x.dtype)
    plan = _plan(bundle, "ranked", {"key": key, "x": x, "lo": lo, "hi": hi}, structure)
    plan.launch()
    assert plan.winner("lo")[:CELLS].tolist() == _winners_by_scan(key, x, CELLS, np.min).tolist()
    assert plan.winner("hi")[:CELLS].tolist() == _winners_by_scan(key, x, CELLS, np.max).tolist()
    assert plan.winner("lo")[CELLS] == -1 and plan.winner("hi")[CELLS] == -1


def test_a_team_run_gives_the_winners_of_the_one_thread_run(bundle, mode):
    key, x = _tied(np.random.default_rng(2), _real(mode))
    found = []
    for structure in (hawk.HostSerial, hawk.HostTeam):
        lo, hi = np.full(CELLS, np.inf, x.dtype), np.full(CELLS, -np.inf, x.dtype)
        plan = _plan(bundle, "ranked", {"key": key, "x": x, "lo": lo, "hi": hi}, structure)
        plan.launch()
        found.append((plan.winner("lo").copy(), plan.winner("hi").copy()))
    assert all(a.tolist() == b.tolist() for a, b in zip(*found))


def test_the_plane_is_the_one_an_unordered_fold_gives(bundle, mode):
    key, x = _tied(np.random.default_rng(3), _real(mode))
    planes = {}
    for name in ("ranked", "unranked"):
        planes[name] = {"lo": np.full(CELLS, np.inf, x.dtype), "hi": np.full(CELLS, -np.inf, x.dtype)}
        _plan(bundle, name, {"key": key, "x": x, **planes[name]}, hawk.HostTeam).launch()
    assert planes["ranked"]["lo"].tobytes() == planes["unranked"]["lo"].tobytes()
    assert planes["ranked"]["hi"].tobytes() == planes["unranked"]["hi"].tobytes()


def test_integer_winners_equal_the_scan(bundle):
    rng = np.random.default_rng(4)
    key = np.where(rng.random(N) < 0.5, 0, rng.integers(0, CELLS, N)).astype(np.int32)
    v = rng.choice(np.array([-3, 4, 9]), N).astype(np.int32)
    lo, hi = np.full(CELLS, np.iinfo(np.int32).max, np.int32), np.full(CELLS, np.iinfo(np.int32).min, np.int32)
    plan = _plan(bundle, "ranked_ints", {"key": key, "v": v, "lo": lo, "hi": hi}, hawk.HostTeam)
    plan.launch()
    assert plan.winner("lo").tolist() == _winners_by_scan(key, v, CELLS, np.min).tolist()
    assert plan.winner("hi").tolist() == _winners_by_scan(key, v, CELLS, np.max).tolist()


def test_a_plane_with_two_axes_has_a_winner_per_element(bundle, mode):
    rng = np.random.default_rng(5)
    real = _real(mode)
    i, k = (np.where(rng.random(N) < 0.5, 1, rng.integers(0, n, N)).astype(np.int32) for n in (3, 4))
    x = rng.choice(np.array([2.0, 3.0, 5.0]), N).astype(real)
    lo = np.full((3, 4), np.inf, real)
    plan = _plan(bundle, "ranked_two_axes", {"i": i, "k": k, "x": x, "lo": lo}, hawk.HostTeam)
    plan.launch()
    want = _winners_by_scan((i * 4 + k).astype(np.int32), x, 12, np.min).reshape(3, 4)
    assert plan.winner("lo").shape == (3, 4)
    assert plan.winner("lo").tolist() == want.tolist()


def test_a_winner_plane_the_caller_binds_is_filled_in_place(bundle, mode):
    key, x = _tied(np.random.default_rng(6), _real(mode))
    lo, hi = np.full(CELLS, np.inf, x.dtype), np.full(CELLS, -np.inf, x.dtype)
    given = np.full(CELLS, np.iinfo(np.uint32).max, np.uint32)
    plan = _plan(bundle, "ranked", {"key": key, "x": x, "lo": lo, "hi": hi, "lo_winner": given}, hawk.HostTeam)
    plan.launch()
    assert given.view(np.int32).tolist() == _winners_by_scan(key, x, CELLS, np.min).tolist()
    assert plan.winner("lo").tolist() == given.view(np.int32).tolist()


def test_a_bound_winner_plane_of_another_shape_than_its_plane_is_refused_by_name(bundle, mode):
    key, x = _tied(np.random.default_rng(6), _real(mode))
    lo, hi = np.full(CELLS, np.inf, x.dtype), np.full(CELLS, -np.inf, x.dtype)
    short = np.full(CELLS - 2, np.iinfo(np.uint32).max, np.uint32)
    with pytest.raises(hawk.HawkError, match="one winner per element of 'lo'"):
        _plan(bundle, "ranked", {"key": key, "x": x, "lo": lo, "hi": hi, "lo_winner": short}, hawk.HostTeam).launch()


def test_the_winner_of_a_plane_is_found_by_the_relation_the_unit_states_not_by_a_name(bundle, mode):
    key, x = _tied(np.random.default_rng(7), _real(mode))
    plane = np.full(CELLS, np.inf, x.dtype)
    plan = _plan(bundle, "ranked_by_an_odd_name", {"key": key, "x": x, "best_winner": plane}, hawk.HostTeam)
    plan.launch()
    assert plan.winner("best_winner").tolist() == _winners_by_scan(key, x, CELLS, np.min).tolist()
    with pytest.raises(HawkError, match="not a plane with winners"):
        plan.winner("best_winner_winner")


def test_equal_values_of_either_zero_tie_and_a_nan_element_has_no_winner(bundle, mode):
    real = _real(mode)
    x = np.array([0.0, -0.0, 5.0, np.nan, 1.0, np.nan, 2.0, 2.0], real)
    key = np.array([0, 0, 0, 1, 1, 2, 3, 3], np.int32)
    lo, hi = np.full(5, np.inf, real), np.full(5, -np.inf, real)
    plan = _plan(bundle, "ranked", {"key": key, "x": x, "lo": lo, "hi": hi}, hawk.HostSerial)
    plan.launch()
    # +0.0 and -0.0 compare equal: lane 0 wins the minimum; a NaN absorbs in the plane, which no lane's value equals; cell 4 is unwritten
    assert np.isnan(lo[1]) and np.isnan(lo[2]) and lo[0] == 0 and lo[3] == 2
    assert plan.winner("lo").tolist() == [0, -1, -1, 6, -1]
    assert plan.winner("hi").tolist() == [2, -1, -1, 6, -1]


def test_ordered_is_refused_by_name_where_it_has_no_winner():
    with pytest.raises(HawkError, match="contract='ordered' records the winning lane of each element of a scalar plane"):
        @kernel
        def vectors(key: Index, v: Vector[2], y: Mutable[Vector[2], "bin":4]):  # noqa: F821
            hawk.scatter(y, contract="ordered")[key].min(v)
    with pytest.raises(HawkError, match=r"has no sum contract; contract= is for \+= and \.add, and 'ordered' for \.min and \.max"):
        @kernel
        def plain(key: Index, x: Scalar, y: Mutable[Scalar]):
            hawk.scatter(y, contract="plain")[key].min(x)
    with pytest.raises(HawkError, match="has no sum contract"):
        @kernel
        def masks(key: Index, b: hawk.UInt32, y: Mutable[hawk.UInt32]):
            hawk.scatter(y, contract="ordered")[key].and_(b)


def test_a_plane_without_winners_is_refused_by_name_when_asked_for_them(bundle, mode):
    real = _real(mode)
    planes = {"key": np.zeros(2, np.int32), "x": np.ones(2, real), "lo": np.full(1, np.inf, real),
              "hi": np.full(1, -np.inf, real)}
    plan = _plan(bundle, "unranked", planes, hawk.HostSerial)
    with pytest.raises(HawkError, match="is not a plane with winners"):
        plan.winner("lo")


@pytest.mark.gpu
def test_the_device_winners_equal_the_scan(bundle, mode):
    import cupy as cp

    key, x = _tied(np.random.default_rng(8), _real(mode))
    lo, hi = cp.full(CELLS, np.inf, x.dtype), cp.full(CELLS, -np.inf, x.dtype)
    plan = _launch(bundle, "ranked", "cuda", {"key": cp.asarray(key), "x": cp.asarray(x), "lo": lo, "hi": hi})
    plan.launch()
    cp.cuda.Device().synchronize()
    assert plan.winner("lo").tolist() == _winners_by_scan(key, x, CELLS, np.min).tolist()
    assert plan.winner("hi").tolist() == _winners_by_scan(key, x, CELLS, np.max).tolist()
