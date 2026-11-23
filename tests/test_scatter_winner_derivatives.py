# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``jvp`` and ``vjp`` of ``hawk.scatter(y, contract="ordered")[k].min(v)`` and ``.max(v)``.

The element of ``y`` is the value of its winning lane, so its tangent is that lane's tangent, and the adjoint of a lane's value
is the element's adjoint when the lane won, zero otherwise. Both derivatives read the winners of the primal run: the caller binds
``plan.winner("y")`` of that run under the winner plane's name (``y_winner``). The oracles are the closed forms written with a
NumPy first-lane scan, a central finite difference on inputs without ties, and the transpose identity between the two modes.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import _launch  # noqa: F401  (keeps the host-only hooks of the suite in force)

import hawk
from hawk import Index, Mutable, Scalar, kernel
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp
from hawk.trace.kernel import Kernel

N = 600
CELLS = 5
UNIT = "0" * 64


@kernel
def folds(key: Index, x: Scalar, lo: Mutable[Scalar], hi: Mutable[Scalar]):
    hawk.scatter(lo, contract="ordered")[key].min(x)
    hawk.scatter(hi, contract="ordered")[key].max(x)


STRUCTURES = pytest.mark.parametrize("structure", [hawk.HostSerial, hawk.HostTeam], ids=["one_thread", "team"])


@pytest.fixture(scope="module")
def built(tmp_path_factory, cache_dir):
    """The bundles of the primal, its forward derivative and its reverse derivative (a bundle holds one execution axis)."""
    tangent = Kernel("folds_jvp", jvp(folds, wrt=("x",), primal_unit=UNIT))
    reverse = Kernel("folds_vjp", vjp(folds, primal_unit=UNIT))
    return {k.name: build_bundle([k], tmp_path_factory.mktemp(k.name), mode="float64", targets=("host",),
                                 cache_dir=cache_dir)
            for k in (folds, tangent, reverse)}


def _plan(built, name, structure, **planes):
    return hawk.plan(hawk.load(built[name].directory, name), structure).bind(**planes)


def _primal(built, structure, key, x, lo, hi):
    """The primal run: the planes it leaves and the winners it records."""
    lo, hi = np.full(CELLS, lo, np.float64), np.full(CELLS, hi, np.float64)
    plan = _plan(built, "folds", structure, key=key, x=x, lo=lo, hi=hi)
    plan.launch()
    return lo, hi, plan.winner("lo").copy(), plan.winner("hi").copy()


def _tied(seed):
    rng = np.random.default_rng(seed)
    key = np.where(rng.random(N) < 0.5, 0, rng.integers(0, CELLS - 1, N)).astype(np.int32)   # the last cell is never hit
    return key, rng.choice(np.array([-1.5, 0.25, 7.0]), N)


def _tangent_by_scan(winner, dx, incoming):
    """The tangent of each element: its winning lane's, else the incoming one (the first-lane scan is in ``winner``)."""
    return np.where(winner >= 0, dx[np.maximum(winner, 0)], incoming)


def _forward(built, structure, key, x, dx, winners, incoming):
    dot_lo, dot_hi = np.full(CELLS, incoming), np.full(CELLS, incoming)
    _plan(built, "folds_jvp", structure, key=key, dot_x=dx, lo_winner=winners[0], hi_winner=winners[1],
          dot_lo=dot_lo, dot_hi=dot_hi).launch()
    return dot_lo, dot_hi


def _reverse(built, structure, key, x, winners, bar_lo, bar_hi):
    bar_x = np.zeros(N)
    _plan(built, "folds_vjp", structure, key=key, lo_winner=winners[0], hi_winner=winners[1],
          bar_lo=bar_lo, bar_hi=bar_hi, bar_x=bar_x).launch()
    return bar_x


@STRUCTURES
def test_the_tangent_of_an_element_is_the_tangent_of_its_winning_lane(built, structure):
    # oracle: the tangent of the lane the first-lane scan names as the winner; an element no lane wrote keeps its incoming tangent
    key, x = _tied(1)
    dx = np.random.default_rng(2).normal(size=N)
    *_, win_lo, win_hi = _primal(built, structure, key, x, np.inf, -np.inf)
    assert (win_lo[:CELLS - 1] >= 0).all() and win_lo[CELLS - 1] == -1
    got_lo, got_hi = _forward(built, structure, key, x, dx, (win_lo, win_hi), 7.5)
    assert got_lo.tolist() == _tangent_by_scan(win_lo, dx, 7.5).tolist()
    assert got_hi.tolist() == _tangent_by_scan(win_hi, dx, 7.5).tolist()
    assert got_lo[CELLS - 1] == 7.5


@STRUCTURES
def test_the_tangent_equals_a_finite_difference_on_inputs_without_ties(built, structure):
    # oracle: the central difference of the primal plane, a value that is exact on a piecewise linear function
    rng = np.random.default_rng(3)
    key = rng.integers(0, CELLS, N).astype(np.int32)
    x, dx, eps = rng.normal(size=N), rng.normal(size=N), 1e-6
    lo, hi, win_lo, win_hi = _primal(built, structure, key, x, np.inf, -np.inf)
    got_lo, got_hi = _forward(built, structure, key, x, dx, (win_lo, win_hi), 0.0)
    ahead, behind = (_primal(built, structure, key, x + s * eps * dx, np.inf, -np.inf) for s in (1, -1))
    assert np.allclose(got_lo, (ahead[0] - behind[0]) / (2 * eps), atol=1e-7)
    assert np.allclose(got_hi, (ahead[1] - behind[1]) / (2 * eps), atol=1e-7)


@STRUCTURES
def test_the_adjoint_of_a_lane_is_its_elements_adjoint_when_it_won_else_zero(built, structure):
    # oracle: the closed form, written with the winners of a first-lane scan, on heavy ties
    key, x = _tied(4)
    bar_lo, bar_hi = np.random.default_rng(5).normal(size=(2, CELLS))
    *_, win_lo, win_hi = _primal(built, structure, key, x, np.inf, -np.inf)
    lanes = np.arange(N)
    want = np.where(win_lo[key] == lanes, bar_lo[key], 0.0) + np.where(win_hi[key] == lanes, bar_hi[key], 0.0)
    got = _reverse(built, structure, key, x, (win_lo, win_hi), bar_lo, bar_hi)
    assert got.tolist() == want.tolist()
    assert np.count_nonzero(got) <= 2 * CELLS


@STRUCTURES
def test_the_reverse_is_the_transpose_of_the_forward_on_heavy_ties(built, structure):
    # oracle: <bar_y, J v> = <J^T bar_y, v> with J built by the forward kernel and J^T by the reverse kernel
    key, x = _tied(6)
    rng = np.random.default_rng(7)
    dx, bar_lo, bar_hi = rng.normal(size=N), rng.normal(size=CELLS), rng.normal(size=CELLS)
    *_, win_lo, win_hi = _primal(built, structure, key, x, np.inf, -np.inf)
    dot_lo, dot_hi = _forward(built, structure, key, x, dx, (win_lo, win_hi), 0.0)
    bar_x = _reverse(built, structure, key, x, (win_lo, win_hi), bar_lo, bar_hi)
    assert np.isclose(bar_lo @ dot_lo + bar_hi @ dot_hi, bar_x @ dx, rtol=1e-12)


@STRUCTURES
def test_a_plane_whose_start_value_wins_keeps_its_incoming_tangent_and_gives_no_lane_an_adjoint(built, structure):
    # the minimum starts below every lane and the maximum above: no lane equals the finished element, so no lane won
    key, x = _tied(8)
    dx = np.random.default_rng(9).normal(size=N)
    lo, hi, win_lo, win_hi = _primal(built, structure, key, x, -100.0, 100.0)
    assert (win_lo == -1).all() and (win_hi == -1).all() and (lo == -100.0).all()
    got_lo, got_hi = _forward(built, structure, key, x, dx, (win_lo, win_hi), 3.0)
    assert (got_lo == 3.0).all() and (got_hi == 3.0).all()
    assert not _reverse(built, structure, key, x, (win_lo, win_hi), np.ones(CELLS), np.ones(CELLS)).any()


@pytest.mark.parametrize("derive", [jvp, vjp], ids=["jvp", "vjp"])
def test_a_fold_without_the_ordered_contract_is_refused_with_the_way_out(derive):
    @kernel
    def unordered(key: Index, x: Scalar, y: Mutable[Scalar]):
        hawk.scatter(y)[key].max(x)

    with pytest.raises(hawk.HawkError, match=r'is a keyed maximum.*write it as `hawk.scatter\(y, contract="ordered"\)'):
        derive(unordered)


@pytest.mark.gpu
def test_the_device_derivatives_equal_the_closed_forms(tmp_path, cache_dir):
    import cupy as cp
    from conftest import refuse_device_row

    refuse_device_row()
    tangent = Kernel("folds_jvp", jvp(folds, wrt=("x",), primal_unit=UNIT))
    reverse = Kernel("folds_vjp", vjp(folds, primal_unit=UNIT))
    on_device = {k.name: build_bundle([k], tmp_path / k.name, mode="float64", targets=("cuda", "host"), cache_dir=cache_dir)
                 for k in (folds, tangent, reverse)}

    def run(name, **planes):
        plan = hawk.plan(hawk.load(on_device[name].directory, name), hawk.DeviceKernel).bind(**planes)
        plan.launch()
        cp.cuda.Device().synchronize()
        return plan

    key, x = _tied(1)
    rng = np.random.default_rng(2)
    dx, bar_lo, bar_hi = rng.normal(size=N), rng.normal(size=CELLS), rng.normal(size=CELLS)
    lo, hi = cp.full(CELLS, np.inf), cp.full(CELLS, -np.inf)
    primal = run("folds", key=cp.asarray(key), x=cp.asarray(x), lo=lo, hi=hi)
    win_lo, win_hi = primal.winner("lo").copy(), primal.winner("hi").copy()
    dot_lo, dot_hi = cp.full(CELLS, 7.5), cp.full(CELLS, 7.5)
    run("folds_jvp", key=cp.asarray(key), dot_x=cp.asarray(dx), lo_winner=cp.asarray(win_lo),
        hi_winner=cp.asarray(win_hi), dot_lo=dot_lo, dot_hi=dot_hi)
    assert dot_lo.get().tolist() == _tangent_by_scan(win_lo, dx, 7.5).tolist()
    assert dot_hi.get().tolist() == _tangent_by_scan(win_hi, dx, 7.5).tolist()
    bar_x = cp.zeros(N)
    run("folds_vjp", key=cp.asarray(key), lo_winner=cp.asarray(win_lo), hi_winner=cp.asarray(win_hi),
        bar_lo=cp.asarray(bar_lo), bar_hi=cp.asarray(bar_hi), bar_x=bar_x)
    lanes = np.arange(N)
    want = np.where(win_lo[key] == lanes, bar_lo[key], 0.0) + np.where(win_hi[key] == lanes, bar_hi[key], 0.0)
    assert bar_x.get().tolist() == want.tolist()
