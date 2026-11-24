# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Sums, minima and maxima over samples under a lane set, against NumPy.

A prefix lane set runs launch position ``t`` as sample ``t`` for ``t < count``; a mapped one runs position ``t`` as sample
``map[t]``. Either way the cells fold the live samples only. The oracle is NumPy over ``x[map[:live]]``. The lane a position
cell stores is the winner's SAMPLE index, the number ``m.sample_index()`` has in that lane (not its launch position); a tie
goes to the lowest sample index, and a mapped set puts that sample at a launch position other than the lowest one.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import _launch

import hawk
from hawk import HawkError, Kernel, Mutable, Scalar, kernel
from hawk.artifact import build_bundle
from hawk.ext import IndexSet, Kind

KINDS = {
    "prefix": Kind("prefix_set", lanes=IndexSet(prefix=True)),
    "mapped": Kind("mapped_set", lanes=IndexSet()),
}
SIZES = (1, 31, 32, 33, 100, 1000)


def _folds(kind):
    @kernel(kind=kind)
    def folds(x: Scalar, s: Mutable[Scalar], lo: Mutable[Scalar], hi: Mutable[Scalar]):
        s = hawk.sum(x)
        lo = hawk.min(x)
        hi = hawk.max(x)

    @kernel(kind=kind)
    def positions(x: Scalar, low: Mutable[Scalar], at: Mutable[Scalar]):
        low, at = hawk.min(x, with_index=True)

    return [Kernel("folds", folds.sinks), Kernel("positions", positions.sinks)]


@pytest.fixture(scope="module", params=list(KINDS))
def built(request, tmp_path_factory, cache_dir):
    kind = KINDS[request.param]
    bundle = build_bundle(
        _folds(kind),
        tmp_path_factory.mktemp(request.param),
        targets=("host",),
        cache_dir=cache_dir,
        kind=kind,
    )
    return bundle, request.param


@pytest.fixture(
    params=[1, 4, None], ids=["one_thread", "four_threads", "default_threads"]
)
def threads(request):
    """The host team size of the launch, restored afterwards."""
    wanted = hawk.host_threads()
    hawk.set_host_threads(request.param)
    yield request.param
    hawk.set_host_threads(wanted)


def _case(form, n, live, tied=False):
    """The values, the planes of the set, the live slice of the values in launch order and the samples they sit at."""
    x = np.random.default_rng(n).uniform(1.0, 2.0, n)
    if tied:
        x = np.round(x * 3.0)
    planes = {"active_count": np.array([live], dtype=np.uint32)}
    order = np.arange(n)
    if form == "mapped":
        order = np.random.default_rng(n + 1).permutation(n)
        planes["active_map"] = order.astype(np.int32)
    return x, planes, x[order[:live]], order[:live]


def _live_counts(n):
    return sorted({0, 1, n // 2, n})


@pytest.mark.parametrize("n", SIZES)
def test_the_cells_of_mixed_folds_fold_the_live_samples_only(built, n, threads):
    bundle, form = built
    for live in _live_counts(n):
        x, planes, kept, _ = _case(form, n, live)
        planes |= {"x": x, "s": np.zeros(1), "lo": np.zeros(1), "hi": np.zeros(1)}
        _launch(bundle, "folds", "host", planes).launch()
        want = {
            "s": kept.sum(),
            "lo": kept.min() if live else np.inf,
            "hi": kept.max() if live else -np.inf,
        }
        assert {k: planes[k][0] for k in want} == pytest.approx(want, rel=1e-12), (
            form,
            n,
            live,
        )


@pytest.mark.parametrize("tied", [False, True], ids=["distinct", "tied"])
@pytest.mark.parametrize("n", SIZES)
def test_the_lane_of_a_winner_is_its_sample_and_a_tie_goes_to_the_lowest_sample(
    built, n, tied, threads
):
    bundle, form = built
    for live in (live for live in _live_counts(n) if live):
        x, planes, kept, samples = _case(form, n, live, tied)
        planes |= {"x": x, "low": np.zeros(1), "at": np.zeros(1)}
        _launch(bundle, "positions", "host", planes).launch()
        winner = samples[kept == kept.min()].min()
        assert (planes["low"][0], planes["at"][0]) == (kept.min(), winner), (
            form,
            n,
            live,
        )


def test_a_sum_under_a_lane_group_above_one_is_refused_by_name(tmp_path, cache_dir):
    kind = KINDS["mapped"]

    @kernel(kind=kind, group=4)
    def grouped(x: Scalar, s: Mutable[Scalar]):
        s = hawk.sum(x)

    with pytest.raises(
        HawkError,
        match="a sum over samples with a lane group above 1 is not written yet",
    ):
        build_bundle(
            [grouped],
            tmp_path,
            targets=("host",),
            cache_dir=cache_dir,
            kind=kind,
        )
