# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""What stops a ``hawk.repeat_while`` beside its guard: a trip count read at every replay, a break cell the body writes, and
the iteration counter the library keeps. Every value row is held against the same loop written in NumPy."""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import HawkError, Mutable, Scalar, UInt32
from hawk.math import where

N = 19
X = np.linspace(0.5, 2.5, N)


@hawk.kernel
def grow(x: Scalar, y: Mutable[Scalar]):
    y = y + x


@hawk.kernel
def tick(c: Mutable[UInt32]):
    c = where(c > 0, c - 1, c)


@hawk.kernel
def raise_flag(flag: Mutable[UInt32]):
    flag = flag * 0 + 1


def cell(value: int) -> np.ndarray:
    """A one-element ``uint32`` cell."""
    return np.array([value], dtype=np.uint32)


def test_a_limit_is_a_trip_count_read_at_every_replay():
    limit, y = cell(3), np.zeros(N)
    loop = hawk.repeat_while(hawk.deploy(grow).bind(x=X, y=y), limit=limit)
    graph = hawk.Graph().add(loop).build()
    for trips in (3, 5, 0):
        limit[0], y[:] = trips, 0.0
        graph.launch()
        np.testing.assert_allclose(y, trips * X, rtol=1e-14)
        assert loop.iterations() == trips


def test_the_counter_reads_the_iterations_of_a_guarded_loop():
    c, y = cell(4), np.zeros(N)
    body = [hawk.deploy(grow).bind(x=X, y=y), hawk.deploy(tick).bind(c=c)]
    loop = hawk.repeat_while(body, hawk.SkipGuard.nonzero(c), 9)
    assert loop.iterations() == 0
    hawk.Graph().add(loop).build().launch()
    np.testing.assert_allclose(y, 4 * X, rtol=1e-14)
    assert loop.iterations() == 4 and loop.iteration_index.dtype == np.uint32


def test_a_break_cell_ends_the_loop_after_the_iteration_that_sets_it():
    stop, y = cell(0), np.zeros(N)
    body = [hawk.deploy(grow).bind(x=X, y=y), hawk.deploy(raise_flag).bind(flag=stop)]
    loop = hawk.repeat_while(body, max_iters=50, brk=stop)
    hawk.Graph().add(loop).build().launch()
    np.testing.assert_allclose(y, X, rtol=1e-14)
    assert loop.iterations() == 1 and stop[0] == 1


def test_a_loop_that_nothing_stops_is_refused_by_name():
    with pytest.raises(HawkError, match="nothing stops the loop"):
        hawk.repeat_while([], max_iters=4)
