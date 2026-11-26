# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The forms a group body takes on the card: vector and matrix outputs per lane, keyed adds and stores from inside a group, a branch
or a loop inside a phase, and an append.

Each row (the ``gpu`` mark) builds the kernels of ``test_group_body`` for the card, runs them and compares with the NumPy oracle of
the host row of the same name; a bundle holds one kind of access, so the kernels that write at other lanes are built apart from
those that write only their own.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from test_group_body import (
    BINS,
    RNG,
    appended,
    counted,
    exits,
    keyed_add,
    keyed_add_compensated,
    keyed_store,
    matrix_out,
    parity,
    pick,
    unrolled,
    vector_out,
)

import hawk
from hawk.artifact import build_bundle


def _bundle(kernels, tmp_path_factory, cache_dir, name):
    """``kernels`` built for the card and the host, once."""
    return build_bundle(
        kernels,
        tmp_path_factory.mktemp(name),
        targets=("cuda", "host"),
        cache_dir=cache_dir,
        host_profile="native",
    )


@pytest.fixture(scope="module")
def wide(tmp_path_factory, cache_dir):
    return _bundle([vector_out, matrix_out], tmp_path_factory, cache_dir, "group_wide_device")


@pytest.fixture(scope="module")
def keyed(tmp_path_factory, cache_dir):
    return _bundle(
        [keyed_add, keyed_store, keyed_add_compensated, appended],
        tmp_path_factory,
        cache_dir,
        "group_keyed_device",
    )


@pytest.fixture(scope="module")
def sample_local(tmp_path_factory, cache_dir):
    return _bundle([parity, unrolled, pick, counted, exits], tmp_path_factory, cache_dir, "group_local_device")


def _run(bundle, name, **planes):
    """The kernel ``name`` run on the card over the NumPy ``planes``; each is copied back into its array."""
    cupy = pytest.importorskip("cupy")
    on_card = {key: cupy.asarray(value) for key, value in planes.items()}
    bound = hawk.plan(hawk.load(bundle.directory, name), hawk.DeviceKernel).bind(**on_card)
    bound.launch()
    bound.wait()
    for key, value in planes.items():
        value[...] = on_card[key].get()


@pytest.mark.gpu
def test_a_vector_output_committed_in_a_group_is_one_vector_per_lane(wide):
    n = 19
    x = RNG.normal(size=n)
    y = np.zeros((3, n))
    _run(wide, "vector_out", x=x, y=y)
    np.testing.assert_allclose(y, np.stack([x, 2.0 * x, (np.arange(n) % 8).astype(float)]), rtol=1e-14)


@pytest.mark.gpu
def test_a_matrix_output_committed_in_a_group_is_one_matrix_per_lane(wide):
    n = 19
    x = RNG.normal(size=n)
    y = np.zeros((4, n))
    _run(wide, "matrix_out", x=x, y=y)
    size = np.repeat([8.0, 8.0, 3.0], [8, 8, 3])
    left, right = np.stack([x, 2.0 * x]), np.stack([(np.arange(n) % 8).astype(float), size])
    np.testing.assert_allclose(y, np.einsum("il,jl->ijl", left, right).reshape(4, n), rtol=1e-14)


@pytest.mark.gpu
def test_a_keyed_add_from_inside_a_group_equals_numpy_add_at_beside_the_group_sum(keyed):
    n = 19
    cell, x = RNG.integers(0, BINS, size=n).astype(np.int32), RNG.normal(size=n)
    acc, total = np.zeros(BINS), np.zeros(n)
    _run(keyed, "keyed_add", cell=cell, x=x, acc=acc, total=total)
    want = np.zeros(BINS)
    np.add.at(want, cell, x)
    np.testing.assert_allclose(acc, want, rtol=0, atol=1e-12)
    np.testing.assert_allclose(total, x * np.repeat([8.0, 8.0, 3.0], [8, 8, 3]), rtol=1e-14)


@pytest.mark.gpu
def test_a_keyed_store_with_distinct_keys_from_inside_a_group_equals_numpy_assignment(keyed):
    n = 19
    cell, x = RNG.permutation(32)[:n].astype(np.int32), RNG.normal(size=n)
    slot, total = np.zeros(32), np.zeros(n)
    _run(keyed, "keyed_store", cell=cell, x=x, slot=slot, total=total)
    want = np.zeros(32)
    want[cell] = x
    np.testing.assert_array_equal(slot, want)


@pytest.mark.gpu
def test_a_compensated_keyed_add_in_a_group_equals_fsum_per_cell_beside_an_output(keyed):
    n = 19
    cell = RNG.integers(0, BINS, size=n).astype(np.int32)
    x = RNG.normal(size=n) * 10.0 ** RNG.integers(-8, 9, size=n)
    acc, total = np.zeros(BINS), np.zeros(n)
    _run(keyed, "keyed_add_compensated", cell=cell, x=x, acc=acc, total=total)
    want = np.array([math.fsum(x[cell == c]) for c in range(BINS)])
    np.testing.assert_allclose(acc, want, rtol=1e-14, atol=1e-14 * np.abs(x).max())


@pytest.mark.gpu
def test_an_append_in_the_scope_stores_the_positive_values_beside_an_output(keyed):
    x = RNG.normal(size=19)
    out, y = np.zeros(64), np.zeros(19)
    _run(keyed, "appended", x=x, out=out, y=y)
    np.testing.assert_array_equal(np.sort(out[: (x > 0).sum()]), np.sort(x[x > 0]))
    np.testing.assert_allclose(y, x * np.repeat([8.0, 8.0, 3.0], [8, 8, 3]), rtol=1e-14)


@pytest.mark.gpu
def test_an_if_in_the_scope_chooses_by_a_select_on_the_rank_of_the_lane(sample_local):
    x = RNG.normal(size=19)
    y = np.zeros(19)
    _run(sample_local, "parity", x=x, y=y)
    np.testing.assert_allclose(y, np.where(np.arange(19) % 8 % 2 == 0, 3.0 * x, -x), rtol=1e-14)


@pytest.mark.gpu
def test_an_if_whose_arms_read_scratch_chooses_between_the_own_and_the_next_lane(sample_local):
    x = RNG.normal(size=19)
    y = np.zeros(19)
    _run(sample_local, "pick", x=x, y=y)
    rolled = np.concatenate([np.roll(x[k : k + 8], -1) for k in range(0, 19, 8)])
    np.testing.assert_allclose(y, np.where(np.arange(19) % 8 % 2 == 0, x, rolled), rtol=1e-14)


@pytest.mark.gpu
def test_a_loop_inside_a_phase_runs_for_every_lane_inside_the_lane_loop(sample_local):
    x = RNG.normal(size=19)
    y = np.zeros(19)
    _run(sample_local, "unrolled", x=x, y=y)
    np.testing.assert_allclose(y, 3.0 * x + np.arange(19) % 8, rtol=1e-14, atol=1e-14)


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["counted", "exits"])
def test_a_loop_in_the_scope_runs_its_trips_and_reads_back_after_it(sample_local, name):
    x = RNG.uniform(1.0, 3.0, size=19)
    n = (np.arange(19) % 5).astype(np.int32)
    y = np.zeros(19)
    _run(sample_local, name, x=x, n=n, y=y)
    np.testing.assert_allclose(y, x * n, rtol=1e-14)
