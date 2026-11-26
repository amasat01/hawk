# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A kernel with a group runs on the card: a block serves a group, each phase a loop over the group's lanes strided over the block's threads.

Every row (the ``gpu`` mark) builds the kernels of ``test_group_host`` for the card, runs them and compares with NumPy written there
(the slice of each group read with ``np.roll``, ``np.clip`` or a mask; ``arange % N`` for a rank); the text rows read the device text:

* a neighbour read, the rank and the size of a lane, two scratch planes (one vector-valued) across a barrier, a partial last group;
* a group over the trailing, the leading or two separated axes of the lane grid;
* the scratch is carved from the block's dynamic shared memory, the barrier is ``__syncthreads()``, and a phase is a strided loop.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_group_host import (
    EDGES,
    RNG,
    _groups,
    _moved,
    _ring,
    _ring_oracle,
    beside,
    fill,
    over_cols,
    over_rows,
    over_two,
    ranked,
    two_phase,
)

import hawk
from hawk import Mutable, Scalar
from hawk.artifact import build_bundle, emit_source

@hawk.kernel
def rank_only(x: Scalar, y: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        y = x + g.rank


KERNELS = (
    [_ring(edge, lanes) for edge in EDGES for lanes in (8, 64)]
    + [ranked, two_phase, over_cols, over_rows, over_two, beside, fill, rank_only]
)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    """Every kernel above built for the card and the host, once."""
    return build_bundle(
        KERNELS,
        tmp_path_factory.mktemp("group_device"),
        targets=("cuda", "host"),
        cache_dir=cache_dir,
        host_profile="native",
    )


def _run(bundle, name, **planes):
    """The kernel ``name`` run on the card over the NumPy ``planes``; each is copied back into its array."""
    cupy = pytest.importorskip("cupy")
    on_card = {key: cupy.asarray(value) for key, value in planes.items()}
    bound = hawk.plan(hawk.load(bundle.directory, name), hawk.DeviceKernel).bind(**on_card)
    bound.launch()
    bound.wait()
    for key, value in planes.items():
        value[...] = on_card[key].get()


def _device_text(kernel):
    """The device text of ``kernel``."""
    return emit_source(kernel, targets=("cuda",)).sources["cuda"].text


def test_the_device_text_of_a_group_is_a_block_per_group_with_strided_phases_and_syncthreads():
    text = _device_text(_ring("clamp", 8))
    assert "extern __shared__ __align__(8) char hawk_scratch[];" in text
    assert "const std::int64_t hawk_g = base + static_cast<std::int64_t>(blockIdx.x);" in text
    assert "hawk_rank = threadIdx.x; hawk_rank < hawk_size; hawk_rank += blockDim.x" in text
    assert text.count("__syncthreads();") == 1
    assert text.index("__syncthreads();") < text.rindex("hawk_rank += blockDim.x")


def test_a_group_with_no_scratch_carves_no_shared_memory():
    assert "__shared__" not in _device_text(rank_only)


@pytest.mark.gpu
@pytest.mark.parametrize("lanes", (8, 64))
@pytest.mark.parametrize("edge", EDGES)
@pytest.mark.parametrize("n", (64, 128, 19, 150))
def test_a_neighbour_read_in_a_group_equals_the_numpy_move_of_its_slice_with_a_partial_last_group(
    bundle, edge, lanes, n
):
    x = RNG.normal(size=n)
    y = np.zeros(n)
    _run(bundle, f"ring_{edge}_{lanes}", x=x, y=y)
    np.testing.assert_allclose(y, _ring_oracle(x, edge, lanes), rtol=1e-14, atol=1e-14)


@pytest.mark.gpu
@pytest.mark.parametrize("n", (5, 8, 300_003), ids=("short only", "one whole", "short last of many"))
def test_a_group_of_eight_is_right_for_a_short_only_group_a_whole_one_and_a_short_last_of_many_blocks(bundle, n):
    x = RNG.normal(size=n)
    y = np.zeros(n)
    _run(bundle, "ring_wrap_8", x=x, y=y)
    np.testing.assert_allclose(y, _ring_oracle(x, "wrap", 8), rtol=1e-14, atol=1e-14)


@pytest.mark.gpu
def test_the_rank_and_the_size_of_a_lane_are_its_place_in_its_group_and_the_length_of_its_group(bundle):
    x = RNG.normal(size=19)
    place, size, y = np.zeros(19, dtype=np.int64), np.zeros(19, dtype=np.int64), np.zeros(19)
    _run(bundle, "ranked", x=x, place=place, size=size, y=y)
    np.testing.assert_array_equal(place, np.arange(19) % 8)
    np.testing.assert_array_equal(size, np.concatenate([np.full(len(g), len(g)) for g in _groups(x, 8)]))
    np.testing.assert_array_equal(y, 2.0 * x)


@pytest.mark.gpu
def test_two_scratch_planes_and_a_vector_valued_one_survive_the_barrier(bundle):
    n = 11
    x, v = RNG.normal(size=n), RNG.normal(size=(3, n))
    out = {name: np.zeros(n) for name in ("first", "second", "third", "z")}
    _run(bundle, "two_phase", x=x, v=v, **out)
    want = {name: np.zeros(n) for name in out}
    for start in range(0, n, 4):
        lanes = slice(start, min(start + 4, n))
        w = v[:, lanes] + x[lanes]
        a = 2.0 * x[lanes]
        moved = np.roll(w, -1, axis=1)
        want["first"][lanes] = moved[0]
        want["second"][lanes] = moved[1] + w[1]
        want["third"][lanes] = moved[2]
        want["z"][lanes] = a[np.clip(np.arange(a.size) - 1, 0, a.size - 1)] + a
    for name in out:
        np.testing.assert_allclose(out[name], want[name], rtol=1e-14, atol=1e-14, err_msg=name)


@pytest.mark.gpu
@pytest.mark.parametrize("rows", (2, 5))
def test_a_group_over_the_trailing_axis_reads_its_neighbour_along_that_axis(bundle, rows):
    x = RNG.normal(size=(rows, 4))
    y = np.zeros_like(x)
    _run(bundle, "over_cols", x=x, y=y)
    np.testing.assert_allclose(y, np.roll(x, -1, axis=1) + 10.0 * np.arange(4), rtol=1e-14)


@pytest.mark.gpu
def test_a_group_over_the_leading_axis_reads_its_neighbour_along_that_axis(bundle):
    x = RNG.normal(size=(3, 5))
    y = np.zeros_like(x)
    _run(bundle, "over_rows", x=x, y=y)
    np.testing.assert_allclose(y, np.roll(x, -1, axis=0) + 10.0 * np.arange(3)[:, None], rtol=1e-14)


@pytest.mark.gpu
def test_a_group_over_two_axes_with_one_between_them_ranks_its_lanes_in_lane_order(bundle):
    x = RNG.normal(size=(2, 3, 4))
    y = np.zeros_like(x)
    _run(bundle, "over_two", x=x, y=y)
    want = np.empty_like(x)
    for b in range(3):
        want[:, b, :] = np.roll(x[:, b, :].reshape(-1), -1).reshape(2, 4)
    np.testing.assert_allclose(y, want, rtol=1e-14)


@pytest.mark.gpu
def test_an_output_beside_a_group_is_written_for_every_lane_as_outside_a_group(bundle):
    x = RNG.normal(size=19)
    y, z = np.zeros(19), np.zeros(19)
    _run(bundle, "beside", x=x, y=y, z=z)
    np.testing.assert_allclose(z, x + 1.0, rtol=1e-14)
    np.testing.assert_allclose(y, np.concatenate([_moved(g, 1, "wrap") for g in _groups(x, 8)]), rtol=1e-14)


@pytest.mark.gpu
def test_a_group_with_no_scratch_runs_its_lanes_by_rank(bundle):
    x = RNG.normal(size=19)
    y = np.zeros(19)
    _run(bundle, "rank_only", x=x, y=y)
    np.testing.assert_allclose(y, x + np.arange(19) % 8, rtol=1e-14)


@pytest.mark.gpu
def test_a_fill_and_a_read_in_two_phases_gives_x(bundle):
    x = np.abs(RNG.normal(size=100))
    y = np.zeros(100)
    _run(bundle, "fill", x=x, y=y)
    np.testing.assert_allclose(y, np.sqrt(x) * np.sqrt(x), rtol=1e-14)
