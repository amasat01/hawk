# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A kernel with a group runs on the host, from the author's ``with hawk.group(...)`` to the values.

Every value is checked against NumPy written here, never against the kernel's own output: a neighbour read inside a group is the
group's slice read with ``np.roll`` (wrap), ``np.clip`` (clamp) or a mask (zero), the lanes of a group over a Grid axis are
the NumPy axis, and the rank and size of a lane are ``arange % N`` and the group's length. The emitted text is pinned by its lines:
a phase that reads a neighbour is cut into an interior loop with plain indices and the lanes at the edges.
"""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import Grid, HawkError, Index, Mutable, Scalar, Vector
from hawk import math as m
from hawk.artifact import build_bundle, emit_source

EDGES = ("clamp", "wrap", "zero")
RNG = np.random.default_rng(20261123)


def _groups(field, lanes):
    """The consecutive groups of ``lanes`` entries of the lane vector ``field``; the last may be shorter."""
    return [field[start : start + lanes] for start in range(0, len(field), lanes)]


def _moved(group, shift, edge):
    """Oracle: the entries of one group read at each place moved by ``shift``, under the edge rule."""
    size = len(group)
    place = np.arange(size) + shift
    if edge == "wrap":
        return np.roll(group, -shift)
    inside = np.clip(place, 0, size - 1)
    return (
        group[inside]
        if edge == "clamp"
        else np.where((place >= 0) & (place < size), group[inside], 0.0)
    )


def _ring(edge, lanes=8):
    """A group of ``lanes`` lanes: each lane adds its two neighbours (after and before) to its own value, under ``edge``."""

    @hawk.kernel
    def ring(x: Scalar, y: Mutable[Scalar]):
        with hawk.group(lanes=lanes) as g:
            t = g.scratch("t", Scalar)
            t[g.rank] = x
            g.barrier()
            y = (
                t[g.rank]
                + 2.0 * t.at(g.rank + 1, edge=edge)
                + 3.0 * t.at(g.rank - 1, edge=edge)
            )

    ring.name = f"ring_{edge}_{lanes}"
    return ring


@hawk.kernel
def ranked(x: Scalar, place: Mutable[Index], size: Mutable[Index], y: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        t = g.scratch("t", Scalar)
        t[g.rank] = x * 2.0
        place = g.rank
        size = g.size
        y = t[g.rank]


@hawk.kernel
def two_phase(
    x: Scalar,
    v: Vector[3],
    first: Mutable[Scalar],
    second: Mutable[Scalar],
    third: Mutable[Scalar],
    z: Mutable[Scalar],
):
    with hawk.group(lanes=4) as g:
        a = g.scratch("a", Scalar)
        w = g.scratch("w", Vector[3])
        a[g.rank] = x * 2.0
        w[g.rank] = v + x
        g.barrier()
        moved = w.at(g.rank + 1, edge="wrap")
        first = moved[0]
        second = moved[1] + w[g.rank][1]
        third = moved[2]
        z = a.at(g.rank - 1, edge="clamp") + a[g.rank]


@hawk.kernel
def over_cols(lane: Grid["row", "col":4], x: Scalar, y: Mutable[Scalar]):  # noqa: F821 axis labels
    with hawk.group(over="col") as g:
        t = g.scratch("t", Scalar)
        t[g.rank] = x
        g.barrier()
        y = t.at(g.rank + 1, edge="wrap") + 10.0 * g.rank


@hawk.kernel
def over_rows(lane: Grid["row":3, "col":5], x: Scalar, y: Mutable[Scalar]):  # noqa: F821 axis labels
    with hawk.group(over="row") as g:
        t = g.scratch("t", Scalar)
        t[g.rank] = x
        g.barrier()
        y = t.at(g.rank + 1, edge="wrap") + 10.0 * g.rank


@hawk.kernel
def over_lead(lane: Grid["row":8, "col":32771], x: Scalar, y: Mutable[Scalar]):  # noqa: F821 axis labels
    with hawk.group(over="row") as g:
        t = g.scratch("t", Scalar)
        t[g.rank] = x
        g.barrier()
        y = t.at(g.rank + 1, edge="wrap") + t.at(g.rank - 1, edge="wrap")


@hawk.kernel
def over_two(lane: Grid["a":2, "b":3, "c":4], x: Scalar, y: Mutable[Scalar]):  # noqa: F821 axis labels
    with hawk.group(over=("a", "c")) as g:
        t = g.scratch("t", Scalar)
        t[g.rank] = x
        g.barrier()
        y = t.at(g.rank + 1, edge="wrap")


@hawk.kernel
def fill(x: Scalar, y: Mutable[Scalar]):
    with hawk.group(lanes=16) as g:
        t = g.scratch("t", Scalar)
        t[g.rank] = m.sqrt(x)
        g.barrier()
        y = t[g.rank] * t[g.rank]


@hawk.kernel
def beside(x: Scalar, y: Mutable[Scalar], z: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        t = g.scratch("t", Scalar)
        t[g.rank] = x
        g.barrier()
        y = t.at(g.rank + 1, edge="wrap")
    z = x + 1.0


@hawk.kernel
def beside_keyed(
    cell: Index,
    x: Scalar,
    y: Mutable[Scalar],
    acc: Mutable[Scalar, "bin":4],  # noqa: F821 axis labels
):
    with hawk.group(lanes=8) as g:
        y = x * m.real(g.size)
    hawk.scatter(acc, contract="plain")[cell] += x


@hawk.kernel
def before_reads_prior(x: Scalar, y: Mutable[Scalar], z: Mutable[Scalar]):
    z = y + x
    with hawk.group(lanes=8) as g:
        y = x * m.real(g.size)


@hawk.kernel
def shared_add(
    cell: Index, x: Scalar, y: Mutable[Scalar], acc: Mutable[Scalar, "bin":4]
):  # noqa: F821 axis labels
    hawk.scatter(acc, contract="plain")[cell] += 4.0 * x
    with hawk.group(lanes=8) as g:
        y = x * m.real(g.size)
    hawk.scatter(acc, contract="plain")[cell] += 2.0 * x


@hawk.kernel
def beside_compensated(
    cell: Index, x: Scalar, y: Mutable[Scalar], acc: Mutable[Scalar, "bin":4]
):  # noqa: F821 axis labels
    with hawk.group(lanes=8) as g:
        y = x * m.real(g.size)
    hawk.scatter(acc)[cell] += x


@hawk.kernel
def after_writes_what_the_body_reads(x: Scalar, w: Mutable[Scalar], y: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        y = w + x * m.real(g.size)
    w = x + 1.0


@hawk.kernel
def cycle(x: Scalar, w: Mutable[Scalar], y: Mutable[Scalar]):
    ahead = y + x
    with hawk.group(lanes=8) as g:
        y = w + x
    w = ahead


KERNELS = [_ring(edge, lanes) for edge in EDGES for lanes in (8, 64)] + [
    ranked,
    two_phase,
    over_cols,
    over_rows,
    over_lead,
    over_two,
    fill,
    beside,
    before_reads_prior,
    after_writes_what_the_body_reads,
]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    """Every kernel above built for the host, once."""
    return build_bundle(
        KERNELS,
        tmp_path_factory.mktemp("group_host"),
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )


@pytest.fixture(scope="module")
def keyed_bundle(tmp_path_factory, cache_dir):
    """The kernel with a keyed add, built apart: a bundle holds one kind of access."""
    return build_bundle(
        [beside_keyed],
        tmp_path_factory.mktemp("group_keyed"),
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )


@pytest.fixture(scope="module")
def shared_bundle(tmp_path_factory, cache_dir):
    """The kernel that adds into one plane from before, inside and after its scope, built apart (a keyed access)."""
    return build_bundle(
        [shared_add, beside_compensated],
        tmp_path_factory.mktemp("group_shared"),
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )


def _loaded(bundle, name):
    return hawk.load(bundle.directory, name)


def _run(bundle, name, **planes):
    hawk.run(_loaded(bundle, name), **planes)


@pytest.mark.parametrize("lanes", (8, 64))
@pytest.mark.parametrize("edge", EDGES)
@pytest.mark.parametrize("n", (64, 128, 19, 150))
def test_a_neighbour_read_in_a_group_equals_the_numpy_move_of_its_slice_with_a_partial_last_group(
    bundle, edge, lanes, n
):
    """Oracle: ``np.roll`` / ``np.clip`` / a mask of each group's slice; the last group's edges act at its real size."""
    x = RNG.normal(size=n)
    y = np.zeros(n)
    _run(bundle, f"ring_{edge}_{lanes}", x=x, y=y)
    want = np.concatenate(
        [
            g + 2.0 * _moved(g, 1, edge) + 3.0 * _moved(g, -1, edge)
            for g in _groups(x, lanes)
        ]
    )
    np.testing.assert_array_equal(y, want)


def test_the_rank_and_the_size_of_a_lane_are_its_place_in_its_group_and_the_length_of_its_group(
    bundle,
):
    """Oracle: ``arange % 8`` and the length of each NumPy slice of 8 (the last is partial)."""
    x = RNG.normal(size=19)
    place, size, y = (
        np.zeros(19, dtype=np.int64),
        np.zeros(19, dtype=np.int64),
        np.zeros(19),
    )
    _run(bundle, "ranked", x=x, place=place, size=size, y=y)
    np.testing.assert_array_equal(place, np.arange(19) % 8)
    np.testing.assert_array_equal(
        size, np.concatenate([np.full(len(g), len(g)) for g in _groups(x, 8)])
    )
    np.testing.assert_array_equal(y, 2.0 * x)


def test_two_scratch_planes_and_a_vector_valued_one_survive_the_barrier(bundle):
    """Oracle: the components of the next lane's vector, read after the barrier, from NumPy slices (11 lanes: the last group has 3)."""
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
        np.testing.assert_allclose(
            out[name], want[name], rtol=1e-15, atol=1e-15, err_msg=name
        )


@pytest.mark.parametrize("rows", (2, 5))
def test_a_group_over_the_trailing_axis_reads_its_neighbour_along_that_axis(
    bundle, rows
):
    """Oracle: ``np.roll`` along axis 1, plus ten times the column (the rank), for two extents of the run-time axis."""
    x = RNG.normal(size=(rows, 4))
    y = np.zeros_like(x)
    _run(bundle, "over_cols", x=x, y=y)
    np.testing.assert_array_equal(y, np.roll(x, -1, axis=1) + 10.0 * np.arange(4))


def test_a_group_over_the_leading_axis_reads_its_neighbour_along_that_axis(bundle):
    """Oracle: ``np.roll`` along axis 0, plus ten times the row (the rank): the lanes of a group are 5 apart."""
    x = RNG.normal(size=(3, 5))
    y = np.zeros_like(x)
    _run(bundle, "over_rows", x=x, y=y)
    np.testing.assert_array_equal(
        y, np.roll(x, -1, axis=0) + 10.0 * np.arange(3)[:, None]
    )


def test_a_group_over_two_axes_with_one_between_them_ranks_its_lanes_in_lane_order(
    bundle,
):
    """Oracle: for each ``b``, the 8 lanes along ``a`` then ``c`` flattened in that order and rolled by one."""
    x = RNG.normal(size=(2, 3, 4))
    y = np.zeros_like(x)
    _run(bundle, "over_two", x=x, y=y)
    want = np.empty_like(x)
    for b in range(3):
        want[:, b, :] = np.roll(x[:, b, :].reshape(-1), -1).reshape(2, 4)
    np.testing.assert_array_equal(y, want)


def test_an_output_beside_a_group_is_written_for_every_lane_as_outside_a_group(bundle):
    """Oracle: ``x + 1`` for the output beside the group, the NumPy roll of each group's slice for the group's (19 lanes)."""
    x = RNG.normal(size=19)
    y, z = np.zeros(19), np.zeros(19)
    _run(bundle, "beside", x=x, y=y, z=z)
    np.testing.assert_array_equal(z, x + 1.0)
    np.testing.assert_array_equal(
        y, np.concatenate([_moved(g, 1, "wrap") for g in _groups(x, 8)])
    )


def test_a_keyed_add_beside_a_group_adds_every_lane_once(keyed_bundle):
    """Oracle: ``np.add.at`` of the lanes' values at their keys, and ``x`` times the length of the lane's group."""
    n = 19
    x, cell = RNG.normal(size=n), RNG.integers(0, 4, size=n).astype(np.int32)
    y, acc = np.zeros(n), np.zeros(4)
    _run(keyed_bundle, "beside_keyed", cell=cell, x=x, y=y, acc=acc)
    want = np.zeros(4)
    np.add.at(want, cell, x)
    np.testing.assert_allclose(acc, want, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(y, x * np.repeat([8.0, 8.0, 3.0], [8, 8, 3]))


def test_a_sink_before_a_group_reads_the_value_the_plane_held_before_the_group_wrote_it(
    bundle,
):
    """Oracle: ``y + x`` of the plane's launch-start value (the author's order: the sink is written before the scope)."""
    x, y = RNG.normal(size=19), RNG.normal(size=19)
    start, z = y.copy(), np.zeros(19)
    _run(bundle, "before_reads_prior", x=x, y=y, z=z)
    np.testing.assert_array_equal(z, start + x)
    np.testing.assert_array_equal(y, x * np.repeat([8.0, 8.0, 3.0], [8, 8, 3]))


def test_a_sink_that_writes_a_plane_the_group_reads_runs_after_the_group(bundle):
    """Oracle: every read sees the launch-start value, so the group reads ``w`` as given, and ``w`` becomes ``x + 1``."""
    x, w = RNG.normal(size=19), RNG.normal(size=19)
    start, y = w.copy(), np.zeros(19)
    _run(bundle, "after_writes_what_the_body_reads", x=x, w=w, y=y)
    np.testing.assert_array_equal(y, start + x * np.repeat([8.0, 8.0, 3.0], [8, 8, 3]))
    np.testing.assert_array_equal(w, x + 1.0)


def test_sinks_that_read_what_the_body_writes_and_write_what_it_reads_are_refused_by_name():
    refusal = _refusal(cycle)
    assert (
        "read a plane the body writes and write a plane the body reads is not written on the host yet"
        in refusal
    )
    assert '"cycle"' in refusal


def test_keyed_adds_before_and_after_a_group_into_one_plane_add_every_lane_six_times(
    shared_bundle,
):
    """Oracle: ``np.add.at`` of ``6 x`` (two adds into one plane: 4 x written before the scope, 2 x after it)."""
    n = 19
    x, cell = RNG.normal(size=n), RNG.integers(0, 4, size=n).astype(np.int32)
    y, acc = np.zeros(n), np.zeros(4)
    _run(shared_bundle, "shared_add", cell=cell, x=x, y=y, acc=acc)
    want = np.zeros(4)
    np.add.at(want, cell, 6.0 * x)
    np.testing.assert_allclose(acc, want, rtol=0, atol=1e-12)


def test_a_compensated_keyed_add_beside_a_group_adds_every_lane_to_what_the_plane_held(
    shared_bundle,
):
    """Oracle: the start value plus ``np.add.at`` of the values (an add keeps what the plane held), within the bound the keyed sums use."""
    n = 19
    x, cell = RNG.normal(size=n), RNG.integers(0, 4, size=n).astype(np.int32)
    y, acc = np.zeros(n), np.full(4, 5.0)
    _run(shared_bundle, "beside_compensated", cell=cell, x=x, y=y, acc=acc)
    want = np.full(4, 5.0)
    np.add.at(want, cell, x)
    np.testing.assert_allclose(acc, want, rtol=0, atol=1e-12)


#: The groups one tile of a launch holds: a group's working set is small enough for the largest tile the host library cuts (4096
#: samples, here groups), so a launch of more groups than this is cut into several tiles.
GROUPS_PER_TILE = 4096


def _determinism_cases():
    """(kernel, planes of one launch, oracle) over more than seven tiles of groups: a flat count group and a group over the
    leading axis of a grid of 32771 columns (4096 * 8 + 3; its groups are the columns)."""
    groups = GROUPS_PER_TILE * 8 + 3
    n = groups * 8 + 5
    x = RNG.normal(size=n)
    flat = np.concatenate(
        [
            g + 2.0 * _moved(g, 1, "wrap") + 3.0 * _moved(g, -1, "wrap")
            for g in _groups(x, 8)
        ]
    )
    columns = 32771
    grid = RNG.normal(size=(8, columns))
    lead = np.roll(grid, -1, axis=0) + np.roll(grid, 1, axis=0)
    return [("ring_wrap_8", x, flat, groups), ("over_lead", grid, lead, columns)]


@pytest.mark.parametrize(
    "case", range(2), ids=["count group", "group over the leading axis"]
)
def test_the_same_launch_gives_the_same_bytes_on_one_four_and_seven_threads_over_many_tiles(
    bundle, case
):
    """Oracle: the NumPy slices; the launch is cut into ``ceil(groups / 4096)`` tiles (a launch reports how many ran), more than
    the seven threads, so the groups really are split among the threads, and every thread count gives the same bytes."""
    name, x, want, groups = _determinism_cases()[case]
    tiles_expected = -(-groups // GROUPS_PER_TILE)
    assert tiles_expected > 7
    results = []
    try:
        for threads in (1, 4, 7):
            hawk.set_host_threads(threads)
            y = np.zeros_like(x)
            tiles = (
                hawk.plan(_loaded(bundle, name), hawk.HostTeam).bind(x=x, y=y).launch()
            )
            assert tiles == tiles_expected
            results.append(y.tobytes())
    finally:
        hawk.set_host_threads(None)
    assert results[0] == results[1] == results[2]
    np.testing.assert_array_equal(np.frombuffer(results[0]).reshape(want.shape), want)


def test_a_fill_and_a_read_in_two_phases_gives_x(bundle):
    """Oracle: ``sqrt(x) * sqrt(x)`` rounded as NumPy does."""
    x = np.abs(RNG.normal(size=100))
    y = np.zeros(100)
    _run(bundle, "fill", x=x, y=y)
    np.testing.assert_array_equal(y, np.sqrt(x) * np.sqrt(x))


def _ring_oracle(x, edge, lanes):
    """Oracle: ``np.roll`` / ``np.clip`` / a mask of each NumPy slice of ``lanes`` entries; the last slice may be short."""
    return np.concatenate(
        [
            g + 2.0 * _moved(g, 1, edge) + 3.0 * _moved(g, -1, edge)
            for g in _groups(x, lanes)
        ]
    )


@pytest.mark.parametrize(
    "n",
    (5, 8, 300_003),
    ids=("only a short group", "one whole group", "short last of many tiles"),
)
@pytest.mark.parametrize("edge", ("wrap", "clamp"))
def test_a_count_group_of_eight_is_right_for_a_short_only_group_a_whole_one_and_a_short_last_of_many_tiles(
    bundle, edge, n
):
    """Oracle: NumPy slices of 8; 300,003 lanes are 37,501 groups, cut into ten tiles by the team, the last group of 3 lanes."""
    x = RNG.normal(size=n)
    y = np.zeros(n)
    _run(bundle, f"ring_{edge}_8", x=x, y=y)
    np.testing.assert_array_equal(y, _ring_oracle(x, edge, 8))


@pytest.mark.parametrize(
    "n",
    (5, 16, 300_003),
    ids=("only a short group", "one whole group", "short last of many tiles"),
)
def test_a_fill_group_of_sixteen_is_right_for_a_short_only_group_a_whole_one_and_a_short_last_of_many_tiles(
    bundle, n
):
    """Oracle: ``sqrt(x) * sqrt(x)`` rounded as NumPy does, lane by lane."""
    x = np.abs(RNG.normal(size=n))
    y = np.zeros(n)
    _run(bundle, "fill", x=x, y=y)
    np.testing.assert_array_equal(y, np.sqrt(x) * np.sqrt(x))


# ---- the text ----


def _host_text(kernel):
    return emit_source(kernel, targets=("host",)).sources["host"].text


def test_a_phase_that_reads_a_neighbour_is_cut_into_an_interior_loop_and_the_edge_lanes():
    """The interior loop reads plain indices; the edge lanes fold them; no scratch index holds a remainder."""
    text = _host_text(_ring("wrap"))
    lines = [line.strip() for line in text.splitlines()]
    assert (
        "for (std::int64_t hawk_rank = hawk_lo1; hawk_rank < hawk_hi1; ++hawk_rank) {"
        in lines
    )
    assert any(
        line.startswith("mut_y[i] = ") and "scr_t[hawk_rank + 1]" in line
        for line in lines
    )
    assert any("scr_t[aether::edge::wrap(hawk_rank + 1, " in line for line in lines)
    assert any(line.startswith("for (std::int64_t hawk_edge = 0;") for line in lines)
    assert not any("%" in line for line in lines if "scr_t" in line)
    interior = lines[
        lines.index(
            "for (std::int64_t hawk_rank = hawk_lo1; hawk_rank < hawk_hi1; ++hawk_rank) {"
        ) :
    ]
    assert "?" not in " ".join(interior[: interior.index("}")])


def test_an_input_plane_is_read_in_the_phase_loops_through_a_hoisted_restrict_pointer():
    """The pointer is declared once before the group loop, and the lane's own element is read through it."""
    lines = [line.strip() for line in _host_text(_ring("wrap")).splitlines()]
    assert "const auto* __restrict hawk_p_psc_x = psc_x.data();" in lines
    assert lines.index("const auto* __restrict hawk_p_psc_x = psc_x.data();") < next(
        k for k, line in enumerate(lines) if line.startswith("for (std::int64_t hawk_g")
    )
    assert any("hawk_p_psc_x[hawk_i]" in line for line in lines)
    assert not any("psc_x[i]" in line for line in lines)


def test_a_plane_the_body_reads_and_the_unit_writes_gets_no_restrict_pointer():
    """``w`` is read by the body and written beside it: it is read through its view; the input ``x`` still has a pointer."""
    text = _host_text(after_writes_what_the_body_reads)
    assert "hawk_p_psc_x = psc_x.data();" in text
    assert "hawk_p_mut_w" not in text and "__restrict hawk_p_mut" not in text


@pytest.mark.parametrize("edge", ("clamp", "zero"))
def test_the_edge_lanes_of_clamp_and_zero_are_written_apart_from_the_interior(edge):
    text = _host_text(_ring(edge))
    fold = {"clamp": "aether::edge::clamp(", "zero": "aether::edge::inside("}[edge]
    assert fold in text
    assert not any("%" in line for line in text.splitlines() if "scr_t" in line)


# ---- what is not written on the host ----


def _refusal(kernel):
    with pytest.raises(HawkError) as raised:
        emit_source(kernel, targets=("host",))
    return str(raised.value)


def test_a_group_reduce_is_refused_by_name_on_the_host():
    @hawk.kernel
    def summed(x: Scalar, y: Mutable[Scalar]):
        with hawk.group(lanes=8) as g:
            y = g.sum(x)

    assert "a group reduce is not written on the host yet" in _refusal(summed)
    assert '"summed"' in _refusal(summed)


def test_a_group_by_key_is_refused_by_name_on_the_host():
    @hawk.kernel
    def keyed(cell: Index, x: Scalar, y: Mutable[Scalar]):
        with hawk.group(key=cell, max_size=16) as g:
            t = g.scratch("t", Scalar)
            t[g.rank] = x
            g.barrier()
            y = t[g.rank]

    assert "a group by key is not written on the host yet" in _refusal(keyed)


def test_two_run_time_lane_axes_cannot_be_authored_so_a_group_over_them_is_not_asked_for():
    """A grid states one run-time axis, its first: the group over two run-time axes the writer refuses by name has no author."""
    with pytest.raises(HawkError, match=r"only the FIRST axis may be bare"):
        Grid["row", "col"]


def test_a_group_for_the_device_is_refused_by_name():
    with pytest.raises(
        HawkError, match=r"a group body is not written for this backend yet"
    ):
        emit_source(_ring("wrap"), targets=("cuda",))
