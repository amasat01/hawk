# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A per-lane input read at another lane: ``u.at(row=…, col=…, edge=…)`` and ``u.near(col=1, edge=…)``.

The read is ONE ``At`` over a plane that states the Grid's axes, with one index per axis and the edge rule; the plane is one
value per lane, lanes in Grid order. Values are checked against NumPy, independent of the code under test: ``np.pad`` with
mode ``"edge"`` (clamp), ``"wrap"`` and constant 0 (zero) for the neighbours of every lane, so every face and corner of the
border is read; the derivatives against a closed form (forward: the tangent read at the same lane; reverse: ``np.add.at`` of
the adjoint at the read lane, nothing for the outside under ``zero``).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from _eval import evaluate

import hawk
from hawk import Grid, HawkError, Mutable, Scalar, Vector, _ir
from hawk import math as m
from hawk.artifact import build_bundle, emit_source
from hawk.diff import jvp, vjp

EDGES = ("clamp", "wrap", "zero")
#: ``np.pad`` mode of each edge rule
PAD = {"clamp": {"mode": "edge"}, "wrap": {"mode": "wrap"}, "zero": {"mode": "constant", "constant_values": 0.0}}
SHAPES = {2: (3, 4), 3: (2, 3, 4)}
RNG = np.random.default_rng(20261122)


def _neighbours(field, offsets, edge):
    """Oracle: ``field`` of every lane read at the lane moved by ``offsets``, from the padded field."""
    padded = np.pad(field, 1, **PAD[edge])
    window = tuple(slice(1 + k, 1 + k + extent) for k, extent in zip(offsets, field.shape))
    return padded[window]


def _named(kernel, name):
    """``kernel`` under ``name``: one body shape is built for several edge rules, and a bundle keys kernels by name."""
    kernel.name = name
    return kernel


def _around(u, lane, axes, form, edge):
    """The reads of ``u`` at every neighbour (offsets -1, 0, 1 on each of ``axes``), by ``form`` ("at" or "near"), in the
    row-major order of the offsets; a plain helper, since a kernel body has no comprehension."""
    reads = []
    for offsets in itertools.product((-1, 0, 1), repeat=len(axes)):
        moved = dict(zip(axes, offsets))
        if form == "near":
            reads.append(u.near(**moved, edge=edge))
        else:
            reads.append(u.at(**{a: getattr(lane, a) + k for a, k in moved.items()}, edge=edge))
    return m.vec(reads)


def _grid2(form, edge):
    """The kernel reading all nine neighbours of a lane of a 2-D Grid."""
    def reads(lane: Grid["row":3, "col":4], u: Scalar, y: Mutable[Vector[9]]):
        y = _around(u, lane, ("row", "col"), form, edge)

    return _named(hawk.kernel(reads), f"reads2_{form}_{edge}")


def _grid3(form, edge):
    """The kernel reading all twenty-seven neighbours of a lane of a 3-D Grid."""
    def reads(lane: Grid["z":2, "row":3, "col":4], u: Scalar, y: Mutable[Vector[27]]):
        y = _around(u, lane, ("z", "row", "col"), form, edge)

    return _named(hawk.kernel(reads), f"reads3_{form}_{edge}")


KERNELS = {(rank, form, edge): (_grid2 if rank == 2 else _grid3)(form, edge)
           for rank in (2, 3) for form in ("at", "near") for edge in EDGES}


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    """Every kernel of :data:`KERNELS` built for the host, once."""
    return build_bundle(list(KERNELS.values()), tmp_path_factory.mktemp("lane_reads"), targets=("host",),
                        cache_dir=cache_dir, host_profile="native")


@pytest.mark.parametrize("edge", EDGES)
@pytest.mark.parametrize("form", ("at", "near"))
@pytest.mark.parametrize("rank", (2, 3))
def test_every_neighbour_of_every_lane_equals_numpy_padding_on_one_thread_and_a_team(bundle, rank, form, edge):
    """Oracle: ``np.pad`` of the field in the mode of the edge rule, sliced by each offset."""
    field = RNG.normal(size=SHAPES[rank])
    offsets = list(itertools.product((-1, 0, 1), repeat=rank))
    for structure in (hawk.HostSerial, hawk.HostTeam):
        out = np.zeros((len(offsets), *field.shape))
        loaded = hawk.load(bundle.directory, KERNELS[rank, form, edge].name)
        hawk.plan(loaded, structure).bind(u=field, y=out).launch()
        for k, offset in enumerate(offsets):
            np.testing.assert_array_equal(out[k], _neighbours(field, offset, edge), err_msg=f"{structure} {offset}")


@hawk.kernel
def laplacian(lane: Grid["row":5, "col":7], u: Scalar, y: Mutable[Scalar]):
    y = (u.near(row=-1) + u.near(row=1) + u.near(col=-1) + u.near(col=1)) - 4.0 * u


def test_a_five_point_stencil_written_with_near_equals_the_numpy_stencil(tmp_path, cache_dir):
    """Oracle: the stencil on the field padded by ``np.pad(mode="edge")``."""
    field = RNG.normal(size=(5, 7))
    bundle = build_bundle([laplacian], tmp_path, targets=("host",), cache_dir=cache_dir, host_profile="native")
    padded = np.pad(field, 1, mode="edge")
    want = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * field
    for structure in (hawk.HostSerial, hawk.HostTeam):
        out = np.zeros_like(field)
        hawk.plan(hawk.load(bundle.directory, "laplacian"), structure).bind(u=field, y=out).launch()
        np.testing.assert_allclose(out, want, rtol=1e-14, atol=1e-14)


@hawk.kernel
def shifted(lane: Grid["row", "col":4], u: Scalar, y: Mutable[Scalar]):
    y = u.near(row=-1, edge="wrap") + 2.0 * u.near(col=1, edge="zero") + u


def test_a_grid_with_a_run_time_axis_reads_neighbours_at_any_extent(tmp_path, cache_dir):
    """Oracle: ``np.pad`` wrap along the run-time axis, constant 0 along the static one, for two extents of one build."""
    bundle = build_bundle([shifted], tmp_path, targets=("host",), cache_dir=cache_dir, host_profile="native")
    loaded = hawk.load(bundle.directory, "shifted")
    for rows in (2, 5):
        field = RNG.normal(size=(rows, 4))
        out = np.zeros_like(field)
        hawk.run(loaded, u=field, y=out)
        want = (_neighbours(field, (-1, 0), "wrap") + 2.0 * _neighbours(field, (0, 1), "zero") + field)
        np.testing.assert_array_equal(out, want)


@pytest.mark.parametrize("edge", EDGES)
def test_a_flat_read_beside_a_read_by_axis_keeps_its_flat_edge_rule(tmp_path, cache_dir, edge):
    """Oracle: ``np.take`` of the flattened field with mode clip / wrap, a zero outside for ``zero``."""
    def both(lane: Grid["row":3, "col":4], u: Scalar, y: Mutable[Scalar]):
        y = u.near(col=1) + u.at(lane.flat + 5, edge=edge)

    both = _named(hawk.kernel(both), f"both_{edge}")
    field = RNG.normal(size=(3, 4))
    bundle = build_bundle([both], tmp_path, targets=("host",), cache_dir=cache_dir, host_profile="native")
    out = np.zeros_like(field)
    hawk.run(hawk.load(bundle.directory, both.name), u=field, y=out)
    flat, at = field.reshape(-1), np.arange(12) + 5
    far = {"clamp": np.take(flat, at, mode="clip"), "wrap": np.take(flat, at, mode="wrap"),
           "zero": np.where(at < 12, np.take(flat, at, mode="clip"), 0.0)}[edge]
    np.testing.assert_array_equal(out, _neighbours(field, (0, 1), "clamp") + far.reshape(3, 4))


# ---- the IR -------------------------------------------------------------------------------------------------------


def _reads(kernel):
    return [n for n in _ir.post_order(kernel.sinks) if _ir.is_a(n, "at")]


def test_a_read_by_axis_is_one_at_with_an_index_per_lane_axis_over_a_plane_that_states_the_axes():
    reads = [n for n in _reads(KERNELS[2, "at", "wrap"]) if n.field_text("edge") == "wrap"]
    assert len(reads) == 9 and all(len(n.operands) == 3 for n in reads)
    assert {tuple(label for label, _ in _ir.field_pairs(n.operands[0], "axes")) for n in reads} == {("row", "col")}


def test_near_builds_exactly_the_at_of_the_coordinates_moved_by_the_offsets():
    @hawk.kernel
    def by_near(lane: Grid["row":3, "col":4], u: Scalar, y: Mutable[Scalar]):
        y = u.near(col=1, edge="wrap")

    @hawk.kernel
    def by_at(lane: Grid["row":3, "col":4], u: Scalar, y: Mutable[Scalar]):
        y = u.at(row=lane.row, col=lane.col + 1, edge="wrap")

    assert evaluate(by_near.sinks, {"u": np.arange(12.0).reshape(3, 4)}, lane=7, lanes=12) == \
        evaluate(by_at.sinks, {"u": np.arange(12.0).reshape(3, 4)}, lane=7, lanes=12)


def test_a_flat_read_states_its_edge_rule_in_the_ir():
    @hawk.kernel
    def gather(lane: hawk.Index, u: Scalar, y: Mutable[Scalar]):
        y = u.at(lane + 1)

    (read,) = _reads(gather)
    assert read.field_text("edge") == "clamp" and len(read.operands) == 2


# ---- the derivatives ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("edge", EDGES)
def test_the_tangent_is_read_and_the_adjoint_added_at_the_read_lane_under_each_edge_rule(edge):
    """Oracle: the tangent field read by padding; the adjoint ``np.add.at`` at the clipped / wrapped / dropped lane."""
    @hawk.kernel
    def product(lane: Grid["row":3, "col":4], u: Scalar, y: Mutable[Scalar]):
        y = u.near(col=1, edge=edge) * u.near(row=-1, edge=edge)

    field, tangent = RNG.normal(size=(3, 4)), RNG.normal(size=(3, 4))
    forward, reverse = jvp(product.sinks), vjp(product.sinks)
    right, up = (_neighbours(f, d, edge) for f in (field,) for d in ((0, 1), (-1, 0)))
    dright, dup = (_neighbours(tangent, d, edge) for d in ((0, 1), (-1, 0)))
    for lane in range(12):
        row, col = divmod(lane, 4)
        env = {"u": field, "dot_u": tangent}
        got = evaluate(forward, env, lane=lane, lanes=12)["dot_y"]
        np.testing.assert_allclose(got, dright[row, col] * up[row, col] + right[row, col] * dup[row, col], rtol=1e-13)
        seed = float(RNG.normal())
        got = evaluate(reverse, {"u": field, "bar_y": seed}, lane=lane, lanes=12, shapes={"bar_u": (3, 4)})["bar_u"]
        want = np.zeros((3, 4))
        for (dr, dc), factor in (((0, 1), up[row, col]), ((-1, 0), right[row, col])):
            where = (row + dr, col + dc)
            if edge == "clamp":
                where = (np.clip(where[0], 0, 2), np.clip(where[1], 0, 3))
            elif edge == "wrap":
                where = (where[0] % 3, where[1] % 4)
            if edge != "zero" or (0 <= where[0] < 3 and 0 <= where[1] < 4):
                np.add.at(want, where, seed * factor)
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=1e-15)


def test_several_lanes_reading_one_lane_add_their_adjoints_into_it_on_the_host():
    """Oracle: ``np.add.at`` over every lane's clamped read lane (the last column is read by two lanes)."""
    @hawk.kernel
    def right(lane: Grid["row":3, "col":4], u: Scalar, y: Mutable[Scalar]):
        y = u.near(col=1)

    seed = RNG.normal(size=(3, 4))
    grad = np.asarray(hawk.Kernel("right_vjp", vjp(right))(bar_y=seed))
    want = np.zeros((3, 4))
    for row, col in itertools.product(range(3), range(4)):
        np.add.at(want, (row, min(col + 1, 3)), seed[row, col])
    np.testing.assert_allclose(grad, want, rtol=1e-13)


# ---- the refusals -------------------------------------------------------------------------------------------------


def test_a_flat_kernel_refuses_near_naming_the_way_out():
    with pytest.raises(HawkError, match=r"near\(\).*no Grid.*at\(i\)"):
        @hawk.kernel
        def flat(x: Scalar, y: Mutable[Scalar]):
            y = x.near(col=1)


def test_near_refuses_an_axis_the_grid_lacks_and_a_non_integer_offset():
    with pytest.raises(HawkError, match=r"near\(\): \['depth'\] are not axes"):
        @hawk.kernel
        def unknown(lane: Grid["row":3, "col":4], u: Scalar, y: Mutable[Scalar]):
            y = u.near(depth=1)

    for offset in (1.5, True, "1"):
        with pytest.raises(HawkError, match=r"near\(\): an offset is a Python integer"):
            @hawk.kernel
            def real(lane: Grid["row":3, "col":4], u: Scalar, y: Mutable[Scalar]):
                y = u.near(col=offset)


def test_an_unknown_edge_rule_is_refused_naming_the_rules():
    with pytest.raises(HawkError, match=r"edge='mirror'.*clamp.*wrap.*zero"):
        @hawk.kernel
        def mirror(lane: Grid["row":3, "col":4], u: Scalar, y: Mutable[Scalar]):
            y = u.near(col=1, edge="mirror")


def test_a_value_computed_in_the_launch_cannot_be_read_at_another_lane():
    with pytest.raises(HawkError, match="only a plane that existed before the launch"):
        @hawk.kernel
        def computed(lane: Grid["row":3, "col":4], u: Scalar, y: Mutable[Scalar]):
            y = (u * 2.0).near(col=1)


# ---- the device text and the device values -------------------------------------------------------------------------


def test_the_device_text_reads_the_plane_through_its_view_under_the_edge_rule():
    text = emit_source(KERNELS[2, "near", "wrap"], targets=("cuda",), device_arch="sm_90").sources["cuda"].text
    assert "aether::edge::wrap(" in text and "psc_u(" in text and "psc_u[i]" not in text
    zero = emit_source(KERNELS[2, "near", "zero"], targets=("cuda",), device_arch="sm_90").sources["cuda"].text
    assert "aether::edge::inside(" in zero


@pytest.mark.gpu
@pytest.mark.parametrize("edge", EDGES)
@pytest.mark.parametrize("rank", (2, 3))
def test_every_neighbour_equals_numpy_padding_on_the_device(tmp_path, cache_dir, rank, edge):
    """Oracle: ``np.pad`` of the field in the mode of the edge rule, sliced by each offset."""
    import cupy as cp
    from conftest import _launch

    kernel = KERNELS[rank, "near", edge]
    field = RNG.normal(size=SHAPES[rank])
    offsets = list(itertools.product((-1, 0, 1), repeat=rank))
    bundle = build_bundle([kernel], tmp_path, targets=("cuda", "host"), cache_dir=cache_dir)
    out = cp.zeros((len(offsets), *field.shape))
    _launch(bundle, kernel.name, "cuda", {"u": cp.asarray(field), "y": out}).launch()
    cp.cuda.Device().synchronize()
    for k, offset in enumerate(offsets):
        np.testing.assert_array_equal(out.get()[k], _neighbours(field, offset, edge))
