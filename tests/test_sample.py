# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``table.sample(x, ...)``: a table read at real coordinates in index units, one to three dimensions.

Every value is compared with ``interpolate`` below, a NumPy interpolation written from the definition (per axis
``i = floor(x)``, ``w = x - i``, the corners ``i`` and ``i + 1`` weighted by ``1 - w`` and ``w``; ``nearest`` reads at
``floor(x + 0.5)``; the edge rule applies to each corner on its own). Derivatives are compared with that function: the
table enters linearly, so its forward and reverse maps are ``interpolate`` itself and its transpose built from unit
tables; a coordinate's derivative is the slope between neighbours, taken as a central difference of ``interpolate`` at
points away from a cell boundary, where the interpolant is exactly linear. Rows with the ``gpu`` mark run the same
kernels on the device (arithmetic form)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from _team_stages import run_staged
from conftest import sidecar_of

import hawk
import hawk.diff
from hawk import HawkError, Kernel, Mutable, Scalar, Table, Vector, kernel
from hawk.artifact import build_bundle

TARGETS = ["host", pytest.param("cuda", marks=pytest.mark.gpu)]
FILTERS = ("nearest", "linear")
EDGES = ("clamp", "wrap", "zero")
#: The extents of the tables of one, two and three axes.
SHAPES = {1: (6,), 2: (5, 4), 3: (4, 3, 2)}
N = 96


def _sampler(dimension: int, filter_: str, edge: str):
    """The kernel ``y = p.sample(x0, ..., filter=..., edge=...)`` over a static table of ``dimension`` axes."""
    if dimension == 1:

        def sampler(p: Table["i":6], x0: Scalar, y: Mutable[Scalar]):  # noqa: F821 axis labels
            y = p.sample(x0, filter=filter_, edge=edge)
    elif dimension == 2:

        def sampler(p: Table["i":5, "j":4], x0: Scalar, x1: Scalar, y: Mutable[Scalar]):  # noqa: F821 axis labels
            y = p.sample(x0, x1, filter=filter_, edge=edge)
    else:

        def sampler(  # noqa: D103
            p: Table["i":4, "j":3, "k":2],  # noqa: F821 axis labels
            x0: Scalar,
            x1: Scalar,
            x2: Scalar,
            y: Mutable[Scalar],
        ):
            y = p.sample(x0, x1, x2, filter=filter_, edge=edge)

    return kernel(sampler)


def interpolate(table, x, filter_, edge):
    """The table at the points ``x`` (one row per axis), by the definition in the module docstring."""
    axes = len(x)
    out = np.zeros(x.shape[1])

    def cell(index):
        """``table`` at integer ``index`` rows under the edge rule, zero where the rule says so."""
        valid = np.ones(x.shape[1], dtype=bool)
        fixed = []
        for k, extent in enumerate(table.shape):
            i = index[k]
            if edge == "clamp":
                i = np.clip(i, 0, extent - 1)
            elif edge == "wrap":
                i = np.mod(i, extent)
            else:
                valid &= (i >= 0) & (i < extent)
                i = np.clip(i, 0, extent - 1)
            fixed.append(i)
        return np.where(valid, table[tuple(fixed)], 0.0)

    if filter_ == "nearest":
        return cell([np.floor(x[k] + 0.5).astype(np.int64) for k in range(axes)])
    low = np.floor(x)
    w = x - low
    for corner in itertools.product((0, 1), repeat=axes):
        weight = np.ones(x.shape[1])
        for k, high in enumerate(corner):
            weight = weight * (w[k] if high else 1.0 - w[k])
        out += weight * cell([low[k].astype(np.int64) + corner[k] for k in range(axes)])
    return out


def _points(dimension, seed=7):
    """Coordinates that include whole numbers, the last cell and points beyond both ends of every axis."""
    rng = np.random.default_rng(seed)
    rows = []
    for extent in SHAPES[dimension]:
        special = [-2.75, -1.0, -0.5, 0.0, 0.25, 1.0, 1.5, extent - 1.0, extent - 0.5, extent, extent + 1.5]
        rows.append(np.concatenate([special, rng.uniform(-2, extent + 2, N - len(special))]))
    return np.array(rows)


def _table(dimension, seed=3):
    return np.random.default_rng(seed).uniform(-2, 2, SHAPES[dimension])


def _built(dimension, filter_, edge, directory, cache_dir):
    """The bundle of the sampler kernel, for the device as well as the host (a row that needs the device is skipped
    without one)."""
    return build_bundle(
        [_sampler(dimension, filter_, edge)],
        directory,
        targets=("cuda", "host"),
        cache_dir=cache_dir,
        host_profile="native",
    )


def _launch(bundle, name, target, planes, out):
    """One launch of kernel ``name`` over the NumPy ``planes``; the plane ``out`` afterwards."""
    planes = {k: v.copy() if k == out else v for k, v in planes.items()}
    if target == "host":
        from hawk import runtime as rt

        rt.run(rt.load(bundle.directory, name, sidecar_of(bundle, name)), **planes)
        return planes[out]
    import cupy as cp

    device = {k: cp.asarray(v) for k, v in planes.items()}
    run_staged(bundle, name, device, target)
    return cp.asnumpy(device[out])


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("edge", EDGES)
@pytest.mark.parametrize("filter_", FILTERS)
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_a_sample_equals_the_numpy_interpolation(tmp_path, cache_dir, dimension, filter_, edge, target):
    bundle = _built(dimension, filter_, edge, tmp_path, cache_dir)
    table, x = _table(dimension), _points(dimension)
    planes = {"p": table, **{f"x{k}": x[k] for k in range(dimension)}, "y": np.zeros(x.shape[1])}
    got = _launch(bundle, "sampler", target, planes, "y")
    np.testing.assert_allclose(got, interpolate(table, x, filter_, edge), rtol=1e-12, atol=1e-12)


def test_a_whole_coordinate_is_exactly_the_stored_element(tmp_path, cache_dir):
    bundle = _built(2, "linear", "clamp", tmp_path, cache_dir)
    table = _table(2)
    grid = np.ascontiguousarray(np.array(list(itertools.product(range(5), range(4)))).T.astype(float))
    planes = {"p": table, "x0": grid[0], "x1": grid[1], "y": np.zeros(grid.shape[1])}
    got = _launch(bundle, "sampler", "host", planes, "y")
    np.testing.assert_array_equal(got, table[grid[0].astype(int), grid[1].astype(int)])


@kernel
def vector_sampler(p: Table[Vector[3], "i":6], x: Scalar, y: Mutable[Vector[3]]):  # noqa: F821 axis labels
    y = p.sample(x)


def test_a_table_of_vectors_is_sampled_component_by_component(tmp_path, cache_dir):
    """Reference: ``np.interp`` of each component over the cell numbers (it holds the end elements beyond them)."""
    from hawk import runtime as rt

    bundle = build_bundle([vector_sampler], tmp_path, cache_dir=cache_dir, targets=("host",), host_profile="native")
    table = np.random.default_rng(4).uniform(-2, 2, (3, 6))
    x = np.array([-1.5, 0.0, 0.5, 2.25, 5.0, 7.5])
    y = np.zeros((3, len(x)))
    rt.run(rt.load(bundle.directory, "vector_sampler"), p=table, x=x, y=y)
    want = np.array([np.interp(x, np.arange(6), component) for component in table])
    np.testing.assert_allclose(y, want, rtol=1e-12, atol=1e-12)


# ---- derivatives -------------------------------------------------------------------------------------


def _slope(table, x, axis, filter_, edge, step=1e-6):
    """The central difference of ``interpolate`` along ``axis``: the slope between neighbours."""
    up, down = x.copy(), x.copy()
    up[axis] += step
    down[axis] -= step
    return (interpolate(table, up, filter_, edge) - interpolate(table, down, filter_, edge)) / (2 * step)


def _away_from_cells(dimension, seed=11):
    """Points whose fraction is in [0.1, 0.9] on every axis, so no point is within a step of a cell boundary."""
    rng = np.random.default_rng(seed)
    return np.array(
        [rng.integers(-3, extent + 3, N) + rng.uniform(0.1, 0.9, N) for extent in SHAPES[dimension]]
    )


def _derivative_bundles(dimension, edge, tmp_path, cache_dir, filter_="linear"):
    primal = _sampler(dimension, filter_, edge)
    built = build_bundle([primal], tmp_path / "p", cache_dir=cache_dir, targets=("host",), host_profile="native")
    forward = Kernel("fwd", hawk.diff.jvp(primal, primal_unit=built.digest))
    reverse = Kernel("rev", hawk.diff.vjp(primal, primal_unit=built.digest))
    return {
        "fwd": build_bundle([forward], tmp_path / "f", cache_dir=cache_dir, targets=("host",), host_profile="native"),
        "rev": build_bundle([reverse], tmp_path / "r", cache_dir=cache_dir, targets=("host",), host_profile="native"),
    }


@pytest.mark.parametrize(
    ("dimension", "edge"), [(1, "clamp"), (2, "clamp"), (2, "wrap"), (2, "zero"), (3, "clamp")]
)
def test_the_forward_derivative_is_the_weighted_table_tangent_plus_the_slopes(
    tmp_path, cache_dir, dimension, edge
):
    bundles = _derivative_bundles(dimension, edge, tmp_path, cache_dir)
    table, x = _table(dimension), _away_from_cells(dimension)
    rng = np.random.default_rng(5)
    dot_p, dot_x = rng.uniform(-1, 1, table.shape), rng.uniform(-1, 1, x.shape)
    planes = {"p": table, **{f"x{k}": x[k] for k in range(dimension)}, "dot_p": dot_p}
    planes |= {f"dot_x{k}": dot_x[k] for k in range(dimension)} | {"dot_y": np.zeros(N)}
    from hawk import runtime as rt

    rt.run(rt.load(bundles["fwd"].directory, "fwd"), **planes)
    want = interpolate(dot_p, x, "linear", edge) + sum(
        _slope(table, x, k, "linear", edge) * dot_x[k] for k in range(dimension)
    )
    np.testing.assert_allclose(planes["dot_y"], want, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    ("dimension", "edge"), [(1, "clamp"), (2, "clamp"), (2, "wrap"), (2, "zero"), (3, "clamp")]
)
def test_the_reverse_derivative_is_the_transpose_of_the_weights_and_the_slopes(
    tmp_path, cache_dir, dimension, edge
):
    bundles = _derivative_bundles(dimension, edge, tmp_path, cache_dir)
    table, x = _table(dimension), _away_from_cells(dimension)
    bar_y = np.random.default_rng(6).uniform(-1, 1, N)
    bar_p, bar_x = np.zeros(table.shape), np.zeros(x.shape)
    planes = {"p": table, **{f"x{k}": x[k] for k in range(dimension)}, "bar_y": bar_y, "bar_p": bar_p}
    planes |= {f"bar_x{k}": bar_x[k] for k in range(dimension)}
    from hawk import runtime as rt

    rt.run(rt.load(bundles["rev"].directory, "rev"), **planes)
    want_p = np.zeros(table.shape)
    for cell in np.ndindex(*table.shape):
        unit = np.zeros(table.shape)
        unit[cell] = 1.0
        want_p[cell] = bar_y @ interpolate(unit, x, "linear", edge)
    np.testing.assert_allclose(planes["bar_p"], want_p, rtol=1e-9, atol=1e-10)
    for k in range(dimension):
        want = bar_y * _slope(table, x, k, "linear", edge)
        np.testing.assert_allclose(planes[f"bar_x{k}"], want, rtol=1e-6, atol=1e-8)


def test_a_nearest_sample_has_no_derivative_in_its_coordinates_and_moves_with_the_table_alone(
    tmp_path, cache_dir
):
    """``floor`` has a zero derivative: the forward kernel declares no plane for a coordinate's tangent (a launch that
    names one is refused), the reverse kernel's adjoint of a coordinate stays zero, and the table's derivatives are
    those of the element read."""
    from hawk import runtime as rt

    bundles = _derivative_bundles(2, "clamp", tmp_path, cache_dir, filter_="nearest")
    table, x = _table(2), _away_from_cells(2)
    rng = np.random.default_rng(8)
    dot_p, bar_y = rng.uniform(-1, 1, table.shape), rng.uniform(-1, 1, N)
    forward = {"dot_p": dot_p, "x0": x[0], "x1": x[1], "dot_y": np.zeros(N)}
    rt.run(rt.load(bundles["fwd"].directory, "fwd"), **forward)
    np.testing.assert_allclose(forward["dot_y"], interpolate(dot_p, x, "nearest", "clamp"), atol=1e-13)
    with pytest.raises(HawkError, match="dot_x0"):
        rt.run(rt.load(bundles["fwd"].directory, "fwd"), **forward, dot_x0=np.zeros(N))
    reverse = {"x0": x[0], "x1": x[1], "bar_y": bar_y, "bar_p": np.zeros(table.shape)}
    reverse |= {"bar_x0": np.zeros(N), "bar_x1": np.zeros(N)}
    rt.run(rt.load(bundles["rev"].directory, "rev"), **reverse)
    assert not reverse["bar_x0"].any() and not reverse["bar_x1"].any()
    want_p = np.zeros(table.shape)
    for cell in np.ndindex(*table.shape):
        unit = np.zeros(table.shape)
        unit[cell] = 1.0
        want_p[cell] = bar_y @ interpolate(unit, x, "nearest", "clamp")
    np.testing.assert_allclose(reverse["bar_p"], want_p, atol=1e-13)


# ---- refusals by name --------------------------------------------------------------------------------


def test_a_wrong_number_of_coordinates_is_refused_by_name():
    def one_of_two(p: Table["i":5, "j":4], x0: Scalar, y: Mutable[Scalar]):  # noqa: F821 axis labels
        y = p.sample(x0)

    with pytest.raises(HawkError, match=r"has 2 axes, so a sample takes one coordinate per axis, got 1"):
        kernel(one_of_two)

    def four(p: Table["i":2], a: Scalar, b: Scalar, c: Scalar, d: Scalar, y: Mutable[Scalar]):  # noqa: F821
        y = p.sample(a, b, c, d)

    with pytest.raises(HawkError, match=r"one to three coordinates, got 4"):
        kernel(four)


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("filter", "cubic", r"filter 'cubic'.*\('nearest', 'linear'\)"),
        ("edge", "unchecked", r"edge rule 'unchecked'.*\('clamp', 'wrap', 'zero'\)"),
        ("edge", "mirror", r"edge rule 'mirror'"),
    ],
)
def test_an_unknown_filter_or_edge_is_refused_naming_the_valid_ones(option, value, message):
    def sampler(p: Table["i":5], x0: Scalar, y: Mutable[Scalar]):  # noqa: F821 axis labels
        y = p.sample(x0, **{option: value})

    with pytest.raises(HawkError, match=message):
        kernel(sampler)
