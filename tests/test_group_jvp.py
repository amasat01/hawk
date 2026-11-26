# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The forward derivative of a kernel with a group, run on the host.

Every tangent is checked against two references that do not use the transform: a central finite difference of the primal
kernel run in float64 with the step ``H`` (truncation ``H**2 / 6 * |f'''|`` and rounding ``eps * |f| / H`` both stay far
below ``FD_TOL``, so the tolerance is ``rtol = 1e-7`` with ``atol = 1e-8``), and the closed-form Jacobian-vector product
written in NumPy here, held to ``rtol = atol = 1e-12`` (a handful of float64 operations). A minimum or a maximum is not
differentiable at a tie, so its ties and its NaN lanes are pinned by the closed form alone.
"""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import Grid, Index, Mutable, Scalar, Vector
from hawk import math as m
from hawk.artifact import build_bundle
from hawk.trace.kernel import Kernel

RNG = np.random.default_rng(20261127)
H = 1e-5
FD_TOL = {"rtol": 1e-7, "atol": 1e-8}
EXACT = {"rtol": 1e-12, "atol": 1e-12}
EDGES = ("clamp", "wrap", "zero")


def _groups(field, lanes):
    """The consecutive groups of ``lanes`` entries of ``field``; the last may be shorter."""
    return [field[start : start + lanes] for start in range(0, len(field), lanes)]


def _per_group(field, lanes, one):
    """``one(group)`` of each group of ``field`` (arrays of the group's length), joined back into one lane vector."""
    return np.concatenate([one(g) for g in _groups(field, lanes)])


def _moved(group, shift, edge):
    """The entries of one group read at each place moved by ``shift``, under the edge rule, at the group's real size."""
    place = np.arange(len(group)) + shift
    if edge == "wrap":
        return np.roll(group, -shift)
    inside = np.clip(place, 0, len(group) - 1)
    if edge == "clamp":
        return group[inside]
    return np.where((place >= 0) & (place < len(group)), group[inside], 0.0)


def _ring(edge):
    """Each lane reads its neighbour's square from scratch under ``edge``."""

    @hawk.kernel
    def ring(x: Scalar, y: Mutable[Scalar]):
        with hawk.group(lanes=8) as g:
            t = g.scratch("t", Scalar)
            t[g.rank] = x * x
            g.barrier()
            y = t[g.rank] * x + 2.0 * t.at(g.rank + 1, edge=edge)

    ring.name = f"ring_{edge}"
    return ring


@hawk.kernel
def two_planes(x: Scalar, u: Scalar, y: Mutable[Scalar]):
    """Two scratch planes, one a vector, read after a barrier at the next lane and at the own lane."""
    with hawk.group(lanes=8) as g:
        a = g.scratch("a", Scalar)
        w = g.scratch("w", Vector[2])
        a[g.rank] = x * x
        w[g.rank] = m.vec([x * u, m.sin(x)])
        g.barrier()
        moved = w.at(g.rank + 1, edge="wrap")
        y = moved[0] * a[g.rank] + moved[1]


def _share(order):
    """Each lane's share of the group's sum of squares, the sum added in ``order``."""

    @hawk.kernel
    def share(x: Scalar, y: Mutable[Scalar]):
        with hawk.group(lanes=8) as g:
            t = g.scratch("t", Scalar)
            t[g.rank] = x * x
            total = g.sum(t[g.rank], order=order)
            y = t[g.rank] / total

    share.name = f"share_{order}"
    return share


@hawk.kernel
def group_min(x: Scalar, y: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        y = g.min(3.0 * x)


@hawk.kernel
def group_max(x: Scalar, y: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        y = g.max(x * x)


@hawk.kernel
def group_flags(x: Scalar, y: Mutable[Scalar]):
    """``any`` and ``all`` pick between two smooth values: they have no tangent of their own."""
    with hawk.group(lanes=8) as g:
        y = (
            m.where(g.any(x > 0.9), x * x, 3.0 * x)
            + m.where(g.all(x > -0.9), 1.0, 0.0) * x
        )


@hawk.kernel
def scanned(x: Scalar, y: Mutable[Scalar]):
    """The running sum of the squares over the group."""
    with hawk.group(lanes=8) as g:
        y = g.scan(x * x)


@hawk.kernel
def over_rows(lane: Grid["row":3, "col":5], x: Scalar, y: Mutable[Scalar]):  # noqa: F821 axis labels
    with hawk.group(over="row") as g:
        t = g.scratch("t", Scalar)
        t[g.rank] = x * x
        g.barrier()
        y = t.at(g.rank + 1, edge="wrap") * x + g.sum(x * x)


SHARES = [_share("tree"), _share("lane")]
PRIMALS = [
    *(_ring(edge) for edge in EDGES),
    two_planes,
    group_min,
    group_max,
    group_flags,
    scanned,
    over_rows,
]


def _tangent(primal):
    """The kernel holding the forward derivative of ``primal`` along all its float inputs."""
    return Kernel(f"{primal.name}_jvp", hawk.diff.jvp(primal))


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    """Every primal and its tangent kernel, built for the host in float64, once."""
    kernels = [k for p in PRIMALS for k in (p, _tangent(p))]
    return build_bundle(
        kernels,
        tmp_path_factory.mktemp("group_jvp"),
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )


@pytest.fixture(scope="module")
def share_bundle(tmp_path_factory, cache_dir):
    """The normalisations and their tangents, built apart from the other kernels."""
    return build_bundle(
        [k for p in SHARES for k in (p, _tangent(p))],
        tmp_path_factory.mktemp("group_jvp_share"),
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )


def _run(bundle, name, **planes):
    hawk.run(hawk.load(bundle.directory, name), **planes)


def _tangent_of(bundle, primal, out_like, **inputs):
    """The tangent plane the tangent kernel of ``primal`` writes, from inputs ``x=..`` and their ``dot_x=..``."""
    dot_y = np.zeros_like(out_like)
    _run(bundle, f"{primal}_jvp", dot_y=dot_y, **inputs)
    return dot_y


def _central(bundle, primal, out_like, inputs, directions):
    """The central finite difference of the primal kernel along ``directions`` with the step ``H``."""

    def at(sign):
        y = np.zeros_like(out_like)
        moved = {k: v + sign * H * directions.get(k, 0.0) for k, v in inputs.items()}
        _run(bundle, primal, y=y, **moved)
        return y

    return (at(+1.0) - at(-1.0)) / (2.0 * H)


def _check(bundle, primal, inputs, closed, n=None):
    """Run the tangent along random directions and compare with the central difference and with ``closed(dot)``."""
    n = len(next(iter(inputs.values()))) if n is None else n
    directions = {k: RNG.normal(size=np.shape(v)) for k, v in inputs.items()}
    dots = {f"dot_{k}": d for k, d in directions.items()}
    out_like = np.zeros(np.shape(next(iter(inputs.values()))))
    got = _tangent_of(bundle, primal, out_like, **inputs, **dots)
    np.testing.assert_allclose(
        got, _central(bundle, primal, out_like, inputs, directions), **FD_TOL
    )
    np.testing.assert_allclose(got, closed(directions), **EXACT)


@pytest.mark.parametrize("edge", EDGES)
def test_the_tangent_of_a_ring_stencil_follows_the_edge_rule_in_a_short_last_group(
    bundle, edge
):
    """Oracle: the closed form ``dy = 2 x dx x + x^2 dx + 2 shift(2 x dx)`` per slice of 8 lanes (19 lanes: the last has 3)."""
    x = RNG.normal(size=19)

    def closed(d):
        dt = 2.0 * x * d["x"]
        shifted = _per_group(dt, 8, lambda g: _moved(g, 1, edge))
        return dt * x + x * x * d["x"] + 2.0 * shifted

    _check(bundle, f"ring_{edge}", {"x": x}, closed)


def test_the_tangent_of_two_scratch_planes_one_a_vector_is_read_at_the_next_lane_and_the_own_lane(
    bundle,
):
    """Oracle: ``y = x_j u_j x_i^2 + sin(x_j)`` with ``j`` the next lane of the slice of 8 (wrapped), differentiated by hand."""
    x, u = RNG.normal(size=19), RNG.normal(size=19)

    def closed(d):
        def wrap(v):
            return _per_group(v, 8, lambda g: _moved(g, 1, "wrap"))

        moved0, dmoved0 = wrap(x * u), wrap(d["x"] * u + x * d["u"])
        dmoved1 = wrap(np.cos(x) * d["x"])
        return dmoved0 * x * x + moved0 * 2.0 * x * d["x"] + dmoved1

    _check(bundle, "two_planes", {"x": x, "u": u}, closed)


@pytest.mark.parametrize("order", ("tree", "lane"))
def test_the_tangent_of_a_normalisation_by_the_group_sum_is_the_quotient_rule_over_the_group(
    share_bundle, order
):
    """Oracle: ``dy = dt / T - t dT / T^2`` with ``t = x^2`` and ``T`` its group sum, per slice of 8 (19 lanes)."""
    x = RNG.normal(size=19)

    def closed(d):
        t, dt = x * x, 2.0 * x * d["x"]
        total = _per_group(t, 8, lambda g: np.full(len(g), g.sum()))
        dtotal = _per_group(dt, 8, lambda g: np.full(len(g), g.sum()))
        return dt / total - t * dtotal / total**2

    _check(share_bundle, f"share_{order}", {"x": x}, closed)


def test_the_tangent_of_a_scan_is_the_scan_of_the_tangents(bundle):
    """Oracle: ``numpy.cumsum`` of ``2 x dx`` per slice of 8 lanes (19 lanes)."""
    x = RNG.normal(size=19)
    _check(
        bundle,
        "scanned",
        {"x": x},
        lambda d: _per_group(2.0 * x * d["x"], 8, np.cumsum),
    )


def test_the_tangent_of_a_group_over_an_axis_follows_the_ring_and_the_sum_along_that_axis(
    bundle,
):
    """Oracle: ``numpy.roll`` and a column sum of the closed form on a 3 x 5 grid, groups along the rows."""
    x = RNG.normal(size=(3, 5))

    def closed(d):
        dt = 2.0 * x * d["x"]
        return (
            np.roll(dt, -1, axis=0) * x
            + np.roll(x * x, -1, axis=0) * d["x"]
            + dt.sum(axis=0)
        )

    _check(bundle, "over_rows", {"x": x}, closed)


def test_the_tangent_of_any_and_all_is_the_tangent_of_the_values_they_choose_between(
    bundle,
):
    """Oracle: the closed form of the chosen branch per group of 8 (group one has a lane above 0.9, group two none)."""
    x = np.concatenate([np.linspace(-0.5, 0.95, 8), np.linspace(-0.5, 0.5, 8)])

    def closed(d):
        any_hit = _per_group(x, 8, lambda g: np.full(len(g), g.max() > 0.9))
        all_hit = _per_group(x, 8, lambda g: np.full(len(g), g.min() > -0.9))
        return np.where(any_hit, 2.0 * x * d["x"], 3.0 * d["x"]) + np.where(
            all_hit, d["x"], 0.0
        )

    _check(bundle, "group_flags", {"x": x}, closed)


def _extreme(bundle, name, x, dx):
    """The tangent plane of ``group_min`` or ``group_max`` for the inputs ``x`` along ``dx``."""
    return _tangent_of(bundle, name, np.zeros_like(x), x=x, dot_x=dx)


def test_the_tangent_of_a_minimum_is_the_tangent_of_the_lane_holding_it(bundle):
    """Oracle: ``3 dx`` of ``numpy.argmin`` per slice of 8, and a central difference of the primal (distinct values)."""
    x = RNG.permutation(16).astype(float) - 5.0
    dx = RNG.normal(size=16)
    got = _extreme(bundle, "group_min", x, dx)
    want = _per_group(
        np.arange(16), 8, lambda g: np.full(len(g), 3.0 * dx[g[np.argmin(x[g])]])
    )
    np.testing.assert_allclose(got, want, **EXACT)
    np.testing.assert_allclose(
        got, _central(bundle, "group_min", x, {"x": x}, {"x": dx}), **FD_TOL
    )


def test_the_tangent_of_a_maximum_is_the_tangent_of_the_lane_holding_it(bundle):
    """Oracle: ``2 x dx`` of ``numpy.argmax`` of the squares per slice of 8, and a central difference (distinct squares)."""
    x = (RNG.permutation(16).astype(float) - 5.5) * np.where(
        RNG.random(16) < 0.5, 1.0, -1.0
    )
    dx = RNG.normal(size=16)
    got = _extreme(bundle, "group_max", x, dx)
    want = _per_group(
        np.arange(16),
        8,
        lambda g: np.full(
            len(g), 2.0 * x[g[np.argmax(x[g] ** 2)]] * dx[g[np.argmax(x[g] ** 2)]]
        ),
    )
    np.testing.assert_allclose(got, want, **EXACT)
    np.testing.assert_allclose(
        got, _central(bundle, "group_max", x, {"x": x}, {"x": dx}), **FD_TOL
    )


def test_a_tie_takes_the_tangent_of_the_first_lane_in_rank_order_whose_tangent_differs(
    bundle,
):
    """Oracle: by construction. Lanes 2 and 5 tie for the minimum of group one, lanes 1 and 6 of group two, each with a
    different tangent; the answer is the tangent of lanes 2 and 1. Lanes 3 and 4 (``2 and -2``) tie for the maximum of the squares."""
    x = np.array([5, 6, 1, 7, 8, 1, 9, 4, 3, 1, 6, 8, 9, 1, 5, 7], dtype=float)
    dx = np.arange(16, dtype=float) + 1.0
    got = _extreme(bundle, "group_min", x, dx)
    np.testing.assert_array_equal(got[:8], np.full(8, 3.0 * dx[2]))
    np.testing.assert_array_equal(got[8:], np.full(8, 3.0 * dx[9]))
    squares = np.array([0, 1, 1, 2, -2, 1, 0, 1, 1, 0, 1, 1, 2, 1, -2, 1], dtype=float)
    got = _extreme(bundle, "group_max", squares, dx)
    np.testing.assert_array_equal(got[:8], np.full(8, 2.0 * squares[3] * dx[3]))
    np.testing.assert_array_equal(got[8:], np.full(8, 2.0 * squares[12] * dx[12]))


@pytest.mark.parametrize("name", ("group_min", "group_max"))
def test_a_nan_lane_never_wins_and_a_group_of_nan_has_a_zero_tangent(bundle, name):
    """Oracle: by construction. Group one has NaN on its first lane (a huge tangent there) and a clear extreme elsewhere;
    group two is all NaN."""
    x = np.array([np.nan, 2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0] + [np.nan] * 8)
    dx = np.arange(16, dtype=float) + 100.0
    got = _extreme(bundle, name, x, dx)
    lane = 2 if name == "group_min" else 7
    scale = 3.0 if name == "group_min" else 2.0 * x[lane]
    np.testing.assert_array_equal(got[:8], np.full(8, scale * dx[lane]))
    np.testing.assert_array_equal(got[8:], np.zeros(8))


def test_the_tangent_of_integer_scratch_and_the_rank_and_size_is_nothing():
    """Oracle: the planes the tangent kernel names are the tangents of float planes only."""

    @hawk.kernel
    def counted(x: Scalar, y: Mutable[Scalar]):
        with hawk.group(lanes=8) as g:
            k = g.scratch("k", hawk.Index)
            k[g.rank] = g.rank
            g.barrier()
            y = x * m.real(k.at(g.rank + 1, edge="wrap")) + m.real(g.size)

    derived = hawk.diff.jvp(counted)
    assert set(derived.tangents) == {"dot_x", "dot_y"}


def test_a_scratch_plane_no_lane_reads_still_gets_its_tangent_plane_beside_it():
    """Oracle: the planes the derivative names, from the kernel's own declarations."""

    @hawk.kernel
    def unread(x: Scalar, y: Mutable[Scalar]):
        with hawk.group(lanes=8) as g:
            t = g.scratch("t", Scalar)
            t[g.rank] = x
            y = x * m.real(g.size)

    derived = hawk.diff.jvp(unread)
    names = {s.field_text("name") for s in derived[0].field_nodes("scratch")}
    assert names == {"t", "dot_t"}


BINS = 5


@hawk.kernel
def keyed_add(
    cell: Index, x: Scalar, acc: Mutable[Scalar, "bin":BINS], total: Mutable[Scalar]
):  # noqa: F821 axis labels
    """Each lane adds the square of its value into the cell it names, beside a per-lane output."""
    with hawk.group(lanes=8) as g:
        hawk.scatter(acc, contract="plain")[cell] += x * x
        total = x * x * m.real(g.size)


@hawk.kernel
def keyed_store(
    cell: Index, x: Scalar, slot: Mutable[Scalar, "bin":32], total: Mutable[Scalar]
):  # noqa: F821 axis labels
    """Each lane stores the square of its value into the cell it names (distinct cells), beside a per-lane output."""
    with hawk.group(lanes=8) as g:
        hawk.scatter(slot, conflict="any")[cell] = x * x
        total = x * x * m.real(g.size)


@pytest.fixture(scope="module")
def keyed_bundle(tmp_path_factory, cache_dir):
    """The keyed kernels and their tangents, built for the host in float64."""
    kernels = [k for p in (keyed_add, keyed_store) for k in (p, _tangent(p))]
    return build_bundle(
        kernels,
        tmp_path_factory.mktemp("group_jvp_keyed"),
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )


@pytest.mark.parametrize(
    ("name", "plane", "extent"),
    [("keyed_add", "acc", BINS), ("keyed_store", "slot", 32)],
)
def test_the_tangent_of_a_keyed_write_in_a_group_lands_in_the_cell_the_primal_writes(
    keyed_bundle, name, plane, extent
):
    """Oracle: ``numpy.add.at`` (or an assignment at distinct cells) of ``2 x dx`` for the keyed plane, and ``2 x dx`` times
    the group size (8, 8, 3 for 19 lanes) for the output; and a central difference of the primal with the step ``H``."""
    n = 19
    cell = (
        RNG.integers(0, extent, size=n)
        if name == "keyed_add"
        else RNG.permutation(extent)[:n]
    ).astype(np.int32)
    x, dx = RNG.normal(size=n), RNG.normal(size=n)
    got_plane, got_total = np.zeros(extent), np.zeros(n)
    _run(
        keyed_bundle,
        f"{name}_jvp",
        cell=cell,
        x=x,
        dot_x=dx,
        **{f"dot_{plane}": got_plane, "dot_total": got_total},
    )
    want = np.zeros(extent)
    if name == "keyed_add":
        np.add.at(want, cell, 2.0 * x * dx)
    else:
        want[cell] = 2.0 * x * dx
    np.testing.assert_allclose(got_plane, want, **EXACT)
    np.testing.assert_allclose(
        got_total, 2.0 * x * dx * np.repeat([8.0, 8.0, 3.0], [8, 8, 3]), **EXACT
    )

    def primal(sign):
        written, total = np.zeros(extent), np.zeros(n)
        _run(
            keyed_bundle,
            name,
            cell=cell,
            x=x + sign * H * dx,
            **{plane: written, "total": total},
        )
        return written, total

    (plane_hi, total_hi), (plane_lo, total_lo) = primal(+1.0), primal(-1.0)
    np.testing.assert_allclose(got_plane, (plane_hi - plane_lo) / (2.0 * H), **FD_TOL)
    np.testing.assert_allclose(got_total, (total_hi - total_lo) / (2.0 * H), **FD_TOL)
