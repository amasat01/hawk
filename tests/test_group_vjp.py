# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The reverse derivative of a kernel with a group, run on the host.

Each kernel of ``test_group_jvp`` that has a reverse derivative is checked against two references that do not use the
transform. (a) The closed-form transpose written in NumPy here, held to ``rtol = atol = 1e-12`` (a handful of float64
operations per lane; a running sum's reverse is the total minus a prefix, which adds one rounding of the size of the total).
(b) The dot-product identity against the forward derivative, ``<J v, w> = <v, J^T w>`` for random ``v`` and ``w``: both
sides add the 19 lane terms ``|v_i| |(J^T w)_i|`` and ``|w_i| |(J v)_i|`` of at most 50 float64 operations each, so they
agree to ``1e-12`` times the sum of those absolute terms, about 9000 times the unit round-off ``eps``. Ties and NaN are pinned
by closed form alone.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_group_jvp import (
    EDGES,
    EXACT,
    RNG,
    SHARES,
    _groups,
    _moved,
    _per_group,
    _ring,
    group_flags,
    group_max,
    group_min,
    over_rows,
    scanned,
    two_planes,
)

import hawk
from hawk import Index, Mutable, Scalar
from hawk import math as m
from hawk.artifact import build_bundle
from hawk.errors import HawkError
from hawk.trace.kernel import Kernel

SHIFTS = (-3, -1, 2, 5)
LANES = (
    19  # two groups of 8 and a last one of 3, so every edge rule meets a short group
)


def _shifted(edge, shift):
    """Each lane reads the square of the lane ``shift`` away from scratch under ``edge``, scaled by its own value."""

    @hawk.kernel
    def shifted(x: Scalar, y: Mutable[Scalar]):
        with hawk.group(lanes=8) as g:
            t = g.scratch("t", Scalar)
            t[g.rank] = x * x
            g.barrier()
            y = x * t.at(g.rank + shift, edge=edge)

    shifted.name = f"shifted_{edge}_{shift + 10}"
    return shifted


@hawk.kernel
def group_max_linear(x: Scalar, y: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        y = g.max(2.0 * x)


@hawk.kernel
def own_read_twice_in_a_phase(x: Scalar, y: Mutable[Scalar]):
    """One slot read at the own rank by a group sum and by a lane expression of the same phase."""
    with hawk.group(lanes=8) as g:
        t = g.scratch("t", Scalar)
        t[g.rank] = x * x
        y = t[g.rank] * g.sum(t[g.rank])


@hawk.kernel
def own_read_in_two_phases(x: Scalar, y: Mutable[Scalar]):
    """One slot read at the own rank by lane expressions of two phases."""
    with hawk.group(lanes=8) as g:
        t = g.scratch("t", Scalar)
        u = g.scratch("u", Scalar)
        t[g.rank] = x * x
        g.barrier()
        u[g.rank] = t[g.rank] * x
        g.barrier()
        y = t[g.rank] + u[g.rank]


@hawk.kernel
def stored_twice(x: Scalar, y: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        t = g.scratch("t", Scalar)
        t[g.rank] = x
        g.barrier()
        t[g.rank] = t[g.rank] * x
        g.barrier()
        y = t[g.rank]


@hawk.kernel
def read_at_a_runtime_index(x: Scalar, k: Index, y: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        t = g.scratch("t", Scalar)
        t[g.rank] = x * x
        g.barrier()
        y = t.at(k, edge="clamp")


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
def keyed_store_any(
    cell: Index, x: Scalar, slot: Mutable[Scalar, "bin":32], total: Mutable[Scalar]
):  # noqa: F821 axis labels
    with hawk.group(lanes=8) as g:
        hawk.scatter(slot, conflict="any")[cell] = x * x
        total = x * x * m.real(g.size)


@hawk.kernel
def outside_sink(x: Scalar, y: Mutable[Scalar], z: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        y = g.sum(x)
    z = x * x


FAMILIES = [
    *(_ring(edge) for edge in EDGES),
    *(_shifted(edge, shift) for edge in EDGES for shift in SHIFTS),
    two_planes,
    group_min,
    group_max,
    group_max_linear,
    own_read_twice_in_a_phase,
    own_read_in_two_phases,
    scanned,
    over_rows,
]


def _reverse(primal):
    """The kernel holding the reverse derivative of ``primal``."""
    return Kernel(f"{primal.name}_vjp", hawk.diff.vjp(primal))


def _forward(primal):
    """The kernel holding the forward derivative of ``primal``."""
    return Kernel(f"{primal.name}_jvp", hawk.diff.jvp(primal))


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    """Every family with its forward and reverse kernels, built for the host in float64, once."""
    kernels = [k for p in FAMILIES for k in (p, _forward(p), _reverse(p))]
    return build_bundle(
        kernels,
        tmp_path_factory.mktemp("group_vjp"),
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )


def _run(bundle, name, **planes):
    hawk.run(hawk.load(bundle.directory, name), **planes)


def _gradients(bundle, primal, w, **inputs):
    """The gradient planes ``bar_<input>`` the reverse kernel of ``primal`` writes, for the output adjoint ``w``."""
    bars = {f"bar_{k}": np.zeros_like(v) for k, v in inputs.items()}
    _run(bundle, f"{primal}_vjp", bar_y=w, **inputs, **bars)
    return bars


def _transpose_of_shift(size, shift, edge):
    """The matrix ``S`` with ``S @ t == _moved(t, shift, edge)`` for a group of ``size`` lanes, built column by column."""
    return np.stack([_moved(np.eye(size)[j], shift, edge) for j in range(size)], axis=1)


def _scattered(field, lanes, shift, edge):
    """The transpose of the shifted read applied per group of ``lanes`` to ``field``: what each slot receives."""
    return np.concatenate(
        [_transpose_of_shift(len(g), shift, edge).T @ g for g in _groups(field, lanes)]
    )


@pytest.mark.parametrize("edge", EDGES)
def test_the_reverse_of_a_ring_stencil_returns_each_read_to_the_slot_it_read_in_a_short_last_group(
    bundle, edge
):
    """Oracle: the transpose of the shift matrix of each slice of 8 lanes (19 lanes: the last has 3), by hand."""
    x, w = RNG.normal(size=LANES), RNG.normal(size=LANES)
    got = _gradients(bundle, f"ring_{edge}", w, x=x)["bar_x"]
    slot = w * x + _scattered(2.0 * w, 8, 1, edge)
    np.testing.assert_allclose(got, w * x * x + 2.0 * x * slot, **EXACT)


@pytest.mark.parametrize("shift", SHIFTS)
@pytest.mark.parametrize("edge", EDGES)
def test_the_reverse_of_a_read_at_any_constant_offset_inverts_the_edge_rule_of_that_read(
    bundle, edge, shift
):
    """Oracle: the transpose of the shift matrix of each slice, offsets beyond the short group included."""
    x, w = RNG.normal(size=LANES), RNG.normal(size=LANES)
    got = _gradients(bundle, f"shifted_{edge}_{shift + 10}", w, x=x)["bar_x"]
    moved = _per_group(x * x, 8, lambda g: _moved(g, shift, edge))
    np.testing.assert_allclose(
        got, w * moved + 2.0 * x * _scattered(w * x, 8, shift, edge), **EXACT
    )


def test_the_reverse_of_two_scratch_planes_one_a_vector_reads_back_the_next_lane_and_the_own_lane(
    bundle,
):
    """Oracle: ``y = x_j u_j x_i^2 + sin(x_j)`` with ``j`` the next lane of the slice (wrapped), transposed by hand."""
    x, u, w = (RNG.normal(size=LANES) for _ in range(3))
    got = _gradients(bundle, "two_planes", w, x=x, u=u)
    moved = _per_group(x * u, 8, lambda g: _moved(g, 1, "wrap"))
    bar_w0 = _scattered(x * x * w, 8, 1, "wrap")
    bar_w1 = _scattered(w, 8, 1, "wrap")
    bar_a = moved * w
    np.testing.assert_allclose(
        got["bar_x"], bar_a * 2.0 * x + bar_w0 * u + bar_w1 * np.cos(x), **EXACT
    )
    np.testing.assert_allclose(got["bar_u"], bar_w0 * x, **EXACT)


def test_a_slot_read_at_the_own_rank_by_a_group_sum_and_a_lane_expression_gets_both_adjoints_once(
    bundle,
):
    """Oracle: ``y = t T`` with ``t = x^2`` and ``T`` its group sum, so ``bar_x = 2 x (w T + sum(w t))`` per slice of 8."""
    x, w = RNG.normal(size=LANES), RNG.normal(size=LANES)
    got = _gradients(bundle, "own_read_twice_in_a_phase", w, x=x)["bar_x"]
    t = x * x
    total = _per_group(t, 8, lambda g: np.full(len(g), g.sum()))
    cross = _per_group(w * t, 8, lambda g: np.full(len(g), g.sum()))
    np.testing.assert_allclose(got, 2.0 * x * (w * total + cross), **EXACT)


def test_a_slot_read_at_the_own_rank_in_two_phases_gets_the_adjoint_of_each_read_once(
    bundle,
):
    """Oracle: ``y = t + t x`` with ``t = x^2``, so ``bar_x = w (2 x + 3 x^2)`` (by hand)."""
    x, w = RNG.normal(size=LANES), RNG.normal(size=LANES)
    got = _gradients(bundle, "own_read_in_two_phases", w, x=x)["bar_x"]
    np.testing.assert_allclose(got, w * (2.0 * x + 3.0 * x * x), **EXACT)


def test_the_reverse_of_a_scan_is_the_reverse_running_sum_of_the_adjoints(bundle):
    """Oracle: ``numpy.cumsum`` of the adjoints taken from the last lane of each slice, times ``2 x``."""
    x, w = RNG.normal(size=LANES), RNG.normal(size=LANES)
    got = _gradients(bundle, "scanned", w, x=x)["bar_x"]
    suffix = _per_group(w, 8, lambda g: np.cumsum(g[::-1])[::-1])
    np.testing.assert_allclose(got, 2.0 * x * suffix, **EXACT)


def test_the_reverse_of_a_group_over_an_axis_follows_the_ring_and_the_sum_along_that_axis(
    bundle,
):
    """Oracle: ``numpy.roll`` and a column sum of the closed form on a 3 x 5 grid, groups along the rows."""
    x, w = RNG.normal(size=(3, 5)), RNG.normal(size=(3, 5))
    got = _gradients(bundle, "over_rows", w, x=x)["bar_x"]
    expected = w * np.roll(x * x, -1, axis=0) + 2.0 * x * (
        np.roll(w * x, 1, axis=0) + w.sum(axis=0)
    )
    np.testing.assert_allclose(got, expected, **EXACT)


def _extreme(bundle, name, x, w, scale):
    """The gradient of ``group_min`` or ``group_max`` for inputs ``x``, and its closed form: the group's summed adjoint
    ``w`` at the first lane holding the extreme of ``scale(x)``, with ``scale'(x)`` there (``scale`` takes ``(values)``)."""
    got = _gradients(bundle, name, w, x=x)["bar_x"]
    pick = np.argmin if name == "group_min" else np.argmax
    want = np.zeros_like(x)
    for start in range(0, len(x), 8):
        group = np.arange(start, min(start + 8, len(x)))
        want[group[pick(scale(x[group]))]] = w[group].sum()
    return got, want


def test_the_reverse_of_a_minimum_gives_the_summed_adjoint_to_the_lane_holding_it(
    bundle,
):
    """Oracle: ``3 sum(w)`` at ``numpy.argmin`` per slice of 8 (distinct values)."""
    x = RNG.permutation(16).astype(float) - 5.0
    got, want = _extreme(bundle, "group_min", x, RNG.normal(size=16), lambda v: v)
    np.testing.assert_allclose(got, 3.0 * want, **EXACT)


def test_the_reverse_of_a_maximum_gives_the_summed_adjoint_to_the_lane_holding_it(
    bundle,
):
    """Oracle: ``2 x sum(w)`` at ``numpy.argmax`` of the squares per slice of 8 (distinct squares)."""
    x = (RNG.permutation(16).astype(float) - 5.5) * np.where(
        RNG.random(16) < 0.5, 1.0, -1.0
    )
    got, want = _extreme(bundle, "group_max", x, RNG.normal(size=16), np.square)
    np.testing.assert_allclose(got, 2.0 * x * want, **EXACT)


def test_a_tie_sends_the_adjoint_to_the_first_lane_in_rank_order(bundle):
    """Oracle: by construction. Lanes 2 and 5 tie for the minimum of group one, lanes 1 and 6 of group two (group
    min is ``3 x``); lanes 3 and 4 (``2`` and ``-2``) tie for the maximum of the squares of group one."""
    x = np.array([5, 6, 1, 7, 8, 1, 9, 4, 3, 1, 6, 8, 9, 1, 5, 7], dtype=float)
    w = np.arange(16, dtype=float) + 1.0
    got = _gradients(bundle, "group_min", w, x=x)["bar_x"]
    want = np.zeros(16)
    want[2], want[9] = 3.0 * w[:8].sum(), 3.0 * w[8:].sum()
    np.testing.assert_array_equal(got, want)
    squares = np.array([0, 1, 1, 2, -2, 1, 0, 1, 1, 0, 1, 1, 2, 1, -2, 1], dtype=float)
    got = _gradients(bundle, "group_max", w, x=squares)["bar_x"]
    want = np.zeros(16)
    want[3], want[12] = 2.0 * squares[3] * w[:8].sum(), 2.0 * squares[12] * w[8:].sum()
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("name", ("group_min", "group_max_linear"))
def test_a_nan_lane_never_wins_and_a_group_of_nan_gives_zero(bundle, name):
    """Oracle: by construction. Group one has NaN on its first lane and a clear extreme elsewhere; group two is all NaN.
    The local derivative of ``3 x`` and ``2 x`` is finite, so the lanes that do not win hold exactly zero."""
    x = np.array([np.nan, 2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0] + [np.nan] * 8)
    w = np.arange(16, dtype=float) + 100.0
    got = _gradients(bundle, name, w, x=x)["bar_x"]
    lane, scale = (2, 3.0) if name == "group_min" else (7, 2.0)
    want = np.zeros(16)
    want[lane] = scale * w[:8].sum()
    np.testing.assert_array_equal(got, want)


def _assert_dot_identity(bundle, primal):
    """``<J v, w> = <v, J^T w>`` for random ``v`` and ``w``, to ``1e-12`` of the sum of the absolute lane terms."""
    inputs = {"x": RNG.normal(size=(3, 5) if primal is over_rows else LANES)}
    if primal is two_planes:
        inputs["u"] = RNG.normal(size=LANES)
    if primal is group_flags:
        inputs["x"] = RNG.uniform(-0.5, 1.0, size=LANES)
    shape = inputs["x"].shape
    v = {k: RNG.normal(size=shape) for k in inputs}
    w = RNG.normal(size=shape)
    jv = np.zeros(shape)
    _run(
        bundle,
        f"{primal.name}_jvp",
        dot_y=jv,
        **inputs,
        **{f"dot_{k}": d for k, d in v.items()},
    )
    bars = _gradients(bundle, primal.name, w, **inputs)
    forward = float(np.sum(w * jv))
    backward = sum(float(np.sum(v[k] * bars[f"bar_{k}"])) for k in inputs)
    scale = np.sum(np.abs(w * jv)) + sum(
        np.sum(np.abs(v[k] * bars[f"bar_{k}"])) for k in inputs
    )
    assert abs(forward - backward) <= 1e-12 * scale


@pytest.mark.parametrize("primal", FAMILIES, ids=lambda k: k.name)
def test_the_reverse_and_the_forward_derivative_agree_in_the_dot_product_identity(
    bundle, primal
):
    """Oracle: the forward derivative of the same kernel, through ``<J v, w> = <v, J^T w>``."""
    _assert_dot_identity(bundle, primal)


@pytest.fixture(scope="module")
def share_bundle(tmp_path_factory, cache_dir):
    """The normalisations by the group sum with their forward and reverse kernels, built for the host in float64."""
    kernels = [k for p in SHARES for k in (p, _forward(p), _reverse(p))]
    return build_bundle(
        kernels,
        tmp_path_factory.mktemp("group_vjp_share"),
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )


@pytest.mark.parametrize("primal", SHARES, ids=lambda k: k.name)
def test_the_reverse_and_the_forward_of_a_normalisation_agree_in_the_dot_product_identity(
    share_bundle, primal
):
    """Oracle: the forward derivative of the same kernel, through ``<J v, w> = <v, J^T w>``."""
    _assert_dot_identity(share_bundle, primal)


@pytest.fixture(scope="module")
def keyed_bundle(tmp_path_factory, cache_dir):
    """The reverse of the keyed kernel, built for the host in float64 beside the bundle that holds its primal (the two read
    and write across lanes in opposite ways, so they cannot share one bundle)."""
    root = tmp_path_factory.mktemp("group_vjp_keyed")
    options = {"targets": ("host",), "cache_dir": cache_dir, "host_profile": "native"}
    home = build_bundle([keyed_add], root / "primal", **options)
    return build_bundle(
        [_reverse(keyed_add)],
        root / "reverse",
        derivative={"keyed_add_vjp": ("keyed_add", home.digest)},
        **options,
    )


def test_the_reverse_of_a_keyed_add_in_a_group_reads_the_adjoint_of_the_cell_the_lane_added_to(
    keyed_bundle,
):
    """Oracle: ``2 x (bar_acc[cell] + bar_total * size)`` with the group size 8, 8, 3 for 19 lanes, by hand."""
    cell = RNG.integers(0, BINS, size=LANES).astype(np.int32)
    x, bar_total, bar_acc = (
        RNG.normal(size=LANES),
        RNG.normal(size=LANES),
        RNG.normal(size=BINS),
    )
    bar_x = np.zeros(LANES)
    _run(
        keyed_bundle,
        "keyed_add_vjp",
        cell=cell,
        x=x,
        bar_acc=bar_acc,
        bar_total=bar_total,
        bar_x=bar_x,
    )
    sizes = np.repeat([8.0, 8.0, 3.0], [8, 8, 3])
    np.testing.assert_allclose(
        bar_x, 2.0 * x * (bar_acc[cell] + bar_total * sizes), **EXACT
    )


@pytest.mark.parametrize(
    ("primal", "reason"),
    [
        (stored_twice, "stored in more than one place"),
        (read_at_a_runtime_index, "not the lane's rank plus a constant"),
        (keyed_store_any, "stores by key under conflict='any'"),
        (outside_sink, "one group holding every output"),
    ],
    ids=lambda p: p.name if hasattr(p, "name") else None,
)
def test_a_reverse_derivative_the_group_cannot_hold_is_refused_by_name(primal, reason):
    """Oracle: the refusal text names the form."""
    with pytest.raises(HawkError, match=reason):
        hawk.diff.vjp(primal)


@pytest.fixture(scope="module")
def blocked_bundle(tmp_path_factory, cache_dir):
    """The kernels that read one group reduce in two places, with their reverse kernels, built apart."""
    kernels = [k for p in (group_flags, *SHARES) for k in (p, _reverse(p))]
    return build_bundle(
        kernels,
        tmp_path_factory.mktemp("group_vjp_blocked"),
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )


def test_the_reverse_of_any_and_all_is_the_reverse_of_the_values_they_choose_between(
    blocked_bundle,
):
    """Oracle: the closed form of the chosen branch per group of 8 (group one has a lane above 0.9, group two none)."""
    x = np.concatenate([np.linspace(-0.5, 0.95, 8), np.linspace(-0.5, 0.5, 8)])
    w = RNG.normal(size=16)
    got = _gradients(blocked_bundle, "group_flags", w, x=x)["bar_x"]
    any_hit = _per_group(x, 8, lambda g: np.full(len(g), g.max() > 0.9))
    all_hit = _per_group(x, 8, lambda g: np.full(len(g), g.min() > -0.9))
    np.testing.assert_allclose(
        got,
        w * (np.where(any_hit, 2.0 * x, 3.0) + np.where(all_hit, 1.0, 0.0)),
        **EXACT,
    )


@pytest.mark.parametrize("order", ("tree", "lane"))
def test_the_reverse_of_a_normalisation_by_the_group_sum_is_the_quotient_rule_transposed(
    blocked_bundle, order
):
    """Oracle: ``bar_t = w / T - sum(w t) / T^2`` and ``bar_x = 2 x bar_t`` with ``t = x^2``, ``T`` its group sum."""
    x, w = RNG.normal(size=LANES), RNG.normal(size=LANES)
    got = _gradients(blocked_bundle, f"share_{order}", w, x=x)["bar_x"]
    t = x * x
    total = _per_group(t, 8, lambda g: np.full(len(g), g.sum()))
    cross = _per_group(w * t, 8, lambda g: np.full(len(g), g.sum()))
    np.testing.assert_allclose(got, 2.0 * x * (w / total - cross / total**2), **EXACT)
