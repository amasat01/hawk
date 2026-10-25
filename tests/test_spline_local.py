# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The LOCAL de Boor spline against the DENSE indicator form, on the host twin
(``tests/_eval.py``): forward value and the derived vjp (``bar_theta``,
``bar_x``), bit for bit where the arithmetic is the same (see
``tests/_spline_forms.py`` for the two forms)."""

from __future__ import annotations

import _spline_forms as S
import numpy as np
import pytest
from _eval import evaluate
from test_math_parity import _ulps

from hawk.diff import vjp


def _lanes(rng):
    th = rng.normal(0.0, 0.3, size=S.W)
    th[0], th[1], th[S.W - 2], th[S.W - 1] = rng.normal(), rng.normal(), 0.0, 1.0
    return th


def _points(rng):
    """Random inside the grid, every knot of it exactly and one ulp either side,
    and the two domain ends."""
    inside = rng.uniform(-0.999, 0.999, size=200)
    knots = np.array([S.knot(q) for q in range(S.K, S.G + S.K + 1)])
    near = np.concatenate([knots, np.nextafter(knots, -9), np.nextafter(knots, 9)])
    return np.concatenate([inside, near, [-1.0, 1.0]])


def _run(cell, x, th, seed=1.0):
    env = {"x": x, "theta": th}
    fwd = evaluate(cell.sinks, env)["y"]
    got = evaluate(vjp(cell, wrt=("x", "theta")), {**env, "bar_y": seed})
    return float(fwd), float(got["bar_x"]), np.asarray(got["bar_theta"])


@pytest.fixture(scope="module")
def cells():
    return S.edge_cell("dense"), S.edge_cell("local"), S.edge_cell("local", False)


def test_local_equals_dense_bitwise_on_the_grid(cells):
    """Forward value and the cotangent of every coefficient and of ``w_b``,
    ``w_s`` are bit for bit the dense form's at every point inside the grid, on
    each knot and one ulp either side of it. The cotangent of ``u`` (``bar_x``
    and the two domain lanes' ``bar_theta``) is a sum of the SAME terms the dense
    form sums, but the dense graph is a different walk, so the reverse pass adds
    them in another order: equal to a few ulps (at most 4), and at a few points only."""
    dense, local, _ = cells
    rng = np.random.default_rng(20261021)
    th = _lanes(rng)
    points = [x for x in _points(rng) if -1.0 <= x < 1.0 or x == -1.0]
    off_by_order = 0
    for x in points:
        want, got = _run(dense, float(x), th), _run(local, float(x), th)
        assert want[0] == got[0], f"forward at {x!r}"
        assert np.array_equal(want[2][:S.W - 2], got[2][:S.W - 2]), f"bar_theta at {x!r}"
        tail = [(want[1], got[1]), *zip(want[2][S.W - 2:], got[2][S.W - 2:])]
        assert _ulps(*map(np.array, zip(*tail))).max() <= 4, f"bar_u at {x!r}"
        off_by_order += any(a != b for a, b in tail)
    assert off_by_order <= len(points) // 20, off_by_order


def test_at_the_domain_end_the_clamped_index_differs_only_by_continuity(cells):
    """At ``u = 1`` the indicator opens the interval past the grid (three live
    coefficients), the local index is clamped to the last grid interval (four);
    the two pieces meet at the knot, so the values agree to rounding, not bit for
    bit. Just below ``-1`` the dense basis is partial and the clamped polynomial
    is not: equal only on the grid."""
    dense, local, _ = cells
    th = _lanes(np.random.default_rng(9))
    want, got = _run(dense, 1.0, th), _run(local, 1.0, th)
    assert np.isclose(want[0], got[0], rtol=1e-14, atol=0)
    assert np.allclose(want[2], got[2], rtol=1e-13, atol=0)


def test_a_plain_floor_misplaces_a_value_on_a_knot(cells):
    """The one source of a difference: ``floor((u + 1) / step)`` rounds, the
    indicator compares against the knot itself, so a value ON a knot can fall in
    the interval below. The corrected index (default) agrees with the
    indicator at every knot, the plain floor does not at all of them."""
    dense, _, plain = cells
    th = _lanes(np.random.default_rng(3))
    knots = [S.knot(q) for q in range(S.K, S.G + S.K + 1)]
    off = [x for x in knots if _run(dense, x, th)[0] != _run(plain, x, th)[0]]
    assert off, "the plain floor was expected to misplace at least one knot"


def test_outside_the_grid_the_local_form_extends_the_end_piece(cells):
    """By design: past the domain the dense form's basis fades to zero over the
    ``K`` extension intervals, the local form (index clamped to the grid)
    continues the end polynomial piece. They agree only up to the end."""
    dense, local, _ = cells
    th = _lanes(np.random.default_rng(5))
    assert _run(dense, 1.3, th)[0] != _run(local, 1.3, th)[0]
