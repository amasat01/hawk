# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.min`` / ``hawk.max`` of a kernel body with ``with_index=True``, and the derivative of a minimum or maximum over samples.

Oracles: ``np.min`` / ``np.argmin`` and ``np.max`` / ``np.argmax`` (the first occurrence: the lowest lane) for the value and
the lane; the NaN-ignoring ``np.nanmin`` / ``np.nanargmin`` where some lane is NaN; a central finite difference of the
minimum for the tangent, and the one-hot vector at ``np.argmin`` for the adjoint. Every plane here is float64, the real type
of the default mode, so a lane number is exact.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import _launch, sidecar_of

import hawk
from hawk import HawkError, Kernel, Mutable, Scalar, kernel
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp

PAIRS = {"min": (np.min, np.argmin), "max": (np.max, np.argmax)}
#: One, two, a few, around the 32-sample row of a cell and the leaf of the runtime's fold, and a large one.
SIZES = (1, 2, 5, 31, 32, 33, 4095, 4096, 4097, 20_000)


@kernel
def lowest(x: Scalar, low: Mutable[Scalar], at: Mutable[Scalar]):
    low, at = hawk.min(x, with_index=True)


@kernel
def highest(x: Scalar, high: Mutable[Scalar], at: Mutable[Scalar]):
    high, at = hawk.max(x, with_index=True)


@kernel
def lowest_masked(x: Scalar, keep: Scalar, low: Mutable[Scalar], at: Mutable[Scalar]):
    low, at = hawk.min(x, where=keep > 0.5, with_index=True)


@kernel
def lowest_and_highest(
    x: Scalar,
    low: Mutable[Scalar],
    low_at: Mutable[Scalar],
    high: Mutable[Scalar],
    high_at: Mutable[Scalar],
):
    low, low_at = hawk.min(x, with_index=True)
    high, high_at = hawk.max(x, with_index=True)


@pytest.fixture(scope="module")
def built(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("reduce_index")
    return build_bundle(
        [
            Kernel("lowest", lowest.sinks),
            Kernel("highest", highest.sinks),
            Kernel("lowest_masked", lowest_masked.sinks),
            Kernel("lowest_and_highest", lowest_and_highest.sinks),
        ],
        root,
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )


def run(built, name, x, extra=None):
    """The value and the lane the kernel ``name`` stores for the plane ``x``."""
    planes = {"x": np.asarray(x, dtype=np.float64), "at": np.full(1, -1.0)}
    planes["low" if name != "highest" else "high"] = np.full(1, np.nan)
    planes.update(extra or {})
    bound = _launch(built, name, "host", planes)
    bound.launch()
    bound.wait()
    return planes["low" if name != "highest" else "high"][0], planes["at"][0]


def planes_of_ties(n):
    return {
        "many ties": ((np.arange(n) * 37) % 101 + 10).astype(np.float64),
        "constant": np.full(n, 50.0),
        "ends": np.where(np.isin(np.arange(n), [0, n - 1]), 3.0, 50.0),
        "random": np.random.default_rng(n).normal(size=n),
    }


@pytest.mark.parametrize("op, name", [("min", "lowest"), ("max", "highest")])
@pytest.mark.parametrize("n", SIZES)
def test_the_value_and_lane_equal_numpy_with_the_first_occurrence_winning_a_tie(
    built, op, name, n
):
    value, lane = PAIRS[op]
    for what, x in planes_of_ties(n).items():
        got, at = run(built, name, x)
        assert got == value(x) and at == lane(x), (what, n, got, at)


@pytest.mark.parametrize("op, name", [("min", "lowest"), ("max", "highest")])
def test_a_nan_never_wins_and_a_plane_of_nothing_but_nan_has_the_lane_n(
    built, op, name
):
    value, lane = {"min": (np.nanmin, np.nanargmin), "max": (np.nanmax, np.nanargmax)}[
        op
    ]
    x = ((np.arange(300) * 37) % 101 + 10).astype(np.float64)
    x[::5] = np.nan
    got, at = run(built, name, x)
    assert got == value(x) and at == lane(x)
    got, at = run(built, name, np.full(300, np.nan))
    assert np.isnan(got) and at == 300


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("shape", list(planes_of_ties(2)))
def test_a_minimum_and_a_maximum_with_their_lanes_share_one_kernel(built, n, shape):
    x = planes_of_ties(n)[shape]
    cells = ("low", "low_at", "high", "high_at")
    planes = {"x": x, **{cell: np.full(1, -1.0) for cell in cells}}
    bound = _launch(built, "lowest_and_highest", "host", planes)
    bound.launch()
    bound.wait()
    got = tuple(planes[cell][0] for cell in cells)
    assert got == (x.min(), x.argmin(), x.max(), x.argmax())


def test_a_mask_leaves_the_masked_lanes_out_and_a_mask_of_nothing_has_the_lane_n(built):
    x = np.array([1.0, -4.0, 3.0, -4.0, 2.0])
    keep = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
    got, at = run(built, "lowest_masked", x, {"keep": keep})
    assert (
        got == -4.0 and at == 3
    )  # lane 1 is masked, so the other -4 is the first occurrence
    got, at = run(built, "lowest_masked", x, {"keep": np.zeros(5)})
    assert at == 5


def test_a_kernel_with_a_position_is_one_stage_and_one_text(built):
    """The value and its position are one fold of pairs over one value per lane: no cell of the kernel is read back by another stage."""
    for name in ("lowest", "highest", "lowest_masked"):
        assert "stages" not in sidecar_of(built, name)
        assert sidecar_of(built, name)["cells"][("at")]["op"].startswith("arg")


def test_the_result_is_the_same_with_one_thread_and_with_the_default(built):
    x = np.random.default_rng(7).integers(0, 40, size=20_000).astype(np.float64)
    wanted = hawk.host_threads()
    try:
        hawk.set_host_threads(1)
        one = run(built, "lowest", x)
        hawk.set_host_threads(4)
        four = run(built, "lowest", x)
        hawk.set_host_threads(None)
        many = run(built, "lowest", x)
    finally:
        hawk.set_host_threads(wanted)
    assert one == four == many == (x.min(), x.argmin())


def test_an_over_with_a_position_is_refused_by_name():
    with pytest.raises(HawkError, match="a position is a lane of the whole run"):

        @kernel
        def k(x: Scalar, low: Mutable[Scalar]):
            low, _ = hawk.min(x, over=0, with_index=True)


# ---- derivatives ----


def term_min(op):
    @kernel
    def cell(x: Scalar, w: Scalar, t: Mutable[Scalar]):
        t = hawk.min(x * w) if op == "min" else hawk.max(x * w)

    return cell


@pytest.mark.parametrize("op", ["min", "max"])
def test_the_tangent_is_the_tangent_of_the_winning_lane(built, tmp_path, cache_dir, op):
    """Oracle: ``w * dx`` at ``np.argmin`` / ``np.argmax`` of ``x * w``, the lowest lane at a tie."""
    cell = term_min(op)
    bundle = build_bundle(
        [Kernel("cell", cell.sinks), Kernel("d", jvp(cell, wrt=("x",)))],
        tmp_path,
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )
    x = np.array([3.0, 1.0, 2.0, 1.0, 5.0])
    w = np.array([1.0, 1.0, 2.0, 1.0, 1.0])
    x = x if op == "min" else np.array([1.0, 9.0, 3.0, 9.0, 2.0])
    dx = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    lane = PAIRS[op][1](x * w)
    planes = {"x": x, "w": w, "dot_x": dx, "dot_t": np.full(1, np.nan)}
    bound = _launch(bundle, "d", "host", planes)
    bound.launch()
    bound.wait()
    assert planes["dot_t"][0] == w[lane] * dx[lane]


@pytest.mark.parametrize("op", ["min", "max"])
def test_the_adjoint_is_one_hot_at_the_winning_lane(built, tmp_path, cache_dir, op):
    """Oracle: ``bar_t * w`` at ``np.argmin`` / ``np.argmax`` of ``x * w`` and zero elsewhere."""
    cell = term_min(op)
    bundle = build_bundle(
        [Kernel("cell", cell.sinks), Kernel("d", vjp(cell, wrt=("x",)))],
        tmp_path,
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )
    x = (
        np.array([3.0, 1.0, 2.0, 1.0, 5.0])
        if op == "min"
        else np.array([1.0, 9.0, 3.0, 9.0, 2.0])
    )
    w = np.array([1.0, 1.0, 2.0, 1.0, 1.0])
    planes = {"x": x, "w": w, "bar_t": 7.0, "bar_x": np.zeros(5)}
    bound = _launch(bundle, "d", "host", planes)
    bound.launch()
    bound.wait()
    want = np.zeros(5)
    lane = PAIRS[op][1](x * w)
    want[lane] = 7.0 * w[lane]
    np.testing.assert_array_equal(planes["bar_x"], want)


# ---- the winner is found again from the term: the term must come out the same bits in every stage ----

#: The settings hawk offers that change how a float expression is evaluated: the optimisation level, and the host profiles
#: that allow a fused multiply-add (``native-vector-math``) or forbid it (``native``, ``portable``; ``native-vector-math-nofma`` is
#: built on x86_64 only, so this host cannot run it).
PROFILES = ("native", "portable", "native-vector-math")
LEVELS = ("O0", "O3")


def sensitive_terms():
    """Per term: its kernels (the term of every lane written out, the minimum and maximum with a position, and the cell whose
    derivatives are taken) over a term that a fused multiply-add and a different evaluation order round differently."""

    def make(name, expression):
        @kernel
        def written(a: Scalar, b: Scalar, c: Scalar, tau: Mutable[Scalar]):
            tau = expression(a, b, c)

        @kernel
        def low(
            a: Scalar, b: Scalar, c: Scalar, v: Mutable[Scalar], at: Mutable[Scalar]
        ):
            v, at = hawk.min(expression(a, b, c), with_index=True)

        @kernel
        def high(
            a: Scalar, b: Scalar, c: Scalar, v: Mutable[Scalar], at: Mutable[Scalar]
        ):
            v, at = hawk.max(expression(a, b, c), with_index=True)

        @kernel
        def cell(a: Scalar, b: Scalar, c: Scalar, t: Mutable[Scalar]):
            t = hawk.min(expression(a, b, c))

        return {
            f"{name}_written": Kernel(f"{name}_written", written.sinks),
            f"{name}_low": Kernel(f"{name}_low", low.sinks),
            f"{name}_high": Kernel(f"{name}_high", high.sinks),
            "primal": Kernel("cell", cell.sinks),
            "cell": cell,
        }

    return {
        "fma": make("fma", lambda a, b, c: a * b + c),
        "root": make("root", lambda a, b, c: hawk.math.sqrt(a) / b + c),
    }


def sensitive_inputs(n):
    """``(1 + d) * (1 - d) - 1`` for a few steps ``d``: the exact product is ``1 - d**2``, which fused with the add keeps ``-d**2``
    and unfused rounds to ``1`` first (a result of zero or ``-2**-53``); the same ``d`` repeats so that ties occur."""
    d = 2.0**-27 * (1 + np.arange(n) % 5)
    rng = np.random.default_rng(n)
    return {"a": 1.0 + d, "b": 1.0 - d, "c": np.full(n, -1.0)}, {
        "a": rng.uniform(1.0, 2.0, n),
        "b": rng.uniform(1.0, 2.0, n),
        "c": rng.normal(size=n),
    }


@pytest.mark.parametrize("opt_level", LEVELS)
@pytest.mark.parametrize("profile", PROFILES)
def test_the_position_and_the_derivatives_follow_the_terms_the_kernel_itself_writes(
    tmp_path, cache_dir, profile, opt_level
):
    """Oracles: the per-lane term a plain kernel writes with the same settings; ``np.argmin`` / ``np.argmax`` of it for the lane,
    and its value at that lane for the result. The position must be below ``n`` whenever a lane exists."""
    for key, kernels in sensitive_terms().items():
        names = {k: v for k, v in kernels.items() if k != "cell"}
        derived = {
            f"{key}_jvp": Kernel(f"{key}_jvp", jvp(kernels["cell"], wrt=("a",))),
            f"{key}_vjp": Kernel(f"{key}_vjp", vjp(kernels["cell"], wrt=("a",))),
        }
        bundle = build_bundle(
            [*names.values(), *derived.values()],
            tmp_path / f"{profile}_{opt_level}_{key}",
            targets=("host",),
            cache_dir=cache_dir,
            options={"host.profile": profile},
            opt_level=opt_level,
        )
        for n in (1, 7, 1000):
            for inputs in sensitive_inputs(n):
                launch = lambda name, planes: (
                    _launch(bundle, name, "host", planes).launch(),
                    planes,
                )[1]  # noqa: E731
                tau = np.zeros(n)
                launch(f"{key}_written", {**inputs, "tau": tau})
                for op, pick in (("low", np.argmin), ("high", np.argmax)):
                    planes = launch(
                        f"{key}_{op}",
                        {**inputs, "v": np.full(1, np.nan), "at": np.full(1, -1.0)},
                    )
                    lane = int(planes["at"][0])
                    assert (
                        lane < n and planes["v"][0] == tau[lane] and lane == pick(tau)
                    ), (key, op, n, lane)
                dot_a = np.arange(1.0, n + 1.0)
                jvp_planes = launch(
                    f"{key}_jvp",
                    {**inputs, "dot_a": dot_a, "dot_t": np.full(1, np.nan)},
                )
                vjp_planes = launch(
                    f"{key}_vjp", {**inputs, "bar_t": 3.0, "bar_a": np.zeros(n)}
                )
                lane = int(np.argmin(tau))
                slope = _slope(key, inputs, lane)
                assert jvp_planes["dot_t"][0] == pytest.approx(
                    slope * dot_a[lane], rel=1e-12
                )
                one_hot = np.zeros(n)
                one_hot[lane] = 3.0 * slope
                np.testing.assert_allclose(vjp_planes["bar_a"], one_hot, rtol=1e-12)


def _slope(key, inputs, lane):
    """The derivative of the term with respect to ``a`` at ``lane``: ``b`` for ``a * b + c``, ``1 / (2 sqrt(a) b)`` for the root."""
    a, b = inputs["a"][lane], inputs["b"][lane]
    return b if key == "fma" else 1.0 / (2.0 * np.sqrt(a) * b)
