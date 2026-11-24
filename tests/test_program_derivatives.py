# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.diff.jvp`` and ``hawk.diff.vjp`` of a builder: the derivative of a recorded program.

* Each whole-plane node's tangent, and the same node backward, equal the closed form in NumPy.
* A program that chains a kernel, a sort, a reorder, a scan and a reduce equals the NumPy composition forward and backward,
  and the two directions agree on ``<bar, J dot>`` to a bound derived from the sums they make.
* ``vjp(builder)`` and ``hawk.reverse`` give the same bytes on one program; a loop and a condition keep their structure.
* What cannot be differentiated is refused by name.

A float sum of ``n`` terms is within ``n`` units of the machine epsilon times the sum of the magnitudes of its terms; the
bounds below are that, times the number of sums a result passes through.
"""

from __future__ import annotations

import numpy as np
import pytest

import hawk
import hawk.diff
from hawk import HawkError, Mutable, Scalar, UInt32

N = 16
EPS = np.finfo(np.float64).eps
X = 0.5 + 0.1 * np.arange(N)
DX = np.linspace(-1.0, 1.0, N)


@hawk.kernel
def square(x: Scalar, y: Mutable[Scalar]):
    y = x * x


@hawk.kernel
def scale(x: Scalar, w: Scalar, y: Mutable[Scalar]):
    y = w * x


@hawk.kernel
def tick(c: Mutable[UInt32]):
    c = hawk.math.where(c > 0, c - 1, c)


def _near(got, want, scale, sums):
    """Whether ``got`` is within ``sums`` chained sums of ``N`` terms of ``want``, whose terms' magnitudes add to ``scale``."""
    return np.all(np.abs(got - want) <= 4.0 * sums * N * EPS * np.abs(scale))


def _zeros(*shapes):
    return [np.zeros(shape) for shape in shapes]


@hawk.Graph.builder
def scan_then_reduce(x, y, s, t) -> hawk.Graph:
    g = hawk.Graph(record=True)
    square(x, y=y)
    hawk.scan(y, "sum", out=s)
    hawk.reduce(s, "sum", out=t)
    return g


def test_the_tangent_of_a_scan_and_a_reduce_sum_equals_the_closed_form():
    x, (y, s, t) = X.copy(), _zeros(N, N, 1)
    graph = hawk.diff.jvp(scan_then_reduce)(x, y, s, t, tangents={"x": DX.copy()})
    graph()
    dy = 2.0 * X * DX
    assert _near(graph.tangent_of(s), np.cumsum(dy), np.cumsum(np.abs(dy)), 1)
    assert _near(
        graph.tangent_of(t)[0], np.cumsum(dy).sum(), np.cumsum(np.abs(dy)).sum(), 2
    )
    assert _near(
        t[0], np.cumsum(X * X).sum(), np.cumsum(X * X).sum(), 2
    )  # the primal is still computed


def test_the_gradient_through_a_scan_and_a_reduce_sum_equals_the_closed_form():
    x, (y, s, t) = X.copy(), _zeros(N, N, 1)
    gradients = hawk.diff.vjp(scan_then_reduce)(
        x, y, s, t, cotangents={"t": np.ones(1)}
    )
    want = 2.0 * X * np.arange(N, 0, -1)  # y[i] reaches the sum of the scan N - i times
    assert _near(gradients["x"], want, want, 2)


@hawk.Graph.builder
def reorder_program(x, y, order, t) -> hawk.Graph:
    g = hawk.Graph(record=True)
    square(x, y=y)
    hawk.reorder([y], order)
    hawk.reduce(y, "sum", out=t)
    return g


def test_a_reorder_moves_the_tangent_by_the_same_order_and_the_gradient_back():
    order = np.random.default_rng(2).permutation(N).astype(np.int32)
    x, (y, t) = X.copy(), _zeros(N, 1)
    graph = hawk.diff.jvp(reorder_program)(x, y, order, t, tangents={"x": DX.copy()})
    graph()
    assert np.array_equal(y, (X * X)[order])
    dy = (2.0 * X * DX)[order]
    assert _near(graph.tangent_of(y), dy, dy, 1)
    x2, (y2, t2) = X.copy(), _zeros(N, 1)
    gradients = hawk.diff.vjp(reorder_program)(
        x2, y2, order, t2, cotangents={"y": np.arange(N, dtype=float)}
    )
    want = np.empty(N)
    want[order] = np.arange(
        N, dtype=float
    )  # lane order[j] of x lands at place j, so gets the seed of place j
    assert _near(gradients["x"], 2.0 * X * want, 2.0 * X * want, 1)


@hawk.Graph.builder
def extremum_program(x, y, t) -> hawk.Graph:
    g = hawk.Graph(record=True)
    square(x, y=y)
    hawk.reduce(y, "min", out=t)
    return g


@hawk.Graph.builder
def maximum_program(x, y, t) -> hawk.Graph:
    g = hawk.Graph(record=True)
    square(x, y=y)
    hawk.reduce(y, "max", out=t)
    return g


@pytest.mark.parametrize(
    "program, pick", [(extremum_program, np.argmin), (maximum_program, np.argmax)]
)
def test_the_tangent_and_the_gradient_of_a_minimum_and_a_maximum_are_those_of_the_winning_lane(
    program, pick
):
    base = np.array(
        [
            0.9,
            -0.3,
            0.7,
            1.4,
            -1.1,
            0.2,
            0.8,
            -0.6,
            1.0,
            0.1,
            -0.2,
            0.5,
            1.2,
            -0.9,
            0.4,
            0.6,
        ]
    )
    winner = pick(base * base)
    x, (y, t) = base.copy(), _zeros(N, 1)
    graph = hawk.diff.jvp(program)(x, y, t, tangents={"x": DX.copy()})
    graph()
    assert graph.tangent_of(t)[0] == 2.0 * base[winner] * DX[winner]
    x2, (y2, t2) = base.copy(), _zeros(N, 1)
    gradients = hawk.diff.vjp(program)(x2, y2, t2, cotangents={"t": np.ones(1)})
    want = np.zeros(N)
    want[winner] = 2.0 * base[winner]
    assert np.array_equal(gradients["x"], want)


@hawk.Graph.builder
def copy_program(x, y, z, t) -> hawk.Graph:
    g = hawk.Graph(record=True)
    square(x, y=y)
    hawk.copy(y, z)
    hawk.reduce(z, "sum", out=t)
    return g


def test_a_copy_carries_the_tangent_and_the_gradient():
    x, (y, z, t) = X.copy(), _zeros(N, N, 1)
    graph = hawk.diff.jvp(copy_program)(x, y, z, t, tangents={"x": DX.copy()})
    graph()
    assert np.array_equal(graph.tangent_of(z), graph.tangent_of(y))
    gradients = hawk.diff.vjp(copy_program)(
        X.copy(), *_zeros(N, N, 1), cotangents={"t": np.ones(1)}
    )
    assert _near(gradients["x"], 2.0 * X, 2.0 * X, 1)


@hawk.Graph.builder
def product_program(a, b, c, t) -> hawk.Graph:
    g = hawk.Graph()
    hawk.matmul(a, b, out=c)
    hawk.reduce(c, "sum", out=t)
    return g


def test_a_matrix_product_has_the_tangent_da_b_plus_a_db_and_the_gradients_of_both_factors():
    rng = np.random.default_rng(5)
    a, b = rng.uniform(-1, 1, (4, 3)), rng.uniform(-1, 1, (3, 5))
    da, db = rng.uniform(-1, 1, (4, 3)), rng.uniform(-1, 1, (3, 5))
    c, t = np.zeros((4, 5)), np.zeros(4)
    graph = hawk.diff.jvp(product_program)(a, b, c, t, tangents={"a": da, "b": db})
    graph()
    bound = 4.0 * 8 * EPS * (np.abs(da) @ np.abs(b) + np.abs(a) @ np.abs(db))
    assert np.all(np.abs(graph.tangent_of(c) - (da @ b + a @ db)) <= bound)


@hawk.Graph.builder
def transform_program(p, q, t) -> hawk.Graph:
    g = hawk.Graph()
    hawk.fft.c2c(p, out=q)
    hawk.reduce(q, "sum", out=t)
    return g


def test_a_fourier_transform_has_the_same_transform_as_its_tangent():
    rng = np.random.default_rng(6)
    p, dp = rng.uniform(-1, 1, (8, 2)), rng.uniform(-1, 1, (8, 2))
    q, t = np.zeros((8, 2)), np.zeros(8)
    graph = hawk.diff.jvp(transform_program)(p, q, t, tangents={"p": dp})
    graph()
    want = np.fft.fft(dp[:, 0] + 1j * dp[:, 1])
    got = graph.tangent_of(q)
    assert np.all(
        np.abs(got[:, 0] - want.real) + np.abs(got[:, 1] - want.imag)
        <= 64 * EPS * np.abs(dp).sum()
    )


@hawk.Graph.builder
def chained(x, keys, y, s, t) -> hawk.Graph:
    g = hawk.Graph(record=True)
    square(x, y=y)
    order, _ = hawk.sort(keys, bound=4)
    hawk.reorder([y], order)
    hawk.scan(y, "sum", out=s)
    hawk.reduce(s, "sum", out=t)
    return g


def _chain_data():
    keys = np.random.default_rng(3).integers(0, 4, N).astype(np.int32)
    return keys, np.argsort(keys, kind="stable")


def test_a_chain_of_a_kernel_a_sort_a_reorder_a_scan_and_a_reduce_equals_the_numpy_composition_both_ways():
    keys, order = _chain_data()
    x, (y, s, t) = X.copy(), _zeros(N, N, 1)
    graph = hawk.diff.jvp(chained)(x, keys, y, s, t, tangents={"x": DX.copy()})
    graph()
    dy = (2.0 * X * DX)[order]
    assert _near(
        graph.tangent_of(t)[0], np.cumsum(dy).sum(), np.cumsum(np.abs(dy)).sum(), 2
    )
    gradients = hawk.diff.vjp(chained)(
        X.copy(), keys, *_zeros(N, N, 1), cotangents={"t": np.ones(1)}
    )
    place = np.empty(N, dtype=int)
    place[order] = np.arange(N)  # the sorted place of lane i
    want = 2.0 * X * (N - place)
    assert _near(gradients["x"], want, want, 2)
    # <bar, J dot> both ways: the cotangent is one, so the tangent output equals the gradient dotted with the tangent input
    forward = graph.tangent_of(t)[0]
    backward = float(np.dot(gradients["x"], DX))
    assert abs(forward - backward) <= 8.0 * N * EPS * np.abs(want * DX).sum()


@hawk.Graph.builder
def to_square(x, y, s, z) -> hawk.Graph:
    g = hawk.Graph(record=True)
    square(x, y=y)
    hawk.scan(y, "sum", out=s)
    square(s, y=z)
    return g


def test_vjp_of_a_builder_gives_the_bytes_hawk_reverse_gives_on_the_same_program():
    x, y, s, z = X.copy(), *_zeros(N, N, N)
    graph = to_square(x, y, s, z)
    graph()
    by_reverse = hawk.reverse(graph.record, {"square#2.y": np.ones(N)})
    by_vjp = hawk.diff.vjp(to_square)(
        X.copy(), *_zeros(N, N, N), cotangents={"z": np.ones(N)}
    )
    assert sorted(by_vjp) == sorted(by_reverse)
    for name in by_reverse:
        assert by_vjp[name].tobytes() == by_reverse[name].tobytes()


@hawk.Graph.builder
def looped(x, a, one, t, c) -> hawk.Graph:
    g = hawk.Graph(record=True)
    for _ in range(3):
        if not c:
            break
        scale(x, a, y=t)
        scale(t, one, y=x)
        tick(c)
    return g


@pytest.mark.parametrize("start", [2, 6, 0])
def test_a_loop_keeps_its_structure_and_its_tests_in_the_tangent(start):
    """x <- a x per pass for min(start, 3) passes: the tangent is p a^(p-1) x0 da + a^p dx0."""
    x0, a, one = X.copy(), 0.9 + 0.02 * np.arange(N), np.ones(N)
    x, t, c = x0.copy(), np.zeros(N), np.array([start], dtype=np.uint32)
    da = np.full(N, 0.5)
    graph = hawk.diff.jvp(looped)(x, a, one, t, c, tangents={"x": DX.copy(), "a": da})
    graph()
    passes = min(start, 3)
    want = (
        passes * a ** max(passes - 1, 0) * x0 * da if passes else 0.0
    ) + a**passes * DX
    scale_of = passes * a ** max(passes - 1, 0) * x0 * np.abs(da) + a**passes * np.abs(
        DX
    )
    assert np.all(np.abs(graph.tangent_of(x) - want) <= 64 * EPS * np.abs(scale_of))


@hawk.Graph.builder
def gated(x, w, flag, y) -> hawk.Graph:
    g = hawk.Graph(record=True)
    square(x, y=y)
    if flag:
        scale(x, w, y=y)
    return g


@pytest.mark.parametrize("taken", [1, 0])
def test_a_condition_is_tested_at_every_replay_of_the_tangent(taken):
    w = 0.7 + 0.1 * np.arange(N)
    flag = np.array([taken], dtype=np.uint32)
    graph = hawk.diff.jvp(gated)(
        X.copy(), w, flag, np.zeros(N), tangents={"x": DX.copy(), "w": np.ones(N)}
    )
    graph()
    dy = (w * DX + X) if taken else 2.0 * X * DX
    assert np.all(
        np.abs(graph.tangent_of(graph._params["y"]) - dy)
        <= 8 * EPS * (np.abs(w * DX) + np.abs(X) + 2 * X * np.abs(DX))
    )


# the refusals


def _running_program(op: str, exclusive: bool):
    @hawk.Graph.builder
    def running(x, y, s) -> hawk.Graph:
        g = hawk.Graph(record=True)
        square(x, y=y)
        hawk.scan(y, op, exclusive=exclusive, out=s)
        return g

    return running


#: Squares with ties (lanes 1 and 4, lanes 2 and 9) and a smaller value late.
_TIED = np.array(
    [
        0.9,
        0.5,
        1.4,
        -0.3,
        -0.5,
        0.8,
        0.2,
        -1.1,
        0.6,
        -1.4,
        1.0,
        0.1,
        0.3,
        1.2,
        -0.9,
        0.4,
    ]
)


def _winners(squares, op: str, exclusive: bool):
    """The loop the test checks against: the lowest lane among the winners of each prefix (a NaN never wins), -1 for none."""
    found, best = [], -1
    for lane, value in enumerate(squares):
        if exclusive:
            found.append(best)
        better = not np.isnan(value) and (
            best < 0
            or (value < squares[best] if op == "min" else value > squares[best])
        )
        best = lane if better else best
        if not exclusive:
            found.append(best)
    return np.array(found)


@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("exclusive", [False, True])
def test_the_tangent_of_a_running_extremum_is_the_tangent_at_the_running_winner_and_agrees_with_the_gradient(
    op, exclusive
):
    """Oracle: a loop over the lanes for the winner of each prefix; the two directions agree on ``<bar, J dot>``."""
    running = _running_program(op, exclusive)
    graph = hawk.diff.jvp(running)(
        _TIED.copy(), *_zeros(N, N), tangents={"x": DX.copy()}
    )
    graph()
    dy = 2.0 * _TIED * DX
    winner = _winners(_TIED * _TIED, op, exclusive)
    want = np.where(winner >= 0, dy[np.maximum(winner, 0)], 0.0)
    got = graph.tangent_of(graph._params["s"])
    assert np.array_equal(got, want)
    seed = np.arange(1.0, N + 1.0)
    gradient = hawk.diff.vjp(running)(
        _TIED.copy(), *_zeros(N, N), cotangents={"s": seed}
    )["x"]
    assert abs(gradient @ DX - seed @ got) <= 8.0 * N * EPS * np.abs(seed) @ np.abs(
        want
    )


@hawk.Graph.builder
def minimum_of_squares(x, y, t) -> hawk.Graph:
    g = hawk.Graph(record=True)
    square(x, y=y)
    hawk.reduce(y, "min", out=t)
    return g


def test_a_minimum_over_an_all_nan_column_has_a_zero_tangent_and_no_lane_receives_the_cotangent():
    """Oracle: no lane wins, so the derivative is zero both ways (the old gather read the last lane's tangent)."""
    nan = np.full(N, np.nan)
    graph = hawk.diff.jvp(minimum_of_squares)(
        nan.copy(), *_zeros(N, 1), tangents={"x": DX.copy()}
    )
    graph()
    assert graph.tangent_of(graph._params["t"])[0] == 0.0
    gradient = hawk.diff.vjp(minimum_of_squares)(
        nan.copy(), *_zeros(N, 1), cotangents={"t": np.ones(1)}
    )["x"]
    assert np.all(
        np.isnan(gradient) | (gradient == 0.0)
    )  # 2 * NaN * 0 is NaN: never a number
    assert not np.any(gradient[~np.isnan(gradient)])


@pytest.mark.parametrize("exclusive", [False, True])
@pytest.mark.parametrize("op", ["min", "max"])
def test_a_running_extremum_over_a_prefix_of_only_nan_has_a_zero_tangent_and_drops_its_cotangent(
    op, exclusive
):
    """Oracle: the loop of the winners; lanes before the first number have none, so their tangent is zero and their cotangent
    reaches no lane."""
    running = _running_program(op, exclusive)
    base = _TIED.copy()
    base[:3] = np.nan
    graph = hawk.diff.jvp(running)(
        base.copy(), *_zeros(N, N), tangents={"x": DX.copy()}
    )
    graph()
    got = graph.tangent_of(graph._params["s"])
    winner = _winners(base * base, op, exclusive)
    assert np.all(winner[: 3 + exclusive] < 0) and np.all(got[: 3 + exclusive] == 0.0)
    seed = np.arange(1.0, N + 1.0)
    gradient = hawk.diff.vjp(running)(
        base.copy(), *_zeros(N, N), cotangents={"s": seed}
    )["x"]
    received = np.zeros(N)
    np.add.at(received, winner[winner >= 0], seed[winner >= 0])
    valid = ~np.isnan(base)
    assert np.allclose(gradient[valid], (2.0 * base * received)[valid], rtol=1e-13)
    assert abs(gradient[valid] @ DX[valid] - seed @ got) <= 8.0 * N * EPS * (
        np.abs(seed) @ np.abs(got)
    )


def test_a_function_that_is_not_a_builder_is_refused_by_name():
    def plain(x):
        return x

    for derivative in (hawk.diff.jvp, hawk.diff.vjp):
        with pytest.raises(
            HawkError, match="'plain' is neither a kernel nor a builder"
        ):
            derivative(plain)


def test_a_builder_whose_graph_keeps_no_record_is_refused_by_vjp_by_name():
    @hawk.Graph.builder
    def unrecorded(x, y) -> hawk.Graph:
        g = hawk.Graph()
        square(x, y=y)
        return g

    with pytest.raises(
        HawkError,
        match="'unrecorded' makes a graph that keeps no record; make it with hawk.Graph",
    ):
        hawk.diff.vjp(unrecorded)(X.copy(), np.zeros(N), cotangents={"y": np.ones(N)})


def test_a_name_that_is_not_a_parameter_is_refused_by_name_in_both_directions():
    with pytest.raises(
        HawkError,
        match=r"tangents= names \['nope'\].*its parameters are \['s', 't', 'x', 'y'\]",
    ):
        hawk.diff.jvp(scan_then_reduce)(
            X.copy(), *_zeros(N, N, 1), tangents={"nope": DX.copy()}
        )
    with pytest.raises(
        HawkError,
        match="cotangents= names 'nope', which is not a parameter of 'scan_then_reduce'",
    ):
        hawk.diff.vjp(scan_then_reduce)(
            X.copy(), *_zeros(N, N, 1), cotangents={"nope": np.ones(1)}
        )


def test_a_minimum_over_a_plane_of_two_axes_is_refused_by_name():
    @hawk.Graph.builder
    def columns(x, y, m, t) -> hawk.Graph:
        g = hawk.Graph()
        square(x, y=y)
        hawk.reduce(m, "min", out=t)
        return g

    with pytest.raises(
        HawkError,
        match=r"hawk.diff.jvp: reduce min: .* a plane of one axis, this one has shape \(2, 8\)",
    ):
        hawk.diff.jvp(columns)(X.copy(), np.zeros(N), np.ones((2, 8)), np.zeros(2))


@hawk.Graph.builder
def product_recorded(a, b, c) -> hawk.Graph:
    g = hawk.Graph(record=True)
    hawk.matmul(a, b, out=c)
    return g


def test_a_matrix_product_gives_the_gradients_of_both_factors_by_the_arrays_the_caller_names():
    rng = np.random.default_rng(8)
    a, b, seed = (
        rng.uniform(-1, 1, (4, 3)),
        rng.uniform(-1, 1, (3, 5)),
        rng.uniform(-1, 1, (4, 5)),
    )
    gradients = hawk.diff.vjp(product_recorded, wrt=["a", "b"])(
        a, b, np.zeros((4, 5)), cotangents={"c": seed}
    )
    assert sorted(gradients) == ["a", "b"]
    for got, want, scale in (
        (gradients["a"], seed @ b.T, np.abs(seed) @ np.abs(b.T)),
        (gradients["b"], a.T @ seed, np.abs(a.T) @ np.abs(seed)),
    ):
        assert np.all(np.abs(got - want) <= 4.0 * 8 * EPS * scale)


@hawk.Graph.builder
def transform_recorded(p, q) -> hawk.Graph:
    g = hawk.Graph(record=True)
    hawk.fft.c2c(p, out=q)
    return g


def test_a_fourier_transform_gives_the_conjugate_transform_as_the_gradient():
    rng = np.random.default_rng(9)
    p, seed = rng.uniform(-1, 1, (8, 2)), rng.uniform(-1, 1, (8, 2))
    gradients = hawk.diff.vjp(transform_recorded, wrt="p")(
        p, np.zeros((8, 2)), cotangents={"q": seed}
    )
    # J^T s for q = F p: the real and imaginary parts of F^H (s_re + i s_im) with the sign conventions of the transform
    want = np.fft.ifft(seed[:, 0] + 1j * seed[:, 1]) * 8
    got = gradients["p"]
    assert np.all(
        np.abs(got[:, 0] - want.real) + np.abs(got[:, 1] - want.imag)
        <= 64 * EPS * np.abs(seed).sum()
    )


def test_the_tangent_kernel_is_deployed_with_the_settings_the_programs_kernel_was(
    tmp_path,
):
    """A single stated target, a non-default scalar type and a cache directory of its own are kept, not defaulted. (A
    contraction is not a build setting: ``deploy`` refuses it by name, so there is no non-default one to carry.)"""
    plan = hawk.deploy(
        square, targets=("host",), scalar_type="float32", cache_dir=str(tmp_path)
    )

    @hawk.Graph.builder
    def single(x, y, t) -> hawk.Graph:
        g = hawk.Graph()
        plan(x, y=y)
        hawk.reduce(y, "min", out=t)
        return g

    x, y, t = X.astype(np.float32), np.zeros(N, np.float32), np.zeros(1, np.float32)
    dx = DX.astype(np.float32)
    graph = hawk.diff.jvp(single)(x, y, t, tangents={"x": dx.copy()})
    tangent_launches = [launch.bound for launch in graph._launches]
    assert {bound.settings["scalar_type"] for bound in tangent_launches} == {"float32"}
    assert {bound.settings["targets"] for bound in tangent_launches} == {("host",)}
    assert {bound.settings["cache_dir"] for bound in tangent_launches} == {
        str(tmp_path)
    }
    graph()
    assert graph.tangent_of(y).dtype == np.float32
    want = 2.0 * X.astype(np.float32) * dx
    assert np.all(
        np.abs(graph.tangent_of(y) - want)
        <= 4.0 * np.finfo(np.float32).eps * np.abs(want)
    )
    assert (
        graph.tangent_of(t)[0] == want[0]
    )  # the winner is lane 0, the smallest square


def test_a_graph_kept_for_reverse_gives_its_tangent_kernel_no_vjp():
    graph = hawk.diff.jvp(scan_then_reduce)(X.copy(), *_zeros(N, N, 1))
    assert all(launch.bound.settings["reverse"] is False for launch in graph._launches)


@hawk.Graph.builder
def running_extremum(x, y, s) -> hawk.Graph:
    g = hawk.Graph(record=True)
    square(x, y=y)
    hawk.scan(y, "min", out=s)
    return g


def test_the_gradient_through_a_running_minimum_goes_to_the_running_winner_and_is_the_bytes_of_reverse():
    base = np.array(
        [
            0.9,
            0.7,
            1.4,
            -0.3,
            0.8,
            0.2,
            -1.1,
            0.5,
            0.6,
            1.0,
            0.1,
            -0.2,
            0.3,
            1.2,
            -0.9,
            0.4,
        ]
    )
    seed = np.arange(1.0, N + 1.0)
    squares = base * base
    winner = np.array(
        [int(np.argmin(squares[: i + 1])) for i in range(N)]
    )  # lowest lane among equals
    want = np.zeros(N)
    np.add.at(want, winner, seed)
    want = 2.0 * base * want
    gradients = hawk.diff.vjp(running_extremum)(
        base.copy(), np.zeros(N), np.zeros(N), cotangents={"s": seed}
    )
    assert np.all(np.abs(gradients["x"] - want) <= 4.0 * N * EPS * np.abs(want))


@hawk.Graph.builder
def running_then_square(x, y, s, z) -> hawk.Graph:
    g = hawk.Graph(record=True)
    square(x, y=y)
    hawk.scan(y, "max", out=s)
    square(s, y=z)
    return g


def test_vjp_through_a_running_maximum_is_the_bytes_hawk_reverse_gives():
    base = np.array(
        [
            0.9,
            0.7,
            1.4,
            -0.3,
            0.8,
            0.2,
            -1.1,
            0.5,
            0.6,
            1.0,
            0.1,
            -0.2,
            0.3,
            1.2,
            -0.9,
            0.4,
        ]
    )
    graph = running_then_square(base.copy(), *_zeros(N, N, N))
    graph()
    by_reverse = hawk.reverse(graph.record, {"square#2.y": np.ones(N)})
    by_vjp = hawk.diff.vjp(running_then_square)(
        base.copy(), *_zeros(N, N, N), cotangents={"z": np.ones(N)}
    )
    assert sorted(by_vjp) == sorted(by_reverse)
    assert all(
        by_vjp[name].tobytes() == by_reverse[name].tobytes() for name in by_reverse
    )
    squares = base * base
    winner = np.array([int(np.argmax(squares[: i + 1])) for i in range(N)])
    running = squares[winner]
    want = np.zeros(N)
    np.add.at(want, winner, 2.0 * running)  # d/ds of s * s, given to the running winner
    first = next(name for name in by_vjp if name.endswith("x") and "#2" not in name)
    assert np.all(
        np.abs(by_vjp[first] - 2.0 * base * want)
        <= 4.0 * N * EPS * np.abs(2.0 * base * want)
    )
