# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.reverse`` over a recorded graph: gradients through a chain, a fork, an IF and a WHILE. Each row names its oracle: the
closed form of the arithmetic the graph records, and central differences of the forward replay of the same graph."""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import HawkError, Index, Mutable, Scalar, Table, UInt32
from hawk.math import where

N = 9
RNG = np.random.default_rng(7)
X = 0.5 + 0.3 * np.arange(N)
W = 1.7 + 0.1 * np.arange(N)
RTOL = 1e-12


@hawk.kernel
def scale(x: Scalar, w: Scalar, y: Mutable[Scalar]):
    y = w * x


@hawk.kernel
def axpb(x: Scalar, a: hawk.Param, b: hawk.Param, y: Mutable[Scalar]):
    y = a * x + b


@hawk.kernel
def square(y: Scalar, z: Mutable[Scalar]):
    z = y * y


@hawk.kernel
def product(a: Scalar, b: Scalar, z: Mutable[Scalar]):
    z = a * b


@hawk.kernel
def shift(x: Scalar, a: hawk.Param, k: hawk.Int32, y: Mutable[Scalar]):
    y = where(k > 0, a * x, x)


@hawk.kernel
def tick(c: Mutable[UInt32]):
    c = where(c > 0, c - 1, c)


def cell(value: int) -> np.ndarray:
    """A one-element condition cell."""
    return np.array([value], dtype=np.uint32)


def central(forward, array, index, step=1e-6):
    """The central difference of ``forward()`` (a number) with respect to ``array[index]``."""
    held = array[index]
    array[index] = held + step
    high = forward()
    array[index] = held - step
    low = forward()
    array[index] = held
    return (high - low) / (2.0 * step)


# 1. a chain through two launches

@hawk.Graph.builder
def chain(x, w, y, z) -> hawk.Graph:
    g = hawk.Graph(record=True)
    scale(x, w, y=y)
    square(y, z=z)
    return g


def test_a_chain_of_two_launches_gives_the_closed_form_and_agrees_with_central_differences():
    x, w, y, z = X.copy(), W.copy(), np.zeros(N), np.zeros(N)
    graph = chain(x, w, y, z)
    assert isinstance(graph.record, hawk.graph.GraphRecord)
    graph()
    gradients = hawk.reverse(graph.record, {"z": np.ones(N)})
    np.testing.assert_allclose(gradients["x"], 2.0 * w * w * x, rtol=RTOL)
    np.testing.assert_allclose(gradients["w"], 2.0 * w * x * x, rtol=RTOL)
    assert sorted(gradients) == ["w", "x"]                       # y is written before it is read: not an input

    def forward():
        graph()
        return float(z.sum())

    for index in (0, 4, N - 1):
        np.testing.assert_allclose(gradients["x"][index], central(forward, x, index), rtol=1e-6)
        np.testing.assert_allclose(gradients["w"][index], central(forward, w, index), rtol=1e-6)


def test_a_cotangent_is_a_seed_per_element_and_a_replay_reads_the_arrays_as_they_are_then():
    x, w, y, z = X.copy(), W.copy(), np.zeros(N), np.zeros(N)
    graph = chain(x, w, y, z)
    cotangent = RNG.standard_normal(N)
    x += 1.0
    graph()
    gradients = hawk.reverse(graph.record, {"z": cotangent})
    np.testing.assert_allclose(gradients["x"], 2.0 * w * w * x * cotangent, rtol=RTOL)


# 2. wrt: every float input by default, a name or a list of names, the refusals by name

def test_wrt_selects_inputs_refuses_the_unknown_and_leaves_an_integer_plane_out():
    x, w, y, z = X.copy(), W.copy(), np.zeros(N), np.zeros(N)
    graph = chain(x, w, y, z)
    graph()
    record = graph.record
    assert sorted(hawk.reverse(record, {"z": np.ones(N)}, wrt="x")) == ["x"]
    assert sorted(hawk.reverse(record, {"z": np.ones(N)}, wrt=["w", "x"])) == ["w", "x"]
    with pytest.raises(HawkError, match=r"wrt 'nope' is not a plane of the graph; its planes are \['w', 'x', 'y', 'z'\]"):
        hawk.reverse(record, {"z": np.ones(N)}, wrt="nope")
    with pytest.raises(HawkError, match=r"wrt 'y' is written by the graph before anything reads it"):
        hawk.reverse(record, {"z": np.ones(N)}, wrt="y")
    with pytest.raises(HawkError, match="cotangent 'nope' is not a plane"):
        hawk.reverse(record, {"nope": np.ones(N)})
    with pytest.raises(HawkError, match="has shape"):
        hawk.reverse(record, {"z": np.ones(N + 1)})
    with pytest.raises(HawkError, match="wrt= selects among a graph's inputs"):
        hawk.reverse(hawk.until_done.__self__ if False else _run_record(), {}, wrt="x")


def _run_record():
    """The record of a recorded run, for the row that a runner's reverse takes no wrt."""
    from hawk.runner import RunRecord

    return RunRecord(None)


def test_a_plane_bound_by_a_number_is_refused_by_name_and_an_integer_plane_is_skipped():
    @hawk.Graph.builder
    def with_number(x, y, c) -> hawk.Graph:
        g = hawk.Graph(record=True)
        axpb(x, a=3.0, b=1.0, y=y)
        tick(c)
        return g

    x, y, c = X.copy(), np.zeros(N), cell(2)
    graph = with_number(x, y, c)
    graph()
    seed = {"y": np.ones(N)}
    np.testing.assert_allclose(hawk.reverse(graph.record, seed)["x"], 3.0, rtol=RTOL)
    assert sorted(hawk.reverse(graph.record, seed)) == ["x"]                    # a number is no input
    assert hawk.reverse(graph.record, seed, wrt="c") == {}                      # an integer plane is left out
    with pytest.raises(HawkError, match="wrt 'a' is bound by value, which has no gradient: pass a one-element array"):
        hawk.reverse(graph.record, seed, wrt="a")


def test_a_kernel_that_reads_an_integer_plane_is_reversed_and_the_integer_plane_has_no_gradient():
    @hawk.Graph.builder
    def with_integer(x, k, y) -> hawk.Graph:
        g = hawk.Graph(record=True)
        shift(x, a=3.0, k=k, y=y)
        return g

    x, k = X.copy(), (np.arange(N) % 2).astype(np.int32)
    graph = with_integer(x, k, np.zeros(N))
    graph()
    gradients = hawk.reverse(graph.record, {"y": np.ones(N)})
    np.testing.assert_allclose(gradients["x"], np.where(k > 0, 3.0, 1.0), rtol=RTOL)   # dy/dx by hand: a where k > 0, else 1
    assert "k" not in gradients


def test_two_launches_that_bind_different_arrays_under_one_name_are_told_apart_by_label():
    @hawk.Graph.builder
    def two_x(x1, x2, w, y, z) -> hawk.Graph:
        g = hawk.Graph(record=True)
        scale(x1, w, y=y)
        scale(x2, w, y=z)
        return g

    x1, x2, w, y, z = X.copy(), X + 1.0, W.copy(), np.zeros(N), np.zeros(N)
    graph = two_x(x1, x2, w, y, z)
    graph()
    with pytest.raises(HawkError, match=r"cotangent 'y' names different arrays.*\['scale\.y', 'scale#2\.y'\]"):
        hawk.reverse(graph.record, {"y": np.ones(N)})
    gradients = hawk.reverse(graph.record, {"scale.y": np.ones(N), "scale#2.y": np.ones(N)})
    np.testing.assert_allclose(gradients["scale.x"], w, rtol=RTOL)
    np.testing.assert_allclose(gradients["scale#2.x"], w, rtol=RTOL)
    np.testing.assert_allclose(gradients["w"], x1 + x2, rtol=RTOL)               # one array under one name: one gradient, summed


# 3. through a fork, an IF and a WHILE

def test_a_gradient_flows_through_a_fork_summing_both_branches():
    @hawk.Graph.builder
    def forked(x, w1, w2, y1, y2, z) -> hawk.Graph:
        g = hawk.Graph(record=True)
        with g.parallel() as p:
            with p.branch():
                scale(x, w1, y=y1)
            with p.branch():
                scale(x, w2, y=y2)
        product(y1, y2, z=z)
        return g

    x, w1, w2 = X.copy(), W.copy(), -0.6 - 0.05 * np.arange(N)
    y1, y2, z = np.zeros(N), np.zeros(N), np.zeros(N)
    graph = forked(x, w1, w2, y1, y2, z)
    graph()
    gradients = hawk.reverse(graph.record, {"z": np.ones(N)})
    np.testing.assert_allclose(gradients["x"], 2.0 * w1 * w2 * x, rtol=RTOL)
    np.testing.assert_allclose(gradients["scale.w"], w2 * x * x, rtol=RTOL)
    np.testing.assert_allclose(gradients["scale#2.w"], w1 * x * x, rtol=RTOL)


@hawk.Graph.builder
def gated(x, x2, w1, w2, flag, y) -> hawk.Graph:
    g = hawk.Graph(record=True)
    scale(x, w1, y=y)
    if flag:
        scale(x2, w2, y=y)
    return g


@pytest.mark.parametrize("taken", [1, 0])
def test_a_gradient_follows_an_if_that_was_taken_and_adds_nothing_for_one_that_was_not(taken):
    x, x2, w1, w2, y = X.copy(), X - 1.5, W.copy(), 0.4 + 0.1 * np.arange(N), np.zeros(N)
    flag = cell(taken)
    graph = gated(x, x2, w1, w2, flag, y)
    graph()
    gradients = hawk.reverse(graph.record, {"y": np.ones(N)})
    zero = np.zeros(N)
    np.testing.assert_allclose(gradients["scale.x"], zero if taken else w1, rtol=RTOL)       # a taken arm overwrites y
    np.testing.assert_allclose(gradients["scale.w"], zero if taken else x, rtol=RTOL)
    np.testing.assert_allclose(gradients["scale#2.x"], w2 if taken else zero, rtol=RTOL)
    np.testing.assert_allclose(gradients["scale#2.w"], x2 if taken else zero, rtol=RTOL)


def test_the_arm_a_replay_took_is_the_one_the_gradient_follows_when_the_cell_changes_between_replays():
    x, x2, w1, w2, y = X.copy(), X - 1.5, W.copy(), 0.4 + 0.1 * np.arange(N), np.zeros(N)
    flag = cell(1)
    graph = gated(x, x2, w1, w2, flag, y)
    for taken in (1, 0, 1):
        flag[0] = taken
        graph()
        gradients = hawk.reverse(graph.record, {"y": np.ones(N)})
        np.testing.assert_allclose(gradients["scale#2.w"], x2 if taken else 0.0, rtol=RTOL)


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
def test_a_gradient_flows_through_a_capped_while_by_the_passes_it_made(start):
    """x <- a x per pass, for min(start, cap) passes: J = sum(x_p) has dJ/dx0 = a^p and dJ/da = p a^(p-1) x0."""
    x0, a, one = X.copy(), 0.9 + 0.02 * np.arange(N), np.ones(N)
    x, t, c = x0.copy(), np.zeros(N), cell(start)
    graph = looped(x, a, one, t, c)
    graph()
    passes = min(start, 3)
    np.testing.assert_allclose(x, a**passes * x0, rtol=RTOL)
    gradients = hawk.reverse(graph.record, {"scale#2.y": np.ones(N)})
    np.testing.assert_allclose(gradients["x"], a**passes, rtol=RTOL)
    np.testing.assert_allclose(gradients["scale.w"], passes * a ** (passes - 1) * x0 if passes else 0.0, rtol=RTOL)

    def forward():
        x[:], c[0] = x0, start
        graph()
        return float(x.sum())

    for index in (0, N - 1):
        np.testing.assert_allclose(gradients["scale.w"][index], central(forward, a, index), rtol=1e-6, atol=1e-9)


# 4. a graph made by hand, and what the library refuses

def test_a_graph_added_by_hand_is_reversed_when_its_plans_carry_the_vjp():
    plan, square_plan = hawk.deploy(scale, reverse=True), hawk.deploy(square, reverse=True)
    x, w, y, z = X.copy(), W.copy(), np.zeros(N), np.zeros(N)
    graph = hawk.Graph(record=True).add(plan.bind(x=x, w=w, y=y), name="first").add(square_plan.bind(y=y, z=z), name="second")
    graph.build().launch()
    gradients = hawk.reverse(graph.record, {"z": np.ones(N)})
    np.testing.assert_allclose(gradients["x"], 2.0 * w * w * x, rtol=RTOL)
    np.testing.assert_allclose(gradients["w"], 2.0 * w * x * x, rtol=RTOL)


def test_a_launch_with_no_vjp_a_plain_graph_a_closed_one_and_a_rebind_are_refused_by_the_library_or_by_name():
    x, w, y, z = X.copy(), W.copy(), np.zeros(N), np.zeros(N)
    no_vjp = hawk.Graph(record=True).add(hawk.deploy(scale).bind(x=x, w=w, y=y)).build()
    no_vjp.launch()
    with pytest.raises(HawkError, match="no vjp"):
        hawk.reverse(no_vjp.record, {"y": np.ones(N)})
    plain = chain(x, w, y, z)
    assert plain.record is not None
    unrecorded = hawk.Graph().add(hawk.deploy(scale, reverse=True).bind(x=x, w=w, y=y)).build()
    assert unrecorded.record is None
    with pytest.raises(HawkError, match="takes the record of a run made with record=True"):
        hawk.reverse(None, {})
    plain.close()
    with pytest.raises(HawkError, match="the graph is closed"):
        hawk.reverse(plain.record, {"z": np.ones(N)})
    graph = chain(x, w, y, z)
    graph.bind(x=X + 1.0)
    with pytest.raises(HawkError, match="rebound"):
        graph()
    with pytest.raises(HawkError, match="not both"):
        hawk.deploy(scale, record=True, reverse=True)


# 7. the report of a reverse and strict

@hawk.kernel
def gather(table: Table[Scalar], where: Index, y: Mutable[Scalar]):
    y = table[where] * 2.0


@hawk.Graph.builder
def gathered(table, where, y) -> hawk.Graph:
    g = hawk.Graph(record=True)
    gather(table, where, y=y)
    return g


TABLE = 1.0 + np.arange(5.0)
WHERE = np.array([0, 1, 1, 2, 2, 2, 3, 4, 4], dtype=np.int32)
#: the closed form of the gradient of ``sum(y)`` by the table: twice the number of lanes that read each cell
GATHER_GRADIENT = 2.0 * np.bincount(WHERE, minlength=5)


def test_a_chain_is_reported_bit_exact_in_one_pass_and_its_result_reads_as_a_mapping():
    x, w, y, z = X.copy(), W.copy(), np.zeros(N), np.zeros(N)
    graph = chain(x, w, y, z)
    graph()
    gradients = hawk.reverse(graph.record, {"z": np.ones(N)})
    assert isinstance(gradients, hawk.runner.Gradients) and len(gradients) == 2
    assert gradients.report == hawk.runner.GradientReport(bitwise={"x": True, "w": True}, chunks=1)
    np.testing.assert_allclose(dict(gradients)["x"], 2.0 * w * w * x, rtol=RTOL)
    assert hawk.reverse(graph.record, {"z": np.ones(N)}, strict=True).report.bitwise == {"x": True, "w": True}


def test_a_gradient_through_a_scatter_add_is_reported_not_bit_exact_and_has_the_closed_form():
    table, where, y = TABLE.copy(), WHERE.copy(), np.zeros(N)
    graph = gathered(table, where, y)
    graph()
    gradients = hawk.reverse(graph.record, {"y": np.ones(N)})
    np.testing.assert_allclose(gradients["table"], GATHER_GRADIENT, rtol=RTOL)
    assert gradients.report.bitwise == {"table": False} and gradients.report.chunks == 1


def test_strict_refuses_a_gradient_through_a_scatter_add_with_the_librarys_text_and_returns_nothing():
    table, where, y = TABLE.copy(), WHERE.copy(), np.zeros(N)
    graph = gathered(table, where, y)
    graph()
    with pytest.raises(HawkError, match="cross-sample scatter-add"):
        hawk.reverse(graph.record, {"y": np.ones(N)}, strict=True)
    np.testing.assert_array_equal(table, TABLE)                        # nothing the caller owns was written
    np.testing.assert_array_equal(where, WHERE)
    np.testing.assert_allclose(y, 2.0 * TABLE[WHERE], rtol=RTOL)
    np.testing.assert_allclose(hawk.reverse(graph.record, {"y": np.ones(N)})["table"], GATHER_GRADIENT, rtol=RTOL)


def test_the_policy_words_are_the_librarys_one_deterministic_order_and_the_others_are_refused_by_name():
    x, w, y, z = X.copy(), W.copy(), np.zeros(N), np.zeros(N)
    graph = chain(x, w, y, z)
    graph()
    for word in ("compensated", "auto"):
        assert hawk.reverse(graph.record, {"z": np.ones(N)}, policy=word).report.bitwise["x"]
    with pytest.raises(HawkError, match="policy='ordered' asks for the sums over samples in plain sample order"):
        hawk.reverse(graph.record, {"z": np.ones(N)}, policy="ordered")
    with pytest.raises(HawkError, match=r"policy='fast' is not one of \['auto', 'compensated', 'ordered'\]"):
        hawk.reverse(graph.record, {"z": np.ones(N)}, policy="fast")
    with pytest.raises(HawkError, match="strict= is True or False"):
        hawk.reverse(graph.record, {"z": np.ones(N)}, strict=1)


# 5. a scan and a reorder inside a WHILE body
#
# Each row differentiates a loop whose body holds the node, against central differences of the forward replay in float64
# (step H, so the rounding error of a difference is about eps * |J| / H, below ``FD_TOL`` for the magnitudes here: |J| below
# 1e4, eps 2.2e-16, H 1e-6) and against a closed form built from the loop's matrix. A scan is piecewise linear and a
# reorder linear, so the difference has no truncation error away from a tie.

H = 1e-6
FD_TOL = 1e-5
PASSES = 3


@hawk.Graph.builder
def scanned(x, t, u, v, one, c, op) -> hawk.Graph:
    g = hawk.Graph(record=True)
    for _ in range(PASSES):
        if not c:
            break
        scale(x, one, y=t)
        hawk.scan(t, "sum", out=u)
        if op == "max":
            hawk.scan(u, "max", out=v)
            scale(v, one, y=x)
        else:
            scale(u, one, y=x)
        tick(c)
    return g


@pytest.mark.parametrize("start", [2, 5, 0])
@pytest.mark.parametrize("op", ["sum", "max"])
def test_a_gradient_flows_through_a_scan_in_a_while_body(op, start):
    """``x <- cumsum(x)`` (or ``maximum.accumulate(cumsum(x))``) per pass, ``min(start, cap)`` passes. Oracles: the loop in
    NumPy for the values; central differences of J = seed . x for the gradient; for the sum, also ``(L^T)^p seed`` with L
    the lower triangle of ones, the Jacobian of one pass."""
    x0, seed = RNG.normal(size=N), RNG.normal(size=N)
    x, t, u, c = x0.copy(), np.zeros(N), np.zeros(N), cell(start)
    graph = scanned(x, t, u, np.zeros(N), np.ones(N), c, op)
    graph()
    passes = min(start, PASSES)
    want = x0.copy()
    for _ in range(passes):
        want = np.cumsum(want)
        want = np.maximum.accumulate(want) if op == "max" else want
    np.testing.assert_allclose(x, want, rtol=1e-12)
    gradients = hawk.reverse(graph.record, {"scale#2.y": seed})
    np.testing.assert_allclose(x, want, rtol=1e-12)  # the reverse leaves the state as the forward left it

    def forward():
        x[:], c[0] = x0, start
        graph()
        return float(seed @ x)

    for index in range(N):
        np.testing.assert_allclose(
            gradients["scale.x"][index], central(forward, x0, index, H), atol=FD_TOL, rtol=FD_TOL
        )
    if op == "sum":
        lower = np.tril(np.ones((N, N)))
        np.testing.assert_allclose(gradients["scale.x"], np.linalg.matrix_power(lower.T, passes) @ seed, rtol=1e-12)


@hawk.Graph.builder
def reordered(x, a, t, one, order, c) -> hawk.Graph:
    g = hawk.Graph(record=True)
    for _ in range(PASSES):
        if not c:
            break
        scale(x, a, y=t)
        hawk.reorder([t], order)
        scale(t, one, y=x)
        tick(c)
    return g


@pytest.mark.parametrize("start", [2, 5, 0])
def test_a_gradient_flows_through_a_reorder_in_a_while_body(start):
    """``x <- (a x)[order]`` per pass. Oracles: ``M = P diag(a)`` with ``P[j, order[j]] = 1``, so ``x_p = M^p x0``,
    ``dJ/dx0 = (M^T)^p seed`` and ``dJ/da`` by central differences of the forward replay."""
    x0, a, seed = X.copy(), W.copy(), RNG.normal(size=N)
    order = RNG.permutation(N).astype(np.int32)
    x, t, c = x0.copy(), np.zeros(N), cell(start)
    graph = reordered(x, a, t, np.ones(N), order, c)
    graph()
    passes = min(start, PASSES)
    matrix = np.eye(N)[order] @ np.diag(a)
    np.testing.assert_allclose(x, np.linalg.matrix_power(matrix, passes) @ x0, rtol=1e-12)
    gradients = hawk.reverse(graph.record, {"scale#2.y": seed})
    np.testing.assert_allclose(gradients["x"], np.linalg.matrix_power(matrix.T, passes) @ seed, rtol=1e-12)

    def forward():
        x[:], c[0] = x0, start
        graph()
        return float(seed @ x)

    for index in (0, N - 1):
        np.testing.assert_allclose(
            gradients["scale.w"][index], central(forward, a, index, H), atol=FD_TOL, rtol=FD_TOL
        )
