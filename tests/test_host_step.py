# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.host_step``: a Python function recorded in a graph beside kernel launches, and its reverse. Each value row names its
oracle: the closed form of the arithmetic, written in NumPy."""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import HawkError, Mutable, Scalar

N = 9
X = 0.5 + 0.3 * np.arange(N)
W = 1.7 + 0.1 * np.arange(N)


@hawk.kernel
def scale(x: Scalar, w: Scalar, y: Mutable[Scalar]):
    y = w * x


@hawk.kernel
def square(y: Scalar, z: Mutable[Scalar]):
    z = y * y


@hawk.host_step
def double(x, y: Mutable):
    np.multiply(x, 2.0, out=y)


@hawk.host_step
def double_returning(x, y: Mutable[Scalar]):
    return 2.0 * x


@hawk.host_step
def triple_square(y, t: Mutable):
    t[...] = 3.0 * y * y


@triple_square.reverse
def triple_square_reverse(value, cotangent, gradient):
    gradient.y += 6.0 * value.y * cotangent.t


def graph_of(*steps, record: bool = False) -> hawk.Graph:
    """A built graph of ``steps``, added by hand."""
    graph = hawk.Graph(record=record)
    for step in steps:
        graph.add(step)
    return graph.build()


# -- running --


def test_a_step_that_writes_its_mutable_argument_in_place_gives_the_closed_form():
    x, y = X.copy(), np.zeros(N)
    graph = graph_of(double(x=x, y=y))
    graph()
    np.testing.assert_array_equal(y, 2.0 * X)
    x += 1.0
    graph()  # a replay reads the arrays as they are then
    np.testing.assert_array_equal(y, 2.0 * (X + 1.0))


def test_a_step_may_return_the_values_of_its_mutable_arguments_in_order():
    x, y = X.copy(), np.zeros(N)
    graph_of(double_returning(x, y))()  # positions bind as keywords do
    np.testing.assert_array_equal(y, 2.0 * X)

    @hawk.host_step
    def split(x, low: Mutable, high: Mutable):
        return x - 1.0, x + 1.0

    low, high = np.zeros(N), np.zeros(N)
    graph_of(split(x, low, high))()
    np.testing.assert_array_equal(low, X - 1.0)
    np.testing.assert_array_equal(high, X + 1.0)


def test_an_argument_with_no_mark_arrives_read_only_and_a_write_to_it_is_refused():
    @hawk.host_step
    def writes_its_input(x, y: Mutable):
        x[0] = -1.0

    x, y = X.copy(), np.zeros(N)
    graph = graph_of(writes_its_input(x=x, y=y))
    with pytest.raises(ValueError, match="read-only"):  # NumPy's own exception, as the step raised it
        graph()
    np.testing.assert_array_equal(x, X)


def test_a_value_error_of_the_step_arrives_as_itself_and_the_graph_stays_usable():
    calls = []

    @hawk.host_step
    def mine(x, y: Mutable):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("mine")
        y[...] = x

    x, y = X.copy(), np.zeros(N)
    graph = graph_of(mine(x=x, y=y))
    with pytest.raises(ValueError, match="^mine$") as raised:
        graph()
    assert type(raised.value) is ValueError
    graph()
    np.testing.assert_array_equal(y, X)


def test_a_step_in_a_builder_sits_next_to_kernels_and_returns_what_it_writes():
    @hawk.Graph.builder
    def build(x, w, y, z, q) -> hawk.Graph:
        g = hawk.Graph()
        scale(x, w, y=y)
        assert triple_square(y, z) is z
        square(z, z=q)
        return g

    x, w, y, z, q = X.copy(), W.copy(), np.zeros(N), np.zeros(N), np.zeros(N)
    build(x, w, y, z, q)()
    np.testing.assert_allclose(q, (3.0 * (W * X) ** 2) ** 2, rtol=1e-12)
    assert "triple_square" in build(x, w, y, z, q).node_labels()


def test_a_builder_step_with_several_mutable_arguments_returns_them_as_a_tuple():
    @hawk.host_step
    def split(x, low: Mutable, high: Mutable):
        return x - 1.0, x + 1.0

    @hawk.Graph.builder
    def build(x, low, high) -> hawk.Graph:
        g = hawk.Graph()
        assert split(x, low, high) == (low, high)
        return g

    build(X.copy(), np.zeros(N), np.zeros(N))


# -- reverse --


def chain(x, w, y, z, q, step) -> hawk.Graph:
    """kernel -> ``step`` -> kernel, recorded for reverse."""
    graph = hawk.Graph(record=True)
    plan = hawk.deploy(scale, reverse=True)
    graph.add(plan.bind(x=x, w=w, y=y))
    graph.add(step(y=y, t=z))
    graph.add(hawk.deploy(square, reverse=True).bind(y=z, z=q))
    return graph.build()


def test_the_gradient_through_kernel_step_kernel_matches_the_closed_form():
    x, w, y, z, q = X.copy(), W.copy(), np.zeros(N), np.zeros(N), np.zeros(N)
    graph = chain(x, w, y, z, q, triple_square)
    graph()
    np.testing.assert_allclose(q, 9.0 * (W * X) ** 4, rtol=1e-12)
    gradients = hawk.reverse(graph.record, {"square.z": np.ones(N)})
    np.testing.assert_allclose(gradients["x"], 36.0 * (W * X) ** 3 * W, rtol=1e-12)
    np.testing.assert_allclose(gradients["w"], 36.0 * (W * X) ** 3 * X, rtol=1e-12)


def test_a_reverse_sees_the_values_the_cotangents_and_names_what_it_may_not_touch():
    seen = {}

    @hawk.host_step
    def probe(a, b: Mutable):
        b[...] = 5.0 * a

    @probe.reverse
    def probe_reverse(value, cotangent, gradient):
        seen["value"] = value.a.copy()
        seen["cotangent"] = cotangent.b.copy()
        seen["read_only"] = not cotangent.b.flags.writeable
        gradient.a += 5.0 * cotangent.b
        with pytest.raises(HawkError, match="cotangent.a does not exist"):
            cotangent.a
        with pytest.raises(HawkError, match="write into the array"):
            gradient.a = np.zeros(N)

    a, b = X.copy(), np.zeros(N)
    graph = graph_of(probe(a=a, b=b), record=True)
    graph()
    seed = np.linspace(1.0, 2.0, N)
    gradients = hawk.reverse(graph.record, {"b": seed})
    np.testing.assert_array_equal(seen["value"], X)
    np.testing.assert_array_equal(seen["cotangent"], seed)
    assert seen["read_only"]
    np.testing.assert_allclose(gradients["a"], 5.0 * seed, rtol=1e-12)


def test_a_gradient_through_a_step_with_no_reverse_is_refused_by_name():
    @hawk.host_step
    def plain(y, t: Mutable):
        t[...] = 3.0 * y * y

    x, w, y, z, q = X.copy(), W.copy(), np.zeros(N), np.zeros(N), np.zeros(N)
    graph = chain(x, w, y, z, q, plain)
    graph()  # a replay needs no reverse
    with pytest.raises(HawkError, match="plain"):
        hawk.reverse(graph.record, {"square.z": np.ones(N)})


# -- failures --


def test_an_exception_of_the_step_arrives_unchanged_and_the_graph_stays_usable():
    class Boom(Exception):
        pass

    calls = []

    @hawk.host_step
    def fails(x, y: Mutable):
        calls.append(1)
        if len(calls) == 1:
            raise Boom("the step failed: 42")
        y[...] = x

    x, y = X.copy(), np.zeros(N)
    graph = graph_of(fails(x=x, y=y))
    with pytest.raises(Boom, match="the step failed: 42"):
        graph()
    graph()
    np.testing.assert_array_equal(y, X)


@pytest.mark.parametrize(
    "returned, match",
    [
        (lambda x: (x, x), "returned 2 values for the Mutable arguments \\['y'\\]"),
        (lambda x: x[:3], "value returned for 'y' has shape \\(3,\\)"),
        (lambda x: x.astype(np.float32), "dtype float32"),
    ],
)
def test_a_returned_value_of_the_wrong_count_shape_or_dtype_is_refused_naming_step_and_argument(
    returned, match
):
    @hawk.host_step
    def wrong(x, y: Mutable):
        return returned(x)

    graph = graph_of(wrong(x=X.copy(), y=np.zeros(N)))
    with pytest.raises(HawkError, match=f"host_step wrong: .*{match}"):
        graph()


def test_a_value_returned_by_a_step_with_no_mutable_argument_is_refused():
    @hawk.host_step
    def reads_only(x):
        return x

    with pytest.raises(
        HawkError,
        match="host_step reads_only: it returned a value but has no Mutable argument",
    ):
        graph_of(reads_only(x=X.copy()))()


def test_binding_refuses_a_wrong_missing_or_repeated_argument_and_a_non_array_by_name():
    x, y = X.copy(), np.zeros(N)
    with pytest.raises(
        HawkError,
        match=r"host_step double: 'z' is not an argument; they are \['x', 'y'\]",
    ):
        double(x=x, z=y)
    with pytest.raises(
        HawkError, match=r"host_step double: give every argument; missing \['y'\]"
    ):
        double(x=x)
    with pytest.raises(
        HawkError, match="host_step double: argument 'x' is given twice"
    ):
        double(x, x=x, y=y)
    with pytest.raises(HawkError, match="host_step double: takes 2 arguments"):
        double(x, y, y)
    with pytest.raises(
        HawkError, match="host_step double: argument 'x' is a float, not an array"
    ):
        double(x=1.0, y=y)


def test_a_step_refuses_what_it_cannot_bind_by_name_and_a_second_reverse():
    with pytest.raises(
        HawkError, match="host_step varying: argument 'rest' cannot be bound by name"
    ):

        @hawk.host_step
        def varying(x, *rest):
            pass

    with pytest.raises(
        HawkError, match="host_step triple_square: it has a reverse already"
    ):
        triple_square.reverse(triple_square_reverse)


def test_a_step_is_named_by_its_function_and_a_label_is_refused():
    graph = hawk.Graph()
    with pytest.raises(HawkError, match="a host step is named by its function"):
        graph.add(double(x=X.copy(), y=np.zeros(N)), name="again")


def test_two_calls_of_one_step_in_a_graph_both_run():
    x, y, z = X.copy(), np.zeros(N), np.zeros(N)
    graph_of(double(x=x, y=y), double(x=y, y=z))()
    np.testing.assert_array_equal(z, 4.0 * X)


def test_rebinding_a_builder_parameter_a_step_holds_is_refused_by_name():
    @hawk.Graph.builder
    def build(x, y) -> hawk.Graph:
        g = hawk.Graph()
        double(x, y)
        return g

    graph = build(X.copy(), np.zeros(N))
    with pytest.raises(
        HawkError,
        match="Graph.bind: 'x' is an argument of a host step, recorded at build",
    ):
        graph.bind(x=X.copy())
