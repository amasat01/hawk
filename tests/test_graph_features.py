# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Graph features of the builder: a ``match`` on a cell with a default arm and with labels that are not 0, 1, 2, ..., and a
counted ``for`` over a Python number, which is recorded once per count. Each value row is held against a NumPy hand loop."""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

import hawk
from hawk import HawkError, Mutable, Param, Scalar

N = 11
X = np.linspace(0.5, 2.5, N)

SOURCE = """
import hawk
from hawk import Mutable, Param, Scalar


@hawk.kernel
def axpb(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


@hawk.Graph.builder
def with_default(x, c, y) -> hawk.Graph:
    g = hawk.Graph()
    match c:
        case 0:
            axpb(x, a=1.0, b=0.0, y=y)
        case 1:
            axpb(x, a=2.0, b=0.0, y=y)
        case _:
            axpb(x, a=5.0, b=0.0, y=y)
    return g


@hawk.Graph.builder
def sparse_labels(x, c, y) -> hawk.Graph:
    g = hawk.Graph()
    match c:
        case 7:
            axpb(x, a=3.0, b=0.0, y=y)
        case 2:
            axpb(x, a=4.0, b=0.0, y=y)
    return g


@hawk.Graph.builder
def counted(x, y) -> hawk.Graph:
    g = hawk.Graph()
    for k in range(4):
        axpb(y if k else x, a=2.0, b=float(k), y=y)
    return g


@hawk.Graph.builder
def too_wide(x, c, y) -> hawk.Graph:
    g = hawk.Graph()
    match c:
        case 5000000000:
            axpb(x, a=1.0, b=0.0, y=y)
    return g
"""


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("features") / "feature_builders.py"
    path.write_text(SOURCE)
    spec = importlib.util.spec_from_file_location("feature_builders", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["feature_builders"] = module
    spec.loader.exec_module(module)
    return module


def cell(value: int, dtype=np.uint32) -> np.ndarray:
    return np.array([value], dtype=dtype)


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the match statement is Python 3.10"
)
def test_a_match_on_a_cell_with_a_default_arm_takes_the_arm_of_the_value_and_the_default_otherwise(
    built,
):
    c, y = cell(0), np.zeros(N)
    graph = built.with_default(X, c, y)
    for value, factor in ((0, 1.0), (1, 2.0), (2, 5.0), (40, 5.0)):
        c[0] = value
        graph()
        np.testing.assert_allclose(y, factor * X, rtol=1e-14)


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the match statement is Python 3.10"
)
def test_a_match_on_a_cell_with_labels_out_of_order_runs_no_arm_for_an_unlisted_value(
    built,
):
    c, y = cell(7), np.zeros(N)
    graph = built.sparse_labels(X, c, y)
    for value, want in ((7, 3.0 * X), (2, 4.0 * X), (5, np.zeros(N))):
        c[0] = value
        y[:] = 0.0
        graph()
        np.testing.assert_allclose(y, want, rtol=1e-14)
    signed = built.sparse_labels(X, cell(2, np.int32), y)
    signed()
    np.testing.assert_allclose(y, 4.0 * X, rtol=1e-14)


@pytest.mark.skipif(
    sys.version_info < (3, 10), reason="the match statement is Python 3.10"
)
def test_a_case_that_does_not_fit_the_cell_dtype_is_refused_by_name(built):
    with pytest.raises(
        HawkError, match="the case 5000000000 does not fit the uint32 cell"
    ):
        built.too_wide(X, cell(0), np.zeros(N))


def test_a_for_over_a_python_count_is_recorded_once_per_count_and_equals_the_hand_loop(
    built,
):
    y = np.zeros(N)
    graph = built.counted(X, y)
    graph()
    want = X
    for k in range(4):
        want = 2.0 * want + float(k)
    np.testing.assert_allclose(y, want, rtol=1e-14)
    assert graph.num_nodes() == 4


@hawk.kernel
def axpb(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


def two_launches(record: bool = False):
    """``y = a x + b`` then ``z = a y + b``, added by hand, built."""
    x, y, z = X.copy(), np.zeros(N), np.zeros(N)
    plan = hawk.deploy(axpb, reverse=record)
    graph = hawk.Graph(record=record)
    graph.add(plan.bind(x=x, a=2.0, b=1.0, y=y)).add(plan.bind(x=y, a=2.0, b=1.0, y=z))
    return graph.build(), z


def test_update_changes_a_value_in_every_launch_or_in_the_launch_it_names_and_the_replay_follows():
    graph, z = two_launches()
    graph.launch()
    np.testing.assert_allclose(z, 2.0 * (2.0 * X + 1.0) + 1.0, rtol=1e-14)
    graph.update(a=3.0).launch()
    np.testing.assert_allclose(z, 3.0 * (3.0 * X + 1.0) + 1.0, rtol=1e-14)
    graph.update({"axpb#2.a": 5.0}).launch()  # only the second launch
    np.testing.assert_allclose(z, 5.0 * (3.0 * X + 1.0) + 1.0, rtol=1e-14)


def test_update_refuses_what_it_cannot_change_by_name():
    graph, _ = two_launches()
    with pytest.raises(
        HawkError,
        match=r"Graph.update: 'nope' is not a value any launch binds by value; they are \['a', 'b'\]",
    ):
        graph.update(nope=1.0)
    with pytest.raises(HawkError, match="'x' is not a value any launch binds by value"):
        graph.update(x=1.0)
    with pytest.raises(HawkError, match="the graph is not built"):
        hawk.Graph().update(a=1.0)


def test_update_of_a_graph_kept_for_reverse_is_refused_with_the_librarys_reason():
    graph, _ = two_launches(record=True)
    with pytest.raises(
        HawkError,
        match="a graph kept for reverse holds the values it was recorded with",
    ):
        graph.update(a=3.0)
