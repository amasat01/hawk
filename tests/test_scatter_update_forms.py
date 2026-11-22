# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The by-key update forms beyond the folds: a store under ``conflict="any"``, the exchanges and the returned old value,
and the derivatives each of them refuses by name, in both directions."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import _launch

import hawk
from hawk import HawkError, Index, Int32, Mutable, Scalar, kernel
from hawk.artifact import build_bundle, emit_source
from hawk.diff import jvp, vjp

N = 3000
CELLS = 4


@kernel
def racing_store(key: Index, x: Scalar, y: Mutable[Scalar]):
    hawk.scatter(y, conflict="any")[key] = x


@kernel
def minimum(key: Index, x: Scalar, y: Mutable[Scalar]):
    hawk.scatter(y)[key].min(x)


@kernel
def maximum(key: Index, x: Scalar, y: Mutable[Scalar]):
    hawk.scatter(y)[key].max(x)


@kernel
def conjunction(key: Index, b: Int32, y: Mutable[Int32]):
    hawk.scatter(y)[key].and_(b)


@kernel
def parity(key: Index, b: Int32, y: Mutable[Int32]):
    hawk.scatter(y)[key].xor(b)


@kernel
def swap(key: Index, x: Scalar, y: Mutable[Scalar], seen: Mutable[Scalar]):
    old = hawk.scatter(y)[key].exchange(x)
    seen = old


@kernel
def claim(key: Index, x: Scalar, y: Mutable[Scalar], seen: Mutable[Scalar]):
    old = hawk.scatter(y)[key].compare_exchange(0.0, x)
    seen = old


@kernel
def ticket(key: Index, x: Scalar, y: Mutable[Scalar], seen: Mutable[Scalar]):
    old = hawk.scatter(y)[key].add(x, old=True)
    seen = old


@pytest.mark.parametrize("derive", [vjp, jvp], ids=["vjp", "jvp"])
@pytest.mark.parametrize(
    "primal, reason",
    [
        (racing_store, r"stores by key under conflict='any': the lane that wins a shared key differs between runs"),
        (minimum, r"is a keyed minimum.*written under the ordered contract, which is not written yet"),
        (maximum, r"is a keyed maximum.*written under the ordered contract, which is not written yet"),
        (conjunction, r"is a bitwise and by key.*an integer mask has no derivative"),
        (parity, r"is a bitwise xor by key.*an integer mask has no derivative"),
        (swap, r"exchanges by key.*the old value it returns depends on the order lanes arrive in"),
        (claim, r"compare-exchanges by key.*the old value it returns depends on the order lanes arrive in"),
        (ticket, r"adds by key and returns the old value.*depends on the order lanes arrive in"),
    ],
    ids=["store_any", "min", "max", "and", "xor", "exchange", "compare_exchange", "add_old"],
)
def test_a_derivative_of_an_update_with_no_derivative_is_refused_by_name(derive, primal, reason):
    with pytest.raises(HawkError, match=reason):
        derive(primal)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    return build_bundle([racing_store], tmp_path_factory.mktemp("update_forms"), mode="float64",
                        targets=("cuda", "host"), cache_dir=cache_dir)


def test_a_store_under_any_leaves_one_lanes_whole_value_in_each_cell(bundle):
    # oracle: the set of values the lanes of each cell offered; a lane's value is a distinct float with many mantissa bits,
    # so a torn value would not be in the set
    rng = np.random.default_rng(5)
    key = rng.integers(0, CELLS - 1, N).astype(np.int32)        # the last cell is never hit
    x = rng.normal(size=N) * np.pi
    y = np.full(CELLS, -7.0)
    _launch(bundle, "racing_store", "host", {"key": key, "x": x, "y": y}).launch()
    for cell in range(CELLS - 1):
        assert y[cell] in x[key == cell]
    assert y[CELLS - 1] == -7.0


@pytest.mark.parametrize("primal, helper", [(racing_store, "exchangeAt")], ids=["store_any"])
def test_the_device_text_stores_under_any_through_one_exchange(primal, helper):
    text = emit_source(primal, targets=("cuda",)).sources["cuda"].text
    assert f"aether::accum::atomic::{helper}(wout_y(" in text


@pytest.mark.parametrize("primal", [swap, claim, ticket], ids=["exchange", "compare_exchange", "add_old"])
def test_an_update_that_returns_the_old_value_is_refused_by_name_until_it_is_written(primal, tmp_path, cache_dir):
    with pytest.raises(HawkError, match=r"returns the old value"):
        build_bundle([primal], tmp_path, mode="float64", targets=("host",), cache_dir=cache_dir)


def test_the_forms_of_the_methods_are_listed_in_a_misuse():
    with pytest.raises(HawkError, match=r"returns nothing: the methods that return the old value are"):
        @kernel
        def k(key: Index, x: Scalar, y: Mutable[Scalar]):
            old = hawk.scatter(y)[key].min(x)
            y = old
    with pytest.raises(HawkError, match=r"conflict= is for a store"):
        @kernel
        def k2(key: Index, x: Scalar, y: Mutable[Scalar]):
            hawk.scatter(y, conflict="any")[key] += x
    with pytest.raises(HawkError, match=r"\.compare_exchange\(\) takes 2 operands"):
        @kernel
        def k3(key: Index, x: Scalar, y: Mutable[Scalar]):
            hawk.scatter(y)[key].compare_exchange(x)
