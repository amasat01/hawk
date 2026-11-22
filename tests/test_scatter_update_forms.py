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


@kernel
def swap_i(key: Index, x: Int32, y: Mutable[Int32], seen: Mutable[Int32]):
    old = hawk.scatter(y)[key].exchange(x)
    seen = old


@kernel
def ticket_i(key: Index, one: Int32, y: Mutable[Int32], seen: Mutable[Int32]):
    old = hawk.scatter(y)[key].add(one, old=True)
    seen = old


@kernel
def claim_i(key: Index, x: Int32, zero: Int32, y: Mutable[Int32], seen: Mutable[Int32]):
    old = hawk.scatter(y)[key].compare_exchange(zero, x)
    seen = old


@pytest.fixture(scope="module")
def old_bundle(tmp_path_factory, cache_dir):
    return build_bundle([swap, claim, ticket, swap_i, ticket_i, claim_i], tmp_path_factory.mktemp("old_values"),
                        mode="float64", targets=("cuda", "host"), cache_dir=cache_dir)


def _run(bundle, name, **planes):
    _launch(bundle, name, "host", planes).launch()


def test_an_exchange_chain_returns_every_stored_value_once(old_bundle):
    # order-free property: the old values returned plus the final cells are the initial cells plus every stored value
    rng = np.random.default_rng(7)
    key = rng.integers(0, CELLS, N).astype(np.int32)
    x = rng.normal(size=N)
    y, seen = np.full(CELLS, 100.0), np.zeros(N)
    _run(old_bundle, "swap", key=key, x=x, y=y, seen=seen)
    for cell in range(CELLS):
        got = np.sort(np.append(seen[key == cell], y[cell]))
        want = np.sort(np.append(x[key == cell], 100.0))
        assert got.tobytes() == want.tobytes()


def test_an_integer_exchange_chain_returns_every_stored_value_once(old_bundle):
    key = np.zeros(N, np.int32)
    x = np.arange(1, N + 1, dtype=np.int32)
    y, seen = np.zeros(1, np.int32), np.zeros(N, np.int32)
    _run(old_bundle, "swap_i", key=key, x=x, y=y, seen=seen)
    assert np.array_equal(np.sort(np.append(seen, y)), np.arange(0, N + 1))


def test_a_compare_exchange_on_a_fresh_cell_has_one_winner_whose_value_stays(old_bundle):
    x = np.arange(1, N + 1, dtype=np.float64)
    y, seen = np.zeros(1), np.full(N, -1.0)
    _run(old_bundle, "claim", key=np.zeros(N, np.int32), x=x, y=y, seen=seen)
    winners = np.flatnonzero(seen == 0.0)
    assert len(winners) == 1 and y[0] == x[winners[0]]


def test_an_integer_compare_exchange_on_a_fresh_cell_has_one_winner(old_bundle):
    x = np.arange(1, N + 1, dtype=np.int32)
    y, seen = np.zeros(1, np.int32), np.full(N, -1, np.int32)
    _run(old_bundle, "claim_i", key=np.zeros(N, np.int32), x=x, zero=np.zeros(N, np.int32), y=y, seen=seen)
    winners = np.flatnonzero(seen == 0)
    assert len(winners) == 1 and y[0] == x[winners[0]]


def test_a_ticket_add_returns_a_permutation_of_the_counts(old_bundle):
    y, seen = np.zeros(1, np.int32), np.full(N, -1, np.int32)
    _run(old_bundle, "ticket_i", key=np.zeros(N, np.int32), one=np.ones(N, np.int32), y=y, seen=seen)
    assert y[0] == N and np.array_equal(np.sort(seen), np.arange(N))


def test_a_ticket_add_into_several_cells_counts_each_cell_and_equals_add_at(old_bundle):
    # oracle for the final plane: np.add.at; per cell the tickets are a permutation of 0..count-1
    rng = np.random.default_rng(8)
    key = rng.integers(0, CELLS, N).astype(np.int32)
    y, seen = np.zeros(CELLS), np.zeros(N)
    _run(old_bundle, "ticket", key=key, x=np.ones(N), y=y, seen=seen)
    want = np.zeros(CELLS)
    np.add.at(want, key, np.ones(N))
    assert np.array_equal(y, want)
    for cell in range(CELLS):
        assert np.array_equal(np.sort(seen[key == cell]), np.arange(int(want[cell])))


@pytest.mark.gpu
def test_the_device_ticket_add_returns_a_permutation_of_the_counts(old_bundle):
    import cupy as cp

    y, seen = cp.zeros(1, cp.int32), cp.full(N, -1, cp.int32)
    _launch(old_bundle, "ticket_i", "cuda", {"key": cp.zeros(N, cp.int32), "one": cp.ones(N, cp.int32), "y": y, "seen": seen}).launch()
    cp.cuda.Device().synchronize()
    assert int(y[0]) == N and np.array_equal(np.sort(seen.get()), np.arange(N))


@pytest.mark.parametrize(
    "primal, helper", [(swap, "exchangeAt"), (claim, "compareExchangeAt"), (ticket, "fetchAddAt")],
    ids=["exchange", "compare_exchange", "add_old"],
)
@pytest.mark.parametrize("target", ["host", "cuda"])
def test_an_update_that_returns_the_old_value_calls_its_helper_once_and_its_readers_read_the_name(primal, helper, target):
    text = emit_source(primal, targets=(target,)).sources[target].text
    assert text.count(f"aether::accum::atomic::{helper}(") == 1
    (declaration,) = [line for line in text.splitlines() if f"atomic::{helper}(" in line]
    assert declaration.strip().startswith("const auto hawk_old_")
    name = declaration.split("=")[0].split()[-1]
    assert any(name in line and "atomic::" not in line for line in text.splitlines())


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
