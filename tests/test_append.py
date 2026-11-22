# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.append`` adds the values of the lanes whose condition holds to a list, in no promised order.

The oracle of every value test is NumPy applied to the inputs: the set of stored values is ``x[cond]``, the count is
``cond.sum()``, and each place below the count is filled once. A launch over one thread and one over a team of threads must
satisfy the same properties.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import _launch

import hawk
from hawk import HawkError, Int32, Mutable, Scalar, Vector, kernel
from hawk.artifact import build_bundle, emit_source
from hawk.diff import jvp, vjp
from hawk.trace.kernel import Kernel

N = 100_000
CAPACITY = 65536
SMALL = 64


@kernel
def keep_positive(x: Scalar, out: Mutable[Scalar, "item":CAPACITY]):  # noqa: F821 axis labels
    slot = hawk.append(out, value=x, when=x > 0.0)


@kernel
def keep_positive_few(x: Scalar, out: Mutable[Scalar, "item":SMALL]):  # noqa: F821 axis labels
    slot = hawk.append(out, value=x, when=x > 0.0)


@kernel
def keep_all(x: Scalar, out: Mutable[Scalar, "item":CAPACITY]):  # noqa: F821 axis labels
    slot = hawk.append(out, value=x)


@kernel
def keep_with_place(
    x: Scalar,
    out: Mutable[Scalar, "item":CAPACITY],  # noqa: F821 axis labels
    place: Mutable[Int32],
):
    slot = hawk.append(out, value=x, when=x > 0.0)
    place = slot


@kernel
def keep_pairs(v: Vector[2], out: Mutable[Vector[2], "item":CAPACITY]):  # noqa: F821 axis labels
    slot = hawk.append(out, value=v, when=v[0] > 0.0)


@kernel
def keep_both_signs(x: Scalar, out: Mutable[Scalar, "item":CAPACITY]):  # noqa: F821 axis labels
    above = hawk.append(out, value=x, when=x > 0.0)
    below = hawk.append(out, value=x, when=x < 0.0)


@kernel
def keep_in_tail(x: Scalar, tail: Mutable[Scalar, "item":CAPACITY]):  # noqa: F821 axis labels
    slot = hawk.append(tail, value=x, when=x > 0.0)


@kernel
def keep_squares(x: Scalar, out: Mutable[Scalar, "item":CAPACITY]):  # noqa: F821 axis labels
    slot = hawk.append(out, value=3.0 * x * x, when=x > 0.0)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    return build_bundle(
        [
            keep_positive,
            keep_positive_few,
            keep_all,
            keep_with_place,
            keep_pairs,
            keep_both_signs,
            keep_in_tail,
        ],
        tmp_path_factory.mktemp("append"),
        mode="float64",
        targets=("cuda", "host"),
        cache_dir=cache_dir,
    )


STRUCTURES = pytest.mark.parametrize(
    "structure", [hawk.HostSerial, hawk.HostTeam], ids=["one_thread", "team"]
)


def _bound(bundle, name, structure, **planes):
    return hawk.plan(hawk.load(bundle.directory, name), structure).bind(**planes)


def _inputs(seed=3):
    return np.random.default_rng(seed).normal(size=N)


@STRUCTURES
def test_the_stored_values_are_the_values_whose_condition_holds_and_the_count_is_their_number(
    bundle, structure
):
    x = _inputs()
    out = np.full(CAPACITY, np.nan)
    run = _bound(bundle, "keep_positive", structure, x=x, out=out)
    run.launch()
    want = x[x > 0.0]
    assert run.count("out") == len(want)
    assert np.array_equal(np.sort(out[: len(want)]), np.sort(want))
    assert np.isnan(
        out[len(want) :]
    ).all()  # every place below the count is filled once, the others untouched


@STRUCTURES
def test_a_list_that_overflows_keeps_capacity_values_of_the_true_set_and_counts_every_lane_that_asked(
    bundle, structure
):
    x = _inputs(4)
    out = np.full(SMALL, np.nan)
    run = _bound(bundle, "keep_positive_few", structure, x=x, out=out)
    run.launch()
    true_set = x[x > 0.0]
    assert run.count("out") == len(true_set) > SMALL
    assert not np.isnan(out).any() and len(np.unique(out)) == SMALL
    assert np.isin(out, true_set).all()


@STRUCTURES
def test_a_lane_without_a_condition_always_takes_a_place(bundle, structure):
    x = _inputs(5)[:CAPACITY]
    out = np.full(CAPACITY, np.nan)
    run = _bound(bundle, "keep_all", structure, x=x, out=out)
    run.launch()
    assert run.count("out") == CAPACITY
    assert np.array_equal(np.sort(out), np.sort(x))


@STRUCTURES
def test_the_slot_of_a_lane_is_its_own_place_in_the_list_and_minus_one_when_its_condition_is_false(
    bundle, structure
):
    # oracle: NumPy identity of the inputs; the lane that asked holds its value at the slot it was given
    x = _inputs(6)[:CAPACITY]
    out, place = np.full(CAPACITY, np.nan), np.zeros(len(x), np.int32)
    run = _bound(bundle, "keep_with_place", structure, x=x, out=out, place=place)
    run.launch()
    asked = x > 0.0
    assert (place[~asked] == -1).all()
    assert np.array_equal(np.sort(place[asked]), np.arange(asked.sum()))
    assert np.array_equal(out[place[asked]], x[asked])


@STRUCTURES
def test_a_vector_value_is_stored_whole_at_one_place(bundle, structure):
    v = np.random.default_rng(7).normal(size=(N, 2))
    out = np.full(
        (2, CAPACITY), np.nan
    )  # the value's components first, then the list axis
    run = _bound(bundle, "keep_pairs", structure, v=np.ascontiguousarray(v.T), out=out)
    run.launch()
    want = v[v[:, 0] > 0.0]
    assert run.count("out") == len(want)
    got = out[:, : len(want)].T
    order = lambda rows: rows[np.lexsort(rows.T)]  # noqa: E731
    assert np.array_equal(order(got), order(want))


@STRUCTURES
def test_two_appends_to_one_list_share_its_count(bundle, structure):
    x = _inputs(8)[:CAPACITY]
    x = x[x != 0.0]
    out = np.full(CAPACITY, np.nan)
    run = _bound(bundle, "keep_both_signs", structure, x=x, out=out)
    run.launch()
    assert run.count("out") == len(x)
    assert np.array_equal(np.sort(out[: len(x)]), np.sort(x))


def test_a_second_launch_continues_the_list_where_the_first_stopped(bundle):
    x = _inputs(9)
    out = np.full(CAPACITY, np.nan)
    run = _bound(bundle, "keep_positive", hawk.HostSerial, x=x[: N // 2], out=out)
    run.launch()
    first = run.count("out")
    run.rebind(x=x[N // 2 :])
    run.launch()
    want = x[x > 0.0]
    assert first == (x[: N // 2] > 0.0).sum() and run.count("out") == len(want)
    assert np.array_equal(np.sort(out[: len(want)]), np.sort(want))


def test_the_count_is_found_by_the_relation_the_unit_states_not_by_the_name_of_a_plane(
    bundle,
):
    loaded = hawk.load(bundle.directory, "keep_in_tail")
    (count_plane,) = loaded.sidecar["counts"]
    assert loaded.sidecar["counts"] == {count_plane: "tail"}
    run = hawk.plan(loaded).bind(x=_inputs(), tail=np.full(CAPACITY, np.nan))
    run.launch()
    assert run.count("tail") == int(run.planes[count_plane][0]) > 0
    with pytest.raises(HawkError, match=r"'x' is not a list that hawk.append fills"):
        run.count("x")


def test_a_run_that_leaves_the_count_out_gets_it_back_beside_the_list(bundle):
    x = _inputs(10)
    result = hawk.plan(hawk.load(bundle.directory, "keep_positive")).run(
        x=x, out=np.full(CAPACITY, np.nan)
    )
    assert int(result["out_count"][0]) == (x > 0.0).sum()


def test_the_host_text_takes_one_ticket_and_stores_only_inside_the_list():
    text = emit_source(keep_positive, targets=("host",)).sources["host"].text
    assert text.count("aether::accum::atomic::fetchAddAt(") == 1
    assert "aether::edge::inside(" in text


def test_the_device_text_takes_one_ticket_and_stores_only_inside_the_list():
    text = emit_source(keep_positive, targets=("cuda",)).sources["cuda"].text
    assert text.count("aether::accum::atomic::fetchAddAt(") == 1
    assert "aether::edge::inside(" in text


@pytest.mark.gpu
def test_the_device_stores_the_true_set_and_counts_it(bundle):
    import cupy as cp

    x = _inputs(11)
    out = cp.full(CAPACITY, cp.nan)
    run = _launch(bundle, "keep_positive", "cuda", {"x": cp.asarray(x), "out": out})
    run.launch()
    cp.cuda.Device().synchronize()
    want = x[x > 0.0]
    assert run.count("out") == len(want)
    assert np.array_equal(np.sort(out.get()[: len(want)]), np.sort(want))


def test_a_list_declared_without_an_axis_is_refused_by_name():
    with pytest.raises(
        HawkError, match=r"'out' is a list, a plane declared with one axis"
    ):

        @kernel
        def flat(x: Scalar, out: Mutable[Scalar]):
            slot = hawk.append(out, value=x)


def test_an_append_inside_a_branch_is_refused_by_name():
    with pytest.raises(
        HawkError,
        match=r"hawk.append\(out, value=v, when=cond\) may not sit inside a branch",
    ):

        @kernel
        def branched(x: Scalar, out: Mutable[Scalar, "item":8]):  # noqa: F821 axis labels
            if x > 0.0:
                slot = hawk.append(out, value=x)


def test_an_append_without_a_value_is_refused_by_name():
    with pytest.raises(HawkError, match=r"`value` is required"):

        @kernel
        def valueless(x: Scalar, out: Mutable[Scalar, "item":8]):  # noqa: F821 axis labels
            slot = hawk.append(out, when=x > 0.0)


@STRUCTURES
def test_the_tangent_of_an_appended_value_sits_at_the_place_of_the_value(
    tmp_path, cache_dir, structure
):
    # oracle: the closed form of the derivative, 3 x^2 -> 6 x dx; each (value, tangent) pair the run stores is the pair NumPy
    # computes for one lane, whichever place the lane was given
    tangent_kernel = Kernel(
        "keep_squares_jvp", jvp(keep_squares, wrt=("x",), primal_unit="0" * 64)
    )
    built = build_bundle(
        [tangent_kernel],
        tmp_path,
        mode="float64",
        targets=("host",),
        cache_dir=cache_dir,
    )
    x = _inputs(12)
    dx = np.random.default_rng(13).normal(size=N)
    out, dot_out = np.full(CAPACITY, np.nan), np.full(CAPACITY, np.nan)
    run = _bound(
        built, "keep_squares_jvp", structure, x=x, dot_x=dx, out=out, dot_out=dot_out
    )
    run.launch()
    asked = x > 0.0
    stored = run.count("out")
    assert stored == asked.sum()
    got = np.stack([out[:stored], dot_out[:stored]], axis=1)
    want = np.stack([3.0 * x[asked] ** 2, 6.0 * x[asked] * dx[asked]], axis=1)
    order = lambda pairs: pairs[np.lexsort(pairs.T)]  # noqa: E731
    assert np.allclose(order(got), order(want), rtol=1e-14, atol=0.0)


def test_the_reverse_derivative_of_an_appending_kernel_is_refused_by_name():
    with pytest.raises(
        HawkError,
        match=r"appends to \['out'\] \(`hawk.append`\): the reverse derivative reads the adjoint",
    ):
        vjp(keep_squares)
