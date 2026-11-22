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


@kernel
def place_squares(
    x: Scalar,
    out: Mutable[Scalar, "item":CAPACITY],  # noqa: F821 axis labels
    place: Mutable[Int32],
):
    slot = hawk.append(out, value=3.0 * x * x, when=x > 0.0)
    place = slot


@kernel
def place_squares_few(
    x: Scalar,
    out: Mutable[Scalar, "item":SMALL],  # noqa: F821 axis labels
    place: Mutable[Int32],
):
    slot = hawk.append(out, value=3.0 * x * x, when=x > 0.0)
    place = slot


@kernel
def place_pairs(
    v: Vector[2],
    out: Mutable[Vector[2], "item":CAPACITY],  # noqa: F821 axis labels
    place: Mutable[Int32],
):
    slot = hawk.append(out, value=v * v[0], when=v[0] > 0.0)
    place = slot


@kernel
def place_both_signs(
    x: Scalar,
    out: Mutable[Scalar, "item":CAPACITY],  # noqa: F821 axis labels
    above: Mutable[Int32],
    below: Mutable[Int32],
):
    first = hawk.append(out, value=3.0 * x * x, when=x > 0.0)
    second = hawk.append(out, value=x * x * x, when=x < 0.0)
    above = first
    below = second


def _reverse(primal, tmp_path, cache_dir):
    """The bundles of ``primal`` and of its reverse derivative (a bundle holds one execution axis), and the derivative's name."""
    derived = Kernel(f"{primal.name}_vjp", vjp(primal, primal_unit="0" * 64))
    both = (
        build_bundle(
            [k],
            tmp_path / k.name,
            mode="float64",
            targets=("host",),
            cache_dir=cache_dir,
        )
        for k in (primal, derived)
    )
    return (*both, derived.name)


def _gradient(built, primal, derived, structure, x, capacity, seed):
    built, reverse_built = built
    """Run ``primal`` for the places its lanes took, then its reverse with ``seed`` as the adjoint of the list."""
    place = np.zeros(len(x), np.int32)
    _bound(
        built, primal, structure, x=x, out=np.full(capacity, np.nan), place=place
    ).launch()
    bar_x = np.zeros(len(x))
    _bound(
        reverse_built, derived, structure, x=x, place=place, bar_out=seed, bar_x=bar_x
    ).launch()
    return place, bar_x


@STRUCTURES
def test_the_gradient_of_an_appended_value_is_its_derivative_times_the_seed_at_the_place_the_lane_took(
    tmp_path, cache_dir, structure
):
    # oracle: the closed form, d(3 x^2)/dx = 6 x, times the seed at the lane's own place; 0 for a lane that appended nothing
    primal_built, reverse_built, derived = _reverse(place_squares, tmp_path, cache_dir)
    x = _inputs(14)
    seed = np.random.default_rng(15).normal(size=CAPACITY)
    place, got = _gradient(
        (primal_built, reverse_built),
        "place_squares",
        derived,
        structure,
        x,
        CAPACITY,
        seed,
    )
    asked = x > 0.0
    want = np.where(asked, 6.0 * x * seed[np.where(asked, place, 0)], 0.0)
    assert np.allclose(got, want, rtol=1e-13, atol=0.0)


@STRUCTURES
def test_a_lane_dropped_by_overflow_gets_a_zero_gradient(
    tmp_path, cache_dir, structure
):
    # oracle: the same closed form; a place at or past the capacity reads a zero adjoint under the zero edge rule
    primal_built, reverse_built, derived = _reverse(
        place_squares_few, tmp_path, cache_dir
    )
    x = _inputs(16)
    seed = np.random.default_rng(17).normal(size=SMALL)
    place, got = _gradient(
        (primal_built, reverse_built),
        "place_squares_few",
        derived,
        structure,
        x,
        SMALL,
        seed,
    )
    kept = (x > 0.0) & (place < SMALL)
    assert kept.sum() == SMALL and ((x > 0.0) & (place >= SMALL)).any()
    want = np.where(kept, 6.0 * x * seed[np.where(kept, place, 0)], 0.0)
    assert np.allclose(got, want, rtol=1e-13, atol=0.0)


@STRUCTURES
def test_the_gradient_of_an_appended_vector_value_reads_the_seed_of_each_component(
    tmp_path, cache_dir, structure
):
    # oracle: the closed form of d(v * v0)/dv: component 0 gets 2 v0 s0 + v1 s1, component 1 gets v0 s1
    primal_built, reverse_built, derived = _reverse(place_pairs, tmp_path, cache_dir)
    v = np.random.default_rng(18).normal(size=(N, 2))
    seed = np.random.default_rng(19).normal(size=(2, CAPACITY))
    place = np.zeros(N, np.int32)
    given = np.ascontiguousarray(v.T)
    _bound(
        primal_built,
        "place_pairs",
        structure,
        v=given,
        out=np.full((2, CAPACITY), np.nan),
        place=place,
    ).launch()
    bar_v = np.zeros((2, N))
    _bound(
        reverse_built,
        derived,
        structure,
        v=given,
        place=place,
        bar_out=seed,
        bar_v=bar_v,
    ).launch()
    asked = v[:, 0] > 0.0
    at = np.where(asked, place, 0)
    s0, s1 = seed[0, at], seed[1, at]
    want = np.stack([2 * v[:, 0] * s0 + v[:, 1] * s1, v[:, 0] * s1]) * asked
    assert np.allclose(bar_v, want, rtol=1e-13, atol=1e-13)


@STRUCTURES
def test_two_appends_to_one_list_each_read_the_place_of_their_own_ticket(
    tmp_path, cache_dir, structure
):
    # oracle: the closed forms 6 x (lanes above zero) and 3 x^2 (lanes below), each times the seed at the lane's own place
    primal_built, reverse_built, derived = _reverse(
        place_both_signs, tmp_path, cache_dir
    )
    x = _inputs(22)[:CAPACITY]
    x = x[x != 0.0]
    seed = np.random.default_rng(23).normal(size=CAPACITY)
    above, below = np.zeros(len(x), np.int32), np.zeros(len(x), np.int32)
    _bound(
        primal_built,
        "place_both_signs",
        structure,
        x=x,
        out=np.full(CAPACITY, np.nan),
        above=above,
        below=below,
    ).launch()
    bar_x = np.zeros(len(x))
    _bound(
        reverse_built,
        derived,
        structure,
        x=x,
        above=above,
        below=below,
        bar_out=seed,
        bar_x=bar_x,
    ).launch()
    positive = x > 0.0
    want = np.where(
        positive,
        6.0 * x * seed[np.where(positive, above, 0)],
        3.0 * x**2 * seed[np.where(positive, 0, below)],
    )
    assert np.allclose(bar_x, want, rtol=1e-13, atol=0.0)


def test_the_device_text_of_the_reverse_reads_the_place_and_takes_no_ticket():
    derived = Kernel("place_squares_vjp", vjp(place_squares, primal_unit="0" * 64))
    text = emit_source(derived, targets=("cuda",)).sources["cuda"].text
    assert "fetchAddAt" not in text and "aether::edge::" in text


@pytest.mark.gpu
def test_the_device_gradient_of_an_appended_value_is_its_derivative_times_the_seed(
    tmp_path, cache_dir
):
    import cupy as cp

    derived = Kernel("place_squares_vjp", vjp(place_squares, primal_unit="0" * 64))
    built = build_bundle(
        [place_squares, derived],
        tmp_path,
        mode="float64",
        targets=("cuda", "host"),
        cache_dir=cache_dir,
    )
    x = _inputs(20)
    seed = np.random.default_rng(21).normal(size=CAPACITY)
    place = cp.zeros(N, cp.int32)
    _launch(
        built,
        "place_squares",
        "cuda",
        {"x": cp.asarray(x), "out": cp.full(CAPACITY, cp.nan), "place": place},
    ).launch()
    bar_x = cp.zeros(N)
    _launch(
        built,
        derived.name,
        "cuda",
        {
            "x": cp.asarray(x),
            "place": place,
            "bar_out": cp.asarray(seed),
            "bar_x": bar_x,
        },
    ).launch()
    cp.cuda.Device().synchronize()
    asked = x > 0.0
    at = np.where(asked, place.get(), 0)
    assert np.allclose(
        bar_x.get(), np.where(asked, 6.0 * x * seed[at], 0.0), rtol=1e-13, atol=0.0
    )


def test_the_reverse_derivative_of_an_appending_kernel_that_stores_no_slot_is_refused_by_name():
    with pytest.raises(
        HawkError, match=r"appends to 'out' without storing the slot the lane took"
    ):
        vjp(keep_squares)
