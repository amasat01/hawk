# ruff: noqa: F821 - the labels in a Grid or a plane annotation are axis labels, not names
"""A lane's several commits to one target commit through one run state per element: the run of a commit is not flushed by the
lane's next commit to another element. Oracle for the values: ``np.add.at`` over the lanes' keys."""

import numpy as np
import pytest

import hawk
from hawk import Grid, Mutable, Scalar, Vector, kernel
from hawk.artifact import build_bundle, emit_source

NETS, PER_NET = 5, 64


def vector_sum(width):
    @kernel
    def vec(
        lane: Grid["net", "sample":PER_NET],
        v: Vector[width],
        acc: Mutable[Vector[width], "net"],
    ):
        hawk.scatter(acc, contract="plain")[lane.net] += v

    vec.name = f"run_vec{width}"
    return vec


@kernel
def two_elements(
    lane: Grid["net", "sample":PER_NET], v: Scalar, acc: Mutable[Scalar, "net", "p":4]
):
    hawk.scatter(acc, contract="plain")[lane.net, 2] += v
    hawk.scatter(acc, contract="plain")[lane.net, 3] += v * 2.0


KERNELS = [vector_sum(3), vector_sum(40), two_elements]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    return build_bundle(
        KERNELS,
        tmp_path_factory.mktemp("run_states"),
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )


def _text(kernel_):
    return emit_source(kernel_, targets=("host",)).sources["host"].text


def test_the_components_of_a_vector_commit_through_one_run_state_each():
    text = _text(KERNELS[0])
    for k in range(3):
        state = "hawk_run_aout_acc" if k == 0 else f"hawk_run_{k}_aout_acc"
        assert (
            f"aether::accum::run::addAt<32>(aout_acc({k}, " in text
            and f", i, {state});" in text
        )
        assert f"auto {state} = aether::accum::run::open(aout_acc);" in text
        assert f"aether::accum::run::close({state});" in text
    assert "hawk_abi::accum_add_at(" not in text


def test_two_elements_of_one_plane_commit_through_two_run_states():
    text = _text(two_elements)
    assert text.count("aether::accum::run::addAt<32>(") == 2
    assert "i, hawk_run_aout_acc);" in text and "i, hawk_run_1_aout_acc);" in text


def test_a_vector_past_the_bound_on_states_commits_the_rest_without_a_run():
    text = _text(KERNELS[1])
    assert text.count("aether::accum::run::addAt<32>(") == 32
    assert text.count("hawk_abi::accum_add_at(") == 8


def test_the_vector_sum_equals_add_at_over_the_lanes_keys(bundle):
    lanes = NETS * PER_NET
    v = np.arange(3 * lanes, dtype=float).reshape(3, lanes) % 7
    acc = np.zeros((3, NETS))
    hawk.run(hawk.load(bundle.directory, "run_vec3"), v=v, acc=acc)
    want = np.zeros((3, NETS))
    np.add.at(want, (slice(None), np.arange(lanes) // PER_NET), v)
    np.testing.assert_array_equal(acc, want)


def test_a_vector_past_the_bound_still_equals_add_at(bundle):
    lanes = NETS * PER_NET
    v = np.arange(40 * lanes, dtype=float).reshape(40, lanes) % 7
    acc = np.zeros((40, NETS))
    hawk.run(hawk.load(bundle.directory, "run_vec40"), v=v, acc=acc)
    want = np.zeros((40, NETS))
    np.add.at(want, (slice(None), np.arange(lanes) // PER_NET), v)
    np.testing.assert_array_equal(acc, want)


def test_two_adds_into_one_plane_equal_add_at(bundle):
    lanes = NETS * PER_NET
    v = np.arange(lanes, dtype=float) % 7
    acc = np.zeros((NETS, 4))
    hawk.run(hawk.load(bundle.directory, "two_elements"), v=v, acc=acc)
    want = np.zeros((NETS, 4))
    np.add.at(want, (np.arange(lanes) // PER_NET, 2), v)
    np.add.at(want, (np.arange(lanes) // PER_NET, 3), v * 2.0)
    np.testing.assert_array_equal(acc, want)
