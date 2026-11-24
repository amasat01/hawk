# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.Stream``: the scope that puts launches on a stream, and the refusals of a host context and of a recording.

The host rows prove what hawk itself decides (the arguments, the refusal inside a recording, that the scope reaches the door
and shows its reason). A host context is one queue, so the rows that need a real stream are marked ``gpu`` and run on a machine
with a card; their oracle is NumPy and the same call made with no stream.
"""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import HawkError, Mutable, Scalar, Vector, _place, streams

RNG = np.random.default_rng(20261126)


@hawk.kernel
def double_v(x: Vector[3], y: Mutable[Vector[3]]):
    y = x * 2.0


@hawk.kernel
def double(x: Scalar, y: Mutable[Scalar]):
    y = x * 2.0


# ---- host rows ----


def test_a_stream_made_on_a_host_context_is_refused_with_the_librarys_reason(
    monkeypatch,
):
    """The host door is what answers a stream made on a host-only context: its reason reaches the caller as it gave it."""
    monkeypatch.setattr(
        streams, "_door_of_device", lambda device: _place.door_of(_place.HOST)
    )
    with pytest.raises(
        HawkError, match=r"eg_stream_new: host-only context: it has one queue"
    ):
        hawk.Stream(device="cuda:0")


def test_a_device_this_machine_does_not_run_is_refused_by_name():
    with pytest.raises(HawkError, match="library does not run here"):
        hawk.Stream(device="cuda:0")


def test_a_device_text_that_names_no_device_target_is_refused_by_name():
    with pytest.raises(
        HawkError, match=r"device='nowhere:0' is not '<target>:<ordinal>'"
    ):
        hawk.Stream(device="nowhere:0")


@pytest.mark.parametrize("arguments", [{}, {"theirs": 3, "device": "cuda:0"}])
def test_a_stream_is_either_adopted_or_made_never_both_and_never_neither(arguments):
    theirs = arguments.pop("theirs", None)
    with pytest.raises(HawkError, match="theirs .* or device="):
        hawk.Stream(theirs, **arguments)


def test_a_stream_handle_is_a_raw_integer_or_an_object_with_cuda_stream():
    class Theirs:
        def __cuda_stream__(self):
            return (0, 0)

    assert hawk.Stream(0)._handle == 0
    assert hawk.Stream(Theirs())._handle == 0
    with pytest.raises(HawkError, match="neither a raw stream handle"):
        hawk.Stream("a stream")


def test_a_scope_inside_a_recording_is_refused_by_name_with_parallel_as_the_way():
    def build(x) -> hawk.Graph:
        g = hawk.Graph()
        with hawk.Stream(0):
            g.y = double_v(x)
        return g

    with pytest.raises(HawkError, match=r"inside a recording.*parallel\(\)"):
        hawk.Graph.builder(build)(RNG.normal(size=(3, 6)))


def test_a_scope_reaches_the_door_of_a_launch_and_leaving_it_leaves_the_door_alone():
    """A host door has no stream to put work on: inside a scope of a raw stream the door's own refusal of ``eg_stream_use`` is
    shown, and the same call outside the scope runs and equals NumPy."""
    x = RNG.normal(size=(3, 6))
    with hawk.Stream(5):
        with pytest.raises(HawkError, match="eg_stream_use"):
            hawk.reduce(x, "sum")
    np.testing.assert_allclose(hawk.reduce(x, "sum"), x.sum(axis=1), rtol=1e-14)


def test_the_outer_scope_is_back_after_an_exception_inside_a_nested_one():
    """The host door refuses every stream, so a launch inside a scope shows that the scope is in force: after the nested one
    raised, the outer is the scope again, and after both a launch runs and equals NumPy."""
    x = RNG.normal(size=(3, 6))
    with hawk.Stream(7) as outer:
        with pytest.raises(RuntimeError, match="inside"):
            with hawk.Stream(8):
                raise RuntimeError("inside")
        assert streams._SCOPE.get()[0] is outer
        with pytest.raises(HawkError, match="eg_stream_use"):
            hawk.reduce(x, "sum")
    assert streams._SCOPE.get() is None
    np.testing.assert_allclose(hawk.reduce(x, "sum"), x.sum(axis=1), rtol=1e-14)


def test_closing_an_adopted_stream_frees_nothing_and_closing_twice_is_harmless():
    stream = hawk.Stream(0)
    stream.close()
    stream.close()


# ---- device rows (run on a machine with a card) ----


@pytest.mark.gpu
def test_a_stream_the_library_owns_runs_a_collective_with_the_values_of_numpy():
    cupy = pytest.importorskip("cupy")
    x = RNG.normal(size=(3, 50))
    stream = hawk.Stream(device="cuda:0")
    with stream:
        got = hawk.reduce(cupy.asarray(x), "sum")
    stream.synchronize()
    np.testing.assert_allclose(cupy.asnumpy(got), x.sum(axis=1), rtol=1e-13)
    np.testing.assert_allclose(
        cupy.asnumpy(got), cupy.asnumpy(hawk.reduce(cupy.asarray(x), "sum")), rtol=0
    )
    stream.close()


@pytest.mark.gpu
def test_a_kernel_call_on_a_stream_equals_the_call_on_the_default_stream_and_numpy():
    cupy = pytest.importorskip("cupy")
    x = RNG.normal(size=(3, 6))
    plain = double_v(cupy.asarray(x))
    stream = hawk.Stream(device="cuda:0")
    with stream:
        queued = double_v(cupy.asarray(x))
    stream.synchronize()
    np.testing.assert_array_equal(cupy.asnumpy(queued), cupy.asnumpy(plain))
    np.testing.assert_array_equal(cupy.asnumpy(queued), 2.0 * x)
    stream.close()


@pytest.mark.gpu
def test_a_graph_replay_on_a_stream_equals_the_replay_on_the_default_stream_and_numpy():
    cupy = pytest.importorskip("cupy")
    x = cupy.asarray(RNG.normal(size=(3, 6)))

    def build(x) -> hawk.Graph:
        g = hawk.Graph()
        g.y = double_v(x)
        return g

    graph = hawk.Graph.builder(build)(x)
    graph.launch()
    graph.synchronize()
    plain = cupy.asnumpy(graph.y)
    stream = hawk.Stream(device="cuda:0")
    with stream:
        graph.launch()
    stream.synchronize()
    np.testing.assert_array_equal(cupy.asnumpy(graph.y), plain)
    np.testing.assert_array_equal(plain, 2.0 * cupy.asnumpy(x))
    graph.close()
    stream.close()


@pytest.mark.gpu
def test_nested_scopes_of_adopted_streams_restore_the_outer_stream_also_after_an_exception():
    """Oracle: the values of NumPy from a launch in each region, and the outer stream's work ordered by its own synchronize."""
    cupy = pytest.importorskip("cupy")
    x = RNG.normal(size=(3, 50))
    outer, inner = cupy.cuda.Stream(), cupy.cuda.Stream()
    with hawk.Stream(outer):
        with pytest.raises(RuntimeError, match="inside"):
            with hawk.Stream(inner):
                first = hawk.reduce(cupy.asarray(x), "sum")
                raise RuntimeError("inside")
        second = hawk.reduce(cupy.asarray(x), "sum")
    third = hawk.reduce(cupy.asarray(x), "sum")
    for got in (first, second, third):
        np.testing.assert_allclose(cupy.asnumpy(got), x.sum(axis=1), rtol=1e-13)
    assert streams._SCOPE.get() is None


@pytest.mark.gpu
def test_a_stream_of_the_library_is_freed_when_closed_and_a_second_close_is_harmless():
    stream = hawk.Stream(device="cuda:0")
    stream.synchronize()
    stream.close()
    stream.close()
