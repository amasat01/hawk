# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.Event``: marks on a stream, the time between two of them, and the refusals of a wrong order and of a recording.

The host rows time a host call with the library's events and with ``time.perf_counter``, the independent reference. The rows
that need two streams or a device stream are marked ``gpu`` and run on a machine with a card.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import hawk
from hawk import HawkError, Mutable, Vector

RNG = np.random.default_rng(20261127)
#: A host reduction over 2,000,000 samples takes about fifteen milliseconds here, long enough for a factor of two to mean
#: something; the floor is what a very short call may differ by (the clock reads and the wrapping of the operand).
SAMPLES = 2_000_000
FLOOR_MS = 1.0
ROUNDS = 5


@hawk.kernel
def double_v(x: Vector[3], y: Mutable[Vector[3]]):
    y = x * 2.0


def test_elapsed_ms_around_a_host_call_is_within_a_factor_of_two_of_the_python_clock():
    """Oracle: ``time.perf_counter`` around the same call, over 2,000,000 samples, five rounds after one warm-up call."""
    x = RNG.normal(size=(1, SAMPLES))
    hawk.reduce(x, "sum")
    start, stop = hawk.Event(), hawk.Event()
    for _ in range(ROUNDS):
        began = time.perf_counter()
        start.record()
        hawk.reduce(x, "sum")
        stop.record()
        wall = (time.perf_counter() - began) * 1e3
        ms = stop.elapsed_ms(since=start)
        assert wall / 2 - FLOOR_MS <= ms <= 2 * wall + FLOOR_MS, (ms, wall)
    start.close()
    stop.close()


def test_elapsed_ms_of_two_marks_in_the_wrong_order_is_negative():
    first, second = hawk.Event(), hawk.Event()
    first.record()
    time.sleep(0.002)
    second.record()
    assert second.elapsed_ms(since=first) >= 2.0
    assert first.elapsed_ms(since=second) <= -2.0


def test_waiting_for_an_event_never_recorded_is_refused_by_name():
    with pytest.raises(HawkError, match=r"Event\.wait: the event was never recorded"):
        hawk.Event().wait()


def test_elapsed_ms_names_the_event_that_was_never_recorded():
    recorded, never = hawk.Event(), hawk.Event()
    recorded.record()
    with pytest.raises(
        HawkError, match=r"elapsed_ms: the stop event was never recorded"
    ):
        never.elapsed_ms(since=recorded)
    with pytest.raises(
        HawkError, match=r"elapsed_ms: the start event \(since=\) was never recorded"
    ):
        recorded.elapsed_ms(since=never)


def test_a_closed_event_is_refused_by_name_and_closing_twice_is_harmless():
    event = hawk.Event()
    event.record()
    event.close()
    event.close()
    with pytest.raises(HawkError, match="the event is closed"):
        event.record()
    with pytest.raises(HawkError, match="never recorded, or is closed"):
        event.wait()


def test_wait_on_the_host_returns_at_once_after_a_record():
    event = hawk.Event()
    event.record()
    event.wait()


@pytest.mark.parametrize("call", ["record", "wait", "elapsed_ms"])
def test_an_event_used_inside_a_recording_is_refused_by_name_with_wait_planes_as_the_way(
    call,
):
    marked = hawk.Event()
    marked.record()

    def build(x) -> hawk.Graph:
        g = hawk.Graph()
        use = {
            "record": marked.record,
            "wait": marked.wait,
            "elapsed_ms": lambda: marked.elapsed_ms(since=marked),
        }
        use[call]()
        g.y = double_v(x)
        return g

    with pytest.raises(HawkError, match=r"inside a recording.*wait\(planes\)"):
        hawk.Graph.builder(build)(RNG.normal(size=(3, 6)))


def test_an_event_in_the_scope_of_an_adopted_stream_nobody_launched_on_is_refused_by_name():
    with hawk.Stream(0):
        with pytest.raises(HawkError, match="no door yet"):
            hawk.Event().record()


# ---- device rows (run on a machine with a card) ----

CHAIN = 40


def chain(x) -> hawk.Graph:
    """A graph of CHAIN launches of ``double_v`` in a row over ``x``: its ``y`` is ``2 ** CHAIN * x``."""
    g = hawk.Graph()
    y = x
    for _ in range(CHAIN):
        y = double_v(y)
    g.y = y
    return g


def double(x) -> hawk.Graph:
    g = hawk.Graph()
    g.y = double_v(x)
    return g


@pytest.mark.gpu
def test_a_reader_that_waits_on_an_event_sees_the_writers_values_in_100_runs():
    """Oracle: NumPy. The writer's stream replays a graph that makes ``2 x``; the reader's stream waits for its event and
    replays a graph that doubles it again, so a read before the write finished would show the previous run's values."""
    cupy = pytest.importorskip("cupy")
    x = cupy.zeros((3, 1_000_000))
    writer_graph = hawk.Graph.builder(double)(x)
    reader_graph = hawk.Graph.builder(double)(writer_graph.y)
    writer, reader, written = (
        hawk.Stream(device="cuda:0"),
        hawk.Stream(device="cuda:0"),
        hawk.Event(),
    )
    for _ in range(100):
        values = RNG.normal(size=(3, 1_000_000))
        x[...] = cupy.asarray(values)
        cupy.cuda.get_current_stream().synchronize()  # cupy's stream does not order against the library streams below
        with writer:
            writer_graph.launch()
            written.record()
        with reader:
            written.wait()
            reader_graph.launch()
        reader.synchronize()
        np.testing.assert_array_equal(cupy.asnumpy(reader_graph.y), 4.0 * values)


@pytest.mark.gpu
def test_two_chains_on_two_streams_overlap_in_time_by_their_event_stamps():
    """Oracle: the event stamps. The second chain starts before the first has stopped, which two chains one after the other
    never show."""
    cupy = pytest.importorskip("cupy")
    streams = [hawk.Stream(device="cuda:0"), hawk.Stream(device="cuda:0")]
    graphs = [hawk.Graph.builder(chain)(cupy.zeros((3, 1_000_000))) for _ in streams]
    starts, stops = [hawk.Event(), hawk.Event()], [hawk.Event(), hawk.Event()]
    for stream, graph, start, stop in zip(streams, graphs, starts, stops):
        with stream:
            start.record()
            graph.launch()
            stop.record()
    for stream in streams:
        stream.synchronize()
    assert starts[1].elapsed_ms(since=starts[0]) < stops[0].elapsed_ms(since=starts[0])
