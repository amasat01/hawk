# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The graph verbs: ``hawk.Graph``, ``hawk.GraphPipeline`` and ``hawk.reverse``. A replay is held against the closed form
of the kernels it records, and a graph run under a team of workers against the serial reference."""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import HawkError, Mutable, Param, Scalar, Terminated
from hawk.artifact import build_bundle

N = 37
GOAL = 3.0
RATE = np.linspace(0.2, 1.0, N)


@hawk.kernel
def axpb(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


@hawk.kernel
def twice(y: Scalar, w: Mutable[Scalar]):
    w = y * 2.0


@hawk.kernel
def ramp(rate: Scalar, goal: Param, terminated: Terminated, x: Mutable[Scalar]):
    xn = x + rate
    x = xn
    terminated = xn >= goal


@pytest.fixture(scope="module")
def loaded(tmp_path_factory, cache_dir):
    """``{kernel name: loaded kernel}`` of one host bundle."""
    bundle = build_bundle([axpb, twice, ramp], tmp_path_factory.mktemp("graph_verbs"), targets=("host",),
                          cache_dir=cache_dir)
    return {k.name: hawk.load(bundle.directory, k.name) for k in (axpb, twice, ramp)}


def _axpb_bound(loaded, x, y, structure=hawk.HostTeam):
    return hawk.plan(loaded["axpb"], structure).bind(x=x, a=3.0, b=-2.0, y=y)


@pytest.mark.parametrize("pipeline", [hawk.Graph, hawk.GraphPipeline], ids=["graph", "pipeline"])
def test_a_replay_runs_the_recorded_launch_and_sees_what_the_planes_hold_then(loaded, pipeline):
    x, y = np.arange(N, dtype=np.float64), np.zeros(N)
    graph = pipeline().add(_axpb_bound(loaded, x, y)).build()
    assert not y.any()                                  # recording runs nothing: only a replay does
    graph.launch()
    np.testing.assert_array_equal(y, 3.0 * x - 2.0)
    x += 1.0
    graph.launch()
    np.testing.assert_array_equal(y, 3.0 * (np.arange(N) + 1.0) - 2.0)
    graph.close()


def test_launch_returns_the_graph_and_n_replays_run_n_times(loaded):
    finished = np.zeros(1, dtype=np.uint32)
    x = np.zeros(N)
    bound = hawk.plan(loaded["ramp"]).bind(rate=RATE, goal=GOAL, terminated=np.zeros(N, dtype=bool), x=x,
                                           finished_count=finished)
    graph = hawk.GraphPipeline().add(bound).build()
    assert graph.launch(3) is graph
    stepped, done = np.zeros(N), np.zeros(N, dtype=bool)
    for _ in range(3):
        stepped = np.where(done, stepped, stepped + RATE)
        done |= stepped >= GOAL
    np.testing.assert_array_equal(x, stepped)
    assert finished[0] == int(done.sum())               # each replay adds the samples it finished
    graph.synchronize()


def test_launches_run_in_the_order_they_were_added(loaded):
    x, y, w = np.arange(N, dtype=np.float64), np.zeros(N), np.zeros(N)
    first = _axpb_bound(loaded, x, y)
    second = hawk.plan(loaded["twice"]).bind(y=y, w=w)
    hawk.GraphPipeline().add(first).add(second).build().launch()
    np.testing.assert_array_equal(w, 2.0 * (3.0 * x - 2.0))
    w[:] = 0.0
    y[:] = 0.0
    hawk.GraphPipeline().add(second).add(first).build().launch()
    np.testing.assert_array_equal(w, 0.0)               # the doubling ran before the values it doubles were made


def test_the_team_and_the_serial_structure_replay_to_the_same_values(loaded):
    x = np.linspace(-1.0, 1.0, N)
    team, serial = np.zeros(N), np.zeros(N)
    hawk.Graph().add(_axpb_bound(loaded, x, team, hawk.HostTeam)).build().launch()
    hawk.Graph().add(_axpb_bound(loaded, x, serial, hawk.HostSerial)).build().launch()
    assert team.tobytes() == serial.tobytes()


def test_a_graph_names_its_launches(loaded):
    x, y = np.arange(N, dtype=np.float64), np.zeros(N)
    graph = hawk.Graph().add(_axpb_bound(loaded, x, y), name="step")
    assert "step" in repr(graph) and "unbuilt" in repr(graph)
    assert "built" in repr(graph.build()) and "unbuilt" not in repr(graph)


def test_python_is_never_recorded_and_a_graph_does_not_grow_once_built(loaded):
    x, y = np.arange(N, dtype=np.float64), np.zeros(N)
    with pytest.raises(HawkError, match="launches and never Python"):
        hawk.GraphPipeline().add(lambda: None)
    with pytest.raises(HawkError, match="takes a bound plan"):
        hawk.Graph().add(hawk.plan(loaded["axpb"]))
    graph = hawk.Graph().add(_axpb_bound(loaded, x, y)).build()
    with pytest.raises(HawkError, match="graph is built"):
        graph.add(_axpb_bound(loaded, x, y))
    with pytest.raises(HawkError, match="graph is built"):
        graph.build()


def test_an_empty_graph_an_unbuilt_one_and_a_closed_one_refuse_by_name(loaded):
    x, y = np.arange(N, dtype=np.float64), np.zeros(N)
    with pytest.raises(HawkError, match="nothing has been added"):
        hawk.GraphPipeline().build()
    unbuilt = hawk.GraphPipeline().add(_axpb_bound(loaded, x, y))
    with pytest.raises(HawkError, match="not built"):
        unbuilt.launch()
    graph = unbuilt.build()
    graph.close()
    graph.close()
    with pytest.raises(HawkError, match="not built, or is closed"):
        graph.launch()


def test_a_graph_keeps_a_record_only_when_asked_and_reverse_refuses_it_by_name(loaded):
    x, y = np.arange(N, dtype=np.float64), np.zeros(N)
    plain = hawk.Graph().add(_axpb_bound(loaded, x, y)).build()
    assert plain.record is None
    recorded = hawk.Graph(record=True).add(_axpb_bound(loaded, x, y))
    assert recorded.record is None                      # nothing is recorded before the build
    recorded.build().launch()
    assert isinstance(recorded.record, hawk.graph.GraphRecord) and recorded.record.graph is recorded
    with pytest.raises(HawkError, match="over a recorded graph"):
        hawk.reverse(recorded.record, {"y": np.ones(N)})
    with pytest.raises(HawkError, match="takes the record of a run made with record=True"):
        hawk.reverse(object(), {"y": np.ones(N)})
