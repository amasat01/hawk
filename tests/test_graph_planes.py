# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""What a graph is made over: the plane store its launches share, the views and kernel a bound plan states, the vjp unit a kernel
built with ``reverse=True`` carries, and the recorder a builder runs under."""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import HawkError, Mutable, Scalar, UInt32, graph_builder
from hawk.diff.recording import DerivedUnit, graph_reverse_unit
from hawk.graph import RepeatWhile, Skippable
from hawk.math import where
from hawk.runtime import PlaneStore

N = 7
X = np.linspace(1.0, 2.0, N)


@hawk.kernel
def scale(x: Scalar, w: Scalar, y: Mutable[Scalar]):
    y = w * x


@hawk.kernel
def tick(c: Mutable[UInt32]):
    c = where(c > 0, c - 1, c)


@hawk.kernel
def by_number(x: Scalar, a: hawk.Param, y: Mutable[Scalar]):
    y = a * x


@hawk.kernel
def only_number(a: hawk.Param, y: Mutable[Scalar]):
    y = a * 2.0


def test_a_store_names_a_region_of_memory_once_whatever_view_asks_for_it():
    array = np.arange(6.0)
    store = PlaneStore(hawk.deploy(scale).kernel._door)
    first = store.plane(array)
    assert store.plane(array) == first and store.plane(array[:]) == first        # one plane per region
    assert store.plane(array[:3]) != first                                        # another extent is another region
    assert PlaneStore.region(array) == PlaneStore.region(array.view())
    assert [view.shape for _plane, view in store.entries()] == [(6,), (3,)]


def test_a_bound_plan_states_its_kernel_and_the_views_the_library_reads():
    x, w, y = X.copy(), X.copy(), np.zeros(N)
    bound = hawk.deploy(scale).bind(x=x, w=w, y=y)
    assert bound.kernel.sidecar["kernel"] == "scale"
    assert set(bound.views) == {"x", "w", "y"} and bound.views["x"] is x
    with pytest.raises(TypeError):
        bound.views["x"] = w                                                      # a read-only view of the bound planes


def test_joining_a_store_moves_the_planes_to_it_and_a_launch_still_writes_the_callers_array():
    x, w, y = X.copy(), X.copy(), np.zeros(N)
    bound = hawk.deploy(scale).bind(x=x, w=w, y=y)
    store = PlaneStore(bound.kernel._door)
    bound.join(store)
    assert {id(view) for _plane, view in store.entries()} == {id(x), id(w), id(y)}
    bound.join(store)                                                              # joining again changes nothing
    bound.launch()
    np.testing.assert_allclose(y, w * x, rtol=1e-15)


def test_a_kernel_built_with_reverse_carries_its_vjp_and_one_without_has_none():
    plain, reverse = hawk.deploy(scale), hawk.deploy(scale, reverse=True)
    assert plain.kernel.graph_vjp() is None
    assert reverse.kernel.graph_vjp() is not None and reverse.kernel.graph_vjp() is reverse.kernel.graph_vjp()
    assert [record["role"] for record in reverse.kernel.sidecar["derived"]] == ["step_vjp"]
    assert sorted(reverse.kernel.gradient_names()) == ["w", "x"]
    assert hawk.deploy(tick, reverse=True).kernel.graph_vjp() is None            # no float plane is written: nothing flows


def test_the_vjp_unit_of_a_kernel_states_the_links_and_leaves_a_number_out():
    unit = graph_reverse_unit(scale)
    assert isinstance(unit, DerivedUnit) and unit.role == "step_vjp" and unit.kernel.name == "scale_vjp"
    links = {link["slot"]: (link["form"], link["of"]) for link in unit.facts["links"]}
    assert links == {"x": ("Value", "x"), "w": ("Value", "w"), "bar_y": ("AdjointIn", "y"), "bar_x": ("Grad", "x"),
                     "bar_w": ("Grad", "w")}
    assert [link["of"] for link in graph_reverse_unit(by_number).facts["links"] if link["form"] == "Grad"] == ["x"]
    assert graph_reverse_unit(tick) is None and graph_reverse_unit(only_number) is None


def test_the_recorder_is_there_only_while_a_builder_runs_and_states_whether_the_graph_is_reversed():
    assert graph_builder.current() is None
    seen = []

    @hawk.Graph.builder
    def watch(x, plain) -> hawk.Graph:
        g = hawk.Graph(record=not plain)
        seen.append(graph_builder.current())
        scale(x, np.ones(N), y=np.zeros(N))
        return g

    watch(X.copy(), True)
    watch(X.copy(), False)
    assert graph_builder.current() is None
    assert [recorder.reverse for recorder in seen] == [False, True]
    assert seen[0].kind == "builder" and seen[0].name == "watch" and isinstance(seen[0], graph_builder.Recorder)
    with pytest.raises(HawkError, match="make the graph"):
        graph_builder.Recorder(None, "builder", "demo").store(None)


def test_the_conditional_forms_hold_their_step_guard_and_cap():
    guard = hawk.SkipGuard.nonzero(np.zeros(1, dtype=np.uint32))
    once, loop = hawk.skippable("step", guard), hawk.repeat_while("step", guard, 3)
    assert isinstance(once, Skippable) and once.step == "step" and once.guard is guard
    assert isinstance(loop, RepeatWhile) and loop.max_iters == 3 and hawk.repeat_while("step", guard).max_iters is None
