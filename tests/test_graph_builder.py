# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The graph builder: kernel calls recorded inside ``@hawk.Graph.builder``, forks with ``wait``, the conditional forms and
``Graph.bind``. Every value row is held against a NumPy computation of the same arithmetic, in the order the graph records it."""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

import hawk
from hawk import HawkError, Mutable, Param, Scalar, UInt32
from hawk.math import where

N = 23
X = np.linspace(0.5, 2.5, N)


@hawk.kernel
def axpb(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


@hawk.kernel
def twice(y: Scalar, w: Mutable[Scalar]):
    w = y * 2.0


@hawk.kernel
def grow(x: Scalar, y: Mutable[Scalar]):
    y = y + x


@hawk.kernel
def tick(c: Mutable[UInt32]):
    c = where(c > 0, c - 1, c)


def close(got, want) -> None:
    """``got`` against ``want`` to the rounding of a fused multiply-add, which the kernels contract and NumPy does not."""
    np.testing.assert_allclose(got, want, rtol=1e-14, atol=1e-15)


def cell(value: int) -> np.ndarray:
    """A one-element condition cell."""
    return np.array([value], dtype=np.uint32)


# 1. a builder records kernel calls and returns arrays; replaying it reads the planes as they are then

def test_a_builder_records_kernel_calls_and_a_replay_runs_them():
    @hawk.Graph.builder
    def chain(x) -> hawk.Graph:
        g = hawk.Graph()
        y = axpb(x, a=3.0, b=-2.0)
        g.result = twice(y)
        return g

    x = X.copy()
    graph = chain(x)
    assert isinstance(graph, hawk.Graph) and isinstance(graph.result, np.ndarray)
    np.testing.assert_array_equal(graph.result, 0.0)           # building runs nothing
    close(graph(), 2.0 * (3.0 * X - 2.0))
    x += 1.0
    close(graph(), 2.0 * (3.0 * (X + 1.0) - 2.0))
    graph.close()


def test_graph_call_replays_n_times():
    @hawk.Graph.builder
    def accumulate(x, y) -> hawk.Graph:
        g = hawk.Graph()
        grow(x, y=y)
        g.result = y
        return g

    y = np.zeros(N)
    out = accumulate(X, y)(3)
    want = np.zeros(N)
    for _ in range(3):
        want = want + X
    np.testing.assert_array_equal(out, want)
    assert out is y                                             # the caller's own array is what the graph returns


def test_a_result_is_a_tree_of_arrays_and_is_read_only_once_built():
    @hawk.Graph.builder
    def two(x) -> hawk.Graph:
        g = hawk.Graph()
        y = axpb(x, a=1.0, b=1.0)
        g.result = {"y": y, "pair": (y, twice(y))}
        return g

    graph = two(X.copy())
    out = graph()
    close(out["y"], X + 1.0)
    close(out["pair"][1], 2.0 * (X + 1.0))
    with pytest.raises(HawkError, match="graph is built"):
        graph.result = None
    with pytest.raises(HawkError, match="tuple, list or dict of arrays"):
        hawk.Graph().__setattr__("result", object())


def test_a_builder_names_its_launches_and_numbers_a_repeated_kernel():
    @hawk.Graph.builder
    def twice_over(x) -> hawk.Graph:
        g = hawk.Graph()
        y = axpb(x, a=1.0, b=0.0)
        axpb(y, a=2.0, b=0.0)
        return g

    assert repr(twice_over(X.copy())) == "<hawk.Graph built: axpb, axpb#2>"


def test_the_builder_forms_refuse_what_they_cannot_record_by_name():
    @hawk.Graph.builder
    def two_graphs(x) -> hawk.Graph:
        hawk.Graph()
        return hawk.Graph()

    @hawk.Graph.builder
    def no_graph(x) -> hawk.Graph:
        return None

    @hawk.Graph.builder
    def not_returned(x) -> hawk.Graph:
        g = hawk.Graph()
        axpb(x, a=1.0, b=0.0)
        return None

    @hawk.Graph.builder
    def before_graph(x) -> hawk.Graph:
        axpb(x, a=1.0, b=0.0)
        return hawk.Graph()

    @hawk.Graph.builder
    def inner(g: hawk.Graph, x):
        axpb(x, a=1.0, b=0.0)

    @hawk.Graph.builder
    def nested(x) -> hawk.Graph:
        g = hawk.Graph()
        two_graphs(x)
        return g

    with pytest.raises(HawkError, match="builds ONE graph|makes ONE graph"):
        two_graphs(X)
    with pytest.raises(HawkError, match="a builder makes its graph"):
        no_graph(X)
    with pytest.raises(HawkError, match="a builder returns its graph"):
        not_returned(X)
    with pytest.raises(HawkError, match="make the graph"):
        before_graph(X)
    with pytest.raises(HawkError, match="a builder inside a builder; make it a block"):
        nested(X)
    with pytest.raises(HawkError, match="takes the open hawk.Graph"):
        inner(object(), X)
    with pytest.raises(HawkError, match="a builder returns its graph"):
        hawk.Graph.builder(_bad_return)


def _bad_return(x) -> int:
    return 0


# 2. a block appends to the graph it is given

def test_a_block_appends_to_the_graph_it_is_given_by_hand_or_inside_a_builder():
    @hawk.Graph.builder
    def stage(g: hawk.Graph, x, y):
        axpb(x, a=2.0, b=1.0, y=y)

    y = np.zeros(N)
    by_hand = hawk.Graph()
    stage(by_hand, X, y)
    by_hand.build().launch()
    close(y, 2.0 * X + 1.0)

    @hawk.Graph.builder
    def whole(x, y) -> hawk.Graph:
        g = hawk.Graph()
        stage(g, x, y)
        stage(g, y, y)
        return g

    y2 = np.zeros(N)
    whole(X, y2)()
    close(y2, 2.0 * (2.0 * X + 1.0) + 1.0)


# 3. forks: automatic edges, the race refused by the library, wait as the way out

def test_a_fork_over_disjoint_planes_equals_the_recording_order_run():
    @hawk.Graph.builder
    def fork(x, y1, y2) -> hawk.Graph:
        g = hawk.Graph()
        with g.parallel() as p:
            with p.branch():
                axpb(x, a=2.0, b=1.0, y=y1)
            with p.branch():
                axpb(x, a=-1.0, b=4.0, y=y2)
        g.result = (y1, y2)
        return g

    y1, y2 = np.zeros(N), np.zeros(N)
    out = fork(X, y1, y2)()
    close(out[0], 2.0 * X + 1.0)
    close(out[1], -1.0 * X + 4.0)


def test_a_sibling_race_is_refused_by_the_librarys_text_and_wait_resolves_it():
    def build(waiting: bool):
        @hawk.Graph.builder
        def fork(x, y, w) -> hawk.Graph:
            g = hawk.Graph()
            with g.parallel() as p:
                with p.branch():
                    axpb(x, a=3.0, b=-2.0, y=y)
                with p.branch() as b:
                    if waiting:
                        b.wait(y)
                    twice(y, w=w)
            g.result = w
            return g

        return fork(X, np.zeros(N), np.zeros(N))

    with pytest.raises(HawkError, match=r"RAW race between node"):
        build(False)
    close(build(True)(), 2.0 * (3.0 * X - 2.0))
    # the refusal ended that recording only: the next graph records
    assert build(True) is not None


def test_a_fork_nests_and_takes_hand_added_launches():
    plan = hawk.deploy(axpb)
    y1, y2, y3 = np.zeros(N), np.zeros(N), np.zeros(N)
    graph = hawk.Graph()
    with graph.parallel() as p:
        with p.branch() as b:
            b.add(plan.bind(x=X, a=1.0, b=1.0, y=y1))
            with b.parallel() as inner:
                with inner.branch() as ib:
                    ib.add(plan.bind(x=X, a=2.0, b=2.0, y=y2))
                with inner.branch() as ib:
                    ib.add(plan.bind(x=X, a=3.0, b=3.0, y=y3), name="third")
        with p.branch() as b:
            b.add(plan.bind(x=X, a=4.0, b=4.0, y=np.zeros(N)))
    graph.build().launch()
    for y, a in ((y1, 1.0), (y2, 2.0), (y3, 3.0)):
        close(y, a * X + a)


def test_the_handle_of_the_scope_is_the_one_that_adds_and_forks():
    plan = hawk.deploy(axpb)
    bound = plan.bind(x=X, a=1.0, b=0.0, y=np.zeros(N))
    graph = hawk.Graph()
    with graph.parallel() as p:
        with pytest.raises(HawkError, match="add steps in a branch"):
            p.add(bound)
        with pytest.raises(HawkError, match="nested fork from a branch"):
            p.parallel()
        with p.branch() as b:
            with pytest.raises(HawkError, match="use the branch handle"):
                graph.add(bound)
            with pytest.raises(HawkError, match="use the branch handle"):
                graph.parallel()
            with pytest.raises(HawkError, match="name the arrays to wait for"):
                b.wait()
            b.add(bound)
    with pytest.raises(HawkError, match="the region is closed"):
        p.branch()
    with pytest.raises(HawkError, match="a branch is still open"):
        with graph.parallel() as q, q.branch():
            graph.build()


# 4. conditionals: a cell is read at every replay, any other condition is decided once at build

def _if_else_graph(flag):
    @hawk.Graph.builder
    def pick(x, flag, y) -> hawk.Graph:
        g = hawk.Graph()
        if flag:
            axpb(x, a=2.0, b=0.0, y=y)
        else:
            axpb(x, a=-1.0, b=0.0, y=y)
        g.result = y
        return g

    return pick(X, flag, np.zeros(N))


def test_an_if_on_a_cell_takes_each_arm_at_replay_and_not_at_build():
    @hawk.Graph.builder
    def gated(x, flag, y) -> hawk.Graph:
        g = hawk.Graph()
        if flag:
            axpb(x, a=5.0, b=0.0, y=y)
        g.result = y
        return g

    @hawk.Graph.builder
    def gated_not(x, flag, y) -> hawk.Graph:
        g = hawk.Graph()
        if not flag:
            axpb(x, a=5.0, b=0.0, y=y)
        g.result = y
        return g

    for build, negated in ((gated, False), (gated_not, True)):
        flag, y = cell(1), np.zeros(N)
        graph = build(X, flag, y)
        for value in (1, 0, 1):
            flag[0], y[:] = value, 0.0
            close(graph(), 5.0 * X if (value != 0) != negated else 0.0)


def test_if_else_and_elif_take_exactly_one_arm_each_replay():
    @hawk.Graph.builder
    def chain(x, first, second, y) -> hawk.Graph:
        g = hawk.Graph()
        if first:
            axpb(x, a=1.0, b=0.0, y=y)
        elif second:
            axpb(x, a=2.0, b=0.0, y=y)
        else:
            axpb(x, a=3.0, b=0.0, y=y)
        g.result = y
        return g

    first, second, y = cell(0), cell(0), np.zeros(N)
    graph = chain(X, first, second, y)
    for (a, b), factor in (((1, 0), 1.0), ((0, 1), 2.0), ((0, 0), 3.0), ((1, 1), 1.0)):
        first[0], second[0] = a, b
        close(graph(), factor * X)


def test_a_python_condition_is_decided_once_at_build():
    for flag, factor in ((True, 2.0), (False, -1.0)):
        graph = _if_else_graph(flag)
        close(graph(), factor * X)
        assert "axpb" in repr(graph) and repr(graph).count("axpb") == 1


def test_a_condition_that_is_not_one_integer_cell_is_refused_by_name():
    @hawk.Graph.builder
    def bad(x, flag) -> hawk.Graph:
        g = hawk.Graph()
        if flag:
            axpb(x, a=1.0, b=0.0)
        return g

    for flag in (np.zeros(2, dtype=np.uint32), np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.int64)):
        with pytest.raises(HawkError, match="a one-element int32 or uint32 array"):
            bad(X, flag)


def test_an_if_chain_may_not_mix_a_cell_with_a_python_value():
    @hawk.Graph.builder
    def mixed(x, flag) -> hawk.Graph:
        g = hawk.Graph()
        if flag:
            axpb(x, a=1.0, b=0.0)
        elif len(x) > 3:
            axpb(x, a=2.0, b=0.0)
        return g

    with pytest.raises(HawkError, match="mixes a cell and a Python value"):
        mixed(X, cell(0))


def test_a_name_assigned_from_a_call_in_an_arm_and_read_after_it_is_refused_by_name():
    @hawk.Graph.builder
    def leak(x, flag) -> hawk.Graph:
        g = hawk.Graph()
        if flag:
            y = axpb(x, a=1.0, b=0.0)
        g.result = y
        return g

    with pytest.raises(HawkError, match=r"y is assigned from a kernel call inside graph control flow"):
        leak(X, cell(1))


def test_a_break_continue_or_return_in_a_recorded_body_is_refused_by_name():
    @hawk.Graph.builder
    def leaves(x, flag) -> hawk.Graph:
        g = hawk.Graph()
        if flag:
            axpb(x, a=1.0, b=0.0)
            return g
        return g

    with pytest.raises(HawkError, match="`break`, `continue` and `return` cannot be recorded"):
        leaves(X, cell(1))
    assert repr(leaves(X, 1)) == "<hawk.Graph built: axpb>"      # a Python condition keeps `return` as Python has it


def test_a_skipguard_carries_an_element_and_a_baseline_into_the_conditional():
    @hawk.Graph.builder
    def guarded(x, flags, count, base, y) -> hawk.Graph:
        g = hawk.Graph()
        if hawk.SkipGuard(flags, 2):
            axpb(x, a=2.0, b=0.0, y=y)
        if not hawk.SkipGuard(count, 0, base, 1):
            axpb(x, a=3.0, b=0.0, y=y)
        g.result = y
        return g

    flags = np.array([9, 9, 0, 9], dtype=np.uint32)
    count, base, y = cell(3), np.array([0, 3], dtype=np.uint32), np.zeros(N)
    graph = guarded(X, flags, count, base, y)
    for flag, value, factor in ((0, 3, 3.0), (1, 3, 3.0), (1, 4, 2.0), (0, 4, 0.0)):
        flags[2], count[0], y[:] = flag, value, 0.0
        close(graph(), factor * X)
    flags[2] = 5
    guard = hawk.SkipGuard.intent(flags, 2)
    assert guard.evaluate() is True and guard.negated().evaluate() is False
    assert hawk.SkipGuard.nonzero(cell(0)).evaluate() is False
    assert hawk.SkipGuard(count, 0, base, 1, negate=True).evaluate() is False


def test_a_while_on_a_cell_repeats_until_the_cell_clears_and_a_for_with_a_break_is_capped():
    @hawk.Graph.builder
    def countdown(x, c, y) -> hawk.Graph:
        g = hawk.Graph()
        for _ in range(4):
            if not c:
                break
            grow(x, y=y)
            tick(c)
        g.result = y
        return g

    @hawk.Graph.builder
    def uncapped(x, c, y) -> hawk.Graph:
        g = hawk.Graph()
        while c:
            grow(x, y=y)
            tick(c)
        g.result = y
        return g

    for start in (0, 2, 4, 6):
        c, y = cell(start), np.zeros(N)
        out = countdown(X, c, y)()
        np.testing.assert_allclose(out, min(start, 4) * X, rtol=1e-14)
        assert c[0] == max(start - 4, 0)                    # the cap stops the loop, not the cell
    with pytest.warns(UserWarning, match="a graph loop with no cap"):
        graph = uncapped(X, cell(3), np.zeros(N))
    np.testing.assert_allclose(graph(), 3 * X, rtol=1e-14)


MATCH_SOURCE = """
import hawk
from hawk import Mutable, Param, Scalar


@hawk.kernel
def axpb(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


@hawk.Graph.builder
def switch(x, c, y) -> hawk.Graph:
    g = hawk.Graph()
    match c:
        case 0:
            axpb(x, a=1.0, b=0.0, y=y)
        case 1:
            axpb(x, a=2.0, b=0.0, y=y)
        case 2:
            axpb(x, a=3.0, b=0.0, y=y)
    g.result = y
    return g


@hawk.Graph.builder
def labels(x, c) -> hawk.Graph:
    g = hawk.Graph()
    match c:
        case 3:
            axpb(x, a=1.0, b=0.0)
    return g


@hawk.Graph.builder
def python_match(x, mode) -> hawk.Graph:
    g = hawk.Graph()
    match mode:
        case 0:
            axpb(x, a=1.0, b=0.0)
        case _:
            axpb(x, a=2.0, b=0.0)
            axpb(x, a=2.0, b=0.0)
    return g
"""


@pytest.mark.skipif(sys.version_info < (3, 10), reason="the match statement is Python 3.10")
def test_a_match_on_a_cell_is_one_switch_and_takes_the_arm_of_its_value(tmp_path):
    path = tmp_path / "match_builders.py"
    path.write_text(MATCH_SOURCE)
    spec = importlib.util.spec_from_file_location("match_builders", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c, y = cell(0), np.zeros(N)
    graph = module.switch(X, c, y)
    for value in (2, 0, 1):
        c[0] = value
        close(graph(), (value + 1.0) * X)
    with pytest.raises(HawkError, match="cases 0, 1, 2, ... in order and no `case _`"):
        module.labels(X, cell(3))
    assert repr(module.python_match(X, 0)).count("axpb") == 1
    assert repr(module.python_match(X, 5)).count("axpb") == 2


# 5. skippable and repeat_while over a SkipGuard

def test_skippable_runs_its_step_only_while_the_guard_holds():
    plan = hawk.deploy(axpb)
    flag, y = cell(1), np.zeros(N)
    step = plan.bind(x=X, a=4.0, b=0.0, y=y)
    graph = hawk.Graph().add(hawk.skippable(step, hawk.SkipGuard.nonzero(flag))).build()
    for value in (1, 0, 1):
        flag[0], y[:] = value, 0.0
        graph.launch()
        close(y, 4.0 * X if value else 0.0)
    with pytest.raises(HawkError, match="the guard is a hawk.SkipGuard"):
        hawk.skippable(step, flag)
    with pytest.raises(HawkError, match="takes a bound plan, a built Graph"):
        hawk.Graph().add(hawk.skippable(lambda: None, hawk.SkipGuard.nonzero(flag)))


def test_repeat_while_runs_its_steps_while_the_guard_holds_up_to_the_cap():
    grow_plan, tick_plan = hawk.deploy(grow), hawk.deploy(tick)
    for start, cap, expect in ((3, None, 3), (5, 2, 2), (0, 4, 0)):
        c, y = cell(start), np.zeros(N)
        body = [grow_plan.bind(x=X, y=y), tick_plan.bind(c=c)]
        graph = hawk.Graph().add(hawk.repeat_while(body, hawk.SkipGuard.nonzero(c), cap)).build()
        graph.launch()
        np.testing.assert_allclose(y, expect * X, rtol=1e-14)
        assert c[0] == start - expect
    for cap in (0, -1, True, 2.0):
        with pytest.raises(HawkError, match="max_iters is a whole number of 1 or more"):
            hawk.repeat_while(body, hawk.SkipGuard.nonzero(c), cap)


def test_skippable_and_repeat_while_take_a_function_inside_a_builder_and_nest():
    @hawk.Graph.builder
    def nested(x, c, flag, y) -> hawk.Graph:
        g = hawk.Graph()
        inner = hawk.skippable(lambda: grow(x, y=y), hawk.SkipGuard.nonzero(flag))
        g.add(hawk.repeat_while([inner, lambda: tick(c)], hawk.SkipGuard.nonzero(c), 6))
        g.result = y
        return g

    c, flag, y = cell(3), cell(1), np.zeros(N)
    graph = nested(X, c, flag, y)
    np.testing.assert_allclose(graph(), 3 * X, rtol=1e-14)
    c[0], flag[0], y[:] = 4, 0, 0.0
    np.testing.assert_allclose(graph(), 0.0)


# 6. Graph.bind swaps the arrays of a builder's parameters with no new record

def test_bind_swaps_same_shape_arrays_and_the_next_replay_reads_them():
    @hawk.Graph.builder
    def affine(x, y, flag) -> hawk.Graph:
        g = hawk.Graph()
        axpb(x, a=2.0, b=1.0, y=y)
        g.result = y
        return g

    x0, y0, flag = X.copy(), np.zeros(N), cell(1)
    graph = affine(x0, y0, flag)
    close(graph(), 2.0 * X + 1.0)
    x1, y1 = X + 10.0, np.zeros(N)
    assert graph.bind(x=x1, y=y1) is graph
    out = graph()
    assert out is y1
    close(y1, 2.0 * (X + 10.0) + 1.0)
    np.testing.assert_allclose(y0, 2.0 * X + 1.0, rtol=1e-14)    # the old output is left as it was
    for bad, match in (({"x": np.zeros(N - 1)}, "must keep its shape and dtype"), ({"x": X.astype(np.float32)}, "must keep its shape"),
                       ({"nope": X}, "is not a parameter of the builder")):
        with pytest.raises(HawkError, match=match):
            graph.bind(**bad)


def test_bind_refuses_a_python_value_a_condition_cell_and_a_graph_made_by_hand():
    @hawk.Graph.builder
    def gated(x, flag, scale) -> hawk.Graph:
        g = hawk.Graph()
        if flag:
            axpb(x, a=scale, b=0.0)
        return g

    graph = gated(X, cell(1), 2.0)
    with pytest.raises(HawkError, match="'scale' is a Python value, decided at build"):
        graph.bind(scale=3.0)
    with pytest.raises(HawkError, match="'flag' is a condition cell, recorded at build"):
        graph.bind(flag=cell(0))
    with pytest.raises(HawkError, match="only a graph made by a builder has parameters"):
        hawk.Graph().bind(x=X)


# 7. a built graph is a step of another

def test_a_built_graph_added_as_a_step_runs_its_launches_where_it_stands_and_an_unbuilt_one_is_refused():
    plan = hawk.deploy(axpb)
    y, w = np.zeros(N), np.zeros(N)
    child = hawk.Graph().add(plan.bind(x=X, a=2.0, b=1.0, y=y)).build()
    parent = hawk.Graph().add(child).add(hawk.deploy(twice).bind(y=y, w=w)).build()
    parent.launch()
    close(w, 2.0 * (2.0 * X + 1.0))
    with pytest.raises(HawkError, match="a graph added as a step is built first"):
        hawk.Graph().add(hawk.Graph())
