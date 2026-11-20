# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Seeing a built graph: ``node_labels``, ``num_nodes``, ``edges``, ``schedule``, ``to_mermaid``, ``to_dot``, ``show`` and ``str``.
hawk only renders what the library describes; the rows pin the structure it gives for a fork with a ``wait`` followed by an
if / else, the same one the library's own rows describe."""

from __future__ import annotations

import io

import numpy as np
import pytest

import hawk
from hawk import HawkError, Mutable, Param, Scalar
from hawk.graph import Edge

N = 11
X = np.linspace(1.0, 2.0, N)


@hawk.kernel
def axpb(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


@hawk.kernel
def twice(y: Scalar, w: Mutable[Scalar]):
    w = y * 2.0


@hawk.Graph.builder
def fork_wait_if(x, flag, y, w, u) -> hawk.Graph:
    g = hawk.Graph()
    with g.parallel() as p:
        with p.branch():
            axpb(x, a=3.0, b=1.0, y=y)
        with p.branch() as b:
            b.wait(y)
            twice(y, w=w)
    if flag:
        axpb(x, a=1.0, b=0.0, y=u)
    else:
        axpb(x, a=2.0, b=0.0, y=u)
    return g


@pytest.fixture
def graph():
    return fork_wait_if(X.copy(), np.array([1], dtype=np.uint32), np.zeros(N), np.zeros(N), np.zeros(N))


def test_the_labels_count_and_levels_of_a_fork_with_a_wait_and_an_if_else(graph):
    assert graph.node_labels() == ["fork", "branch", "axpb", "branch", "wait y", "twice", "join", "if / else flag[0]", "then",
                                   "axpb#2", "else", "axpb#3"]
    assert graph.num_nodes() == 12
    assert graph.schedule() == ((0,), (1, 3), (2,), (4,), (5,), (6,), (7,), (8, 10), (9, 11))


def test_the_edges_are_the_librarys_hazards_waits_joins_and_orderings(graph):
    edges = graph.edges()
    assert all(isinstance(edge, Edge) for edge in edges)
    assert edges == (Edge(0, 1, None, "ORDER"), Edge(1, 2, None, "ORDER"), Edge(0, 3, None, "ORDER"), Edge(3, 4, None, "ORDER"),
                     Edge(2, 4, "y", "WAIT"),            # the writer in the first branch to the wait of the second
                     Edge(3, 5, None, "ORDER"), Edge(4, 5, None, "ORDER"), Edge(2, 6, None, "JOIN"), Edge(5, 6, None, "JOIN"),
                     Edge(6, 7, None, "ORDER"), Edge(7, 8, None, "ORDER"), Edge(8, 9, None, "ORDER"), Edge(7, 10, None, "ORDER"),
                     Edge(10, 11, None, "ORDER"))


def test_a_sequence_of_launches_has_the_hazard_edges_the_library_derives_from_its_planes():
    y, w = np.zeros(N), np.zeros(N)
    plan, twice_plan = hawk.deploy(axpb), hawk.deploy(twice)
    pipeline = (hawk.GraphPipeline().add(plan.bind(x=X, a=1.0, b=0.0, y=y)).add(twice_plan.bind(y=y, w=w))
                .add(plan.bind(x=X, a=2.0, b=0.0, y=y)).build())
    # axpb binds y as a Mutable, which the unit reads and writes: its second launch is after the first in all three ways
    assert {(e.src, e.dst, e.plane, e.kind) for e in pipeline.edges()} == {
        (0, 1, "y", "RAW"), (1, 2, "y", "WAR"), (0, 2, "y", "RAW"), (0, 2, "y", "WAR"), (0, 2, "y", "WAW")}
    assert pipeline.schedule() == ((0,), (1,), (2,))
    assert pipeline.to_mermaid().count("-.->|y WAW|") == 1 and "n1 -.->|y WAR| n2" in pipeline.to_mermaid()


def test_text_view_indents_by_enclosure_and_lists_planes_and_edges(graph):
    text = str(graph)
    lines = text.splitlines()
    assert lines[0] == "hawk.Graph  4 launches · 12 nodes · 9 levels"
    assert lines[1].split() == ["0", "L0", "fork"]
    assert lines[3].split() == ["2", "L2", "axpb", "(y:rw,", "x:r)"]
    assert lines[9].split() == ["8", "L7", "then"]
    depth = {int(line.split()[0]): (len(line) - len(line.lstrip(" 0123456789L"))) for line in lines[1:13]}
    assert depth[0] == depth[6] == depth[7] and depth[2] > depth[1] > depth[0]       # the join and the if sit at the top
    assert lines[13] == "edges" and lines[18].split() == ["2", "->", "4", "WAIT", "y"]
    out = io.StringIO()
    graph.show(file=out)
    assert out.getvalue() == text


def test_mermaid_and_dot_draw_each_node_and_each_edge_by_kind(graph):
    mermaid = graph.to_mermaid().splitlines()
    assert mermaid[0] == "flowchart TD"
    assert sum(line.lstrip().startswith("n") and "[" in line for line in mermaid) == 12
    assert '    n4["4 wait y"]' in mermaid
    assert "    n2 ==>|y WAIT| n4" in mermaid
    assert "    n2 --> n6" in mermaid and "    n6 -.-> n7" in mermaid
    dot = graph.to_dot().splitlines()
    assert dot[0] == "digraph hawk_graph {" and dot[-1] == "}"
    assert '    n7 [label="7 if / else flag[0]", shape=diamond];' in dot
    assert '    n2 [label="2 axpb", shape=box];' in dot
    assert '    n2 -> n4 [style=bold, label="y"];' in dot
    assert '    n2 -> n6 [style=solid];' in dot


def test_a_while_a_negated_if_and_a_switch_are_labelled_by_their_test():
    @hawk.Graph.builder
    def conditional(x, flag, y) -> hawk.Graph:
        g = hawk.Graph()
        if not flag:
            axpb(x, a=1.0, b=0.0, y=y)
        for _ in range(4):
            if not flag:
                break
            axpb(x, a=1.0, b=0.0, y=y)
        return g

    graph = conditional(X.copy(), np.array([1], dtype=np.uint32), np.zeros(N))
    assert graph.node_labels() == ["if not flag[0]", "then", "axpb", "while flag[0] (at most 4)", "body", "axpb#2"]


def test_a_graph_that_is_not_built_or_is_closed_is_refused_by_name(graph):
    unbuilt = hawk.Graph().add(hawk.deploy(axpb).bind(x=X, a=1.0, b=0.0, y=np.zeros(N)))
    for verb in ("node_labels", "num_nodes", "edges", "schedule", "to_mermaid", "to_dot", "__str__"):
        with pytest.raises(HawkError, match="not built, or is closed; the library describes a recording"):
            getattr(unbuilt, verb)()
    graph.close()
    with pytest.raises(HawkError, match="not built, or is closed"):
        graph.edges()


def test_a_plain_pipeline_is_described_too():
    pipeline = hawk.GraphPipeline().add(hawk.deploy(axpb).bind(x=X, a=1.0, b=0.0, y=np.zeros(N))).build()
    assert pipeline.node_labels() == ["axpb"] and pipeline.num_nodes() == 1 and pipeline.edges() == ()
    assert pipeline.schedule() == ((0,),) and str(pipeline).splitlines()[0] == "hawk.Graph  1 launches · 1 nodes · 1 levels"
