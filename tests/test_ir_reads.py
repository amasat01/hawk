# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""What ``hawk._ir`` states about its nodes, contexts and refusals, with oracles that are written out.

Each row builds a small graph through the library and compares what is read back with a number or a text stated here: the
rows of cells a plane count names, the scalars a type holds, the order of a walk with children first, the nodes a foreign
context refuses, the refusal a library error becomes.
"""

from __future__ import annotations

import pytest

from hawk import HawkError, _ir

SCALAR = ((), "f64", None)


@pytest.fixture
def graph():
    """``(context, a, b, s, t)`` with ``s = a + b`` and ``t = s * s``: one shared sub-expression."""
    context = _ir.new_context()
    a = _ir.leaf(context, "vocab_read", "per_sample", "a", SCALAR)
    b = _ir.leaf(context, "vocab_read", "per_sample", "b", SCALAR)
    s = _ir.make("add", [a, b])
    return context, a, b, s, _ir.make("mul", [s, s])


def test_a_cell_of_one_column_has_one_plane_and_a_cell_of_three_has_three_numbered_ones():
    assert _ir.planes("g", 1) == ("g",)
    assert _ir.planes("g", 3) == ("g_0", "g_1", "g_2")


@pytest.mark.parametrize("shape,count", [((), 1), ((3,), 3), ((2, 4), 8)])
def test_the_scalar_count_of_a_type_is_the_product_of_its_extents(shape, count):
    assert _ir.element_count(_ir.TensorType(shape, "f64")) == count


def test_a_type_equals_a_hand_built_one_only_with_a_tuple_shape(graph):
    _context, a, *_ = graph
    assert _ir.type_of(a) == _ir.TensorType((), "f64")
    assert _ir.TensorType((3,), "f64") != _ir.TensorType([3], "f64")


def test_a_builder_without_an_operand_asks_for_the_context():
    with pytest.raises(TypeError, match="pass context="):
        _ir.make_node("LoopIndex", SCALAR)


def test_post_order_lists_children_first_each_once_and_a_stop_node_is_a_leaf_of_the_walk(
    graph,
):
    _context, a, b, s, t = graph
    assert _ir.post_order([t]) == [a, b, s, t]
    assert _ir.post_order([t], stop=[s]) == [s, t]


def test_a_node_of_another_context_is_refused_at_once(graph):
    _context, a, *_ = graph
    stranger = _ir.const(_ir.new_context(), 1.0, SCALAR)
    with pytest.raises(TypeError, match="another Ctx"):
        _ir.make("add", [a, stranger])


def test_adopt_copies_a_foreign_root_with_its_graph_and_keeps_a_root_that_is_already_there(
    graph,
):
    _context, _a, _b, _s, t = graph
    home = _ir.new_context()
    own = _ir.leaf(home, "vocab_read", "per_sample", "own", SCALAR)
    copy, kept = _ir.adopt(home, [t, own])
    assert copy.ctx is home and kept == own
    assert len(_ir.post_order([copy])) == 4


def test_a_sink_rebuilt_over_another_value_keeps_its_name_and_a_node_that_commits_nothing_is_refused(
    graph,
):
    _context, _a, _b, s, t = graph
    sink = _ir.make_node("Assign", SCALAR, [s], name="y")
    assert _ir.with_value(sink, t).field_text("name") == "y"
    with pytest.raises(HawkError):
        _ir.with_value(s, s)


def test_a_loop_value_reads_back_its_loop_and_slot_and_a_slot_outside_the_loop_is_refused():
    context = _ir.new_context()
    index = _ir.make_node(
        "LoopIndex",
        ((), "int32", None),
        context=context,
        depth=0,
        name="_",
        bounds=(0, 3),
    )
    carry = _ir.make_node(
        "LoopCarry", SCALAR, context=context, depth=0, slot=0, name="c"
    )
    first = _ir.const(context, 0.0, SCALAR)
    loop = _ir.loop(
        index,
        [carry],
        [first],
        [_ir.make("add", [carry, first])],
        start=0,
        stop=3,
        step=1,
    )
    value = _ir.loop_value(loop, 0)
    assert _ir.loop_value_of(value) == (loop, 0)
    with pytest.raises(HawkError, match="outside the loop"):
        _ir.loop_value(loop, 5)


def test_a_library_refusal_reaches_the_caller_as_a_hawk_error_with_the_library_text_and_its_code(
    graph,
):
    context, *_ = graph
    factor = _ir.leaf(
        context, "vocab_read", "vec_in", "l", ((2, 2), "f64", "lu_factor")
    )
    with pytest.raises(HawkError) as refused:
        _ir.as_matrix(factor)
    with pytest.raises(ValueError) as library:
        context.as_matrix(factor)
    assert str(refused.value) == str(library.value) and isinstance(
        refused.value.__cause__, ValueError
    )
    assert hasattr(refused.value, "code")


def test_an_operation_the_vocabulary_does_not_hold_is_refused_naming_it_and_the_vocabulary(
    graph,
):
    _context, a, *_ = graph
    with pytest.raises(
        HawkError, match=r"unknown op kind 'nosuchop'; the vocabulary is \('add'"
    ):
        _ir.make("nosuchop", [a])
