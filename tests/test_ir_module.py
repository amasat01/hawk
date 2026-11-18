# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk._ir`` against the Python IR package: one graph built on both sides answers alike.

The oracle is ``raptor.ir`` (the package ``_ir`` replaces). A graph is built twice, once with the
old classes and once with ``_ir``, and the two are compared as plain data: class, type, operands,
fields. Where the old package cannot build a kind (a dialect's), the test states the library's own
documented behaviour instead.
"""

from __future__ import annotations

import json
from fractions import Fraction

import pytest
from raptor import ir as old
from raptor.ir import access as old_access
from raptor.ir import cells as old_cells
from raptor.ir import compound as old_compound
from raptor.ir import linalg as old_linalg
from raptor.ir import loops as old_loops
from raptor.ir import nodes as old_nodes
from raptor.ir import ops as old_ops
from raptor.ir import types as old_types
from raptor.ir import walk as old_walk
from raptor.wings.base import block as old_block
from raptor.wings.base import cells as old_cells_wing
from raptor.wings.base import logs as old_logs
from raptor.wings.base import slotwise as old_slotwise

from hawk import _ir
from hawk._errors import HawkError

F64 = ((), "f64", None)
I32 = ((), "i32", None)
OLD_F64 = old_types.TensorType((), "f64")
OLD_I32 = old_types.TensorType((), "i32")


def old_shape(node):
    """Return the old node as plain data: its class name, its type and its operands, recursively."""
    t = node.ttype
    return (
        type(node).__name__,
        (tuple(t.shape), t.dtype, t.tag),
        tuple(old_shape(o) for o in node.operands),
    )


def new_shape(node):
    """Return the ``_ir`` node as the same plain data."""
    shape, dtype, tag = node.ttype
    return (node.cls, (tuple(shape), dtype, tag), tuple(new_shape(o) for o in node.operands))


@pytest.fixture
def ctx():
    """Make a fresh context."""
    return _ir.new_context()


def twin_sum(ctx):
    """Return the graph ``y = (x + 1.0) < 2.0 ? x .

    X + 1.0``, written to the old package and to ``_ir``.
    """
    ox = old.Leaf("vocab_read", "vec_in", "x", OLD_F64)
    oone = old.Const(1.0, OLD_F64)
    osum = old.make("add", [ox, oone])
    osel = old.Select(old.make("lt", [osum, old.Const(2.0, OLD_F64)]), ox, osum, OLD_F64)
    oy = old.Assign("y", osel)
    nx = _ir.leaf(ctx, "vocab_read", "vec_in", "x", F64)
    none = _ir.const(ctx, 1.0, F64)
    nsum = _ir.make("add", [nx, none])
    cond = _ir.make("lt", [nsum, _ir.const(ctx, 2.0, F64)])
    nsel = _ir.make_node("Select", F64, [cond, nx, nsum])
    ny = _ir.make_node("Assign", F64, [nsel], name="y")
    return (ox, osum, osel, oy), (nx, nsum, nsel, ny)


def twin_loop(ctx):
    """Make a running sum over three trips (an accumulator) and a product (a recurrence).

    On both sides.
    """
    ox = old.Leaf("vocab_read", "vec_in", "x", OLD_F64)
    oi, oc = old.LoopIndex(0, "k", OLD_I32), old.LoopCarry(0, 0, "s", OLD_F64)
    nx = _ir.leaf(ctx, "vocab_read", "vec_in", "x", F64)
    ni = _ir.make_node("LoopIndex", I32, context=ctx, depth=0, name="k", bounds=None)
    nc = _ir.make_node("LoopCarry", F64, context=ctx, depth=0, slot=0, name="s")
    out = {}
    for label, kind in (("acc", "add"), ("rec", "mul")):
        old_loop = old_loops.build_loop(
            oi, [oc], [old.Const(0.0, OLD_F64)], [old.make(kind, [oc, ox])], start=0, stop=3, step=1
        )
        new_loop = _ir.loop(
            ni,
            [nc],
            [_ir.const(ctx, 0.0, F64)],
            [_ir.make(kind, [nc, nx])],
            start=0,
            stop=3,
            step=1,
        )
        out[label] = (old_loop, new_loop)
    return out, (ox, oi, oc), (nx, ni, nc)


# ---- the constants ----

OLD_CONSTANT_HOMES = (old_nodes, old_access, old_types, old_cells)


@pytest.mark.parametrize(
    "name",
    [
        "COMPARE", "LOGICAL", "PREDICATE", "CONTRACTS", "SCATTER_CONTRACTS", "KEYED_AUTO",
        "LEAF_ROLES",
        "INT_PLANE_DTYPES", "NAMESPACE_SEP", "PARAM_CELL", "RAW_KIND", "CELL_PLANE", "RAW_LANGUAGE",
        "GATHER_ROLES", "PER_SAMPLE_ROLES", "ACCESS_CLASSES", "COMPANION_SUFFIX", "ADJOINT_PREFIX",
    ],
)  # fmt: skip
def test_the_constant_equals_the_old_value(name):
    """Each of the 18 constants the library answers equals the old package's value.

    Tuple or string.
    """
    home = next(m for m in OLD_CONSTANT_HOMES if hasattr(m, name))
    assert getattr(_ir, name) == getattr(home, name)


# ---- plain records and the rules local to the module ----


@pytest.mark.parametrize("name", ["INT_BINARY", "INT_UNARY"])
def test_the_integer_op_lists_equal_the_old_ones(name):
    """The integer op kind lists equal the old package's."""
    assert getattr(_ir, name) == getattr(old_ops, name)


def test_the_type_and_slot_records_read_as_the_old_dataclasses_do():
    """TensorType and Slot hold the old fields in the old order.

    And a node's type reads as a TensorType.
    """
    t = _ir.TensorType((2,), "f64", "xs")
    ot = old_types.TensorType((2,), "f64", "xs")
    assert (t.shape, t.dtype, t.tag) == (ot.shape, ot.dtype, ot.tag)
    assert _ir.TensorType((), "f64").tag is None
    slot, oslot = _ir.Slot("vec_in", "x", t), old_types.Slot("vec_in", "x", ot)
    assert (slot.role, slot.name, tuple(slot.ttype)) == (
        oslot.role,
        oslot.name,
        (ot.shape, ot.dtype, ot.tag),
    )


def test_planes_and_element_count_equal_the_old_rules():
    """Return the row planes of a cell and the scalar count of a type follow the old rules."""
    for cell, columns in (("g", 1), ("g", 3)):
        assert _ir.planes(cell, columns) == old_cells.planes(cell, columns)
    for shape in ((), (3,), (2, 4)):
        assert _ir.element_count(_ir.TensorType(shape, "f64")) == old_nodes.element_count(
            old_types.TensorType(shape, "f64")
        )


def test_compensates_equals_the_old_rule(ctx):
    """Which contracts commit through a companion plane."""
    for contract in ("auto", "compensated", "keyed_auto", "ordered", "fast"):
        assert _ir.compensates(ctx, contract) == old_nodes.compensates(contract)


def test_loop_class_labels_and_record_equal_the_old_ones():
    """Return the two class labels and the record keep the old names and defaults."""
    assert (_ir.ACCUMULATOR, _ir.RECURRENCE) == (old_loops.ACCUMULATOR, old_loops.RECURRENCE)
    assert _ir.LoopClass(_ir.RECURRENCE) == (old_loops.LoopClass(old_loops.RECURRENCE).name, ())


# ---- builders and kind tests: the node families ----


def test_builders_make_what_the_old_classes_make(ctx):
    """Leaf, constant, op, select and sink built through ``_ir`` match the old nodes in class.

    Type and operands.
    """
    old_nodes_, new_nodes = twin_sum(ctx)
    assert [new_shape(n) for n in new_nodes] == [old_shape(n) for n in old_nodes_]


def test_an_op_over_other_types_infers_the_old_type(ctx):
    """Return the result type of ops over vectors and integers is the old inference."""
    ov = old.Leaf("vocab_read", "vec_in", "v", old_types.TensorType((3,), "f64"))
    nv = _ir.leaf(ctx, "vocab_read", "vec_in", "v", ((3,), "f64", None))
    for kind in ("add", "mul"):
        assert new_shape(_ir.make(kind, [nv, nv])) == old_shape(old.make(kind, [ov, ov]))
    assert _ir.type_of(_ir.make("add", [nv, nv])) == (_ir.TensorType((3,), "f64", None))


def test_dispatch_at_and_the_other_node_classes_build_as_the_old_ones_do(ctx):
    """Dispatch, At, RoleConst, SampleSum.

    WideWrite-free sinks and sample reads match the old nodes.
    """
    ox = old.Leaf("vocab_read", "vec_in", "x", OLD_F64)
    nx = _ir.leaf(ctx, "vocab_read", "vec_in", "x", F64)
    oone, none = old.Const(1.0, OLD_F64), _ir.const(ctx, 1.0, F64)
    osel_i, nsel_i = old.Const(1, OLD_I32), _ir.const(ctx, 1, I32)
    odis = old.Dispatch(osel_i, [ox, oone], OLD_F64)
    ndis = _ir.make_node("Dispatch", F64, [nsel_i, nx, none])
    oplane = old.Leaf("table_read", "lookup", "t", old_types.TensorType((4,), "f64"))
    nplane = _ir.leaf(ctx, "table_read", "lookup", "t", ((4,), "f64", None))
    oat = old.At(oplane, old.RoleConst(0, "other", OLD_I32), OLD_F64)
    nat = _ir.make_node(
        "At", F64, [nplane, _ir.make_node("RoleConst", I32, context=ctx, literal=0, label="other")]
    )
    osum = old.SampleSum("s", ox)
    nsum = _ir.make_node("SampleSum", F64, [nx], name="s", op="sum", cell="s", columns=1)
    oidx, nidx = old.SampleIndex(), _ir.make_node("SampleIndex", I32, context=ctx)
    pairs = [(odis, ndis), (oat, nat), (osum, nsum), (oidx, nidx)]
    assert [new_shape(n) for _, n in pairs] == [old_shape(o) for o, _ in pairs]


def test_a_reduce_over_samples_states_its_feather(ctx):
    """ReduceSamples.

    The old class has the feather and ``_ir`` reads it so, with the fields the old node holds.
    """
    ox = old.Leaf("vocab_read", "vec_in", "x", OLD_F64)
    nx = _ir.leaf(ctx, "vocab_read", "vec_in", "x", F64)
    ored = old_nodes.ReduceSamples("sum", ox)
    nred = _ir.make_node(
        "ReduceSamples", F64, [nx], op="sum", contract="auto", key=None, count=None, kept=None
    )
    assert new_shape(nred) == old_shape(ored)
    assert (
        _ir.has_feather(nred, "reduces_samples")
        == old_nodes.has_feather(ored, "reduces_samples")
        is True
    )


def test_the_kind_test_agrees_with_isinstance_over_every_family(ctx):
    """``is_a`` with the library's kind name equals ``isinstance`` of the old class.

    Node by node.
    """
    (ox, osum, osel, oy), (nx, nsum, nsel, ny) = twin_sum(ctx)
    loops, _, _ = twin_loop(ctx)
    table = [
        ("leaf", old.Leaf), ("op", old.Op), ("select", old.Select), ("assign", old.Assign),
        ("loop", old.Loop), ("const", old.Const), ("sink", old.Sink),
    ]  # fmt: skip
    pairs = list(zip((ox, osum, osel, oy), (nx, nsum, nsel, ny))) + [loops["acc"]]
    for old_node, new_node in pairs:
        for kind, cls in table:
            assert _ir.is_a(new_node, kind) == isinstance(old_node, cls), (
                kind,
                type(old_node).__name__,
            )
        assert _ir.is_a(new_node, "no_such_kind", "leaf", "op", "select", "assign", "loop") is True


def test_the_feathers_equal_the_old_ones_node_by_node(ctx):
    """``has_feather`` over every feather name equals the old answer on every family of the graph.

    ``has_feather`` over every feather name equals the old answer on every family of the graph.
    """
    (ox, osum, osel, oy), (nx, nsum, nsel, ny) = twin_sum(ctx)
    for old_node, new_node in zip((ox, osum, osel, oy), (nx, nsum, nsel, ny)):
        for feather in old_nodes.FEATHERS:
            assert _ir.has_feather(new_node, feather) == old_nodes.has_feather(old_node, feather), (
                feather
            )
    with pytest.raises(HawkError):
        _ir.has_feather(nx, "no_such_feather")


def test_a_dialect_kind_is_declared_built_and_tested_by_the_library(ctx):
    """Check a dialect against the library's documented behaviour (the old loader differs in text).

    A declared kind builds under its CamelCase class name, answers ``is_a`` with its own name and
    states its feathers; an undeclared class is refused by name.
    """
    manifest = {
        "family": "raptor", "name": "probe", "parent": None, "version": "0", "contract": "1",
        "reads": [],
        "dialect": "probe",
        "kinds": [{
            "name": "probe_lane", "family": "terminal", "feathers": ["static_index"],
            "operands": {"min": 0, "max": 0}, "result": ["fixed", "i32", []],
            "spelling": {"dialect": "probe", "name": "probe_lane_value", "args": []},
        }],
        "entries": {}, "emission_modules": [], "runtime_modules": [], "vocabulary": {},
    }  # fmt: skip
    with pytest.raises(HawkError):
        _ir.make_node("ProbeLane", I32, context=ctx)
    _ir.declare_dialect(ctx, json.dumps(manifest), ("probe",))
    lane = _ir.make_node("ProbeLane", I32, context=ctx)
    assert _ir.is_a(lane, "probe_lane") and not _ir.is_a(lane, "leaf")
    assert _ir.has_feather(lane, "static_index") and not _ir.has_feather(lane, "fresh_read")


# ---- parameter cells and sink rewrites ----


def test_the_parameter_cell_read_and_its_names_equal_the_old_ones(ctx):
    """``param_cell`` builds the old read (per cell or not).

    ``is_param_cell`` and the name list agree.
    """
    for per_cell in (False, True):
        oc = old_cells.param_cell("gain", OLD_F64, per_cell)
        nc = _ir.param_cell(ctx, "gain", F64, per_cell)
        assert new_shape(nc) == old_shape(oc)
        assert _ir.is_param_cell(nc) == old_nodes.is_param_cell(oc) == (not per_cell)
    ox = old.Leaf("vocab_read", "vec_in", "x", OLD_F64)
    onames = old_nodes.param_cell_names(
        [old_cells.param_cell(n, OLD_F64) for n in ("b", "a", "b")] + [ox]
    )
    nx = _ir.leaf(ctx, "vocab_read", "vec_in", "x", F64)
    nnames = _ir.param_cell_names([_ir.param_cell(ctx, n, F64) for n in ("b", "a", "b")] + [nx])
    assert nnames == onames == ("b", "a")


def test_swap_replaces_inside_the_committed_values_as_the_old_swap_does(ctx):
    """``swap`` maps nodes in the sink values (with_value rebuilding the sink).

    And a sink with no value is refused.
    """
    (ox, osum, osel, oy), (nx, nsum, nsel, ny) = twin_sum(ctx)
    oz, nz = old.Const(5.0, OLD_F64), _ir.const(ctx, 5.0, F64)
    (oswapped,) = old_cells.swap([oy], {id(ox): oz})
    (nswapped,) = _ir.swap([ny], {nx: nz})
    assert new_shape(nswapped) == old_shape(oswapped)
    assert nswapped.cls == "Assign" and nswapped.field_text("name") == "y"
    assert nsel.operands[1] == nx, "the original graph is kept"


def test_with_value_commits_the_value_and_refuses_a_non_sink_as_the_old_one_does(ctx):
    """Make a sink rebuilt over another value keeps its name.

    A node that commits nothing is refused.
    """
    (ox, osum, osel, oy), (nx, nsum, nsel, ny) = twin_sum(ctx)
    moved, omoved = _ir.with_value(ny, nx), old_cells.with_value(oy, ox)
    assert new_shape(moved) == old_shape(omoved) and moved.field_text("name") == "y"
    with pytest.raises(HawkError):
        _ir.with_value(nx, nx)


def test_as_matrix_projects_a_cholesky_factor_and_refuses_another_factor(ctx):
    """Return the factor projection.

    The old package has no Python twin, so the library's documented answers are the oracle.
    """
    plain = _ir.leaf(ctx, "vocab_read", "vec_in", "x", F64)
    assert _ir.as_matrix(plain) == plain
    chol = _ir.leaf(ctx, "vocab_read", "vec_in", "c", ((2, 2), "f64", "chol_factor"))
    lu = _ir.leaf(ctx, "vocab_read", "vec_in", "l", ((2, 2), "f64", "lu_factor"))
    assert _ir.as_matrix(chol).cls == "Op"
    with pytest.raises(HawkError):
        _ir.as_matrix(lu)


# ---- walks over operands ----


def test_post_order_lists_children_first_each_once_and_honours_stop(ctx):
    """``post_order`` equals the old walk on a shared subexpression.

    With and without a stop node.
    """
    (ox, osum, osel, oy), (nx, nsum, nsel, ny) = twin_sum(ctx)
    assert [new_shape(n) for n in _ir.post_order([ny])] == [
        old_shape(n) for n in old_walk.post_order([oy])
    ]
    assert [new_shape(n) for n in _ir.post_order([ny], stop=[nsum])] == [
        old_shape(n) for n in old_walk.post_order([oy], stop={id(osum)})
    ]


def test_rebuild_and_replace_change_operands_only_like_the_old_ones(ctx):
    """``rebuild`` and ``replace`` give a node over other operands; the original is untouched."""
    (ox, osum, osel, oy), (nx, nsum, nsel, ny) = twin_sum(ctx)
    oz, nz = old.Const(9.0, OLD_F64), _ir.const(ctx, 9.0, F64)
    expected = old_shape(old_walk.rebuild(osum, [oz, ox]))
    assert new_shape(_ir.rebuild(nsum, [nz, nx])) == expected
    assert new_shape(_ir.replace(nsum, operands=[nz, nx])) == expected
    assert old_shape(old_nodes.replace(osum, operands=[oz, ox])) == expected
    assert nsum.operands[1] != nz


def test_substitute_replaces_by_node_and_keeps_the_original(ctx):
    """``substitute`` maps nodes across the graph like the old one keyed by id."""
    (ox, osum, osel, oy), (nx, nsum, nsel, ny) = twin_sum(ctx)
    oz, nz = old.Const(5.0, OLD_F64), _ir.const(ctx, 5.0, F64)
    (oroot,) = old_walk.substitute([oy], {id(ox): oz})
    (nroot,) = _ir.substitute([ny], {nx: nz})
    assert new_shape(nroot) == old_shape(oroot)
    assert _ir.substitute([ny], {}) == (ny,)


def test_structural_keys_partition_the_nodes_as_the_old_keys_do(ctx):
    """Two nodes of one value share a key (nan equals nan, -0.0 is not 0.0), on both sides."""
    values = (1.5, 1.5, 2.5, 0.0, -0.0, float("nan"), float("nan"))
    olds = [old.Const(v, OLD_F64) for v in values]
    news = [_ir.const(ctx, v, F64) for v in values]
    memo: dict = {}
    old_walk.structural_keys(olds, memo)
    nkeys = _ir.structural_keys(news)

    def classes(keys):
        return [[j for j, other in enumerate(keys) if other == key] for key in keys]

    assert classes([memo[id(n)] for n in olds]) == classes([nkeys[n] for n in news])
    assert _ir.structural_keys([]) == {}


# ---- the canonical walk ----


def test_canonical_nodes_gives_the_old_order_and_positions(ctx):
    """Return the node order and the position of every reached node equal the old traversal's."""
    (ox, osum, osel, oy), (nx, nsum, nsel, ny) = twin_sum(ctx)
    oorder, opos = old_walk.canonical_nodes([oy])
    norder, npos = _ir.canonical_nodes([ny])
    assert [new_shape(n) for n in norder] == [old_shape(n) for n in oorder]
    assert sorted(npos.values()) == sorted(opos.values())
    assert [npos[n] for n in norder] == list(range(len(norder)))


def test_a_walk_is_a_number_with_its_context_and_answers_the_dispatch_and_body_questions(ctx):
    """``canonical`` returns a Walk.

    Dispatch nodes and loop-body membership agree with the old listings.
    """
    ox = old.Leaf("vocab_read", "vec_in", "x", OLD_F64)
    nx = _ir.leaf(ctx, "vocab_read", "vec_in", "x", F64)
    odis = old.Dispatch(old.Const(1, OLD_I32), [ox, old.Const(1.0, OLD_F64)], OLD_F64)
    ndis = _ir.make_node("Dispatch", F64, [_ir.const(ctx, 1, I32), nx, _ir.const(ctx, 1.0, F64)])
    oy, ny = old.Assign("y", odis), _ir.make_node("Assign", F64, [ndis], name="y")
    owalk, nwalk = old_walk.canonical([oy]), _ir.canonical([ny])
    assert nwalk.context is ctx
    assert len(_ir.dispatch_nodes(nwalk)) == len(old_walk.dispatch_groups(owalk)) == 1
    assert _ir.dispatch_nodes(nwalk) == (ndis,)
    assert old_walk.loop_body_groups(owalk) == frozenset() and not _ir.in_loop_body(nwalk, ndis)
    loops, _, (_, _, nc) = twin_loop(ctx)
    acc = loops["acc"][1]
    walk = _ir.canonical([_ir.make_node("Assign", F64, [acc], name="y")])
    body = _ir.body_nodes(acc)
    assert body and all(_ir.in_loop_body(walk, n) for n in body)
    assert len(_ir.body_roots(acc)) == 1


# ---- loops ----


def test_the_loop_builder_matches_the_old_boundary_and_operands(ctx):
    """``loop`` computes the same operands.

    (initial values then boundary values) as ``build_loop``.
    """
    loops, _, _ = twin_loop(ctx)
    for old_loop, new_loop in loops.values():
        assert new_shape(new_loop) == old_shape(old_loop)
        assert (new_loop.field_int("start"), new_loop.field_int("stop")) == (
            old_loop.start,
            old_loop.stop,
        )


def test_the_loop_classification_equals_the_old_pair(ctx):
    """Make an accumulator answers its addends, a recurrence none.

    ``addends`` flattens an add chain alike.
    """
    loops, _, _ = twin_loop(ctx)
    oacc, nacc = loops["acc"]
    orec, nrec = loops["rec"]
    oc, nc = old_loops.classify_loop(oacc), _ir.classify_loop(nacc)
    assert nc.name == oc.name == _ir.ACCUMULATOR
    assert [new_shape(a) for a in nc.addends] == [old_shape(a) for a in oc.addends]
    assert _ir.classify_loop(nrec) == _ir.LoopClass(_ir.RECURRENCE)
    assert old_loops.classify_loop(orec).name == _ir.RECURRENCE
    assert [new_shape(a) for a in _ir.addends(nacc.field_nodes("nexts")[0])] == [
        old_shape(a) for a in old_loops.addends(oacc.nexts[0])
    ]


def test_exit_kwargs_read_back_what_the_builder_was_given(ctx):
    """``exit_kwargs`` equals the old dict on a plain loop and on one that breaks.

    Counts a tail and names a mode.
    """
    loops, (ox, oi, oc), (nx, ni, nc) = twin_loop(ctx)
    old_plain, new_plain = loops["acc"]
    assert _ir.exit_kwargs(new_plain) == {
        "exit_cond": None, "exit_values": (), "count": None, "count_tail": False, "dynamic": None,
    }  # fmt: skip
    assert {k: v for k, v in old_loops.exit_kwargs(old_plain).items() if k != "exit_values"} == {
        "exit_cond": None, "count": None, "count_tail": False, "dynamic": None,
    }  # fmt: skip
    ocond = old.make("lt", [ox, old.Const(1.0, OLD_F64)])
    ncond = _ir.make("lt", [nx, _ir.const(ctx, 1.0, F64)])
    onext, nnext = old.make("add", [oc, ox]), _ir.make("add", [nc, nx])
    obreak = old_loops.build_loop(
        oi, [oc], [old.Const(0.0, OLD_F64)], [onext], start=0, stop=3, step=1,
        exit_cond=ocond, exit_values=[onext], count_tail=True, dynamic=None,
    )  # fmt: skip
    nbreak = _ir.loop(ni, [nc], [_ir.const(ctx, 0.0, F64)], [nnext], start=0, stop=3, step=1,
                      exit_cond=ncond, exit_values=[nnext], count_tail=True)  # fmt: skip
    oe, ne = old_loops.exit_kwargs(obreak), _ir.exit_kwargs(nbreak)
    assert new_shape(ne["exit_cond"]) == old_shape(oe["exit_cond"])
    assert [new_shape(v) for v in ne["exit_values"]] == [old_shape(v) for v in oe["exit_values"]]
    assert (ne["count"], ne["count_tail"], ne["dynamic"]) == (
        oe["count"],
        oe["count_tail"],
        oe["dynamic"],
    )


def nested_pair(ctx, leaving):
    """Return the same outer loop on both sides whose body holds an inner loop that does.

    (or does not) leave early.
    """
    _, (ox, oi, oc), (nx, ni, nc) = twin_loop(ctx)
    oj, ojc = old.LoopIndex(1, "j", OLD_I32), old.LoopCarry(1, 0, "t", OLD_F64)
    nj = _ir.make_node("LoopIndex", I32, context=ctx, depth=1, name="j", bounds=None)
    njc = _ir.make_node("LoopCarry", F64, context=ctx, depth=1, slot=0, name="t")
    onext, nnext = old.make("add", [ojc, ox]), _ir.make("add", [njc, nx])
    oextra = (
        dict(exit_cond=old.make("lt", [ox, old.Const(1.0, OLD_F64)]), exit_values=[onext])
        if leaving
        else {}
    )
    nextra = (
        dict(exit_cond=_ir.make("lt", [nx, _ir.const(ctx, 1.0, F64)]), exit_values=[nnext])
        if leaving
        else {}
    )
    oinner = old_loops.build_loop(oj, [ojc], [oc], [onext], start=0, stop=2, step=1, **oextra)
    ninner = _ir.loop(nj, [njc], [nc], [nnext], start=0, stop=2, step=1, **nextra)
    oouter = old_loops.build_loop(
        oi,
        [oc],
        [old.Const(0.0, OLD_F64)],
        [old.make("add", [oc, oinner])],
        start=0,
        stop=3,
        step=1,
    )
    nouter = _ir.loop(ni, [nc], [_ir.const(ctx, 0.0, F64)], [_ir.make("add", [nc, ninner])],
                      start=0, stop=3, step=1)  # fmt: skip
    return oouter, nouter


def test_a_nested_loop_that_leaves_early_is_refused_with_the_old_text(ctx):
    """Refuse a nested break with the old words and a plain nesting with nothing.

    ``refuse_unsupported_nesting`` answers nothing for a plain nesting and refuses a break with the
    old words.
    """
    old_ok, new_ok = nested_pair(ctx, leaving=False)
    old_loops.refuse_unsupported_nesting(old_ok)
    assert _ir.refuse_unsupported_nesting(new_ok) is None
    old_bad, new_bad = nested_pair(ctx, leaving=True)
    with pytest.raises(Exception) as old_refusal:
        old_loops.refuse_unsupported_nesting(old_bad)
    with pytest.raises(HawkError) as new_refusal:
        _ir.refuse_unsupported_nesting(new_bad)
    assert str(new_refusal.value) == str(old_refusal.value)


# ---- index forms ----


def test_the_index_form_equals_the_old_record_on_affine_and_quotient_expressions(ctx):
    """Coefficient, offset, labels.

    Anonymity and symbol equal the old ``IndexForm`` expression by expression.
    """
    oi, ni = (
        old.LoopIndex(0, "k", OLD_I32),
        _ir.make_node("LoopIndex", I32, context=ctx, depth=0, name="k", bounds=None),
    )

    def both(build_old, build_new):
        return build_old(old.Const, oi), build_new(_ir.const, ni)

    def number(c, k):
        return c(k, OLD_I32) if c is old.Const else c(ctx, k, I32)

    cases = [
        lambda c, v, mk: mk("add", [mk("mul", [v, number(c, 4)]), number(c, 2)]),
        lambda c, v, mk: mk(
            "div", [mk("add", [mk("mul", [v, number(c, 4)]), number(c, 2)]), number(c, 2)]
        ),
        lambda c, v, mk: number(c, 7),
        lambda c, v, mk: mk("mul", [v, v]),
        lambda c, v, mk: v,
    ]
    for case in cases:
        o, n = case(old.Const, oi, old.make), case(_ir.const, ni, _ir.make)
        for quotients in (True, False):
            oform = old_access.index_form(o, oi, quotients=quotients)
            nform = _ir.index_form(n, ni, quotients=quotients)
            assert (nform is None) == (oform is None)
            if oform is not None:
                assert tuple(nform) == tuple(oform)
                assert isinstance(nform.coefficient, Fraction)
    assert _ir.index_form(number(_ir.const, 7)) == old_access.index_form(number(old.Const, 7))


# ---- compound quantities ----


def wires(wire_type):
    """Two wires of one quantity."""
    return [wire_type(name="x", role="vec_in"), wire_type(name="m", role="mat_in", shape=(2, 2))]


def test_a_quantity_prefix_wire_names_slots_and_leaves_equal_the_old_ones(ctx):
    """prefix, wire_name, expand, leaves and read follow the old class, instanced or not.

    Packed either way.
    """
    for name, pack in ((None, "auto"), ("p", "tuple")):
        oq = old_compound.Quantity("pos", name, wires=wires(old_types.Wire), pack=pack)
        nq = _ir.Quantity("pos", name, wires=wires(old_types.Wire), pack=pack)
        assert nq.prefix == oq.prefix
        assert [nq.wire_name(w) for w in nq.wires] == [oq.wire_name(w) for w in oq.wires]
        assert [(s.role, s.name) for s in nq.expand()] == [(s.role, s.name) for s in oq.expand()]
        assert [(tuple(s.ttype.shape), s.ttype.dtype) for s in nq.expand()] == [
            (tuple(s.ttype.shape), s.ttype.dtype) for s in oq.expand()
        ]
        nleaves, oleaves = nq.leaves(ctx), oq.leaves()
        assert [new_shape(n) for n in nleaves] == [old_shape(o) for o in oleaves]
        assert [n.field_text("name") for n in nleaves] == [o.name for o in oleaves]
        assert len(nq.read(ctx)) == len(oq.read()) == 2
        assert [w[0] for w in nq.read_with(ctx, lambda leaf: ("w", leaf))] == ["w", "w"]


def test_a_quantity_with_one_wire_reads_as_the_wire_and_a_bad_declaration_is_refused():
    """``auto`` with a single wire reads the wire itself.

    The library refuses a duplicate wire by name.
    """
    one = _ir.Quantity("q", wires=[old_types.Wire(name="x", role="vec_in")])
    assert one.read_with(_ir.new_context(), lambda leaf: leaf).cls == "Leaf"
    with pytest.raises(HawkError, match="duplicate wire name 'x'"):
        _ir.Quantity("q", wires=[old_types.Wire(name="x", role="vec_in")] * 2)
    with pytest.raises(Exception, match="duplicate wire name"):
        old_compound.Quantity("q", wires=[old_types.Wire(name="x", role="vec_in")] * 2)


def test_a_quantity_leaf_binds_through_the_library_walk(ctx):
    """Return the leaves of a quantity carry its record.

    So the canonical walk accepts them as one namespace.
    """
    nq = _ir.Quantity("pos", wires=wires(old_types.Wire))
    leaves = nq.leaves(ctx)
    total = _ir.make("add", [leaves[0], leaves[0]])
    walk = _ir.canonical([_ir.make_node("Assign", F64, [total], name="y")])
    assert walk.context is ctx


# ---- contexts ----


def test_adopt_copies_foreign_roots_and_keeps_the_ones_already_there(ctx):
    """Make a root of another context is copied into the receiving one with its graph.

    A root of this one is kept.
    """
    (_, _, _, _), (nx, nsum, nsel, ny) = twin_sum(ctx)
    home = _ir.new_context()
    local = _ir.const(home, 3.0, F64)
    adopted = _ir.adopt(home, [ny, local])
    assert adopted[0].ctx is home and adopted[1] == local
    assert new_shape(adopted[0]) == new_shape(ny)
    assert _ir.adopt(home, [local]) == (local,)


def test_a_node_of_another_context_is_refused_by_name(ctx):
    """Mixing contexts fails loudly (the library's refusal).

    The missed ``adopt`` is found at once.
    """
    other = _ir.new_context()
    stranger = _ir.const(other, 1.0, F64)
    mine = _ir.const(ctx, 1.0, F64)
    with pytest.raises(TypeError, match="another Ctx"):
        _ir.make("add", [mine, stranger])


def test_a_builder_without_an_operand_asks_for_the_context():
    """``make_node`` with no operand and no context says how to proceed."""
    with pytest.raises(TypeError, match="pass context="):
        _ir.make_node("LoopIndex", I32)


# ---- the refusal class ----


def test_a_library_refusal_reaches_the_caller_as_hawk_error_with_the_library_text(ctx):
    """Every refusal of the library is a ``HawkError`` carrying its text.

    A wrong argument shape stays a ``TypeError``.
    """
    lu = _ir.leaf(ctx, "vocab_read", "vec_in", "l", ((2, 2), "f64", "lu_factor"))
    with pytest.raises(HawkError) as refused:
        _ir.as_matrix(lu)
    with pytest.raises(ValueError) as library:
        ctx.as_matrix(lu)
    assert str(refused.value) == str(library.value)
    assert isinstance(refused.value.__cause__, ValueError)
    with pytest.raises(TypeError):
        _ir.make_node("LoopIndex", I32, context=ctx, depth=object())


# ---- node methods the old nodes had ----


def reduce_pair(ctx, with_key):
    """Build a keyed reduction over samples on both sides: the old node and the ``_ir`` node."""
    ox = old.Leaf("vocab_read", "vec_in", "x", OLD_F64)
    nx = _ir.leaf(ctx, "vocab_read", "vec_in", "x", F64)
    okey = old.Leaf("vocab_read", "per_sample", "k", OLD_I32) if with_key else None
    nkey = _ir.leaf(ctx, "vocab_read", "per_sample", "k", I32) if with_key else None
    ocount = old.Const(2, OLD_I32)
    ored = old_nodes.ReduceSamples("sum", ox, key=okey, count=ocount, kept=(("a", None),))
    refs = [nx, _ir.const(ctx, 2, I32)] + ([nkey] if with_key else [])
    key_text = ", ('key', #2)" if with_key else ""
    head = "(('op', 'sum'), ('value', #0), ('count', #1), ('contract', 'auto'), ('kept', (('a', None),))"
    fields = f"{head}{key_text})"
    return ored, ctx.new("ReduceSamples", fields, refs)


@pytest.mark.parametrize("with_key", [False, True])
def test_recount_equals_the_old_recount(ctx, with_key):
    """``recount`` gives the old node's result: same operand, new count, key and the rest kept."""
    ored, nred = reduce_pair(ctx, with_key)
    expected = ored.recount(old.Const(7, OLD_I32))
    got = _ir.recount(nred, _ir.const(ctx, 7, I32))
    assert new_shape(got) == old_shape(expected)
    assert new_shape(got.field_nodes("count")[0]) == old_shape(expected.count)
    assert (got.field_kind("key") == 0) == (expected.key is None)
    assert (got.field_text("op"), got.field_text("contract")) == (expected.op, expected.contract)
    assert _ir.field_pairs(got, "kept") == tuple(expected.kept)


def accum_pair(ctx, indexed):
    """Build a keyed accumulate with a ``reduced`` edge on both sides."""
    ox = old.Leaf("vocab_read", "vec_in", "x", OLD_F64)
    nx = _ir.leaf(ctx, "vocab_read", "vec_in", "x", F64)
    oi, ni = old.Const(0, OLD_I32), _ir.const(ctx, 0, I32)
    oreduced = old.Const(3, OLD_I32)
    oacc = old.AccumWrite("a", ox, oi if indexed else None, contract="keyed_auto", reduced=oreduced)
    index = ", ('index', #1)" if indexed else ""
    refs = [nx] + ([ni] if indexed else []) + [_ir.const(ctx, 3, I32)]
    last = 2 if indexed else 1
    fields = (
        f"(('name', 'a'), ('value', #0){index}, ('contract', 'keyed_auto'), ('reduced', #{last}))"
    )
    return (ox, oi, oacc), (nx, ni, ctx.new("AccumWrite", fields, refs))


@pytest.mark.parametrize("indexed", [False, True])
def test_unreduced_equals_the_old_rebuilt_that_drops_the_edge(ctx, indexed):
    """``unreduced`` equals ``rebuilt(ops, {})`` (the ``reduced`` edge dropped).

    ``rebuild`` keeps the edge.
    """
    (ox, oi, oacc), (nx, ni, nacc) = accum_pair(ctx, indexed)
    oz, nz = old.Const(9.0, OLD_F64), _ir.const(ctx, 9.0, F64)
    expected = oacc.rebuilt((oz, *oacc.operands[1:]), {})
    got = _ir.unreduced(nacc, (nz, *nacc.operands[1:]))
    assert new_shape(got) == old_shape(expected)
    assert expected.edges() == {} and got.field_kind("reduced") == 0
    assert (got.field_text("name"), got.field_text("contract")) == (
        expected.name,
        expected.contract,
    )
    assert _ir.rebuild(nacc, [nz, *nacc.operands[1:]]).field_kind("reduced") != 0
    plain = _ir.make_node("Assign", F64, [nz], name="y")
    assert new_shape(_ir.unreduced(plain, [nz])) == new_shape(_ir.rebuild(plain, [nz]))


def test_the_flat_index_of_a_read_is_the_operand_without_its_clamp(ctx):
    """``raw_index`` is operand 1 of the read, its clamp peeled.

    What the callers spell with ``_ir``.
    """
    oplane = old.Leaf("table_read", "lookup", "t", old_types.TensorType((4,), "f64"))
    nplane = _ir.leaf(ctx, "table_read", "lookup", "t", ((4,), "f64", None))
    oraw, nraw = old.Const(1, OLD_I32), _ir.const(ctx, 1, I32)
    for clamped in (False, True):
        oidx = old.make("clamp_index", [oraw, old.Const(3, OLD_I32)]) if clamped else oraw
        nidx = _ir.make("clamp_index", [nraw, _ir.const(ctx, 3, I32)]) if clamped else nraw
        oread = old.At(oplane, oidx, OLD_F64)
        nread = _ir.make_node("At", F64, [nplane, nidx])
        index = nread.operands[1]
        peeled = (
            index.operands[0]
            if _ir.is_a(index, "op") and index.field_text("kind") == "clamp_index"
            else index
        )
        assert new_shape(peeled) == old_shape(oread.raw_index)


# ---- the questions the split left open ----


def every_family(ctx):
    """One node of each family the tracer builds, on both sides."""
    (ox, osum, osel, oy), (nx, nsum, nsel, ny) = twin_sum(ctx)
    loops, _, _ = twin_loop(ctx)
    ored, nred = reduce_pair(ctx, True)
    (_, _, oacc), (_, _, nacc) = accum_pair(ctx, True)
    oc = old.Const(1.0, OLD_F64)
    olds = (ox, osum, osel, oy, loops["acc"][0], ored, oacc, oc)
    news = (nx, nsum, nsel, ny, loops["acc"][1], nred, nacc, _ir.const(ctx, 1.0, F64))
    return list(zip(olds, news))


def test_is_a_node_and_is_a_op_equal_isinstance_on_every_family(ctx):
    """``is_a(n, "node")`` is true for every node and ``is_a(n, "op")`` only for ops.

    As ``isinstance`` was.
    """
    for old_node, new_node in every_family(ctx):
        assert _ir.is_a(new_node, "node") == isinstance(old_node, old.Node) is True
        assert _ir.is_a(new_node, "op") == isinstance(old_node, old.Op), type(old_node).__name__
        assert _ir.is_a(new_node, "op", "leaf") == isinstance(old_node, (old.Op, old.Leaf))


def test_the_kind_of_a_node_reads_as_its_kind_field(ctx):
    """``n.kind`` of an op, a leaf and a sink is the text field ``kind`` (``cls`` is the class.

    Not the kind).
    """
    (ox, osum, osel, oy), (nx, nsum, nsel, ny) = twin_sum(ctx)
    for old_node, new_node in ((osum, nsum), (ox, nx)):
        assert new_node.field_text("kind") == old_node.kind
    assert nsum.cls == "Op" and nsum.field_text("kind") == "add"


def test_a_type_equals_a_hand_built_one_only_with_a_tuple_shape(ctx):
    """``type_of`` equals ``TensorType`` of a tuple shape; a list shape differs.

    As it did for the old dataclass.
    """
    vec = _ir.leaf(ctx, "vocab_read", "vec_in", "v", ((3,), "f64", None))
    assert _ir.type_of(vec) == _ir.TensorType((3,), "f64")
    assert _ir.type_of(vec) != _ir.TensorType([3], "f64")
    assert (old_types.TensorType((3,), "f64") == old_types.TensorType([3], "f64")) is False
    assert hash(_ir.type_of(vec)) == hash(_ir.TensorType((3,), "f64"))


def test_a_declared_slot_is_bound_by_the_walk_even_when_the_body_does_not_read_it(ctx):
    """``declared`` reaches the library as Slot records.

    The guard's mask is in the argument list.
    """
    (_, _, _, oy), (_, _, _, ny) = twin_sum(ctx)
    mask = _ir.Slot("terminated", "alive", _ir.TensorType((), "i32", None))
    walk = _ir.canonical([ny], declared=[mask])
    plain = _ir.canonical([ny])
    assert walk.context.walk_binds(walk.number, "terminated", "alive")
    assert not plain.context.walk_binds(plain.number, "terminated", "alive")
    oldwalk = old_walk.canonical([oy], [old_types.Slot("terminated", "alive", OLD_I32)])
    assert ("terminated", "alive") in [tuple(a)[:2] for a in oldwalk.arg_spec]


# ---- reads on a walk ----


def test_a_walk_answers_its_digest_argument_list_and_slot_numbers_as_the_old_walk_does(ctx):
    """``digest``, ``arg_spec`` and ``slot_of`` equal the old walk's, with a loop and a select."""
    (ox, osum, osel, oy), (nx, nsum, nsel, ny) = twin_sum(ctx)
    loops, _, _ = twin_loop(ctx)
    for old_root, new_root in (
        (oy, ny),
        (
            old.Assign("z", loops["acc"][0]),
            _ir.make_node("Assign", F64, [loops["acc"][1]], name="z"),
        ),
    ):
        owalk, nwalk = old_walk.canonical([old_root]), _ir.canonical([new_root])
        assert _ir.digest(nwalk) == owalk.digest
        assert _ir.arg_spec(nwalk) == tuple(owalk.arg_spec)
        assert _ir.slot_of(nwalk) == dict(owalk.slot_of)


# ---- handles, records and the reduction builder ----


def test_the_node_class_is_the_handle_every_built_node_is_an_instance_of(ctx):
    """``Node`` is the class of what the builders return; a plain value is not one."""
    leaf = _ir.leaf(ctx, "vocab_read", "vec_in", "x", F64)
    assert isinstance(leaf, _ir.Node) and isinstance(_ir.make("add", [leaf, leaf]), _ir.Node)
    assert not isinstance(3, _ir.Node) and not isinstance((leaf,), _ir.Node)


def test_the_wire_record_equals_the_old_wire_field_by_field():
    """``Wire`` has the old fields, order and defaults."""
    wire, owire = _ir.Wire("x", "vec_in"), old_types.Wire(name="x", role="vec_in")
    assert tuple(wire) == (owire.name, owire.role, owire.shape, owire.dtype, owire.tag)
    assert _ir.Wire._fields == ("name", "role", "shape", "dtype", "tag")


@pytest.mark.parametrize("keyed", [False, True])
def test_reduce_samples_carries_key_and_count_edges_as_the_old_node_does(ctx, keyed):
    """The builder gives the old node's operands, edges, op, contract and kept axes."""
    ox = old.Leaf("vocab_read", "vec_in", "x", OLD_F64)
    nx = _ir.leaf(ctx, "vocab_read", "vec_in", "x", F64)
    okey = old.Leaf("vocab_read", "per_sample", "k", OLD_I32) if keyed else None
    nkey = _ir.leaf(ctx, "vocab_read", "per_sample", "k", I32) if keyed else None
    ored = old_nodes.ReduceSamples(
        "sum", ox, key=okey, count=old.Const(4, OLD_I32), contract="compensated", kept=(("a", None),)
    )
    nred = _ir.reduce_samples(
        "sum", nx, key=nkey, count=_ir.const(ctx, 4, I32), contract="compensated", kept=(("a", None),)
    )
    assert new_shape(nred) == old_shape(ored)
    assert new_shape(nred.field_nodes("count")[0]) == old_shape(ored.count)
    assert (nred.field_kind("key") != 0) == keyed
    assert nred.field_text("contract") == ored.contract and _ir.field_pairs(nred, "kept") == (("a", None),)


def raw_pair(ctx, text="x + 1", language="aether", reads=("a",), returns=((), "f64", None)):
    """Build a raw-device op whose literal is the record the library writes its key text from."""
    ox = old.Leaf("vocab_read", "vec_in", "x", OLD_F64)
    nx = _ir.leaf(ctx, "vocab_read", "vec_in", "x", F64)
    block = _ir.RawBlock("blk", "sample_local", text, language, reads, (), _ir.TensorType(*returns))
    nraw = _ir.make_node("Op", returns, [nx], kind=_ir.RAW_KIND, literal=block)
    return ox, nx, nraw


def test_a_raw_block_literal_is_written_as_a_record_and_distinct_blocks_are_distinct_nodes(ctx):
    """Two blocks differing in text, language, reads or return type give four more digests."""

    def digest(node):
        return _ir.digest(_ir.canonical([_ir.make_node("Assign", F64, [node], name="y")]))

    base = digest(raw_pair(ctx)[2])
    assert digest(raw_pair(ctx)[2]) == base
    changed = [
        raw_pair(ctx, text="x + 2"),
        raw_pair(ctx, language="cuda"),
        raw_pair(ctx, reads=("a", "b")),
        raw_pair(ctx, returns=((), "f32", None)),
    ]
    digests = {digest(v[2]) for v in changed}
    assert base not in digests and len(digests) == 4


def test_a_loop_value_reads_back_its_loop_and_slot_and_refuses_a_slot_outside(ctx):
    """``loop_value`` builds the old ``LoopValue``; ``loop_value_of`` reads it back."""
    loops, _, _ = twin_loop(ctx)
    old_loop, new_loop = loops["acc"]
    expected, got = old.LoopValue(old_loop, 0), _ir.loop_value(new_loop, 0)
    assert new_shape(got) == old_shape(expected) and got.field_text("kind") == "loop_value"
    assert _ir.loop_value_of(got) == (new_loop, 0) == (new_loop, expected.slot)
    with pytest.raises(HawkError, match="outside the loop"):
        _ir.loop_value(new_loop, 5)


# ---- constants of the roles, the canonical order, the dump text, the kept tuple ----


@pytest.mark.parametrize("name", ["ROLE_ORDER", "SINK_ROLES"])
def test_the_role_constants_equal_the_old_ones(name):
    """The role order and the sink roles equal the old package's."""
    assert getattr(_ir, name) == getattr(old_nodes, name)


def test_walk_order_equals_the_old_canonical_order(ctx):
    """``walk_order`` is the old walk's node order, class and type node by node."""
    (_, _, _, oy), (_, _, _, ny) = twin_sum(ctx)
    assert [new_shape(n) for n in _ir.walk_order(_ir.canonical([ny]))] == [
        old_shape(n) for n in old_walk.canonical([oy]).order
    ]


def test_dump_text_is_the_context_dump_and_copies_foreign_roots_in(ctx):
    """One context's roots give its own dump; roots of two contexts give one loadable text."""
    (_, _, _, _), (_, _, _, ny) = twin_sum(ctx)
    assert _ir.dump_text([ny]) == ctx.dump([ny])
    other = _ir.new_context()
    stranger = _ir.const(other, 3.0, F64)
    text = _ir.dump_text([ny, stranger])
    loaded = _ir.new_context().load(text)
    assert [new_shape(n) for n in loaded] == [new_shape(ny), new_shape(stranger)]


def test_kept_axes_are_pairs_and_a_recount_carries_them(ctx):
    """The kept lane axes are (label, extent) pairs; ``recount`` rebuilds the reduction with the same pairs."""
    nx = _ir.leaf(ctx, "vocab_read", "vec_in", "x", F64)
    kept = (("i", 3), ("j", None))
    reduced = _ir.reduce_samples("sum", nx, kept=kept)
    assert _ir.field_pairs(reduced, "kept") == kept
    assert _ir.field_pairs(_ir.recount(reduced, _ir.const(ctx, 6, I32)), "kept") == kept


# ---- the wings' kinds, rewrites and predicates ----


def test_a_fresh_context_has_the_wing_kinds_declared():
    """The dispatch strategies.

    Block sums and ordered writes build in a context made by ``new_context``.
    """
    fresh = _ir.new_context()
    assert all(_ir.kind_reserved(fresh, name) for name in ("block_sum", "predicated_dispatch"))
    x = _ir.leaf(fresh, "vocab_read", "vec_in", "x", F64)
    arms = [x, _ir.const(fresh, 1.0, F64)]
    node = _ir.make_node("PredicatedDispatch", F64, [_ir.const(fresh, 0, I32), *arms])
    assert _ir.is_a(node, "predicated_dispatch") and _ir.has_feather(node, "evaluates_all_branches")


@pytest.mark.parametrize("name", ["OP_KINDS", "CAST", "LINALG_OPS"])
def test_the_op_kind_lists_equal_the_old_ones(name):
    """The lists the library states equal the old package's, in order."""
    home = old_linalg if name == "LINALG_OPS" else old_ops
    assert getattr(_ir, name) == getattr(home, name)


def test_the_exit_form_of_a_loop_equals_the_old_derived_property(ctx):
    """``head`` (exit values are the carries), ``tail`` (the nexts), ``middle`` (other).

    None without a break.
    """
    loops, (ox, oi, oc), (nx, ni, nc) = twin_loop(ctx)
    onext, nnext = old.make("add", [oc, ox]), _ir.make("add", [nc, nx])
    ocond, ncond = old.make("lt", [ox, oc]), _ir.make("lt", [nx, nc])
    ozero, nzero = old.Const(0.0, OLD_F64), _ir.const(ctx, 0.0, F64)
    other_old, other_new = old.make("mul", [oc, ox]), _ir.make("mul", [nc, nx])
    for expected, old_values, new_values in (
        ("head", [oc], [nc]), ("tail", [onext], [nnext]), ("middle", [other_old], [other_new])
    ):  # fmt: skip
        oloop = old_loops.build_loop(oi, [oc], [ozero], [onext], start=0, stop=3, step=1,
                                     exit_cond=ocond, exit_values=old_values)  # fmt: skip
        nloop = _ir.loop(ni, [nc], [nzero], [nnext], start=0, stop=3, step=1,
                         exit_cond=ncond, exit_values=new_values)  # fmt: skip
        assert _ir.exit_form(nloop) == oloop.exit_form == expected
    assert _ir.exit_form(loops["acc"][1]) is None and loops["acc"][0].exit_form is None


def payloads(ctx):
    """Build keyed-sum payloads on both sides: a gram product, a normal product, an outer.

    A vector, a sum.
    """
    out = []
    mat_t = old_types.TensorType((4, 3), "f64")
    om = old.Leaf("vocab_read", "per_sample", "M", mat_t)
    nm = _ir.leaf(ctx, "vocab_read", "per_sample", "M", ((4, 3), "f64", None))
    orr = old.Leaf("vocab_read", "per_sample", "r", old_types.TensorType((4,), "f64"))
    nr = _ir.leaf(ctx, "vocab_read", "per_sample", "r", ((4,), "f64", None))
    ov3, nv3 = (
        old.Leaf("vocab_read", "per_sample", "v", old_types.TensorType((3,), "f64")),
        _ir.leaf(ctx, "vocab_read", "per_sample", "v", ((3,), "f64", None)),
    )
    omt, nmt = old.make("transpose", [om]), _ir.make("transpose", [nm])
    out.append((old.make("mm", [omt, om]), _ir.make("mm", [nmt, nm])))
    out.append((old.make("mv", [omt, orr]), _ir.make("mv", [nmt, nr])))
    out.append((old.make("outer", [ov3, ov3]), _ir.make("outer", [nv3, nv3])))
    out.append((ov3, nv3))
    out.append((old.make("add", [ov3, ov3]), _ir.make("add", [nv3, nv3])))
    out.append((om, nm))
    return out


def test_the_block_predicates_equal_the_old_functions(ctx):
    """Factors, rows, the ordered case and the block shape agree with the old wing over payloads.

    Contracts, limits.
    """
    for old_payload, new_payload in payloads(ctx):
        found = old_block.factors(old_payload)
        if found is None:
            assert _ir.block_factors(new_payload) == ()
        else:
            u, v, rows = found
            got = _ir.block_factors(new_payload)
            assert [new_shape(n) for n in got] == [old_shape(u), old_shape(v)]
            assert _ir.block_factor_rows(new_payload) == rows
        for contract in ("auto", "ordered"):
            for limit in (None, 2, 3, 4):
                want_case = old_block.ordered_case(old_payload, contract, limit)
                assert _ir.block_ordered_case(new_payload, contract, limit) == want_case
                assert _ir.block_shaped(new_payload, contract, limit) == old_block.block_shaped(
                    old_payload, contract, limit
                )
    with pytest.raises(HawkError):
        _ir.block_factor_rows(payloads(ctx)[3][1])


def test_slotwise_rewrites_like_the_old_pass_and_reads_chains_and_zeros(ctx):
    """``slotwise`` keeps an unchanged root and picks a component of a vector.

    Chains and zeros read as before.
    """
    a_old, b_old = (
        old.Leaf("vocab_read", "vec_in", "a", OLD_F64),
        old.Leaf("vocab_read", "vec_in", "b", OLD_F64),
    )
    a_new, b_new = (
        _ir.leaf(ctx, "vocab_read", "vec_in", "a", F64),
        _ir.leaf(ctx, "vocab_read", "vec_in", "b", F64),
    )
    ovec, nvec = old.make("vec", [a_old, b_old]), _ir.make("vec", [a_new, b_new])
    opicked, npicked = old.make("component", [ovec], 1), _ir.make("component", [nvec], 1)
    got = _ir.slotwise([npicked, nvec, a_new])
    want = old_slotwise.slotwise([opicked, ovec, a_old])
    assert [new_shape(n) for n in got] == [old_shape(n) for n in want]
    assert _ir.slotwise([]) == ()
    zero_old, zero_new = old.Const(0.0, OLD_F64), _ir.const(ctx, 0.0, F64)
    splat_old, splat_new = old.make("splat", [zero_old], (2,)), _ir.make("splat", [zero_new], (2,))
    assert _ir.is_structural_zero(splat_new) == old_slotwise.is_structural_zero(splat_old) is True
    assert _ir.is_structural_zero(nvec) == old_slotwise.is_structural_zero(ovec) is False
    base, links = _ir.slotwise_chain(nvec)
    obase, olinks = old_slotwise.chain(ovec)
    assert base == nvec and len(links) == len(olinks) == 0 and new_shape(base) == old_shape(obase)


@pytest.mark.parametrize("width", [None, 3])
def test_cells_stored_makes_the_old_sample_sum_sinks(ctx, width):
    """A scalar sum is one sink; a vector gives one sink per component.

    Planes ``out_0`` ... as before.
    """
    ttype = (
        (((), "f64", None), OLD_F64)
        if width is None
        else ((((width,), "f64", None)), old_types.TensorType((width,), "f64"))
    )
    new_t, old_t = ttype
    ox, nx = (
        old.Leaf("vocab_read", "per_sample", "x", old_t),
        _ir.leaf(ctx, "vocab_read", "per_sample", "x", new_t),
    )
    ored = old_nodes.ReduceSamples("sum", ox)
    nred = _ir.reduce_samples("sum", nx)
    got, want = _ir.cells_stored(nred, "out"), old_cells_wing.stored(ored, "out")
    assert [new_shape(n) for n in got] == [old_shape(n) for n in want]
    assert [n.field_text("name") for n in got] == [n.name for n in want]
    with pytest.raises(HawkError):
        _ir.cells_stored(nx, "out")


def ordered_pair(ctx, width=None):
    """Build two ordered accumulates into one plane on both sides.

    (a vector value when ``width``).
    """
    shape = () if width is None else (width,)
    old_x = old.Leaf("vocab_read", "per_sample", "lx", old_types.TensorType(shape, "f64"))
    new_x = _ir.leaf(ctx, "vocab_read", "per_sample", "lx", (shape, "f64", None))
    old_key = old.Leaf("vocab_read", "per_sample", "lk", old_types.TensorType((), "i64"))
    new_key = _ir.leaf(ctx, "vocab_read", "per_sample", "lk", ((), "i64", None))
    kept = (("net", None),)
    olds, news = [], []
    for cell in ("ca", "cb"):
        old_cell = old.Leaf("vocab_read", "per_sample", cell, old_types.TensorType((), "i64"))
        new_cell = _ir.leaf(ctx, "vocab_read", "per_sample", cell, ((), "i64", None))
        olds.append(old_logs.OrderedAccumWrite("g", old_x, old_cell, old_key, kept))
        news.append(
            _ir.make_node(
                "OrderedAccumWrite",
                (shape, "f64", None),
                [new_x, new_cell, new_key],
                name="g",
                role="accum_out",
                kept=kept,
            )  # fmt: skip
        )
    return olds, news, old_x, new_x


def test_logs_ordered_replaces_ordered_accumulates_by_their_log_stores(ctx):
    """Two sites make the old number of stores; the library refuses a vector value.

    The old pass accepted a vector value; the refusal is the library's documented behaviour.
    """
    olds, news, old_x, new_x = ordered_pair(ctx)
    got, want = _ir.logs_ordered(news), old_logs.log_ordered(olds)
    assert len(got) == len(want) == 4
    assert [n.cls for n in got] == ["LogStore"] * 4
    assert _ir.logs_ordered([new_x]) == (new_x,) and _ir.logs_ordered([]) == ()
    _, vector_news, _, _ = ordered_pair(ctx, width=3)
    with pytest.raises(HawkError, match="is not a scalar"):
        _ir.logs_ordered(vector_news)


# ---- the kernel's walk and its typed reads ----


def test_minted_ops_equal_the_old_list():
    """The op kinds a traced body mints equal the old package's list."""
    assert _ir.MINTED_OPS == old_nodes.MINTED_OPS


def sum_and_plain(ctx):
    """Build two sample sums into one plane name ``s`` (summing) and an assign.

    For the walk tests.
    """
    x = _ir.leaf(ctx, "vocab_read", "per_sample", "x", F64)
    total = _ir.make_node("SampleSum", F64, [x], name="s", op="sum", cell="s", columns=1)
    clash = _ir.make_node("SampleSum", F64, [x], name="s_comp", op="sum", cell="s_comp", columns=1)
    return x, total, clash


def test_a_summing_sample_sum_gives_the_companion_slot_and_a_collision_is_refused(ctx):
    """The kernel's walk binds ``<name>_comp`` for a summing sample sum.

    An existing plane of that name is refused.
    """
    _, total, clash = sum_and_plain(ctx)
    walk = _ir.cells_walk([total])
    assert any(name == "s_comp" for _, name in _ir.arg_spec(walk))
    with pytest.raises(HawkError, match="its compensation plane 's_comp' collides"):
        _ir.cells_walk([total, clash])
    assert _ir.cells_walk([total]).context is ctx


def test_a_finishing_kernel_binds_the_finished_count_slot_and_declared_slots_are_kept(ctx):
    """A ``Finish`` sink adds the finished-count slot.

    A declared slot is bound whether read or not.
    """
    x = _ir.leaf(ctx, "vocab_read", "per_sample", "x", F64)
    plain = _ir.make_node("Assign", F64, [x], name="o")
    finish = _ir.make_node("Finish", ((), "bool", None), [_ir.make("gt", [x, x])], name="done")
    assert ("lookup", "finished_count") not in _ir.arg_spec(_ir.cells_walk([plain]))
    assert ("lookup", "finished_count") in _ir.arg_spec(_ir.cells_walk([plain, finish]))
    mask = _ir.Slot("terminated", "alive", _ir.TensorType((), "i32", None))
    assert ("terminated", "alive") in _ir.arg_spec(_ir.cells_walk([plain], declared=[mask]))


def auto_kernel(ctx):
    """Build an assign of an ``auto`` dispatch of two branches over a per-sample selector."""
    x = _ir.leaf(ctx, "vocab_read", "per_sample", "x", F64)
    selector = _ir.leaf(ctx, "vocab_read", "per_sample", "sel", I32)
    arms = [_ir.make("add", [x, x]), _ir.make("mul", [x, x])]
    auto = _ir.make_node("AutoDispatch", F64, [selector, *arms])
    return _ir.make_node("Assign", F64, [auto], name="o"), auto


def test_an_auto_dispatch_gives_a_mix_slot_a_group_record_and_a_static_selector(ctx):
    """The walk binds ``dispatch0_mix`` (uint32).

    Reports strategy ``auto`` and a static selector.
    """
    sink, auto = auto_kernel(ctx)
    walk = _ir.cells_walk([sink])
    assert ("lookup", "dispatch0_mix") in _ir.arg_spec(walk)
    assert _ir.slot_type(walk, "lookup", "dispatch0_mix") == _ir.TensorType((), "uint32", None)
    (group,) = _ir.dispatch_groups(walk)
    assert group == _ir.DispatchGroup("dispatch0", 2, "auto", auto)
    assert _ir.static_selector(walk, 0) is True
    with pytest.raises(HawkError):
        _ir.slot_type(walk, "lookup", "no_such")
    with pytest.raises(HawkError):
        _ir.static_selector(walk, 3)


def test_a_plain_dispatch_is_the_switch_group_and_has_no_mix_slot(ctx):
    """A core dispatch reports strategy ``switch`` and binds no branch-mix slot."""
    x = _ir.leaf(ctx, "vocab_read", "per_sample", "x", F64)
    selector = _ir.leaf(ctx, "vocab_read", "per_sample", "sel", I32)
    core = _ir.make_node(
        "Dispatch", F64, [selector, _ir.make("add", [x, x]), _ir.make("mul", [x, x])]
    )
    walk = _ir.cells_walk([_ir.make_node("Assign", F64, [core], name="o")])
    assert [g.strategy for g in _ir.dispatch_groups(walk)] == ["switch"]
    assert all(name != "dispatch0_mix" for _, name in _ir.arg_spec(walk))


def test_a_plane_read_at_launch_start_is_reported_as_a_prior_read(ctx):
    """A ``prior_read`` leaf (the launch-start value of a plane) is reported.

    A plain kernel has none.
    """
    old_y = _ir.leaf(ctx, "prior_read", "mutable", "y", F64)
    bump = _ir.make("add", [old_y, _ir.const(ctx, 1.0, F64)])
    walk = _ir.cells_walk([_ir.make_node("Assign", F64, [bump], name="y")])
    assert _ir.prior_reads(walk) == ("y",)
    sink, _ = auto_kernel(ctx)
    assert _ir.prior_reads(_ir.cells_walk([sink])) == ()


def test_the_dispatch_strategies_are_the_libraries_list():
    """``switch`` first, then one strategy per wing kind, in manifest order."""
    assert _ir.dispatch_strategies() == ("switch", "predicated", "segmented", "auto")


def test_expand_stores_replaces_a_block_write_by_per_entry_accumulates(ctx):
    """A vector block write becomes three accumulates; other sinks stay in order."""
    x = _ir.leaf(ctx, "vocab_read", "per_sample", "x", F64)
    plain = _ir.make_node("Assign", F64, [x], name="o")
    vec3 = _ir.leaf(ctx, "vocab_read", "per_sample", "v", ((3,), "f64", None))
    block = _ir.make_node(
        "BlockSum", ((3,), "f64", None),
        [vec3, _ir.const(ctx, 0, I32), _ir.const(ctx, 1, I32)],
        contract="auto", kept=(),
    )  # fmt: skip
    write = _ir.make_node("BlockSumWrite", F64, [block], name="g")
    out = _ir.expand_stores([plain, write, plain])
    assert len(out) == 5 and out[0] == plain and out[4] == plain
    assert [n.cls for n in out[1:4]] == ["AccumWrite"] * 3 and _ir.expand_stores([]) == ()


def test_dump_numbers_are_the_root_lines_of_the_dump(ctx):
    """Each root's number is the ``root|#p`` line of the dump, in the order of the roots."""
    x = _ir.leaf(ctx, "vocab_read", "per_sample", "x", F64)
    picked = _ir.make("add", [x, x])
    lines = [
        int(line[len("root|#") :])
        for line in ctx.dump([picked, x]).splitlines()
        if line.startswith("root|#")
    ]
    assert _ir.dump_numbers([picked, x]) == tuple(lines) and _ir.dump_numbers([]) == ()


def test_the_wing_names_the_mix_slots_and_the_field_pairs_are_read_through_the_library(ctx):
    """The reserved planes and the costly kinds are the library's; a dispatch that counts its mix names its slot."""
    assert (_ir.FUSED_STEPS_PLANE, _ir.FINISHED_PLANE) == ("fused_steps", "finished_count")
    assert "sin" in _ir.SPECIAL_KINDS and "add" not in _ir.SPECIAL_KINDS
    x = _ir.leaf(ctx, "vocab_read", "per_sample", "x", F64)
    k = _ir.leaf(ctx, "vocab_read", "per_sample", "k", I32)
    sinks = {}
    for cls in ("AutoDispatch", "Dispatch"):
        node = _ir.make_node(cls, F64, [k, x, _ir.make("add", [x, x])])
        sinks[cls] = _ir.make_node("Assign", F64, [node], name=cls.lower())
    assert _ir.mix_slots(_ir.cells_walk([sinks["AutoDispatch"]])) == {"dispatch0": "dispatch0_mix"}
    assert _ir.mix_slots(_ir.cells_walk([sinks["Dispatch"]])) == {}
    with pytest.raises(HawkError, match="list of pairs"):
        _ir.field_pairs(sinks["Dispatch"], "name")


@pytest.mark.parametrize("indexed", [False, True])
def test_accum_write_builds_the_old_keyed_accumulate(ctx, indexed):
    """Name, contract, value and the optional lane index equal the old ``AccumWrite``."""
    ox, oi = (
        old.Leaf("vocab_read", "per_sample", "x", OLD_F64),
        old.Leaf("vocab_read", "per_sample", "k", OLD_I32),
    )
    nx, ni = (
        _ir.leaf(ctx, "vocab_read", "per_sample", "x", F64),
        _ir.leaf(ctx, "vocab_read", "per_sample", "k", I32),
    )
    expected = old.AccumWrite("e", ox, oi if indexed else None, contract="compensated")
    got = _ir.accum_write("e", nx, index=ni if indexed else None, contract="compensated")
    assert new_shape(got) == old_shape(expected)
    assert (got.field_text("name"), got.field_text("contract")) == ("e", expected.contract)
