# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""What ``hawk._ir`` builds, rewrites and walks, with expected values written out by hand.

Each row builds a small graph through the library and compares what is read back (class, type, operands, fields) with a
tree or a text stated in the test: the keyed accumulate of a lane index, the dispatch and sample nodes, the count a
reduction is recounted to, the operands a rebuild swaps, the argument list of a walk. The node library has no public
form in ``hawk``, so these rows read it through ``hawk._ir``.
"""

from __future__ import annotations

import pytest
from _kernels import CORPUS

from hawk import HawkError, _ir
from hawk.trace.kernel import _declared

F64 = ((), "f64", None)
I32 = ((), "i32", None)
LEAF_F64 = ("Leaf", F64, ())
CONST_F64 = ("Const", F64, ())
CONST_I32 = ("Const", I32, ())


def shape(node):
    """Return the node as plain data: its class name, its type and its operands, recursively."""
    return (node.cls, tuple(node.ttype), tuple(shape(o) for o in node.operands))


@pytest.fixture
def ctx():
    """Make a fresh context."""
    return _ir.new_context()


def sum_graph(ctx):
    """Return ``(x, x + 1, select, sink)`` for ``y = (x + 1) < 2 ? x : x + 1``."""
    x = _ir.leaf(ctx, "vocab_read", "plane_in", "x", F64)
    total = _ir.make("add", [x, _ir.const(ctx, 1.0, F64)])
    cond = _ir.make("lt", [total, _ir.const(ctx, 2.0, F64)])
    pick = _ir.make_node("Select", F64, [cond, x, total])
    return x, total, pick, _ir.make_node("Assign", F64, [pick], name="y")


def loop_sink(ctx, kind="add"):
    """Return the sink ``z`` of a three-trip loop whose carried value is ``carry <kind> x``."""
    x = _ir.leaf(ctx, "vocab_read", "plane_in", "x", F64)
    index = _ir.make_node("LoopIndex", I32, context=ctx, depth=0, name="k", bounds=None)
    carry = _ir.make_node("LoopCarry", F64, context=ctx, depth=0, slot=0, name="s")
    loop = _ir.loop(
        index,
        [carry],
        [_ir.const(ctx, 0.0, F64)],
        [_ir.make(kind, [carry, x])],
        start=0,
        stop=3,
        step=1,
    )
    return _ir.make_node("Assign", F64, [loop], name="z")


# ---- the flat index of a read, a raw block, the keyed accumulate ----


@pytest.mark.parametrize("clamped", [False, True])
def test_the_flat_index_of_a_read_is_operand_one_with_its_clamp_peeled(ctx, clamped):
    """The index a read states is operand 1; a ``clamp_index`` around it wraps the raw index as operand 0."""
    table = _ir.leaf(ctx, "table_read", "lookup", "t", ((4,), "f64", None))
    raw = _ir.const(ctx, 1, I32)
    index = _ir.make("clamp_index", [raw, _ir.const(ctx, 3, I32)]) if clamped else raw
    read = _ir.make_node("At", F64, [table, index])
    stated = read.operands[1]
    assert (stated.field_text("kind") == "clamp_index") is clamped
    peeled = stated.operands[0] if _ir.is_a(stated, "op") else stated
    assert peeled == raw


@pytest.mark.parametrize("edge", ["clamp", "wrap", "zero", "unchecked"])
def test_the_index_of_a_read_is_operand_one_and_its_edge_rule_is_a_field(ctx, edge):
    """A read states its index as operand 1 and what happens outside the plane as a field, with no operation around the index."""
    table = _ir.leaf(ctx, "table_read", "lookup", "t", ((4,), "f64", None))
    raw = _ir.const(ctx, 1, I32)
    read = _ir.make_node("At", F64, [table, raw], edge=edge)
    assert read.operands[1] == raw
    assert read.field_text("edge") == edge


def raw_block_digest(ctx, text="x + 1", language="aether", reads=("a",), returns=F64):
    """Return the digest of the walk of a sink over a raw device block with the stated record."""
    x = _ir.leaf(ctx, "vocab_read", "plane_in", "x", F64)
    block = _ir.RawBlock(
        "blk", "sample_local", text, language, reads, (), _ir.TensorType(*returns)
    )
    raw = _ir.make_node("Op", returns, [x], kind=_ir.RAW_KIND, literal=block)
    return _ir.digest(_ir.canonical([_ir.make_node("Assign", F64, [raw], name="y")]))


def test_a_raw_block_literal_is_written_as_a_record_and_distinct_blocks_are_distinct_nodes(
    ctx,
):
    """Two blocks differing in text, language, reads or return type give four more digests than one block built twice."""
    base = raw_block_digest(ctx)
    assert raw_block_digest(_ir.new_context()) == base
    changed = {
        raw_block_digest(ctx, text="x + 2"),
        raw_block_digest(ctx, language="cuda"),
        raw_block_digest(ctx, reads=("a", "b")),
        raw_block_digest(ctx, returns=((), "f32", None)),
    }
    assert base not in changed and len(changed) == 4


@pytest.mark.parametrize("indexed", [False, True])
def test_accum_write_builds_a_keyed_accumulate_of_its_name_contract_value_and_optional_lane_index(
    ctx, indexed
):
    """The node has class ``AccumWrite``, the value (then the lane index) as operands, and the name and contract as text."""
    value = _ir.leaf(ctx, "vocab_read", "plane_in", "x", F64)
    lane = _ir.leaf(ctx, "vocab_read", "plane_in", "k", I32)
    node = _ir.accum_write(
        "e", value, indices=(lane,) if indexed else (), contract="compensated"
    )
    operands = (LEAF_F64, ("Leaf", I32, ())) if indexed else (LEAF_F64,)
    assert shape(node) == ("AccumWrite", F64, operands)
    assert (node.field_text("name"), node.field_text("contract")) == (
        "e",
        "compensated",
    )
    assert (node.field_kind("index") != 0) is indexed


# ---- the other node classes ----


def test_dispatch_at_role_constant_sample_sum_and_sample_index_build_with_their_operands_and_types(
    ctx,
):
    """Each builder returns the class asked for, typed as stated, over exactly the operands given."""
    x = _ir.leaf(ctx, "vocab_read", "plane_in", "x", F64)
    selector = _ir.const(ctx, 1, I32)
    dispatch = _ir.make_node("Dispatch", F64, [selector, x, _ir.const(ctx, 1.0, F64)])
    table = _ir.leaf(ctx, "table_read", "lookup", "t", ((4,), "f64", None))
    role = _ir.make_node("RoleConst", I32, context=ctx, literal=0, label="other")
    read = _ir.make_node("At", F64, [table, role])
    total = _ir.make_node(
        "SampleSum", F64, [x], name="s", op="sum", cell="s", columns=1
    )
    lane = _ir.make_node("SampleIndex", I32, context=ctx)
    assert shape(dispatch) == ("Dispatch", F64, (CONST_I32, LEAF_F64, CONST_F64))
    assert shape(read) == (
        "At",
        F64,
        (("Leaf", ((4,), "f64", None), ()), ("RoleConst", I32, ())),
    )
    assert shape(total) == ("SampleSum", F64, (LEAF_F64,))
    assert shape(lane) == ("SampleIndex", I32, ())
    assert (total.field_text("name"), total.field_text("op")) == ("s", "sum")


# ---- recount, unreduced and the sample reduction ----


def reduction_record(ctx, keyed):
    """Return a sum over samples written as a record: operand 0 the value, 1 the count 2, 2 the key when keyed."""
    value = _ir.leaf(ctx, "vocab_read", "plane_in", "x", F64)
    refs = [value, _ir.const(ctx, 2, I32)]
    key = ""
    if keyed:
        refs.append(_ir.leaf(ctx, "vocab_read", "plane_in", "k", I32))
        key = ", ('key', #2)"
    head = "(('op', 'sum'), ('value', #0), ('count', #1), ('contract', 'auto'), ('kept', (('a', None),))"
    return ctx.new("ReduceSamples", f"{head}{key})", refs)


@pytest.mark.parametrize("keyed", [False, True])
def test_recount_swaps_the_count_and_keeps_the_value_the_key_the_op_the_contract_and_the_kept_axes(
    ctx, keyed
):
    """Recounting a reduction over 2 samples to 7 changes the count edge and nothing else."""
    reduction = reduction_record(ctx, keyed)
    seven = _ir.const(ctx, 7, I32)
    got = _ir.recount(reduction, seven)
    assert tuple(got.field_nodes("count")) == (seven,)
    assert got.operands[0] == reduction.operands[0]
    assert (got.field_kind("key") != 0) is keyed
    assert (got.field_text("op"), got.field_text("contract")) == ("sum", "auto")
    assert _ir.field_pairs(got, "kept") == (("a", None),)
    assert reduction.field_nodes("count")[0] != seven


def accum_with_reduced_edge(ctx, indexed):
    """Return a keyed accumulate ``a`` carrying a ``reduced`` edge (the constant 3), written as a record."""
    refs = [_ir.leaf(ctx, "vocab_read", "plane_in", "x", F64)]
    index = ""
    if indexed:
        refs.append(_ir.const(ctx, 0, I32))
        index = ", ('indices', (#1, ))"
    refs.append(_ir.const(ctx, 3, I32))
    fields = f"(('name', 'a'), ('value', #0){index}, ('contract', 'keyed_auto'), ('reduced', #{len(refs) - 1}))"
    return ctx.new("AccumWrite", fields, refs)


@pytest.mark.parametrize("indexed", [False, True])
def test_unreduced_drops_the_reduced_edge_where_rebuild_keeps_it(ctx, indexed):
    """``unreduced`` over a new value has no ``reduced`` edge; ``rebuild`` over the same operands keeps it."""
    accum = accum_with_reduced_edge(ctx, indexed)
    other = _ir.const(ctx, 9.0, F64)
    operands = (other, *accum.operands[1:])
    got = _ir.unreduced(accum, operands)
    assert got.field_kind("reduced") == 0
    assert got.operands[0] == other
    assert (got.field_text("name"), got.field_text("contract")) == ("a", "keyed_auto")
    assert _ir.rebuild(accum, list(operands)).field_kind("reduced") != 0
    plain = _ir.make_node("Assign", F64, [other], name="y")
    assert shape(_ir.unreduced(plain, [other])) == shape(_ir.rebuild(plain, [other]))


@pytest.mark.parametrize("keyed", [False, True])
def test_reduce_samples_carries_the_key_and_count_edges_the_op_the_contract_and_the_kept_axes(
    ctx, keyed
):
    """The builder gives operands (value, key when keyed), the count edge, and the text fields stated."""
    value = _ir.leaf(ctx, "vocab_read", "plane_in", "x", F64)
    key = _ir.leaf(ctx, "vocab_read", "plane_in", "k", I32) if keyed else None
    count = _ir.const(ctx, 4, I32)
    got = _ir.reduce_samples(
        "sum", value, key=key, count=count, contract="compensated", kept=(("a", None),)
    )
    assert got.cls == "ReduceSamples" and got.operands[0] == value
    assert tuple(got.field_nodes("count")) == (count,)
    assert (got.field_kind("key") != 0) is keyed
    assert (got.field_text("op"), got.field_text("contract")) == ("sum", "compensated")
    assert _ir.field_pairs(got, "kept") == (("a", None),)


# ---- swap, rebuild, replace, ordered logs ----


def test_swap_replaces_a_node_inside_the_committed_value_and_keeps_the_sink_name_and_the_original(
    ctx,
):
    """Swapping the leaf ``x`` for the constant 5 turns every read of ``x`` in ``y`` into that constant."""
    x, _total, pick, sink = sum_graph(ctx)
    five = _ir.const(ctx, 5.0, F64)
    (swapped,) = _ir.swap([sink], {x: five})
    add_const = ("Op", F64, (CONST_F64, CONST_F64))
    compare = ("Op", ((), "bool", None), (add_const, CONST_F64))
    assert shape(swapped) == (
        "Assign",
        F64,
        (("Select", F64, (compare, CONST_F64, add_const)),),
    )
    assert swapped.field_text("name") == "y"
    assert pick.operands[1] == x


def test_rebuild_and_replace_change_the_operands_only_and_leave_the_original_untouched(
    ctx,
):
    """Rebuilding ``x + 1`` over ``(9, x)`` gives ``9 + x``: same class and type, the stated operands."""
    x, total, *_ = sum_graph(ctx)
    nine = _ir.const(ctx, 9.0, F64)
    expected = ("Op", F64, (CONST_F64, LEAF_F64))
    rebuilt = _ir.rebuild(total, [nine, x])
    replaced = _ir.replace(total, operands=[nine, x])
    assert shape(rebuilt) == shape(replaced) == expected
    assert rebuilt.field_text("kind") == replaced.field_text("kind") == "add"
    assert rebuilt.operands[0] == nine and total.operands[0] == x


def ordered_logs(ctx, width=None):
    """Return two ordered accumulates into plane ``g`` and the value leaf (a vector of ``width`` when given)."""
    ttype = (() if width is None else (width,), "f64", None)
    value = _ir.leaf(ctx, "vocab_read", "plane_in", "lx", ttype)
    key = _ir.leaf(ctx, "vocab_read", "plane_in", "lk", ((), "i64", None))
    sites = []
    for cell in ("ca", "cb"):
        ordinal = _ir.leaf(ctx, "vocab_read", "plane_in", cell, ((), "i64", None))
        sites.append(
            _ir.make_node(
                "OrderedAccumWrite",
                ttype,
                [value, ordinal, key],
                name="g",
                role="accum_out",
                kept=(("net", None),),
            )
        )
    return sites, value


def test_logs_ordered_replaces_two_ordered_accumulates_by_four_log_stores_and_refuses_a_vector_value(
    ctx,
):
    """Two sites give two stores each (four ``LogStore`` nodes); other nodes pass through; a vector value is refused."""
    sites, value = ordered_logs(ctx)
    stores = _ir.logs_ordered(sites)
    assert [node.cls for node in stores] == ["LogStore"] * 4
    assert _ir.logs_ordered([value]) == (value,) and _ir.logs_ordered([]) == ()
    vector_sites, _ = ordered_logs(ctx, width=3)
    with pytest.raises(HawkError, match="is not a scalar"):
        _ir.logs_ordered(vector_sites)


# ---- the walk: digest, argument list, slot numbers, declared slots ----


def test_a_walk_answers_its_argument_list_and_slot_numbers_for_a_select_and_for_a_loop(
    ctx,
):
    """The sink is slot 0, the read ``x`` slot 1; the argument list names each as (role, name) in slot order."""
    _x, _total, _pick, sink = sum_graph(ctx)
    for root in (sink, loop_sink(ctx)):
        name = root.field_text("name")
        walk = _ir.canonical([root])
        assert _ir.arg_spec(walk) == (("mutable", name), ("plane_in", "x"))
        assert _ir.slot_of(walk) == {("mutable", name): 0, ("plane_in", "x"): 1}


def test_a_walk_digest_is_the_same_for_one_graph_built_in_two_contexts_and_differs_for_another_graph(
    ctx,
):
    """The digest names the graph, not the context: equal across two builds, different for ``mul`` against ``add``."""

    def digest(context, kind):
        return _ir.digest(_ir.canonical([loop_sink(context, kind)]))

    assert digest(ctx, "add") == digest(_ir.new_context(), "add")
    assert digest(ctx, "add") != digest(ctx, "mul")
    assert len(digest(ctx, "add")) == 64


def test_a_declared_slot_is_bound_by_the_walk_even_when_the_body_does_not_read_it(ctx):
    """A slot declared by the caller is bound in the walk and listed in its argument list; without it, it is not."""
    _x, _total, _pick, sink = sum_graph(ctx)
    mask = _ir.Slot("terminated", "alive", _ir.TensorType((), "i32", None))
    declared = _ir.canonical([sink], declared=[mask])
    plain = _ir.canonical([sink])
    assert declared.context.walk_binds(declared.number, "terminated", "alive")
    assert not plain.context.walk_binds(plain.number, "terminated", "alive")
    assert ("terminated", "alive") in [tuple(arg)[:2] for arg in _ir.arg_spec(declared)]
    assert ("terminated", "alive") not in [
        tuple(arg)[:2] for arg in _ir.arg_spec(plain)
    ]


# ---- rebuilding every node over its own operands ----


def rebuilt_all(sinks):
    """The sinks rebuilt bottom-up: every node over the rebuilt copies of its operands."""
    rebuilt = {}
    for node in _ir.post_order(sinks):
        rebuilt[node] = _ir.rebuild(
            node, [rebuilt[operand] for operand in node.operands]
        )
    return tuple(rebuilt[sink] for sink in sinks)


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_rebuilding_every_node_of_a_kernel_over_its_own_operands_reproduces_its_walk_digest(
    name,
):
    """The corpus kernels cover every node family the tracer builds; the digest of the rebuilt graph is the digest of the original."""
    kernel = CORPUS[name]
    declared = _declared(kernel.planes)
    rebuilt = _ir.cells_walk(rebuilt_all(kernel.sinks), declared)
    assert _ir.digest(rebuilt) == _ir.digest(_ir.cells_walk(kernel.sinks, declared))
