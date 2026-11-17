"""Check the binding.

Run: PYTHONPATH=<binding>/build HOST_LIB=<library>
ABI1_LIB=<binding>/build/libabi1.<dylib or so>
CONTRACT0_LIB=<binding>/build/libcontract0.<dylib or so> <python> test_binding.py
"""

import gc
import json
import os
import sys

import _osprey as o

LEAF = (
    "(('kind', 'vocab_read'), ('role', 'vec_in'), ('name', '%s'), ('quantity', None), "
    "('wire_index', 0), ('sorted', False))"
)
ADD = "(('kind', 'add'), ('literal', None))"
WANT = "e5528739b51eb34fa87421825d26571b381966750395a5e74f9b131dd037afff"
done = []


# a typed host unit of the probe expression: well formed, so its refusals are the writers' own
UNIT = {
    "v": 1,
    "name": "axpb",
    "target": "Host",
    "scalar": "Float64",
    "kind": {"guard": "Default", "lanes": False, "sink": "Plain"},
    "form": "Plain",
    "sinks": [3],
    "walk": {
        "access": "SampleLocal",
        "slots": [{"role": "Mutable", "name": "y", "dtype": "F64", "shape": [], "index": 0}],
    },
    "body": "Walk",
    "rows": 1,
    "build": {
        "form": {"lazy_leaf_width": 32, "rolled_fold_min": 16, "lean_width": 4, "uniform_run": 32},
        "fold_form": "Auto",
        "lanes": {"run_max": 32, "block_run": 1024, "tile": [64, 1024]},
        "linalg_max_dim": 32,
        "cells_row": 32,
        "blend": True,
        "warps": 1,
    },
    "machine": {"isa": "Arm64", "exact_libm": True, "routed_pow": False, "select_weight": {}},
    "exports": "Derived",
    "literal_table": False,
}


def header_abi():
    """The abi number the loader's header defines (the one source of it)."""
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "include", "osprey_host.h")) as header:
        return next(int(line.split()[2]) for line in header if line.startswith("#define OB_ABI "))


def mk(c, cls, ops, fields):
    """Make one f64 scalar node of the class in the context."""
    return c.make(cls, "f64", [], None, ops, fields)


def expr(c):
    """Build the probe expression a * b + 2 as an Assign to y; return the Assign."""
    a, b = mk(c, "Leaf", [], LEAF % "a"), mk(c, "Leaf", [], LEAF % "b")
    s = mk(c, "Op", [a, b], ADD)
    e = mk(c, "Op", [s, mk(c, "Const", [], "(('literal', 2.0),)")], ADD)
    return mk(c, "Assign", [e], "(('name', 'y'),)")


def raises(exc, f, what):
    """Call f, expect the exception, note the check; return the exception."""
    try:
        f()
    except exc as e:
        done.append(what)
        return e
    raise AssertionError(f"{what}: expected {exc.__name__}")


c = o.Ctx()
y = expr(c)
w = c.canonical([y], "[]", None, "[]")
assert c.digest(w) == WANT, c.digest(w)
done.append("probe digest")
text = c.dump([y])
c2 = o.Ctx()
(r,) = c2.load(text)
assert c2.digest(c2.canonical([r], "[]", None, "[]")) == WANT
done.append("dump/load roundtrip into a second Ctx")
e = raises(
    ValueError,
    lambda: c.make("NoSuchClass", "f64", [], None, [], "()"),
    "refusal raises ValueError",
)
assert str(e), "empty message"
raises(AttributeError, lambda: setattr(y, "extra", 1), "node.extra = 1")
raises(TypeError, lambda: type("S", (o.Ctx,), {}), "subclass Ctx")
raises(TypeError, lambda: type("S", (o.Node,), {}), "subclass Node")
raises(TypeError, lambda: o.Node(), "Node()")
# the flag that freezes a type exists from 3.10; the 3.9 build leaves the types patchable
if sys.version_info >= (3, 10):
    raises(TypeError, lambda: setattr(o.Ctx, "new_method", 1), "Ctx.new_method = ...")
    raises(TypeError, lambda: setattr(o.Node, "new_method", 1), "Node.new_method = ...")
raises(TypeError, lambda: c2.canonical([y], "[]", None, "[]"), "Node of another Ctx")
pub = sorted(
    n
    for n in dir(o)
    if not n.startswith("_") and not n.startswith("EG_") and n not in ("Door", "Handle")
)
assert pub == [
    "ABI_VERSION",
    "BadRequest",
    "CONTRACT_VERSION",
    "Ctx",
    "Fault",
    "Node",
    "NotWritten",
], pub
assert issubclass(o.NotWritten, ValueError) and issubclass(o.BadRequest, ValueError)
assert issubclass(o.Fault, RuntimeError) and not issubclass(o.Fault, ValueError)
done.append("public names " + ",".join(pub))

# ---- the front end's surface: read back what was made, build a loop and a substitution ----
x = mk(c, "Leaf", [], LEAF % "x")
assert (x.cls, x.ttype, x.operands) == ("Leaf", ((), "f64", None), ())
done.append("node.cls, node.ttype, node.operands")
assert x == x and x != y and hash(x) == hash(x) and x != 3
assert x.operands == () and y.operands[0] == y.operands[0]
done.append("nodes compare and hash by handle and context")
assert x.ctx is c and y.operands[0].ctx is c and c2.load(text)[0].ctx is c2
raises(AttributeError, lambda: setattr(x, "ctx", c2), "node.ctx = ...")
# a node keeps its context alive: the context object dies only after the last node that reads it
probe = o.Ctx()
survivor = mk(probe, "Leaf", [], LEAF % "s")
del probe
gc.collect()
assert survivor.ctx.is_a(survivor, "leaf") and survivor.operands == ()
done.append("node.ctx: the context of the node, read-only, kept alive by the node")
tagged = c.make("Leaf", "f64", [3, 2], "t", [], LEAF % "tagged")
assert tagged.ttype == ((3, 2), "f64", "t"), tagged.ttype
done.append("node.ttype with a shape and a tag")


def literal_node(text):
    """An op whose literal field holds the value of the key text."""
    return mk(c, "Op", [x], "(('kind', 'add'), ('literal', %s))" % text)


for text, kind, reader, want in [
    ("None", 0, "field_text", None),
    ("'abc'", 1, "field_text", "abc"),
    ("7", 2, "field_int", 7),
    ("2.5", 3, "field_real", 2.5),
    ("True", 4, "field_flag", True),
    ("('a', 'b')", 6, "field_names", ("a", "b")),
    ("(1, -2, 3)", 7, "field_ints", (1, -2, 3)),
    ("()", 7, "field_ints", ()),
]:
    n = literal_node(text)
    assert n.field_kind("literal") == kind, (text, n.field_kind("literal"))
    assert getattr(n, reader)("literal") == want, (text, reader)
raises(ValueError, lambda: literal_node("'abc'").field_int("literal"), "a reader of another kind")
raises(ValueError, lambda: x.field_text("no_such_field"), "a field the node does not have")
done.append("every field kind reads back typed (absent, text, int, real, flag, names, ints)")

idx = c.make("LoopIndex", "i32", [], None, [], "(('depth', 0), ('name', 'k'), ('bounds', None))")
carry = mk(c, "LoopCarry", [], "(('depth', 0), ('slot', 0), ('name', 's'))")
init = mk(c, "Const", [], "(('literal', 0.0),)")
step_add = mk(c, "Op", [carry, x], ADD)
loop = c.new_loop(idx, [carry], [init], [step_add], 0, 3, 1, [], None, [], None, 0, None)
assert loop.cls == "Loop" and loop.operands == (init, x), loop.operands
assert loop.field_nodes("carries") == (carry,) and loop.field_kind("carries") == 5
assert loop.field_int("start") == 0 and loop.field_int("stop") == 3
assert loop.field_ints("body_sinks") == () and loop.field_kind("body_sinks") == 7
assert loop.field_text("exit_cond") is None and loop.field_text("dynamic") is None
done.append("new_loop: several lists, optional nodes, integers; its operands and fields read back")

# ---- the query surface: constants, kinds, feathers, parameter cells, sinks, matrices ----
assert c.const_names("COMPARE") == ("lt", "le", "gt", "ge", "eq", "ne"), c.const_names("COMPARE")
assert c.const_names("PARAM_CELL") == ("param_cell",)
# the lists the core composes (linear-algebra kinds, the cast kind, every op kind a body can mint),
# composed here as the previous package composed them
linalg_ops = tuple(
    k for n in ("FACTORISATIONS", "CONSUMERS", "PROJECTIONS") for k in c.const_names("LINALG_" + n)
)
assert len(linalg_ops) == 20 and c.const_names("LINALG_OPS") == linalg_ops
assert c.const_names("CAST") == ("cast",)
op_parts = (
    "ELEMENTWISE_BINARY UNARY_MATH COMPARE LOGICAL VECTOR_OPS MATRIX_OPS LINALG_OPS LSTSQ_OPS "
    "RAY_OPS QUAT_OPS RANDOM_OPS"
).split()
op_kinds = sum((c.const_names(n) for n in op_parts), ()) + ("select", "dispatch", "at")
op_kinds += sum(
    (c.const_names(n) for n in "ELEMENTWISE_TERNARY PREDICATE INT_BINARY INT_UNARY".split()), ()
)
op_kinds += c.const_names("CAST")
assert len(op_kinds) == 121 and c.const_names("OP_KINDS") == op_kinds
done.append("const_names: LINALG_OPS, CAST and OP_KINDS")
raises(ValueError, lambda: c.const_names("NO_SUCH_CONSTANT"), "const_names of a name nobody owns")
done.append("const_names: a list, a string, a refusal")
assert c.const_names("ACCESS_CLASSES")[-1] == "mapreduce" and c.const_names("ADJOINT_PREFIX") == (
    "bar",
)
done.append("const_names of the constants the manifest gained")
assert c.compensates("compensated") and c.compensates("keyed_auto") and not c.compensates("ordered")
done.append("compensates")
assert c.is_a(x, "leaf") and not c.is_a(x, "op") and not c.is_a(x, "NoSuchKind")
done.append("is_a")
sink = mk(c, "Assign", [x], "(('name', 'out'),)")
assert not c.has_feather(sink, "commits_value") and not c.has_feather(x, "reduces_samples")
reduced = mk(
    c,
    "ReduceSamples",
    [x],
    "(('op', 'sum'), ('contract', 'auto'), ('key', None), ('count', None), ('kept', None))",
)
assert c.has_feather(reduced, "reduces_samples") and not c.has_feather(reduced, "commits_value")
raises(ValueError, lambda: c.has_feather(x, "no_such_feather"), "has_feather of an unknown feather")
done.append("has_feather")
# a declared dialect kind states its feathers: the simt wing declares a static-index terminal
wings = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "raptor", "rs", "wings", "manifests"
)
for manifest in ("base.json", "simt.json"):
    with open(os.path.join(wings, manifest)) as text:
        c.declare_dialect(text.read(), "('base', 'simt')")
lane = c.make("WarpLane", "i32", [], None, [], "()")
assert c.has_feather(lane, "static_index") and not c.has_feather(lane, "fresh_read")
done.append("has_feather of a declared dialect kind")
table = c.make(
    "Leaf",
    "f64",
    [4],
    None,
    [],
    LEAF.replace("vocab_read", "table_read").replace("vec_in", "lookup") % "t",
)
cell = c.make("RoleConst", "i32", [], None, [], "(('literal', 0), ('label', 'param_cell'))")
other = c.make("RoleConst", "i32", [], None, [], "(('literal', 0), ('label', 'other'))")
assert c.is_param_cell(c.make("At", "f64", [], None, [table, cell], "()"))
assert not c.is_param_cell(c.make("At", "f64", [], None, [table, other], "()"))
assert not c.is_param_cell(x)
done.append("is_param_cell")
read = c.param_cell("gain", "f64", [], None, False)
assert c.is_param_cell(read) and read.cls == "At" and read.ttype == ((), "f64", None)
assert not c.is_param_cell(c.param_cell("gain", "f64", [], None, True))
assert c.param_cell("offsets", "f64", [3], "xs", False).ttype == ((3,), "f64", "xs")
done.append("param_cell")


def ints(value):
    """A 32-bit integer constant."""
    return c.make("Const", "i32", [], None, [], "(('literal', %d),)" % value)


strided = c.make_op("add", [c.make_op("mul", [idx, ints(4)], ""), ints(2)], "")
assert c.index_form(strided, idx, True) == ("form", "4/1", "2/1", "1", "-")
halved = c.make_op("div", [strided, ints(2)], "")
assert c.index_form(halved, idx, True) == ("form", "2/1", "1/1", "1", "-")
assert c.index_form(halved, idx, False) == ("none",)
assert c.index_form(ints(7), None, True) == ("form", "0/1", "7/1", "0", "-")
done.append("index_form")
assert c.quantity_check("pos", "auto", "(('vec_in', 'x'), ('mat_in', 'm'))") is None
e = raises(
    ValueError,
    lambda: c.quantity_check("pos", "auto", "(('vec_in', 'x'), ('vec_in', 'x'))"),
    "a duplicate wire",
)
assert str(e) == "Quantity 'pos': duplicate wire name 'x'", str(e)
done.append("quantity_check")
lit = lambda text: c.make("Const", "f64", [], None, [], "(('literal', %s),)" % text)  # noqa: E731
twin_a, twin_b, other, negative, nan_a, nan_b = (
    lit(t) for t in ("1.5", "1.5", "2.5", "-0.0", "nan", "nan")
)
keys = c.structural_keys([twin_a, twin_b, other, lit("0.0"), negative, nan_a, nan_b])
assert (
    keys[0] == keys[1] and keys[2] not in keys[:2] and keys[3] != keys[4] and keys[5] == keys[6]
), keys
assert c.structural_keys([]) == ()
done.append("structural_keys: a value twice is one key; -0.0 is not 0.0; nan is nan")
moved = c.with_value(sink, y)
assert moved.cls == "Assign" and moved.operands == (y,) and moved.field_text("name") == "out"
raises(ValueError, lambda: c.with_value(x, y), "with_value on a sink that commits no value")
done.append("with_value")
chol = c.make("Leaf", "f64", [2, 2], "chol_factor", [], LEAF % "chol")
lu = c.make("Leaf", "f64", [2, 2], "lu_factor", [], LEAF % "lu")
assert c.as_matrix(x) == x and c.as_matrix(chol).cls == "Op"
raises(ValueError, lambda: c.as_matrix(lu), "as_matrix of an LU factor")
done.append("as_matrix")

# ---- the loop rules: the terms of a sum, nested loops, accumulator or recurrence ----
assert c.add_terms(step_add) == (carry, x) and c.add_terms(x) == (x,)
done.append("add_terms")
assert c.refuse_nesting(loop) is None
done.append("refuse_nesting")
assert c.loop_is_accumulator(loop) is True and c.loop_addends(loop) == (x,)
done.append("loop_is_accumulator and loop_addends of an accumulator")
scaled = mk(c, "Op", [carry, x], "(('kind', 'mul'), ('literal', None))")
recurrence = c.new_loop(idx, [carry], [init], [scaled], 0, 3, 1, [], None, [], None, 0, None)
assert c.loop_is_accumulator(recurrence) is False
raises(ValueError, lambda: c.loop_addends(recurrence), "loop_addends of a recurrence")
done.append("loop_is_accumulator of a recurrence")
# where a break sits, as the exit_form: it yields the carries (head), the next values (tail), others
assert recurrence.field_text("exit_form") is None
breaks = c.make_op("gt", [carry, x], "")
for yields, form in (([carry], "head"), ([scaled], "tail"), ([x], "middle")):
    broken = c.new_loop(idx, [carry], [init], [scaled], 0, 3, 1, [], breaks, yields, None, 0, None)
    assert broken.field_text("exit_form") == form, (form, broken.field_text("exit_form"))
done.append("exit_form: none, head, tail, middle")
raises(
    ValueError,
    lambda: c.new_loop(
        idx, [carry], [init, init], [step_add], 0, 3, 1, [], None, [], None, 0, None
    ),
    "new_loop: lists of one group of different length",
)
raises(
    TypeError,
    lambda: c.new_loop(idx, [carry], [init], [step_add], 0, 3, 1, [], 5, [], None, 0, None),
    "new_loop: an optional node that is not a Node",
)
other = mk(c, "Leaf", [], LEAF % "z")
(swapped,) = c.substitute([step_add], [x], [other])
assert swapped.operands == (carry, other) and step_add.operands == (carry, x)
assert c.substitute([step_add], [], []) == (step_add,)
raises(
    ValueError,
    lambda: c.substitute([step_add], [x], [other, other]),
    "substitute: lists of one group of different length",
)
done.append("substitute: the mapped node is replaced, the original is kept")
vec = c.make("Leaf", "f64", [2], None, [], LEAF % "v")
zero = mk(c, "Const", [], "(('literal', 0.0),)")


def add_at(base, i, v):
    """An add_at op of two elements over the base."""
    fields = "(('kind', 'add_at'), ('literal', None))"
    return c.make("Op", "f64", [2], None, [base, i, v], fields)


first = add_at(vec, zero, x)
second = add_at(first, zero, other)
assert c.slotwise_chain(second) == (vec, (zero, x, zero, other))
assert c.slotwise_chain(vec) == (vec, ())
assert (
    c.slotwise_is_structural_zero(
        c.make("Op", "f64", [2], None, [zero], "(('kind', 'splat'), ('literal', None))")
    )
    is True
)
assert c.slotwise_is_structural_zero(vec) is False
done.append("slotwise_chain and slotwise_is_structural_zero")
mul = c.make_op_like("mul", [x, other], None)
carried = c.make_op_like("mul", [x, other], literal_node("(1, 2)"))
assert mul.field_kind("literal") == 0 and carried.field_ints("literal") == (1, 2)
raises(
    ValueError,
    lambda: c.make_op_like("no_such_kind", [x], None),
    "make_op_like: a kind that does not exist",
)
done.append("make_op_like: with no literal and with the literal of another node")

# ---- the derivative rules: the manifest's templates over the nodes of the context ----
vec3 = c.make("Leaf", "f64", [3], None, [], LEAF % "v")
tan3 = c.make("Leaf", "f64", [3], None, [], LEAF % "tv")
adj = mk(c, "Leaf", [], LEAF % "g")
tan = mk(c, "Leaf", [], LEAF % "tx")
assert len(c.rule_kinds("unary_d")) == 26 and c.rule_kinds("core")[:2] == ("fma", "clip")
assert c.rule_kinds("no_such_table") == ()
assert c.rule_modes("zero")[:2] == ("lt", "le") and c.rule_modes("walker") == ("at", "dispatch")
done.append("rule_kinds and rule_modes")
negated = c.make_op("neg", [x], "")
(slope,) = c.rule_factor("unary_d", "neg", "n x", [negated, x], "")
assert slope.cls == "Const" and slope.field_real("literal") == -1.0
raises(
    ValueError,
    lambda: c.rule_factor("unary_d", "neg", "n", [negated, x], ""),
    "rule_factor: a name for two nodes",
)
raises(
    ValueError,
    lambda: c.rule_factor("unary_d", "no_such", "n x", [negated, x], ""),
    "rule_factor: no template",
)
done.append("rule_factor: a factor, and its refusals")
product = c.make_op("mul", [x, adj], "")
reverse = c.rule_factor("bin_vjp", "mul", "n g a b", [product, adj, x, adj], "")
assert len(reverse) == 4 and (reverse[0], reverse[2]) == (x, adj)
assert reverse[1].field_text("kind") == "mul"
forward = c.rule_factor("bin_jvp", "mul", "n a b ta", [product, x, adj, tan], "tb")
assert len(forward) == 1 and forward[0].field_text("kind") == "mul"
done.append("rule_factor: the pairs of bin_vjp and the live entries of bin_jvp")
scalar_product = c.make_op("dot", [vec3, vec3], "")
back = c.rule_vjp("core", "dot", scalar_product, adj)
assert len(back) == 4 and back[0] == vec3 and back[2] == vec3
done.append("rule_vjp")
length = c.make_op("norm", [vec3], "")
(rate,) = c.rule_jvp("core", "norm", length, [vec3], [tan3])
assert rate.ttype == ((), "f64", None)
raises(
    ValueError,
    lambda: c.rule_jvp("core", "norm", length, [], []),
    "rule_jvp: no tangent to a kind that needs one",
)
done.append("rule_jvp: a tangent, and a refusal")
total = c.new("ReduceSamples", "(('op', 'sum'), ('value', #0))", [x])
child, summed = c.reduce_vjp(total, adj)
assert child == x and summed.cls == "ReduceSamples" and summed.operands == (adj,)
(summed,) = c.reduce_jvp(total, [x], [tan])
assert summed.operands == (tan,) and c.reduce_jvp(total, [], []) == ()
largest = c.new("ReduceSamples", "(('op', 'max'), ('value', #0))", [x])
e = raises(ValueError, lambda: c.reduce_vjp(largest, adj), "reduce_vjp of a max")
assert "no derivative for the max over samples" in str(e), str(e)
done.append("reduce_vjp and reduce_jvp")
widened = c.entrywise("add", [vec3, x])
assert widened.field_text("kind") == "vec" and len(widened.operands) == 3
assert widened.ttype == ((3,), "f64", None)
assert c.entrywise("add", [x, adj]).field_text("kind") == "add"
done.append("entrywise: per entry over a vector, and as itself over scalars")
assert c.zero_of("f64", []).cls == "Const"
filled = c.zero_of("f64", [3])
assert filled.field_text("kind") == "splat" and filled.ttype == ((3,), "f64", None)
done.append("zero_of")

# ---- the checks of a derivative template, and the kind names a primitive may not take ----
inside = mk(c, "Leaf", [], LEAF.replace("vocab_read", "uniform") % "$in0")
stranger = mk(c, "Leaf", [], LEAF.replace("vocab_read", "uniform") % "zz")
closed = c.make_op("add", [inside, inside], "")
assert c.rule_template_check("p", [closed], "$in0 $dot0 $out") is None
e = raises(
    ValueError,
    lambda: c.rule_template_check("p", [c.make_op("add", [inside, stranger], "")], "$in0 $out"),
    "rule_template_check: a free leaf that is not a placeholder",
)
assert str(e).startswith("primitive 'p': a rule template reads"), str(e)
assert "'zz'" in str(e) and "(['$in0', '$out'])" in str(e), str(e)
done.append("rule_template_check: a closed template, and a refusal by name")
assert c.kind_reserved("primitive") and c.kind_reserved("assign")
assert not c.kind_reserved("my_primitive") and not c.kind_reserved("add")
done.append("kind_reserved")

# ---- the kinds the wings declare: block sums, ordered writes, dispatch strategies ----
wing_kinds = (
    "block_sum",
    "block_sum_write",
    "ordered_accum_write",
    "predicated_dispatch",
    "segmented_dispatch",
    "auto_dispatch",
)
fresh = o.Ctx()
assert not any(fresh.kind_reserved(k) for k in wing_kinds)
assert fresh.declare_kinds() is None and all(fresh.kind_reserved(k) for k in wing_kinds)
assert fresh.declare_kinds() is None
done.append("declare_kinds: the wing kinds exist after it, and a second call changes nothing")

# ---- the block predicates of a keyed-sum payload (values recorded from the previous package) ----
mat = c.make("Leaf", "f64", [4, 3], None, [], LEAF % "M")
rhs = c.make("Leaf", "f64", [4], None, [], LEAF % "r")
mat_t = c.make_op("transpose", [mat], "")
gram, normal = c.make_op("mm", [mat_t, mat], ""), c.make_op("mv", [mat_t, rhs], "")
assert c.block_factors(gram) == (mat, mat) and c.block_factor_rows(gram) == 4
assert c.block_factors(normal) == (mat, rhs) and c.block_factor_rows(normal) == 4
outer = c.make_op("outer", [vec3, vec3], "")
assert c.block_factors(outer) == (vec3, vec3) and c.block_factor_rows(outer) == 1
assert c.block_factors(c.make_op("add", [vec3, vec3], "")) == () and c.block_factors(vec3) == ()
raises(
    ValueError, lambda: c.block_factor_rows(vec3), "block_factor_rows of a payload without factors"
)
done.append("block_factors and block_factor_rows")
assert c.block_ordered_case(vec3, "ordered", []) and c.block_ordered_case(vec3, "ordered", [3])
assert not c.block_ordered_case(vec3, "ordered", [2]) and not c.block_ordered_case(vec3, "auto", [])
assert not c.block_ordered_case(mat, "ordered", [])
done.append("block_ordered_case")
assert c.block_shaped(gram, "auto", []) and c.block_shaped(gram, "auto", [3])
assert not c.block_shaped(gram, "auto", [2]) and not c.block_shaped(vec3, "auto", [])
assert c.block_shaped(vec3, "ordered", []) and not c.block_shaped(mat, "ordered", [])
done.append("block_shaped")

# ---- the slotwise rewrite (values recorded from the previous package's pass) ----
zero = c.make("Const", "f64", [], None, [], "(('literal', 0.0),)")
parts = [mk(c, "Leaf", [], LEAF % n) for n in ("sa", "sb", "sc")]
wide, zeroed = c.make_op("vec", parts, ""), c.make_op("vec", [zero, parts[1], zero], "")
(picked,) = c.slotwise([c.make_op("component", [wide], "1")])
assert picked == parts[1]
(summed,) = c.slotwise([c.make_op("sum", [zeroed], "")])
assert summed == parts[1]
(added,) = c.slotwise([c.make_op("add", [wide, zeroed], "")])
assert added.field_text("kind") == "vec" and len(added.operands) == 3
# the zeros fold away: slot 0 and 2 are the parts themselves, slot 1 the sum of the two
assert added.operands[0] == parts[0] and added.operands[2] == parts[2]
assert (
    added.operands[1].field_text("kind") == "add" and added.operands[1].operands == (parts[1],) * 2
)
assert c.slotwise([wide, vec3]) == (wide, vec3) and c.slotwise([]) == ()
done.append("slotwise: component, sum and add per slot, unchanged roots, no roots")

# ---- the log lowering of ordered accumulates ----
PER_SAMPLE = LEAF.replace("vec_in", "per_sample")
sample_x = c.make("Leaf", "f64", [], None, [], PER_SAMPLE % "lx")
sample_k = c.make("Leaf", "i64", [], None, [], PER_SAMPLE % "lk")
cell_a = c.make("Leaf", "i64", [], None, [], PER_SAMPLE % "ca")
cell_b = c.make("Leaf", "i64", [], None, [], PER_SAMPLE % "cb")
ORDERED = "(('name', 'g'), ('role', 'accum_out'), ('kept', (('net', None),)))"
site_a = c.make("OrderedAccumWrite", "f64", [], None, [sample_x, cell_a, sample_k], ORDERED)
site_b = c.make("OrderedAccumWrite", "f64", [], None, [sample_x, cell_b, sample_k], ORDERED)
logged = c.logs_ordered([site_a, site_b])
assert len(logged) == 4 and [n.cls for n in logged] == ["LogStore"] * 4
assert c.logs_ordered([sample_x]) == (sample_x,) and c.logs_ordered([]) == ()
vector = c.make("Leaf", "f64", [3], None, [], PER_SAMPLE % "lv")
not_scalar = c.make("OrderedAccumWrite", "f64", [3], None, [vector, sample_k, sample_k], ORDERED)
e = raises(ValueError, lambda: c.logs_ordered([not_scalar]), "logs_ordered of a vector accumulate")
assert "the one into g is not a scalar" in str(e), str(e)
done.append(
    "logs_ordered: two sites make four stores, other sinks pass, a vector is refused by name"
)

# ---- the cell sinks of a reduction over samples (shapes recorded from the previous package) ----
total = c.new("ReduceSamples", "(('op', 'sum'), ('value', #0))", [x])
(cell,) = c.cells_stored(total, "out")
assert cell.cls == "SampleSum" and cell.field_text("name") == "out" and cell.operands == (x,)
columns = c.cells_stored(c.new("ReduceSamples", "(('op', 'sum'), ('value', #0))", [vec3]), "out")
assert [n.field_text("name") for n in columns] == ["out_0", "out_1", "out_2"]
assert all(n.field_text("cell") == "out" and n.field_int("columns") == 3 for n in columns)
raises(ValueError, lambda: c.cells_stored(x, "out"), "cells_stored of a node that is no reduction")
done.append("cells_stored: a scalar sum, one sink per component, a refusal")

# ---- the kernel's walk: companion, finished-count and branch-mix slots ----
plain = c.new("Assign", "(('name', 'o'), ('value', #0))", [x])
finish = c.new("Finish", "(('name', 'done'), ('value', #0))", [c.make_op("gt", [x, x], "")])
assert not c.walk_binds(c.cells_walk([plain], "()", None), "lookup", "finished_count")
assert c.walk_binds(c.cells_walk([plain, finish], "()", None), "lookup", "finished_count")
sum_a = c.new("SampleSum", "(('name', 's'), ('value', #0), ('op', 'sum'))", [x])
sum_b = c.new("SampleSum", "(('name', 's_comp'), ('value', #0), ('op', 'sum'))", [x])
e = raises(ValueError, lambda: c.cells_walk([sum_a, sum_b], "()", None), "cells_walk: a collision")
assert str(e) == (
    "sample_sum 's': its compensation plane 's_comp' collides with a plane of the same name"
), str(e)
done.append("cells_walk: the finished-count slot, and the compensation collision")

# ---- the typed reads of a walk: slot types, prior reads, strategies, static selectors ----
selector = c.make("Leaf", "i32", [], None, [], PER_SAMPLE % "sel")
cases_ = [c.make_op("add", [x, x], ""), c.make_op("mul", [x, x], "")]
core = c.new(
    "Dispatch",
    "(('selector', #0), ('branches', (#1, #2)), "
    "('ttype', TensorType(shape=(), dtype='f64', tag=None)))",
    [selector, *cases_],
)
auto = c.make("AutoDispatch", "f64", [], None, core.operands, "()")
walked = c.cells_walk([c.new("Assign", "(('name', 'o'), ('value', #0))", [auto])], "()", None)
assert c.walk_strategies(walked) == ("auto",) and c.walk_static_selector(walked, 0) is True
assert c.walk_slot_type(walked, "lookup", "dispatch0_mix") == ((), "uint32", None)
assert c.walk_prior(walked) == ()
# the previous package named the mix slot of an automatic group `dispatch<i>_mix`; the walk bound
# no mix slot for any other group
assert c.walk_mix_slots(walked) == (("dispatch0", "dispatch0_mix"),)
raises(
    ValueError,
    lambda: c.walk_slot_type(walked, "lookup", "no_such"),
    "walk_slot_type of a slot the walk does not bind",
)
raises(
    ValueError,
    lambda: c.walk_static_selector(walked, 3),
    "walk_static_selector of a dispatch the walk lacks",
)
done.append("walk_slot_type, walk_prior, walk_strategies and walk_static_selector")


# the mix slot of each dispatch group: `dispatch<i>_mix` for an automatic group, None (never a
# text) for any other
def groups_of(*forms):
    """The mix slots of the walk of a kernel that assigns each dispatch form to its own plane."""
    sinks = [
        c.new("Assign", f"(('name', 'o{i}'), ('value', #0))", [dispatch])
        for i, dispatch in enumerate(forms)
    ]
    return c.walk_mix_slots(c.cells_walk(sinks, "()", None))


def form(word):
    """The dispatch of the strategy word over the selector and the two cases above."""
    if word == "switch":
        return core
    return c.make(word.capitalize() + "Dispatch", "f64", [], None, core.operands, "()")


mixes = (("switch", None), ("predicated", None), ("segmented", None), ("auto", "dispatch0_mix"))
for word, mix in mixes:
    assert groups_of(form(word)) == (("dispatch0", mix),), word
assert c.walk_mix_slots(c.cells_walk([plain], "()", None)) == ()
done.append("walk_mix_slots: an automatic group names its mix slot, every other form is None")

# a sample_position read inside a loop body is in the flat order of the walk, as in the previous
# walk's `order`, so a scan of the order answers what its `reads_sample_position` answered (True
# for a read in a carry's next value and for one in a body sink)
position = c.make("SamplePosition", "i32", [], None, [], "()")
k = c.make("LoopIndex", "i32", [], None, [], "(('depth', 0), ('name', 'k'), ('bounds', None))")
carry_i = c.make("LoopCarry", "i32", [], None, [], "(('depth', 0), ('slot', 0), ('name', 's'))")
zero_i = c.make("Const", "i32", [], None, [], "(('literal', 0),)")
body_next = c.make("Op", "i32", [], None, [carry_i, position], ADD)
by_next = c.new_loop(k, [carry_i], [zero_i], [body_next], 0, 3, 1, [], None, [], None, 0, None)
body_sink = c.new("Assign", "(('name', 't'), ('value', #0))", [position])
by_sink = c.new_loop(
    k, [carry_i], [zero_i], [carry_i], 0, 3, 1, [body_sink], None, [], None, 0, None
)
for looped in (by_next, by_sink):
    walk = c.cells_walk([c.new("Assign", "(('name', 'o'), ('value', #0))", [looped])], "()", None)
    assert [n.cls for n in c.walk_order(walk) if n.cls == "SamplePosition"] == ["SamplePosition"]
unread = c.cells_walk([c.new("Assign", "(('name', 'o'), ('value', #0))", [zero_i])], "()", None)
assert not any(n.cls == "SamplePosition" for n in c.walk_order(unread))
done.append("walk_order holds a sample_position read inside a loop body, as the old order did")

# ---- block-sum writes expanded into per-entry accumulates ----
c.declare_kinds()
zero_key = c.make("Const", "i32", [], None, [], "(('literal', 0),)")
one_count = c.make("Const", "i32", [], None, [], "(('literal', 1),)")
blocks = c.make(
    "BlockSum",
    "f64",
    [3],
    None,
    [vec3, zero_key, one_count],
    "(('contract', 'auto'), ('kept', ()))",
)
stored_block = c.make("BlockSumWrite", "f64", [], None, [blocks], "(('name', 'g'),)")
expanded = c.blocks_expand_stores([plain, stored_block, plain])
assert len(expanded) == 5 and expanded[0] == plain and expanded[4] == plain
assert [n.cls for n in expanded[1:4]] == ["AccumWrite"] * 3 and c.blocks_expand_stores([]) == ()
done.append(
    "blocks_expand_stores: a vector block makes three accumulates, other sinks stay in order"
)

# ---- the dispatch strategies, derived from the wing manifests ----
assert c.dispatch_strategies() == ("switch", "predicated", "segmented", "auto")
done.append("dispatch_strategies")

# ---- the number of a root in its dump ----
dumped = c.dump([picked, x])
lines = [int(line[len("root|#") :]) for line in dumped.splitlines() if line.startswith("root|#")]
assert list(c.dump_numbers([picked, x])) == lines and c.dump_numbers([]) == ()
done.append("dump_numbers: the line of each root in the dump")

# ---- tuple fields of pairs: an absent part is None, never a text ----
pair_key = c.make("Leaf", "i32", [], None, [], LEAF % "pk")
for kept, held in (
    ("(('net', None),)", (("net", None),)),
    ("((None, 4), ('a', None))", ((None, 4), ("a", None))),
    ("()", ()),
):
    params = f"(('op', 'sum'), ('value', #0), ('key', #1), ('count', #1), ('kept', {kept}))"
    assert c.new("ReduceSamples", params, [x, pair_key]).field_pairs("kept") == held, kept
assert c.new("ReduceSamples", "(('op', 'sum'), ('value', #0))", [x]).field_pairs("kept") == ()
bounded = c.make(
    "LoopIndex", "int32", [], None, [], "(('depth', 0), ('name', '_'), ('bounds', (0, 3)))"
)
assert bounded.field_ints("bounds") == (0, 3)
done.append("field_pairs: kept axes with absent parts, and the empty tuple")

# ---- the names the wings own ----
special = c.wing_names("SPECIAL_KINDS")
assert "exp" in special and "pow" in special and "add" not in special and len(special) > 20
assert c.wing_names("FUSED_STEPS_PLANE") == ("fused_steps",)
assert c.wing_names("FINISHED_PLANE") == ("finished_count",)
raises(
    ValueError,
    lambda: c.wing_names("NO_SUCH_NAME"),
    "wing_names of a name no wing owns",
)
done.append("wing_names: SPECIAL_KINDS and the reserved planes")

# ---- the backend library entry: open by path, name, identity, render, close (HOST_LIB names the
# library) ----
lib_path = os.environ.get("HOST_LIB")
if lib_path:
    raises(
        ValueError,
        lambda: c.backend_open("/nonexistent/libnothing.so"),
        "backend_open: a path that is not a library",
    )
    e = raises(
        ValueError,
        lambda: c.backend_open(
            "/usr/lib/libSystem.B.dylib" if sys.platform == "darwin" else "libc.so.6"
        ),
        "backend_open: a library without osprey_v1",
    )
    assert str(e).startswith("open: ") and "osprey_v1" in str(e), str(e)
    old_abi = os.environ.get("ABI1_LIB")
    if old_abi:
        e = raises(
            ValueError,
            lambda: c.backend_open(old_abi),
            "backend_open: abi number 1 is refused by name",
        )
        assert str(e) == f"open: abi number 1, expected {header_abi()}", str(e)
    other_contract = os.environ.get("CONTRACT0_LIB")
    if other_contract:
        e = raises(
            ValueError,
            lambda: c.backend_open(other_contract),
            "backend_open: another contract version is refused by name",
        )
        want = "open: contract version %d, expected %d"
        assert str(e) == want % (o.CONTRACT_VERSION + 1, o.CONTRACT_VERSION), str(e)
    lib = c.backend_open(lib_path)
    done.append("backend_open")
    assert c.backend_name(lib) == "host"
    done.append("backend_name")
    ident = c.backend_identity(lib)
    assert len(ident) == 64 and int(ident, 16) >= 0, ident
    done.append("backend_identity")
    e = raises(
        ValueError,
        lambda: c.backend_render(lib, "not a dump", "", "()", None, "()", [], "{}"),
        "backend_render: a bad dump",
    )
    assert str(e).startswith("load: "), str(e)
    c3 = o.Ctx()
    a3, b3 = mk(c3, "Leaf", [], LEAF % "a"), mk(c3, "Leaf", [], LEAF % "b")
    s3 = mk(c3, "Op", [a3, b3], ADD)
    every = [a3, b3, s3, mk(c3, "Assign", [s3], "(('name', 'y'),)")]
    render = lambda contract: c3.backend_render(  # noqa: E731
        lib, c3.dump(every), "", "[]", None, "[]", [3], contract
    )
    e = raises(o.BadRequest, lambda: render("{}"), "backend_render: a bad request")
    assert str(e).startswith("render: not in this build"), str(e)
    banded = json.dumps({**UNIT, "scalar": "Banded"})
    e = raises(o.NotWritten, lambda: render(banded), "backend_render: a construct not written")
    assert str(e) == "render: not in this build: banded mode", str(e)
    lost = json.dumps({**UNIT, "sinks": [99]})
    e = raises(o.Fault, lambda: render(lost), "backend_render: a fault of the compiled side")
    assert str(e).startswith("render: least squares (panic: "), str(e)
    assert render(json.dumps(UNIT))[0].startswith("// HAWK-emitted host translation unit")
    # a dump whose only root is the sink names the same nodes by position: the same text comes back
    sink_only = c3.backend_render(
        lib, c3.dump(every[3:]), "", "[]", None, "[]", [3], json.dumps(UNIT)
    )
    assert sink_only[0] == render(json.dumps(UNIT))[0], "a sink-only dump renders the same unit"
    done.append("backend_render")
    e = raises(
        o.BadRequest,
        lambda: c.backend_set_text(lib, '{"v": 1, "units": []}'),
        "backend_set_text: an empty list of units",
    )
    assert str(e) == "set_text: a unit set without a unit", str(e)
    done.append("backend_set_text")
    facts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "facts")
    shipped = c.backend_card(lib, facts_dir, "build.json")
    assert json.loads(shipped)["lanes"]["team_threads"] == 128, shipped
    e = raises(
        o.BadRequest,
        lambda: c.backend_card(lib, facts_dir, "cpu.json"),
        "backend_card: another name",
    )
    assert "cpu.json" in str(e), str(e)
    done.append("backend_card")
    assert c.backend_close(lib) is None
    done.append("backend_close")
try:
    import numpy  # noqa: F401
except ImportError:
    print("skipped: the door rows: numpy is not installed for this interpreter")
else:
    import test_door

    test_door.run(done, raises)
print("\n".join("ok: " + d for d in done))
print("test_binding: %d checks passed" % len(done))
