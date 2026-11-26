"""Check the binding.

Run: PYTHONPATH=<binding>/build HOST_LIB=<cpu library> [CUDA_LIB=<cuda library>]
ABI1_LIB=<binding>/build/libabi1.<dylib or so>
CONTRACT0_LIB=<binding>/build/libcontract0.<dylib or so> <python> test_binding.py
"""

import gc
import json
import os
import sys
import tempfile

import _osprey as o

LEAF = (
    "(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', '%s'), ('quantity', None), "
    "('wire_index', 0), ('sorted', False))"
)
ADD = "(('kind', 'add'), ('literal', None))"
WANT = "5822d803d90146cc81ffa03b8c357ee07e1b1faba2e520342aac9851898ed233"
done = []
not_run = []


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
        "slots": [
            {
                "role": "Mutable",
                "name": "y",
                "dtype": "F64",
                "shape": [],
                "index": 0,
                "zeroed": False,
            }
        ],
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
    "planed_scratch": [],
}


def header_abi():
    """The abi number the loader's header defines (the one source of it)."""
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "..", "include", "osprey.h")) as header:
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
foreign = raises(o.NotWritten, lambda: c2.canonical([y], "[]", None, "[]"), "Node of another Ctx")
assert foreign.code == "foreign_node", foreign.code
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

param = c.make("Param", "f64", [], None, [], "(('name', 'v'), ('access', 'read'))")
fn = c.new(
    "Function",
    "(('name', 'ident'), ('scope', 'lane'), ('params', (#0, )), ('body', #1), ('natives', ()), ('jvp', None), ('vjp', ()), "
    "('returns', TensorType(shape=(), dtype='f64', tag=None)))",
    [param, param],
)
call = c.make("Call", "f64", [], None, [x], "(('function', ('ident', '%s')), )" % fn.field_text("digest"))
assert call.field_nodes("callee") == (fn,), call.field_nodes("callee")
raises(ValueError, lambda: x.field_nodes("callee"), "a node that is no call has no callee")
done.append("a call answers the function it names")

# ---- the query surface: constants, kinds, feathers, parameter cells, sinks, matrices ----
assert c.const_names("COMPARE") == ("lt", "le", "gt", "ge", "eq", "ne"), c.const_names("COMPARE")
assert c.const_names("PARAM_CELL") == ("param_cell",)
# the two companion suffixes are two constants: the keyed accumulate's and the summing sample sum's
assert c.const_names("COMPANION_SUFFIX") == ("_c",)
assert c.const_names("SAMPLE_SUM_COMP_SUFFIX") == ("_comp",)
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


def update_of(**fields):
    """An update node of the plane `u` whose fields are `fields`: a float plane, an integer one for an or (a bit operation)."""
    text = "".join(f", ('{k}', '{v}')" for k, v in fields.items())
    dtype, value = ("i32", c.make("Leaf", "i32", [], None, [], LEAF % "bits")) if fields.get("op") == "or" else ("f64", x)
    return c.make("Update", dtype, [], None, [value], f"(('name', 'u'){text}, ('reduced', None))")


assert c.compensates(update_of(op="add", summation="compensated"))
assert not c.compensates(update_of(op="add", order="lane")) and not c.compensates(
    update_of(op="store")
)
assert not c.compensates(update_of(op="add", summation="plain")) and not c.compensates(
    update_of(op="or")
)
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
    LEAF.replace("vocab_read", "table_read").replace("plane_in", "lookup") % "t",
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
assert c.quantity_check("pos", "auto", "(('plane_in', 'x'), ('plane_in', 'm'))") is None
e = raises(
    ValueError,
    lambda: c.quantity_check("pos", "auto", "(('plane_in', 'x'), ('plane_in', 'x'))"),
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

# ---- the kinds the wings declare: block sums, ordered writes, dispatch strategies ----
fresh = o.Ctx()


def block_sum():
    """Make a block sum with no operands: refused, and the refusal says whether the kind is declared."""
    fields = "(('contract', 'auto'), ('kept', ()))"
    return fresh.make("BlockSum", "f64", [3], None, [], fields)


unknown = raises(o.NotWritten, block_sum, "a wing class is unknown before declare_kinds")
assert str(unknown) == 'unknown class "BlockSum"', str(unknown)
assert fresh.declare_kinds() is None
declared = raises(o.NotWritten, block_sum, "a wing class is known after declare_kinds")
assert str(declared) == "kind 'block_sum': takes 3..3 operand(s), got 0", str(declared)
assert fresh.declare_kinds() is None
again = raises(o.NotWritten, block_sum, "a second declare_kinds changes nothing")
assert str(again) == str(declared)
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
PER_SAMPLE = LEAF
sample_x = c.make("Leaf", "f64", [], None, [], PER_SAMPLE % "lx")
sample_k = c.make("Leaf", "i64", [], None, [], PER_SAMPLE % "lk")
cell_a = c.make("Leaf", "i64", [], None, [], PER_SAMPLE % "ca")
cell_b = c.make("Leaf", "i64", [], None, [], PER_SAMPLE % "cb")
ORDERED = "(('name', 'g'), ('role', 'plane_out'), ('kept', (('net', None),)))"
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

# the other reads of a walk, with the values the previous walk and wing gave for the same kernels
# (recorded in the library's walk.json): the selector reads of a group, the access class, the
# companion pairs of the summing sample sums, and the slot names a group owns
assert c.walk_selector_reads(walked, 0) == (("plane_in", "sel"),)
raises(
    ValueError,
    lambda: c.walk_selector_reads(walked, 3),
    "walk_selector_reads of a dispatch the walk lacks",
)
assert c.walk_access_class(walked) == "sample_local"
assert c.walk_companions(c.cells_walk([sum_a], "()", None)) == (("s", "s_comp"),)
assert c.walk_companions(c.cells_walk([plain], "()", None)) == ()
# the companions of the compensated keyed accumulates: {<plane>_c: plane} of the previous
# walk, once per plane
ACCUM = (
    "(('kind', 'update'), ('name', '%s'), ('role', 'plane_out'), "
    "('op', 'add'), ('summation', '%s'), ('reduced', None))"
)
accums = [
    c.make("Update", "f64", [], None, [x], ACCUM % pair)
    for pair in (("a", "compensated"), ("b", "auto"), ("a", "compensated"), ("c", "compensated"))
]
assert c.walk_accum_companions(c.cells_walk(accums, "()", None)) == (("a_c", "a"), ("c_c", "c"))
assert c.walk_accum_companions(c.cells_walk([plain], "()", None)) == ()
assert c.group_plane_names("dispatch3") == (
    "dispatch3_offsets",
    "dispatch3_class",
    "dispatch3_stale",
    "dispatch3_mix",
)
done.append(
    "walk_selector_reads, walk_access_class, walk_companions, walk_accum_companions, "
    "group_plane_names"
)

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

# a refusal may carry a code, a promise callers can match on; the text is not one. Substituting
# into a loop that commits once per iteration and a node of another document are the refusals with a code; every other
# refusal has code None
one_i = c.make("Const", "i32", [], None, [], "(('literal', 1),)")
refused = raises(
    o.NotWritten,
    lambda: c.substitute([by_sink], [zero_i], [one_i]),
    "substitute into a derived loop",
)
assert refused.code == "derived_loop_rewrite", refused.code
assert str(refused) == (
    "a loop that commits once per iteration (a derived loop) cannot be rewritten by substitution"
), str(refused)
other_refusal = raises(o.NotWritten, lambda: c.with_value(x, y), "a refusal without a code")
assert other_refusal.code is None
done.append("NotWritten carries the code of a refusal that has one, None for every other")

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
assert [n.cls for n in expanded[1:4]] == ["Update"] * 3 and c.blocks_expand_stores([]) == ()
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
    ("(('key', None), ('bin', 9))", (("key", None), ("bin", 9))),
    ("()", ()),
):
    params = f"(('op', 'sum'), ('value', #0), ('key', #1), ('count', #1), ('kept', {kept}))"
    assert c.new("ReduceSamples", params, [x, pair_key]).field_pairs("kept") == held, kept
assert c.new("ReduceSamples", "(('op', 'sum'), ('value', #0))", [x]).field_pairs("kept") == ()
bare = c.new("ReduceSamples", "(('op', 'sum'), ('value', #0), ('kept', ('key', 'bin')))", [x])
e = raises(ValueError, lambda: bare.field_pairs("kept"), "field_pairs: bare names are no pairs")
assert "not a pair" in str(e), str(e)
bounded = c.make(
    "LoopIndex", "int32", [], None, [], "(('depth', 0), ('name', '_'), ('bounds', (0, 3)))"
)
assert bounded.field_ints("bounds") == (0, 3)
done.append("field_pairs: kept axes with absent parts, the empty tuple, bare names refused")

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
    assert c.backend_name(lib) == "cpu"
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
    render_in = lambda cx, nodes, opened, contract: cx.backend_render(  # noqa: E731
        opened, cx.dump(nodes), "", "[]", None, "[]", [3], contract
    )[0]
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
    facts_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "osprey_cpu", "facts"
    )
    shipped = c.backend_card(lib, facts_dir, "build.json")
    assert json.loads(shipped)["lanes"]["team_threads"] == 128, shipped
    e = raises(
        o.BadRequest,
        lambda: c.backend_card(lib, facts_dir, "cpu.json"),
        "backend_card: another name",
    )
    assert "cpu.json" in str(e), str(e)
    done.append("backend_card")
    cuda_path = os.environ.get("CUDA_LIB")
    if cuda_path:
        import test_sideways

        repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        test_sideways.run(
            done,
            o,
            raises,
            {"cpu": lib_path, "cuda": cuda_path},
            {
                "cpu": os.path.join(repo, "osprey_cpu", "facts"),
                "cuda": os.path.join(repo, "osprey_cuda", "facts"),
            },
            lambda opened: render_in(c3, every, opened, json.dumps(UNIT)),
        )
    dist = os.environ.get("OSPREY_DIST")
    if dist:
        import test_end_to_end

        def make_nodes():
            """Return a context, the nodes of y = a + b and the sink position."""
            cx = o.Ctx()
            leaf = test_end_to_end.LEAF
            a, b = mk(cx, "Leaf", [], leaf % "a"), mk(cx, "Leaf", [], leaf % "b")
            total = mk(cx, "Op", [a, b], ADD)
            return cx, [a, b, total, mk(cx, "Assign", [total], test_end_to_end.ASSIGN)], [3]

        folders = {
            "cpu": os.path.join(dist, "osprey_cpu"),
            "cuda": os.path.join(dist, "osprey_cuda"),
        }
        test_end_to_end.run(done, o, raises, folders, header_abi(), make_nodes)
    assert c.backend_close(lib) is None
    done.append("backend_close")
# ---- the fold threshold: the rule is the library's ----
assert c.fold_exceeds(3, 2) is True and c.fold_exceeds(2, 2) is False
e = raises(ValueError, lambda: c.fold_exceeds(3, None), "fold_exceeds: no threshold")
assert "fold threshold" in str(e), str(e)
sum3 = c.make_op("sum", [vec3], "")  # a reduction over the three elements of vec3
assert c.fold_rolled(sum3, 2, "auto") is True and c.fold_rolled(sum3, 3, "auto") is False
assert c.fold_rolled(sum3, None, "rolled") is True and c.fold_rolled(sum3, None, "static") is False
assert c.fold_rolled(vec3, 2, "auto") is False
raises(ValueError, lambda: c.fold_rolled(sum3, None, "auto"), "fold_rolled: auto with no threshold")
done.append("fold_exceeds, fold_rolled: n > limit, the form words, a refusal with no threshold")

# ---- the key run of a keyed sink: a constant key is one run over all the lanes ----
keyed = list(expanded[1:4])
assert c.lanes_key_run_in_kernel(keyed, keyed[0], 32, 1024) == ("proven", 32, True)
done.append("lanes_key_run_in_kernel: a constant key is proven over the whole run")

# ---- warp uniformity: a lane index differs across a warp, a constant does not ----
lane = c.make("SampleIndex", "i32", [], None, [], "()")
one_i = c.make("Const", "i32", [], None, [], "(('literal', 1),)")
lane_sink = c.make("Assign", "i32", [], None, [lane], "(('name', 'o'),)")
assert c.lanes_warp_uniform([lane_sink], 32, [lane, one_i]) == (False, True)
assert c.lanes_warp_uniform([lane_sink], None, [lane, one_i]) == (False, True)
done.append("lanes_warp_uniform: a lane index is not uniform, a constant is, with or without run")

# ---- wide placement: a wide value stored whole to a mutable is placed in its plane ----
wide_in = [c.make("Leaf", "f64", [100], None, [], LEAF % n) for n in ("wa", "wb")]
wide_sum = c.make("Op", "f64", [100], None, wide_in, "(('kind', 'add'), ('literal', None))")
wide_fields = "(('name', 'wy'), ('role', 'mutable'))"
wide_sink = c.make("Assign", "f64", [100], None, [wide_sum], wide_fields)
facts = ((2, 4, 0, 8, 4096), 64, 16, "auto", "")  # runtime_min, wide_min, rolled_min, form, taken
assert c.wide_place([wide_sink], 1, *facts) == ((wide_sink,), False)
placed, changed = c.wide_place([wide_sink], 0, *facts)
(resident,) = placed[0].operands
assert changed is True and resident != wide_sum
assert c.wide_is_resident(resident) is True and c.wide_is_resident(wide_sum) is False
plane = resident.operands[1]
assert c.wide_is_scratch(plane) is True and c.wide_is_scratch(wide_in[0]) is False
assert c.wide_is_tiled(plane) is False and c.wide_is_prior(plane) is False
((row_plane, layout, width),) = c.wide_record([resident])
assert (row_plane, layout, width) == (plane, "plane_major", 100)
RW_ELEMENT = {"none": 0, "proxy": 1, "input": 2, "parts": 3, "splat": 4, "elementwise": 5}
assert c.wide_element_kind(wide_in[0]) == RW_ELEMENT["input"]
assert c.wide_element_kind(wide_sum) == RW_ELEMENT["elementwise"]
assert c.wide_element_kind(resident) == RW_ELEMENT["proxy"]
# a run of reads is a gather; its last fact says whether the reads clamp their index,
# which is the default edge rule of a read
for edge, clamps in (("(('edge', 'unchecked'),)", False), ("()", True)):
    reads = [c.make("At", "f64", [], None, [wide_in[0], ints(k)], edge) for k in (2, 4, 6)]
    run3 = c.make("Op", "f64", [3], None, reads, "(('kind', 'vec'), ('literal', None))")
    assert c.wide_gather(run3) == (wide_in[0], None, 2, 2, clamps, ())  # indices 2, 4, 6, no lead index
assert c.wide_gather(wide_sum) is None
done.append(
    "wide_place, wide_is_resident, wide_is_scratch, wide_is_tiled, wide_is_prior, wide_record, "
    "wide_element_kind, wide_gather: placement of a stored wide value and a gather of an index run"
)

# ---- the loop boundary: the maximal values of a body that do not depend on its own nodes ----
free_value = c.make_op("mul", [x, other], "")
body_next = c.make_op("add", [carry, free_value], "")
assert c.loop_boundary([body_next], [carry], []) == (free_value,)
assert c.loop_boundary([body_next], [], []) == (body_next,)
done.append("loop_boundary: the invariant product is the boundary, nothing is own then the root is")

# ---- the matrices a walk reads only through a gather ----
matrix = c.make("Leaf", "f64", [2, 3], None, [], LEAF % "gm")
row = c.make("At", "f64", [3], None, [matrix, ints(1)], "()")
gather_sink = c.make("Assign", "f64", [3], None, [c.make_op("neg", [row], "")], "(('name', 'gy'),)")
assert c.walk_gather_mats(c.cells_walk([gather_sink], "()", None)) == ("gm",)
assert c.walk_gather_mats(c.cells_walk([plain], "()", None)) == ()
done.append("walk_gather_mats: a matrix read only through a gather is named")


# ---- the quantities a walk reaches: (namespace, wires, slots) rows in the walk's order ----
def wire_leaf(slug, wires, at):
    """A read leaf of wire `at` of the quantity `slug`, carrying the whole declaration."""
    wire = "Wire(name='%s', role='plane_in', shape=(), dtype='f64', tag=None)"
    declared = ", ".join(wire % w for w in wires)
    quantity = "Quantity(slug='%s', pack='auto', wires=(%s,))" % (slug, declared)
    fields = (
        "(('kind', 'vocab_read'), ('role', 'plane_in'), ('name', '%s__%s'), ('quantity', %s), "
        "('wire_index', %d), ('sorted', False))" % (slug, wires[at], quantity, at)
    )
    return c.make("Leaf", "f64", [], None, [], fields)


pos_x, pos_v = (wire_leaf("pos", ("x", "v"), i) for i in (0, 1))
acc_w = wire_leaf("acc", ("w",), 0)
quantity_sum = c.make_op("add", [c.make_op("add", [pos_x, pos_v], ""), acc_w], "")
quantity_sink = c.make(
    "Assign", "f64", [], None, [quantity_sum], "(('name', 'qy'), ('role', 'mutable'))"
)
quantity_walk = c.cells_walk([quantity_sink], "()", None)
# Slots are ordered by role (mutable first, then plane_in); inside a role the wires of the reached
# quantities come first, in the order reached and then declared: qy 0, pos__x 1, pos__v 2, acc__w 3.
assert c.walk_quantity_wires(quantity_walk) == (("pos", ("pos__x", "pos__v")), ("acc", ("acc__w",)))
assert c.walk_quantity_slots(quantity_walk) == (("pos", (1, 2)), ("acc", (3,)))
assert c.walk_quantity_wires(c.cells_walk([plain], "()", None)) == ()
done.append("walk_quantity_wires, walk_quantity_slots: one row per reached quantity, in walk order")

# ---- the target of a compensated sink: a name and whether it is the sink's own column ----
comp_out = "(('name', 'out'), ('role', 'mutable'))"
comp_sinks = [
    c.make("Assign", "f64", [], None, [x], comp_out),
    c.make("Assign", "f64", [], None, [x], comp_out.replace("'out'", "'out_comp'")),
]
# the one output besides "out_comp" is "out"; an assignment commits to the lane's own column
assert c.split_compensated_target("compensated", "out_comp", None, "", comp_sinks) == ("out", True)
assert c.split_compensated_target("compensated", "out_comp", "out", "", comp_sinks) == ("out", True)
assert c.split_compensated_target("plain", "out_comp", None, "", comp_sinks) is None
raises(
    ValueError,
    lambda: c.split_compensated_target("compensated", "none", None, "", comp_sinks),
    "split_compensated_target: two candidate outputs",
)
done.append("split_compensated_target: the output besides the companion, None for another sink")

# ---- the manifest calls: what a valid manifest is, from folder paths and raw target names ----
ROW = {
    "request": "host",
    "target": "Host",
    "about": "a",
    "device": False,
    "entry": "host",
    "format": "host",
}


def osprey_folder(root, name, rows, **extra):
    """A folder under root holding a manifest of the OSPREY name."""
    folder = os.path.join(root, name)
    os.makedirs(folder)
    manifest = {"name": name, "version": "0", "door": 11, "targets": rows, **extra}
    with open(os.path.join(folder, "osprey.json"), "w") as out:
        out.write(json.dumps(manifest))
    return folder


assert c.backend_target_id(" Cu-Da ") == "cuda"
done.append("backend_target_id: one spelling of a target id")
e = raises(
    ValueError,
    lambda: c.backend_target_id("9x"),
    "backend_target_id: a name outside the grammar is refused",
)
assert str(e).startswith("the target id '9x' is not a target id"), str(e)
with tempfile.TemporaryDirectory() as root:
    good = osprey_folder(root, "cpu", [ROW])
    twin = osprey_folder(root, "twin", [{**ROW, "request": "Ho-st"}])
    wings = osprey_folder(root, "wings", [ROW], wings={})
    answer = json.loads(c.backend_roster(json.dumps([good, wings])))
    assert [m["name"] for m in answer["ospreys"]] == ["cpu"], answer
    assert answer["refused"][wings].endswith("a Python wing is refused"), answer
    done.append("backend_roster: the checked manifests and the folders refused with their reason")
    e = raises(
        ValueError,
        lambda: c.backend_roster(json.dumps([good, twin])),
        "backend_roster: two ospreys serving one id are refused",
    )
    assert str(e) == "target 'host' is served by both 'cpu' and 'twin'; keep one", str(e)
    made = c.backend_roster(json.dumps([good, wings]))
    assert json.loads(c.backend_serving(made, "HOST")) == {"id": "host", "osprey": "cpu"}
    done.append("backend_serving: the OSPREY that serves a target in any spelling")
    e = raises(
        ValueError,
        lambda: c.backend_serving(made, "tpu"),
        "backend_serving: a target nobody serves is refused naming the served and the refused",
    )
    head = "target 'tpu' is served by no OSPREY found; the served targets are ('host',) "
    assert str(e).startswith(head + f"(searched: {good}, {wings}); refused: "), str(e)

try:
    import numpy  # noqa: F401
except ImportError:
    print("skipped: the door rows: numpy is not installed for this interpreter")
else:
    import test_door

    # the door is the runtime of the cpu library: two opened handles of it, and the cuda
    # library's, which has none
    backends = {"cpu": c.backend_open(lib_path), "second": c.backend_open(lib_path)}
    if os.environ.get("CUDA_LIB"):
        backends["cuda"] = c.backend_open(os.environ["CUDA_LIB"])
    test_door.run(done, raises, not_run, backends)
    for opened in backends.values():
        c.backend_close(opened)
print("\n".join("ok: " + d for d in done))
if not_run:
    print("\n".join("not run: " + r for r in not_run))
    print("test_binding: %d rows not run" % len(not_run))
print("test_binding: %d checks passed" % len(done))
