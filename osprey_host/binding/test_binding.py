"""Check the binding.

Run: PYTHONPATH=<binding>/build HOST_LIB=<library>
ABI1_LIB=<binding>/build/libabi1.<dylib or so>
CONTRACT0_LIB=<binding>/build/libcontract0.<dylib or so> <python> test_binding.py
"""

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
pub = sorted(n for n in dir(o) if not n.startswith("_"))
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
    assert str(e) == "render: not in this build: scalar mode BANDED", str(e)
    lost = json.dumps({**UNIT, "sinks": [99]})
    e = raises(o.Fault, lambda: render(lost), "backend_render: a fault of the compiled side")
    assert str(e).startswith("render: least squares (panic: "), str(e)
    assert render(json.dumps(UNIT))[0].startswith("// HAWK-emitted host translation unit")
    done.append("backend_render")
    assert c.backend_close(lib) is None
    done.append("backend_close")
print("\n".join("ok: " + d for d in done))
print("test_binding: %d checks passed" % len(done))
