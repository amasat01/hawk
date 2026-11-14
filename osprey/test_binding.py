# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Run: PYTHONPATH=<osprey>/build python3 test_binding.py"""
import _osprey as o

LEAF = "(('kind', 'vocab_read'), ('role', 'vec_in'), ('name', '%s'), ('quantity', None), ('wire_index', 0), ('sorted', False))"
ADD = "(('kind', 'add'), ('literal', None))"
WANT = "e5528739b51eb34fa87421825d26571b381966750395a5e74f9b131dd037afff"
done = []


def mk(c, cls, ops, fields):
    return c.make(cls, "f64", [], None, ops, fields)


def expr(c):
    a, b = mk(c, "Leaf", [], LEAF % "a"), mk(c, "Leaf", [], LEAF % "b")
    s = mk(c, "Op", [a, b], ADD)
    e = mk(c, "Op", [s, mk(c, "Const", [], "(('literal', 2.0),)")], ADD)
    return mk(c, "Assign", [e], "(('name', 'y'),)")


def raises(exc, f, what):
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
done.append("reference digest")
assert (y.cls, y.ttype, len(y.operands), y.field("name"), y.operands[0].cls) == ("Assign", "TensorType(shape=(), dtype='f64', tag=None)", 1, "'y'", "Op")
done.append("node getters")
text = c.dump([y])
c2 = o.Ctx()
(r,) = c2.load(text)
assert c2.digest(c2.canonical([r], "[]", None, "[]")) == WANT
done.append("dump/load roundtrip into a second Ctx")
assert [n.cls for n in map(lambda h: h, c.walk_order(w))] == ["Leaf", "Leaf", "Op", "Const", "Op", "Assign"]
done.append("walk_order")
e = raises(ValueError, lambda: c.make("NoSuchClass", "f64", [], None, [], "()"), "refusal raises ValueError")
assert str(e), "empty message"
raises(AttributeError, lambda: setattr(y, "extra", 1), "node.extra = 1")
raises(TypeError, lambda: type("S", (o.Ctx,), {}), "subclass Ctx")
raises(TypeError, lambda: type("S", (o.Node,), {}), "subclass Node")
raises(TypeError, lambda: o.Node(), "Node()")
raises(TypeError, lambda: setattr(o.Ctx, "new_method", 1), "Ctx.new_method = ...")
raises(TypeError, lambda: setattr(o.Node, "new_method", 1), "Node.new_method = ...")
raises(TypeError, lambda: c2.canonical([y], "[]", None, "[]"), "Node of another Ctx")
raises(TypeError, lambda: c2.operands(y), "Node of another Ctx (operands)")
pub = sorted(n for n in dir(o) if not n.startswith("_"))
assert pub == ["ABI_VERSION", "Ctx", "Node"], pub
done.append("public names " + ",".join(pub))
print("\n".join("ok: " + d for d in done))
print("test_binding: %d checks passed" % len(done))
