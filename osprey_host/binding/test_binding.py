"""Check the binding.

Run: PYTHONPATH=<binding>/build HOST_LIB=<library>
ABI1_LIB=<binding>/build/libabi1.<dylib or so>
CONTRACT0_LIB=<binding>/build/libcontract0.<dylib or so> <python> test_binding.py
"""

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
raises(TypeError, lambda: setattr(o.Ctx, "new_method", 1), "Ctx.new_method = ...")
raises(TypeError, lambda: setattr(o.Node, "new_method", 1), "Node.new_method = ...")
raises(TypeError, lambda: c2.canonical([y], "[]", None, "[]"), "Node of another Ctx")
pub = sorted(n for n in dir(o) if not n.startswith("_"))
assert pub == ["ABI_VERSION", "CONTRACT_VERSION", "Ctx", "Node"], pub
done.append("public names " + ",".join(pub))

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
        assert str(e) == "open: abi number 1, expected 4", str(e)
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
    e = raises(
        ValueError,
        lambda: c3.backend_render(lib, c3.dump(every), "", "[]", None, "[]", [3], "{}"),
        "backend_render: a refusal by name",
    )
    assert str(e).startswith("render: not in this build"), str(e)
    done.append("backend_render")
    assert c.backend_close(lib) is None
    done.append("backend_close")
print("\n".join("ok: " + d for d in done))
print("test_binding: %d checks passed" % len(done))
