"""Every node class has a state/edges/from_state recipe, and rebuilding every node reproduces the Walk."""
import _kernels as K
import lowering.box  # noqa: F401  (BoxStore)
import simt.lane  # noqa: F401
import lowering.team  # noqa: F401
import pytest
from raptor.ir.nodes import Node
from lowering.cells import walk as canonical
from hawk.trace.kernel import _declared


def subclasses(cls):
    for c in cls.__subclasses__():
        yield c
        yield from subclasses(c)


ALL = sorted(set(subclasses(Node)), key=lambda c: (c.__module__, c.__name__))


def _own_slots(cls):
    out = []
    for k in cls.__mro__:
        if k not in (Node, object):
            slots = vars(k).get("__slots__", ())
            out += [slots] if isinstance(slots, str) else list(slots)
    return out


@pytest.mark.parametrize("cls", ALL, ids=lambda c: f"{c.__module__}.{c.__name__}")
def test_class_has_a_recipe(cls):
    has = any("from_state" in vars(k) for k in cls.__mro__ if k is not Node)
    assert has or not _own_slots(cls), f"{cls.__name__} holds {_own_slots(cls)} and has no recipe"


def _rebuild_all(sinks):
    done = {}

    def go(n):
        if id(n) not in done:
            ops = tuple(go(o) for o in n.operands)
            edges = {k: None if v is None else tuple(go(c) for c in v) if isinstance(v, tuple) else go(v)
                     for k, v in n.edges().items()}
            done[id(n)] = n.rebuilt(ops, edges)
        return done[id(n)]

    return tuple(go(s) for s in sinks)


@pytest.mark.parametrize("name", sorted(K.CORPUS))
def test_rebuilt_reproduces_the_walk(name):
    k = K.CORPUS[name]
    declared = _declared(k.planes)
    assert canonical(_rebuild_all(k.sinks), declared).digest == canonical(k.sinks, declared).digest


def test_core_imports_no_dialect():
    """``import hawk`` loads no wing; the writer loads only raptor and the wings its manifest names (``lowering``), nothing of hawk;
    the lowering component names no writer."""
    import subprocess
    import sys

    def run(code):
        return subprocess.run([sys.executable, "-c", "import sys\n" + code], capture_output=True, text=True, check=True).stdout.strip()

    wings = "print(sorted(m for m in sys.modules if m.startswith('aether.wings') or m == 'lowering' or m.startswith('lowering.')))"
    assert run("import hawk\n" + wings) == "[]"
    assert run("import raptor.ir\n" + wings) == "[]"
    every = "import json, importlib\nfrom aether.wings.base import MANIFEST\nfor m in MANIFEST['emission_modules']: importlib.import_module(m)\n"
    assert run(every + "print([m for m in sys.modules if m == 'hawk' or m.startswith('hawk.')])") == "[]"
    assert run("import lowering.split\nprint([m for m in sys.modules if m.startswith('aether.wings')])") == "[]"
