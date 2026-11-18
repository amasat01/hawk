"""Rebuilding every node over its own operands reproduces the walk."""
import _kernels as K
import pytest

from hawk import _ir
from hawk.trace.kernel import _declared


def _rebuild_all(sinks):
    """The sinks rebuilt bottom-up: every node over the rebuilt copies of its operands."""
    rebuilt = {}
    for node in _ir.post_order(sinks):
        rebuilt[node] = _ir.rebuild(node, [rebuilt[operand] for operand in node.operands])
    return tuple(rebuilt[sink] for sink in sinks)


@pytest.mark.parametrize("name", sorted(K.CORPUS))
def test_rebuilt_reproduces_the_walk(name):
    k = K.CORPUS[name]
    declared = _declared(k.planes)
    rebuilt = _ir.cells_walk(_rebuild_all(k.sinks), declared)
    assert _ir.digest(rebuilt) == _ir.digest(_ir.cells_walk(k.sinks, declared))


def test_core_imports_no_dialect():
    """``import hawk`` loads no wing; the writer loads only raptor and the wings its manifest names (``raptor.wings.base``), nothing of hawk;
    the base wing names no writer."""
    import subprocess
    import sys

    def run(code):
        return subprocess.run([sys.executable, "-c", "import sys\n" + code], capture_output=True, text=True, check=True).stdout.strip()

    wings = "print(sorted(m for m in sys.modules if m.startswith('aether.wings') or m == 'raptor.wings.base' or m.startswith('raptor.wings.base.')))"
    assert run("import hawk\n" + wings) == "[]"
    assert run("import raptor.ir\n" + wings) == "[]"
    every = "import json, importlib\nfrom aether.wings.base import MANIFEST\nfor m in MANIFEST['emission_modules']: importlib.import_module(m)\n"
    assert run(every + "print([m for m in sys.modules if m == 'hawk' or m.startswith('hawk.')])") == "[]"
    assert run("import raptor.wings.base.split\nprint([m for m in sys.modules if m.startswith('aether.wings')])") == "[]"
