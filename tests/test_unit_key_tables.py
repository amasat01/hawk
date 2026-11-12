# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The unit key covers the shared tables a unit is emitted with: an import-time edit of a spelling table moves it."""
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent

_KEY = """
import sys
{plant}
import hawk
from _specs import HOST
from hawk.artifact.unit_cache import _unit_key
from hawk.emit import scalar_mode
from hawk.ext import Kind
from hawk import Mutable, Scalar

@hawk.kernel
def step(x: Scalar, y: Mutable[Scalar]):
    y = x + 1.0

k = step
print(_unit_key(["step"], [k.walk.digest], [""], targets=("host",), smode=scalar_mode("float64"), kind=Kind("m"), defines=(),
                cache_dir="", device_arch="", derivative_blocks=[{{}}], layout_sizes_override=None, groups=(1,), spec=HOST))
"""


def _key(plant: str) -> str:
    code = _KEY.format(plant=plant)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(HERE.parent), str(HERE), str(HERE.parent / "extensions" / "hawk_cuda"), os.environ.get("PYTHONPATH", "")])}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-600:]
    return out.stdout.strip().splitlines()[-1]


def test_a_planted_edit_of_a_shared_spelling_table_moves_the_unit_key():
    base = _key("")
    assert base == _key("")
    edited = _key("import hawk.emit.spelling as s; s._INFIX['add'] = '+ 0 +'")
    assert edited != base
    text = _key("import hawk.emit.wrapper as w; w._INCLUDES = w._INCLUDES + '// planted\\n'")
    assert text != base


def test_a_device_only_table_edit_leaves_a_host_only_unit_key_unchanged():
    base = _key("")
    device_edit = _key("import hawk_cuda.emit as e; e.PLANTED_TABLE = {'add': 'planted'}")
    assert device_edit == base
    lowering_edit = _key("import lowering.wrapper_text as w; w.PLANTED_TABLE = {'add': 'planted'}")
    assert lowering_edit != base          # the host row uses the component: its tables belong to the host unit


def test_a_planted_edit_of_a_sidecar_or_manifest_or_pass_table_moves_the_unit_key():
    """The tables a sidecar, a manifest or a pass writes the unit with are declared emission tables: an edit moves the key."""
    base = _key("")
    for plant in ("import hawk.artifact.sidecar as m; m.PARAMS_SCHEMA = m.PARAMS_SCHEMA + 1",
                  "import hawk.artifact.sidecar as m; m.MUTABLE_DTYPES = tuple(m.MUTABLE_DTYPES) + ('planted',)",
                  "import hawk.artifact.manifest as m; m.ENTRY_KINDS = tuple(m.ENTRY_KINDS) + ('planted',)",
                  "import hawk._plane_layout as m; m.MARK_ATTR = m.MARK_ATTR + 'x'",
                  "import hawk.passes.vectorise as m; m.UNROLL_LIMIT = m.UNROLL_LIMIT + 1"):
        assert _key(plant) != base, plant
