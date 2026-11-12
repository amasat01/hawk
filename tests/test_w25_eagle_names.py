"""Every hawk name eagle's Python package imports resolves against this tree (a renamed or moved name is a failure here, not at
the first run of the eagle path that uses it), and eagle takes its build spec and its measured rows from the kernel it is given."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "eagle_hawk_names.py"


def test_every_hawk_name_eagle_imports_resolves():
    res = subprocess.run([sys.executable, str(SCRIPT), "--tree", str(ROOT)], capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "unresolved: 0" in res.stdout


def test_the_steps_reverse_choice_is_plain_data_applied_by_raptor():
    from raptor.ir.steps_class import choose

    from hawk.targets.steps_reverse import table_for

    rows = table_for(False)["rows"]
    assert choose(rows, 4, 4, 8 / 1) == "fused"       # inside the measured batch range
    assert choose(rows, 4, 4, 4096) == "per_step"     # beyond it
    assert table_for(True)["regime"] == "per_resident"


def test_eagle_refuses_a_missing_build_spec_by_name():
    import eagle._dispatch_door as door

    class NoSpec:
        pass

    with pytest.raises(ValueError, match="build spec is missing: pass spec="):
        door.host_spec(NoSpec())
    assert door.host_spec(NoSpec(), spec="given") == "given"


def test_a_kernel_holds_the_host_spec_eagle_takes():
    import hawk
    import eagle._dispatch_door as door

    @hawk.kernel
    def k(x: hawk.Mutable[hawk.Scalar]):
        x = x + 1.0

    assert door.host_spec(k) is not None
