"""Every first-party name the sources import resolves against this tree (a renamed or moved name is a failure here, not at
the first run of the path that uses it), and eagle takes its build spec and its measured rows from the kernel it is given."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "names_check.py"


def _names(label, *sources):
    """Run the name check; its report on a red, and the toolchain-blocked names (apart from the unresolved ones) always in the output."""
    args = [sys.executable, str(SCRIPT), "--label", label, "--tree", str(ROOT)]
    for kind, value in sources:
        args += [kind, str(value)]
    res = subprocess.run(args, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    print(res.stdout)


def _eagle_tests():
    """Eagle's own test files: ``$EAGLE_PYTHON_TESTS``, else the sibling checkout of this workspace; none when neither exists."""
    import os

    found = Path(os.environ.get("EAGLE_PYTHON_TESTS", ROOT.parent / "eagle" / "python" / "tests"))
    return [("--src", found)] if found.is_dir() else []


def test_every_first_party_name_eagle_imports_resolves():
    _names("eagle", ("--package", "eagle"), *_eagle_tests())


def test_every_first_party_name_the_test_files_import_resolves():
    _names("hawk tests", ("--src", ROOT / "tests"), *(("--src", d) for d in sorted((ROOT / "components").glob("*/tests"))))


def test_every_first_party_name_the_backends_import_resolves():
    _names("backends", *(("--src", d) for d in sorted((ROOT / "extensions").glob("*"))))


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
