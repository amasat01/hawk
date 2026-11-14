# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``import hawk`` still imports no heavy tier: ``hawk/__init__.py`` binds
authoring names lazily (PEP 562 ``__getattr__``), so importing the package
alone does not import ``raptor.ir``, ``hawk.trace``, ``hawk.diff``,
``aether.wings.base`` or ``hawk.compile``. The pinned module set is an exact
equality, checked in a fresh subprocess, and allows only ``hawk``,
``hawk._core`` (the binding the import-time self-check reads) and
``hawk._contracts`` (the layout it compares against).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent

_BARE = ("import sys, json; import hawk; "
         "print(json.dumps(sorted(m for m in sys.modules if m.startswith('hawk'))))")
_TOUCHED = ("import sys, json; import hawk; k = hawk.kernel; "
            "print(json.dumps(sorted(m for m in sys.modules if m.startswith('hawk'))))")


def _modules(probe: str) -> list[str]:
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                            text=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


#: What ``import hawk`` may pull in, and nothing else.
_ALLOWED_BARE = ["hawk", "hawk._contracts", "hawk._core"]

#: Modules only a wheel build writes: a wheel install has no source tree beside
#: the package, so its self-check compares against the stamped digest module
#: instead, and the version comes from the generated module (see hawk/__init__.py).
_WHEEL_GENERATED = ("hawk._build_digest", "hawk._version")


def _without_wheel_digest(mods: list[str]) -> list[str]:
    return [m for m in mods if m not in _WHEEL_GENERATED]


def test_importing_hawk_imports_no_heavy_module():
    assert _without_wheel_digest(_modules(_BARE)) == _ALLOWED_BARE, (
        "importing hawk must not import raptor.ir / hawk.trace / hawk.diff "
        "/ hawk.emit / hawk.compile — the authoring names are bound lazily, and the "
        "only modules the self-check may add are the binding it reads and the "
        "constants it compares against"
    )


def test_the_import_time_self_check_actually_ran():
    """Non-vacuity for the row above: the set is allowed to contain ``_core``
    only BECAUSE the check reads it. If the check were removed, the set would
    shrink and this row — not the equality above — is what says so."""
    probe = ("import hawk, json; "
             "print(json.dumps([hawk._core.build_digest(), "
             "sorted(hawk._core.build_info())]))")
    digest, info_keys = _modules(probe)
    assert len(digest) == 64, f"the stamped build digest is not a sha256: {digest!r}"
    assert "aether_include" in info_keys and "eagle_include" in info_keys, info_keys


def test_hawk_imports_with_a_version_when_the_generated_module_is_absent():
    """A checkout that was never built has no ``hawk/_version.py``: the version
    then comes from the installed metadata, else the ``0.0.0+unknown`` placeholder."""
    probe = ("import sys; sys.modules['hawk._version'] = None; import hawk, json; "
             "print(json.dumps(hawk.__version__))")
    version = _modules(probe)
    assert isinstance(version, str) and version, version


def test_touching_an_authoring_name_binds_the_tracer():
    modules = _modules(_TOUCHED)
    assert "hawk.trace" in modules, (
        f"touching hawk.kernel must resolve through hawk.trace; got {modules}"
    )


def _run(probe: str, *, extension_path: bool) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if extension_path:
        env["PYTHONPATH"] = os.pathsep.join([str(_REPO / "staging"), env.get("PYTHONPATH", "")])
    else:       # the optional C extension is not importable here (it may be installed: hide it by name)
        probe = "import sys; sys.modules['ospreys'] = None; " + probe
    return subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)


_ALL_PROBE = ("import hawk, json; "
              "print(json.dumps([n for n in hawk.__all__ if n != '__version__' and getattr(hawk, n, None) is None]))")


@pytest.mark.parametrize("extension_path", [False, True], ids=["without-ospreys.c", "with-ospreys.c"])
def test_every_authoring_name_resolves(extension_path):
    r = _run(_ALL_PROBE, extension_path=extension_path)
    assert r.returncode == 0, r.stderr[-800:]
    assert json.loads(r.stdout.strip().splitlines()[-1]) == [], "every name in hawk.__all__ must resolve"


@pytest.mark.parametrize("extension_path", [False, True], ids=["without-ospreys.c", "with-ospreys.c"])
def test_star_import_never_needs_an_optional_backend(extension_path):
    r = _run("from hawk import *", extension_path=extension_path)
    assert r.returncode == 0, r.stderr[-800:]


def test_the_embedded_door_serves_the_extension_and_refuses_by_name_without_it():
    present = _run("import hawk; print(hasattr(hawk.embedded, 'export'))", extension_path=True)
    assert present.returncode == 0 and present.stdout.strip() == "True", present.stderr[-800:]
    absent = _run("import hawk; hawk.embedded", extension_path=False)
    assert absent.returncode != 0 and "'c'" in absent.stderr and "install" in absent.stderr, absent.stderr[-800:]
    assert "embedded" not in _run("import hawk; print(hawk.__all__)", extension_path=False).stdout
