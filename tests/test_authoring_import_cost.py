# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``import hawk`` still imports no heavy tier: ``hawk/__init__.py`` binds
authoring names lazily (PEP 562 ``__getattr__``), so importing the package
alone does not import ``hawk.ir``, ``hawk.trace``, ``hawk.diff``,
``hawk.emit`` or ``hawk.compile``. The pinned module set is an exact
equality, checked in a fresh subprocess, and allows only ``hawk``,
``hawk._core`` (the binding the import-time self-check reads) and
``hawk._contracts`` (the layout it compares against).
"""

from __future__ import annotations

import json
import subprocess
import sys

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

#: A wheel install has no source tree beside the package, so its self-check
#: compares against the stamped digest module instead (see hawk/__init__.py).
_WHEEL_DIGEST = "hawk._build_digest"


def _without_wheel_digest(mods: list[str]) -> list[str]:
    return [m for m in mods if m != _WHEEL_DIGEST]


def test_importing_hawk_imports_no_heavy_module():
    assert _without_wheel_digest(_modules(_BARE)) == _ALLOWED_BARE, (
        "importing hawk must not import hawk.ir / hawk.trace / hawk.diff "
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


def test_touching_an_authoring_name_binds_the_tracer():
    modules = _modules(_TOUCHED)
    assert "hawk.trace" in modules and "hawk.ir" in modules, (
        f"touching hawk.kernel must resolve through hawk.trace; got {modules}"
    )


def test_every_authoring_name_resolves():
    probe = ("import hawk, json; "
             "print(json.dumps([n for n in hawk.__all__ "
             "if n != '__version__' and getattr(hawk, n, None) is None]))")
    assert _modules(probe) == [], "every name in hawk.__all__ must resolve"
