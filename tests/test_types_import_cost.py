# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``import hawk.types`` pulls in no other hawk module.

``hawk.types`` is the LIGHT tier — a dependency-free module of
small typed records (:class:`TensorType`, :class:`Wire`, :class:`Slot`) that
a downstream policy core can describe its own constants with, without
paying for HAWK's heavy IR/emit/compile
machinery — the exact duplication-pressure shape that module's ``DeviceStamp``
comment names ("a policy core that pulled in a kernel-authoring DSL to
describe its own constants would be paying a large import for a three-field
dataclass").

Run in a FRESH subprocess (never in-process ``sys.modules`` inspection): this
pytest session may have already imported other ``hawk.*`` modules via earlier
tests, which would make an in-process check pass or fail for the wrong
reason.

 note: importing ``hawk.types`` imports the ``hawk`` PACKAGE first, and
``hawk/__init__`` now performs the import-time self-check — which reads
``hawk._core`` and compares it against ``hawk._contracts``. Those two join the
pinned set; the light tier's claim is unchanged and still an EQUALITY, so the
heavy tiers (``ir``/``trace``/``diff``/``emit``/``compile``) still cannot slip
in. What the light tier promises is that a policy core describing its own
constants pays HAWK's package import and no kernel-authoring machinery — and
the package import is what makes safe.

This test previously failed when ``hawk/types.py``
carried a deliberate, unused ``from . import _contracts``.
"""

from __future__ import annotations

import json
import subprocess
import sys

_PROBE = (
    "import sys, json; "
    "import hawk.types; "
    "mods = sorted(m for m in sys.modules if m == 'hawk' or m.startswith('hawk.')); "
    "print(json.dumps(mods))"
)


def _hawk_modules_after_importing_types() -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_import_hawk_types_pulls_in_nothing_else():
    # a wheel install's self-check also reads the stamped digest module
    mods = [m for m in _hawk_modules_after_importing_types() if m != "hawk._build_digest"]
    assert mods == ["hawk", "hawk._contracts", "hawk._core", "hawk.types"], (
        "hawk.types must be the light tier: importing it should pull "
        "in only the package, the two self-check modules and hawk.types itself, "
        f"got {mods}"
    )
