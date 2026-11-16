# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Every module of an emitting package that holds plain module-level tables is declared by its manifest: an ``emission_modules``
entry (its tables enter the unit key) or a ``runtime_modules`` entry (memos and probes). An unclassified module fails by name."""
import importlib
import json
import pathlib
import pkgutil

import pytest

from hawk import registry

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _manifest(path):
    d = json.loads((ROOT / path).read_text())
    return set(d.get("emission_modules", ())), set(d.get("runtime_modules", ()))


def _hawk():
    b = registry._builtin()
    return set(b["emission_modules"]), set(b["runtime_modules"])


def test_a_table_outside_hawk_emit_that_an_artifact_is_written_with_is_declared():
    """The sidecar and manifest tables, the plane layouts and the pass constants reach an emitted artifact: they are declared too."""
    emission, _ = _hawk()
    assert {"hawk.artifact.sidecar", "hawk.artifact.manifest", "hawk._plane_layout"} <= emission
