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
from hawk.artifact.unit_cache import _plain

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _manifest(path):
    d = json.loads((ROOT / path).read_text())
    return set(d.get("emission_modules", ())), set(d.get("runtime_modules", ()))


def _hawk():
    b = registry._builtin()
    return set(b["emission_modules"]), set(b["runtime_modules"])


PACKAGES = {
    "hawk_c": lambda: _manifest("extensions/hawk_c/hawk_c/osprey.json"),
    "hawk_cuda": lambda: _manifest("extensions/hawk_cuda/hawk_cuda/osprey.json"),
    "lowering": lambda: _manifest("components/lowering/lowering/wing.json"),
    "simt": lambda: _manifest("components/simt/simt/wing.json"),
    "aether.wings.simt": lambda: _manifest("staging/aether/wings/simt/wing.json"),
    "aether.wings.base": lambda: _manifest("staging/aether/wings/base/wing.json"),
    "aether.wings.cpu": lambda: _manifest("staging/aether/wings/cpu/wing.json"),
    "raptor.wings.cpu": lambda: _manifest("staging/raptor/wings/cpu/wing.json"),
}


def _holds_tables(mod) -> bool:
    return any(not k.startswith("__") and _plain(v) is not None and not callable(v) for k, v in vars(mod).items())


@pytest.mark.parametrize("package", sorted(PACKAGES))
def test_every_module_with_plain_tables_is_classified(package):
    emission, runtime = PACKAGES[package]()
    assert not emission & runtime, f"{package}: modules both emission and runtime: {sorted(emission & runtime)}"
    pkg = importlib.import_module(package)
    unclassified = []
    for info in pkgutil.walk_packages(pkg.__path__, package + "."):
        try:
            mod = importlib.import_module(info.name)
        except ImportError:
            continue
        if _holds_tables(mod) and info.name not in emission | runtime:
            unclassified.append(info.name)
    assert not unclassified, f"modules with plain tables that no manifest classifies: {unclassified}"
    for name in emission | runtime:
        importlib.import_module(name)         # a declared module exists


def test_a_table_outside_hawk_emit_that_an_artifact_is_written_with_is_declared():
    """The sidecar and manifest tables, the plane layouts and the pass constants reach an emitted artifact: they are declared too."""
    emission, _ = _hawk()
    assert {"hawk.artifact.sidecar", "hawk.artifact.manifest", "hawk._plane_layout"} <= emission
