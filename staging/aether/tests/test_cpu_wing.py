# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The cpu aether wing: its manifest names the wings it builds on, and every module imports with no hawk loaded."""
import importlib
import sys

import aether.wings.cpu as wing


def test_the_manifest_names_the_wings_it_builds_on_and_every_module_imports_without_hawk():
    assert wing.MANIFEST["components"] == ["aether_base", "raptor_cpu", "lowering"]
    for name in wing.MANIFEST["emission_modules"]:
        importlib.import_module(name)
    assert not [m for m in sys.modules if m == "hawk" or m.startswith("hawk.")]
