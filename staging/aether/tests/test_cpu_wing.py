# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The cpu aether wing: its manifest names its parent wing, and every module imports with no hawk loaded."""
import importlib
import sys

import aether.wings.cpu as wing


def test_the_manifest_names_its_parent_and_every_module_imports_without_hawk():
    assert wing.MANIFEST["parent"] == "base"
    for name in wing.MANIFEST["emission_modules"]:
        importlib.import_module(name)
    assert not [m for m in sys.modules if m == "hawk" or m.startswith("hawk.")]
