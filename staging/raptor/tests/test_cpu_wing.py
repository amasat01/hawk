# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The cpu raptor wing: its manifest, its one module, and no hawk."""
import importlib
import sys

import raptor.wings.cpu as wing


def test_the_manifest_declares_the_module_and_it_imports_without_hawk():
    assert wing.MANIFEST["parent"] == "base"
    for name in wing.MANIFEST["emission_modules"]:
        mod = importlib.import_module(name)
    assert {"lane_form_access", "interchanged_loops", "UNROLL_LIMIT"} <= set(vars(mod))
    assert not [m for m in sys.modules if m == "hawk" or m.startswith("hawk.")]
