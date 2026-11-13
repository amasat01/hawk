# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The SIMT aether wing: its spellings are data in its manifest, its texts import with raptor and the SIMT raptor wing only."""
import importlib

import aether.wings.simt as wing


def test_the_manifest_declares_the_spellings_and_every_module_imports_without_hawk():
    from aether.wings.simt import aether

    assert {n for _, n in aether.SPELLINGS} == {"warp_lane_value", "warp_store"}
    assert wing.MANIFEST["spellings"] == [{"writer": "aether", "dialect": "simt", "ref": "aether.wings.simt.aether"}]
    for part, names in wing.MANIFEST["entries"].items():
        mod = importlib.import_module(f"aether.wings.simt.{part}")
        assert all(hasattr(mod, n) for n in names), part
    for name in wing.MANIFEST["emission_modules"]:
        importlib.import_module(name)


def test_wrapper_text_helpers():
    from aether.wings.simt import wrapper_text as w

    assert w.WARP_SUM_BANDED != w.WARP_SUM and w.BANDED_SUM_INCLUDE.startswith("#include")


def test_the_wing_builds_on_the_simt_raptor_wing():
    assert wing.MANIFEST["components"] == ["simt", "aether_base"]
