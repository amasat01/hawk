# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The SIMT aether wing: its spellings are data in its manifest, its texts import with raptor and the SIMT raptor wing only."""
import importlib

import aether.wings.simt as wing


def test_the_wing_builds_on_the_simt_raptor_wing():
    assert wing.MANIFEST["parent"] == "base"
