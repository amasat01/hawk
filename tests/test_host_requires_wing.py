# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The host backend's manifest names its wings; without the raptor base wing that its chain reaches, a host build is refused by name."""
import subprocess
import sys

from hawk import registry

_BLOCK = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "raptor.wings.base" or name.startswith("raptor.wings.base."):
            raise ImportError("the raptor base wing is absent")
sys.meta_path.insert(0, Block())
from hawk.targets import build_spec
try:
    build_spec(("host",), "")
    print("BUILT")
except Exception as e:
    print(type(e).__name__, e)
"""


def test_the_host_row_names_its_wings():
    from hawk.targets import TARGETS

    assert TARGETS["host"].wings == {"raptor": {"x86_64": "x86", "arm64": "arm"}, "aether": "cpu", "eagle": None}


def test_a_host_build_without_the_base_wing_is_refused_by_name():
    out = subprocess.run([sys.executable, "-c", _BLOCK], capture_output=True, text=True, check=True).stdout
    assert out.startswith("RaptorError") and "the raptor base wing is absent" in out, out


def test_the_manifest_key_is_checked():
    assert "wings" in registry.traits("host") and "facts" not in registry.traits("host")
