# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The host backend's manifest declares the cpu aether wing, which builds on the lowering component; without it a host build is refused by name."""
import subprocess
import sys

from hawk import registry

_BLOCK = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "lowering" or name.startswith("lowering."):
            raise ImportError("the lowering component is absent")
sys.meta_path.insert(0, Block())
from hawk.targets import build_spec
try:
    build_spec(("host",), "")
    print("BUILT")
except Exception as e:
    print(type(e).__name__, e)
"""


def test_the_host_row_declares_the_component():
    from hawk.targets import TARGETS

    assert TARGETS["host"].components == ("aether_cpu",)


def test_a_host_build_without_the_component_is_refused_by_name():
    out = subprocess.run([sys.executable, "-c", _BLOCK], capture_output=True, text=True, check=True).stdout
    assert out.startswith("RaptorError") and "lowering component is absent" in out, out


def test_the_manifest_key_is_checked():
    assert "components" in registry.traits("host")
