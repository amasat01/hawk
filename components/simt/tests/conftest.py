# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The component's tests run with hawk ABSENT: a meta-path block refuses any import of it."""
import importlib.abc
import sys


class _NoHawk(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "hawk" or name.startswith(("hawk.", "hawk_")):
            raise ImportError(f"hawk is absent in the component's tests: {name}")
        return None


sys.meta_path.insert(0, _NoHawk())
for _m in [m for m in sys.modules if m == "hawk" or m.startswith(("hawk.", "hawk_"))]:
    del sys.modules[_m]
