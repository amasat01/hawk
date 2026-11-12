# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""eagle's torch level reads the row width of its cells scratch from the built kernel's sidecar (torch itself is replaced by a
stub: this runs without it)."""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest


def _level_class():
    """``eagle.frameworks.torch`` imported over a stub ``torch`` (only those two entries of ``sys.modules`` are touched: a compiled
    module must not be imported twice)."""
    import eagle.interop  # noqa: F401  (its compiled core loads once, before the stub)

    had = sys.modules.get("torch")
    sys.modules["torch"] = mock.MagicMock()
    try:
        mod = __import__("eagle.frameworks.torch", fromlist=["_Level"])
    finally:
        sys.modules.pop("torch")
        if had is not None:
            sys.modules["torch"] = had
        sys.modules.pop("eagle.frameworks.torch", None)
    return mod, mod._Level


def test_the_torch_cells_scratch_takes_its_row_from_the_built_kernels_sidecar():
    mod, level = _level_class()
    for built, want in ((types.SimpleNamespace(sidecar={"cell_row": 16}), 16),                    # a loaded host kernel
                        (types.SimpleNamespace(plugin=types.SimpleNamespace(sidecar={"cell_row": 8})), 8)):   # a device plan
        me = types.SimpleNamespace(cells={"s": 1}, owner=types.SimpleNamespace(policy="compensated"), _load_host=lambda m: None,
                                   _built_for=lambda mode, index, make, built=built: built)
        seen = {}
        with mock.patch.object(mod._cell_rows, "scratch", lambda owner, n, xp, dt, pol, cells=None, row=None: seen.setdefault("row", row)):
            level._cell_rows(me, 100, "float64", types.SimpleNamespace(type="cpu", index=None))
        assert seen["row"] == want


def test_a_built_kernel_without_a_row_is_refused_by_name():
    mod, level = _level_class()
    me = types.SimpleNamespace(cells={"s": 1}, owner=types.SimpleNamespace(policy="compensated"),
                               _built_for=lambda *a: types.SimpleNamespace(sidecar={}))
    with pytest.raises(ValueError, match="cell_row"):
        level._cell_rows(me, 100, "float64", types.SimpleNamespace(type="cpu", index=None))
    sys.modules.pop("eagle.frameworks.torch", None)
