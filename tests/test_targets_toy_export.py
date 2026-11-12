# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A new export-only target is one extension: declared in a manifest and found through ``TARGETS`` (a view
of the backend registry), it builds through ``hawk.build(targets=...)`` with no change to the
build, and is refused by name once the row is gone."""

from __future__ import annotations

import _kernels as K
import pytest
from _extensions import backend, copy_of, make

import hawk
from hawk import registry
from hawk.artifact import reset_unit_memo
from hawk.ir import HawkError
from hawk.targets import TARGETS

TOY = copy_of("hawk_c.backend", "toy", "BACKEND.about, BACKEND.artifact, BACKEND.source = 'a toy export', '.toyh', '.toy'\n")


def test_an_export_row_builds_and_leaves_its_file(tmp_path, cache_dir, monkeypatch):
    assert not hasattr(TARGETS, "__setitem__")           # a read-only view, not the dict it once was
    reset_unit_memo()
    make(tmp_path, "toy-ext", {"backends": [backend("toy", "toy_backend:BACKEND", builds=False)]},
         modules={"toy_backend": TOY})
    monkeypatch.syspath_prepend(str(tmp_path))
    registry.reset()
    try:
        bundle = hawk.build([K.drag], tmp_path / "u", targets=("toy",),
                            cache_dir=cache_dir)
    finally:
        monkeypatch.undo()
        registry.reset()
        reset_unit_memo()
    (art,) = bundle.artifacts
    entry = art.entries["toy"]
    assert entry.source == tmp_path / "u" / "drag.toy"
    assert entry.artifact == tmp_path / "u" / "drag.toyh"
    assert entry.source.read_text().strip() and entry.artifact.read_text().strip()
    assert bundle.manifest is None
    with pytest.raises(HawkError, match="unknown target 'toy'"):
        hawk.build([K.drag], tmp_path / "v", targets=("toy",), cache_dir=cache_dir)
