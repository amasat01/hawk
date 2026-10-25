# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A new export-only target is one row: registered in ``TARGETS`` (a plain
dict), it builds through ``hawk.build(targets=...)`` with no change to the
build, and is refused by name once the row is gone."""

from __future__ import annotations

from dataclasses import replace

import _kernels as K
import pytest

import hawk
from hawk.artifact import reset_unit_memo
from hawk.ir import HawkError
from hawk.targets import TARGETS


def test_an_export_row_builds_and_leaves_its_file(tmp_path, cache_dir):
    assert type(TARGETS) is dict
    reset_unit_memo()
    TARGETS["toy"] = replace(TARGETS["c"], id="toy", about="a toy export",
                             artifact=".toyh", source=".toy")
    try:
        bundle = hawk.build([K.drag], tmp_path / "u", targets=("toy",),
                            cache_dir=cache_dir)
    finally:
        del TARGETS["toy"]
        reset_unit_memo()
    (art,) = bundle.artifacts
    entry = art.entries["toy"]
    assert entry.source == tmp_path / "u" / "drag.toy"
    assert entry.artifact == tmp_path / "u" / "drag.toyh"
    assert entry.source.read_text().strip() and entry.artifact.read_text().strip()
    assert bundle.manifest is None
    with pytest.raises(HawkError, match="unknown target 'toy'"):
        hawk.build([K.drag], tmp_path / "v", targets=("toy",), cache_dir=cache_dir)
