# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.artifact.build`` with a target name no OSPREY serves (``"device"`` is eagle's word) names the
targets that are served."""

from __future__ import annotations

import pytest

import hawk
from hawk.artifact import build
from hawk.errors import HawkError


@hawk.kernel
def _identity(a: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    y = a


def test_unknown_target_lists_the_served_names(tmp_path):
    with pytest.raises(HawkError) as err:
        build(_identity, tmp_path / "unit", targets=("device",))
    text = str(err.value)
    assert "target 'device' is served by no OSPREY found" in text
    assert "the served targets are" in text


def test_no_target_is_refused_and_writes_nothing(tmp_path):
    """``targets=()`` must not publish an empty unit."""
    with pytest.raises(HawkError, match="no target to build"):
        hawk.build([_identity], tmp_path / "u", targets=())
    assert not (tmp_path / "u").exists()
