# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.artifact.build`` with a target name it does not know (``"device"`` is eagle's word) says
which names are valid, in plain words."""

from __future__ import annotations

import pytest

import hawk
from hawk.artifact import build
from hawk.ir import HawkError


@hawk.kernel
def _identity(a: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    y = a


def test_unknown_target_lists_the_valid_names(tmp_path):
    with pytest.raises(HawkError) as err:
        build(_identity, tmp_path / "unit", targets=("device",))
    text = str(err.value)
    assert "unknown target 'device'" in text
    assert "'cuda', 'host' and 'c'" in text
    assert "GPU" in text and "CPU" in text
