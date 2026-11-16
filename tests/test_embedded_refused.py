# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The embedded C target is refused by name at every door that asks for it: one message."""

import hawk
import hawk.artifact
import pytest
from hawk import Mutable, Scalar
from hawk._errors import EMBEDDED_C, HawkError


@hawk.kernel
def _scale(x: Scalar, y: Mutable[Scalar]):
    y = 2.0 * x


def _doors(tmp_path, cache_dir):
    return {
        "build_bundle": lambda: hawk.artifact.build_bundle([_scale], tmp_path / "b", targets=("c",), cache_dir=cache_dir),
        "build": lambda: hawk.artifact.build(_scale, tmp_path / "u", targets=("host", "c"), cache_dir=cache_dir),
        "emit_source": lambda: hawk.artifact.emit_source(_scale, targets=("c",)),
        "embedded_profile": lambda: hawk.artifact.build_bundle([_scale], tmp_path / "p", targets=("host",), cache_dir=cache_dir, embedded_profile=object()),
        "option": lambda: hawk.artifact.build_bundle([_scale], tmp_path / "o", targets=("host",), cache_dir=cache_dir, options={"c.profile": 1}),
        "hawk.embedded": lambda: getattr(hawk, "embedded"),
    }


@pytest.mark.parametrize("door", ["build_bundle", "build", "emit_source", "embedded_profile", "option", "hawk.embedded"])
def test_every_door_to_the_c_target_says_it_is_not_available(door, tmp_path, cache_dir):
    with pytest.raises(HawkError) as err:
        _doors(tmp_path, cache_dir)[door]()
    assert str(err.value) == EMBEDDED_C
