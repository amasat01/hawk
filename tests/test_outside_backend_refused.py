# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A back end registered from Python (an OSPREY outside hawk) names a target the build refuses: its text would be written by Python."""

from pathlib import Path

import hawk
import hawk.artifact
import pytest
from hawk import Mutable, Param, Scalar
from hawk._errors import HawkError

PLUGIN = Path(__file__).parent / "plugins" / "backend_host2"


@hawk.kernel
def _affine(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


@pytest.fixture
def host2(monkeypatch):
    from hawk import registry

    monkeypatch.syspath_prepend(str(PLUGIN))            # also drops the import caches
    registry.reset()
    yield
    monkeypatch.undo()
    registry.reset()


@pytest.mark.parametrize("door", ["build", "build_bundle", "emit_source"])
def test_an_outside_backend_is_refused_at_the_build(door, host2, tmp_path, cache_dir):
    calls = {"build": lambda: hawk.artifact.build(_affine, tmp_path / "u", targets=("host2",), cache_dir=cache_dir),
             "build_bundle": lambda: hawk.artifact.build_bundle([_affine], tmp_path / "b", targets=("host2",), cache_dir=cache_dir),
             "emit_source": lambda: hawk.artifact.emit_source(_affine, targets=("host2",))}
    with pytest.raises(HawkError, match=r"target 'host2' is refused: a back end registered from Python is not built"):
        calls[door]()
