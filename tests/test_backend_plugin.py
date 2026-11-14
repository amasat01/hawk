# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A backend shipped outside hawk registers through its OSPREY manifest
(entry point group ``ospreys``) and, named in ``targets=``, builds with ``hawk.build(targets=...)``, with no change to hawk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import hawk
from hawk import Mutable, Param, Scalar
from hawk.artifact import plugins

PLUGIN = Path(__file__).parent / "plugins" / "backend_host2"
N = 33


@hawk.kernel
def h2_affine(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
    y = a * x + b


@pytest.fixture
def host2(monkeypatch):
    from hawk import registry

    monkeypatch.syspath_prepend(str(PLUGIN))            # also drops the import caches
    registry.reset()
    yield
    monkeypatch.undo()
    registry.reset()


def test_an_outside_backend_builds_and_runs_like_host(host2, tmp_path, cache_dir):
    eexec = pytest.importorskip("eagle.exec")
    from eagle import plan as eplan

    from hawk.targets import TARGETS

    assert "host2" in TARGETS
    x = np.random.default_rng(5).uniform(0.5, 2.0, N)
    runs = {}
    for target in ("host", "host2"):
        bundle = hawk.build([h2_affine], tmp_path / target, targets=(target,), cache_dir=cache_dir)
        (art,) = bundle.artifacts
        assert art.entries[target].source.read_text().rstrip().endswith("// host2" if target == "host2" else "}")
        got = plugins(bundle)[art.name]
        runs[target] = np.asarray(eplan.plan(got, structure=eexec.HostTeam).run(x=x, a=1.5, b=0.25))
    assert runs["host2"].tobytes() == runs["host"].tobytes()
    assert np.allclose(runs["host2"], 1.5 * x + 0.25)
