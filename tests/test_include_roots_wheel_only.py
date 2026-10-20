# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The public header roots on a wheel-only install.

With no real header tree reachable (no override, no prefix, no checkout, no
source-distribution copy), ``hawk.compile.aether_include()`` and
``eagle_include()`` return the sealed payload's served root like
``include_flags()``; the strict resolvers still refuse, and the provenance
marker still reads ``sealed:`` there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hawk.compile as hc
from hawk import _roots
from hawk.artifact.sidecar import _root_or_sealed_marker
from hawk.compile import toolchain as tc


@pytest.fixture
def wheel_only(monkeypatch, tmp_path):
    for var in ("CONDA_PREFIX", "PREFIX", "HAWK_AETHER_INCLUDE", "HAWK_EAGLE_INCLUDE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(_roots, "_prefixes", lambda: [])
    monkeypatch.setattr(_roots, "WORKSPACE", tmp_path)
    monkeypatch.setattr(_roots, "SDIST_INCLUDE", tmp_path / "no_sdist")


def test_public_roots_fall_back_to_the_sealed_root(wheel_only):
    with pytest.raises(_roots.RootError):
        _roots.aether_include()
    aether, eagle = hc.aether_include(), hc.eagle_include()
    assert (Path(aether) / "aether" / "typedefs.h").is_file()
    assert (Path(eagle) / "plugin" / "gref_abi.h").is_file()
    assert tc.include_flags() == ["-I", aether, "-I", eagle]


def test_the_provenance_marker_still_reads_sealed(wheel_only):
    assert _root_or_sealed_marker(lambda: tc._resolve(_roots.aether_include)).startswith("sealed:")
