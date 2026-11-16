# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A host unit compiled against the aether SOURCE tree reads its linalg
dimension cap from the wing's generated header.

The default host build compiles against the sealed payload of the installed
aether, so a change in the aether sources (the cap read from the wing's
header) is only exercised when ``$HAWK_AETHER_INCLUDE`` names that source
tree. ``$W25_AETHER_SRC`` names it for this row; it is one probe of the
gate's toolchain registry (tests/conftest.py): unset, the row is reported
toolchain-absent for ``aether-src`` (never skipped, never failed).
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import pytest
from conftest import sidecar_of
from _linalg_kernels import chol_kernel

from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.compile import drivers

LANES = 8


@pytest.mark.needs_toolchain("aether-src")
@pytest.mark.repo_local  # reads a sibling aether source tree, which a wheel install does not carry
def test_a_cholesky_unit_built_against_the_aether_source_tree_matches_numpy(tmp_path, monkeypatch):
    src = os.environ["W25_AETHER_SRC"]
    assert (pathlib.Path(src) / "aether" / "linalg" / "detail" / "Common.h").is_file(), src
    monkeypatch.setenv("HAWK_AETHER_INCLUDE", src)
    assert not drivers._Inputs().sealed

    seen = []
    real = drivers.lookup_key

    def spy(source, target, mode, compiler, flags):
        seen.append([str(f) for f in flags])
        return real(source, target, mode, compiler, flags)
    monkeypatch.setattr(drivers, "lookup_key", spy)
    lines = []
    run = drivers.run_compiler

    def run_spy(argv, what, *rest, **named):
        lines.append([str(a) for a in argv])
        return run(argv, what, *rest, **named)
    monkeypatch.setattr(drivers, "run_compiler", run_spy)

    kernel = chol_kernel(3)
    bundle = build_bundle([kernel], tmp_path / "b", targets=("host",), cache_dir=str(tmp_path / "cache"),
                          host_profile="native")
    assert seen, "no host compile happened"
    # the key names the wing by name and header content, never by where it is installed ...
    assert any(f.startswith("facts:") for f in seen[0]), seen[0]
    # ... and the compile line hands the unit that wing's generated header
    assert lines, "no compiler ran"
    inc = [a[2:] for a in lines[0] if a.startswith("-I")]
    assert any((pathlib.Path(d) / "raptor_facts.h").is_file() for d in inc), lines[0]

    rng = np.random.default_rng(11)
    g = rng.standard_normal((LANES, 3, 3))
    spd = g @ g.transpose(0, 2, 1) + 3.0 * np.eye(3)
    b = rng.standard_normal((LANES, 3))
    rhs = rng.standard_normal((LANES, 3, 2))
    plane = lambda a: np.ascontiguousarray(a.reshape(LANES, -1).T)
    out = {"x": np.zeros((3, LANES)), "X": np.zeros((6, LANES)), "L": np.zeros((9, LANES)),
           "info": np.zeros(LANES, dtype=np.int32)}
    rt.run(rt.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name)),
           A=plane(spd), b=plane(b), B=plane(rhs), **out)
    for lane in range(LANES):
        np.testing.assert_allclose(out["L"][:, lane].reshape(3, 3), np.linalg.cholesky(spd[lane]),
                                   rtol=1e-12, atol=1e-12)
    assert not out["info"].any()
