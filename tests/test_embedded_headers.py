# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A host build needs no header tree on the machine: the numerics headers a unit includes are embedded in the OSPREY.

The build runs with every variable and prefix that could name a header folder unset, the shape of a machine that has only
the installed wheels, and the kernel it builds (vector math and a reduction) equals NumPy.
"""

from __future__ import annotations

import numpy as np
import pytest

import hawk
import hawk.math as M
from hawk import Mutable, Scalar, runtime
from hawk.artifact import build_bundle

HEADER_VARIABLES = (
    "HAWK_AETHER_INCLUDE",
    "HAWK_EAGLE_INCLUDE",
    "CONDA_PREFIX",
    "PREFIX",
)


@hawk.kernel
def wave(x: Scalar, y: Mutable[Scalar]):
    y = M.exp(x) + M.sin(x) * M.cos(x)


@pytest.mark.parametrize("profile", ["native", "portable"])
def test_a_host_unit_with_vector_math_builds_with_no_header_tree_named_and_equals_numpy(
    profile, monkeypatch, tmp_path
):
    """Oracle: NumPy's ``exp(x) + sin(x) * cos(x)``, to a few ulps (the library's vector math is not NumPy's)."""
    for variable in HEADER_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    bundle = build_bundle(
        [wave],
        tmp_path / "bundle",
        targets=("host",),
        cache_dir=str(tmp_path / "store"),
        options={"host.profile": profile},
    )
    loaded = runtime.load(bundle.directory, "wave")
    x = np.random.default_rng(7).uniform(-3.0, 3.0, 1000)
    y = np.full_like(x, np.nan)
    runtime.run(loaded, x=x, y=y)
    np.testing.assert_allclose(
        y, np.exp(x) + np.sin(x) * np.cos(x), rtol=1e-13, atol=1e-15
    )
