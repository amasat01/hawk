# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A plane bound to the host runtime under a name the kernel does not take is
refused BY KIND: a plane the kernel declares but never uses (pruned from the
compiled kernel, ``unused_planes`` in the sidecar) is named apart from a name
it never declared."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import sidecar_of

import hawk
import hawk.runtime as rt
from hawk import HawkError, Mutable, Scalar
from hawk.artifact import build_bundle

N = 8


@hawk.kernel
def drops_a_plane(x: Scalar, spare: Scalar, y: Mutable[Scalar]):
    y = x + 1.0


def _loaded(tmp_path):
    bundle = build_bundle([drops_a_plane], tmp_path, targets=("host",), host_profile="native")
    meta = sidecar_of(bundle, drops_a_plane.name)
    return rt.load(bundle.directory, drops_a_plane.name, meta), meta


def test_the_sidecar_lists_the_declared_but_unused_plane(tmp_path):
    _loaded_kernel, meta = _loaded(tmp_path)
    assert meta["unused_planes"] == ["spare"]


def test_a_pruned_plane_and_an_undeclared_one_are_named_apart(tmp_path):
    kernel, _meta = _loaded(tmp_path)
    x, y = np.arange(float(N)), np.zeros(N)
    with pytest.raises(HawkError, match=r"\['spare'\] are declared but unused"):
        rt.run(kernel, x=x, spare=np.zeros(N), y=y)
    with pytest.raises(HawkError, match=r"\['grams'\] are not planes it declares"):
        rt.run(kernel, x=x, grams=np.zeros(N), y=y)
    with pytest.raises(HawkError, match=r"declared but unused.*\['grams'\] are not declared at all"):
        rt.run(kernel, x=x, spare=np.zeros(N), grams=np.zeros(N), y=y)
    rt.run(kernel, x=x, y=y)
    np.testing.assert_array_equal(y, x + 1.0)
