# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A bool the IR types as bool is a bool in the emitted C++: a lane row of the host loop interchange holds it as it is.

A binary search over a knot table keeps its branch conditions in lane rows; the derivatives of a kernel that uses the found
interval compiled to an ``int`` where the IR said ``bool`` (``static_assert: a lane row holds its value's own type``).
"""

import _oracle as O
import pytest
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import Mutable, Scalar, Table
from hawk.artifact import build_bundle

K = 14
N = 37


@hawk.kernel
def search(z: Scalar, nodes: Table["i":K], y: Mutable[Scalar]):  # noqa: F821 - an axis name
    below = z < nodes.at(0.0)
    low = 0.0
    high = 14.0
    for _ in range(10):
        mid = m.floor((low + high) / 2.0)
        if z < nodes.at(mid):
            high = mid
        else:
            low = mid
    iz = m.floor((low + high) / 2.0)
    iz = m.select(iz < 2.0, 2.0, iz)
    iz = m.select(iz > 13.0, 13.0, iz)
    x0 = nodes.at(iz - 1.0)
    x1 = nodes.at(iz)
    x2 = nodes.at(iz + 1.0)
    t = (z - x1) / (x2 - x1)
    left = z < x1
    w = m.select(left, (z - x0) / (x1 - x0), t)
    inside = m.select(below, False, m.select(z >= x0, z <= x2, left))
    acc = 0.0
    for j in range(40):
        acc = acc + m.select(inside, w, 0.5) * m.select(below, x0, x1) * (z - x2) * j
    y = m.select(inside, w * w * (3.0 - 2.0 * w), acc)


def _derived(kind):
    from hawk.diff import jvp, vjp
    from hawk.trace.kernel import Kernel

    return Kernel("search_" + kind, {"jvp": jvp, "vjp": vjp}[kind](search, wrt=("z",), primal_unit="0" * 64))


@pytest.mark.parametrize("wide_form", ["thread", "auto"])
@pytest.mark.parametrize("which", ["primal", "jvp", "vjp"])
def test_a_binary_search_kernel_and_its_derivatives_build_on_the_host_interchange(which, wide_form, tmp_path, cache_dir):
    k = search if which == "primal" else _derived(which)
    b = build_bundle([k], tmp_path / "b", targets=("host",), cache_dir=cache_dir, host_profile="native", wide_form=wide_form)
    O.load(b.directory, k.name, sidecar_of(b, k.name))


@hawk.kernel
def flagged(z: Scalar, nodes: Table["i":K], y: Mutable[Scalar]):  # noqa: F821 - an axis name
    seen = z < nodes.at(0.0)
    acc = z
    for j in range(20):
        hit = z < nodes.at(j % K)
        seen = m.select(hit, True, seen)
        acc = m.select(seen, acc * 1.5, acc + z)
    y = m.select(seen, acc, -acc)


@pytest.mark.parametrize("which", ["primal", "jvp", "vjp"])
def test_a_bool_carried_through_a_loop_builds_on_the_host_interchange(which, tmp_path, cache_dir):
    from hawk.diff import jvp, vjp
    from hawk.trace.kernel import Kernel

    k = flagged if which == "primal" else Kernel("flagged_" + which, {"jvp": jvp, "vjp": vjp}[which](flagged, wrt=("z",), primal_unit="0" * 64))
    b = build_bundle([k], tmp_path / "b", targets=("host",), cache_dir=cache_dir, host_profile="native", wide_form="thread")
    O.load(b.directory, k.name, sidecar_of(b, k.name))


def _basis(z, nodes, iz):
    b = [1.0, 0.0, 0.0, 0.0]
    for d in [1, 2, 3]:
        saved = 0.0
        for r in [0, 1, 2][:d]:
            lo = nodes.at(iz + (r + 1 - d))
            hi = nodes.at(iz + (r + 1))
            den = hi - lo
            ratio = m.select(den == 0.0, 0.0, 1.0 / den)
            tmp = b[r] * ratio
            b[r] = saved + (hi - z) * tmp
            saved = (z - lo) * tmp
        b[d] = saved
    return b


@hawk.kernel
def spline(z: Scalar, nodes: Table["i":K], y: Mutable[Scalar]):  # noqa: F821 - an axis name
    low = 0.0
    high = 14.0
    for _ in range(10):
        mid = m.floor((low + high) / 2.0)
        if z < nodes.at(mid):
            high = mid
        else:
            low = mid
    iz = m.floor((low + high) / 2.0)
    iz = m.select(iz < 2.0, 2.0, iz)
    iz = m.select(iz > 11.0, 11.0, iz)
    b = _basis(z, nodes, iz)
    y = b[0] + 2.0 * b[1] + 3.0 * b[2] + b[3]


@pytest.mark.parametrize("which", ["primal", "jvp", "vjp"])
@pytest.mark.parametrize("wide_form", ["thread", "auto"])
def test_a_spline_after_a_search_builds_on_the_host_interchange(which, wide_form, tmp_path, cache_dir):
    from hawk.diff import jvp, vjp
    from hawk.trace.kernel import Kernel

    k = spline if which == "primal" else Kernel("spline_" + which, {"jvp": jvp, "vjp": vjp}[which](spline, wrt=("z",), primal_unit="0" * 64))
    b = build_bundle([k], tmp_path / "b", targets=("host",), cache_dir=cache_dir, host_profile="native", wide_form=wide_form)
    O.load(b.directory, k.name, sidecar_of(b, k.name))
