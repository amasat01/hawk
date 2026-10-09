# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ambiguous ``(w, w)`` plane: refused, and resolved two ways, zero-copy.

A per-sample plane of width ``w`` whose shape is ``(w, w)`` reads both as the
native component-major ``(w, N=w)`` and as a sample-major ``(N=w, w)``; it is
refused unless the caller says which axis holds the samples, per call
(``layout="samples_first" | "samples_last"``) or per array
(``hawk.samples_first(x)`` / ``hawk.samples_last(x)``). The per-array marker
wins and is honoured for any shape; one that contradicts the shape is refused.
hawk binds by address, so ``samples_first`` still needs the array's transpose to
be C-contiguous (a free view); a C-contiguous sample-major array stays refused
as a copy."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import sidecar_of

import hawk
from hawk import runtime as rt
from hawk.ir import HawkError


def _vec3(built):
    bundle = built["vec3"]
    return rt.load(bundle.directory, "vec3_scale", sidecar_of(bundle, "vec3_scale"))


def _sample_major(n):
    return np.arange(3 * n, dtype=float).reshape(n, 3) * 0.5 + 1.0


def _reference(built, n=4):
    """The unambiguous run: an (n, 3) F-contiguous sample-major plane, n = 4."""
    sm = _sample_major(n)
    y = np.zeros((3, n))
    rt.run(_vec3(built), x=np.asfortranarray(sm), a=2.0, y=y.T)
    return sm, y


def test_square_plane_is_refused_naming_both_readings_and_both_fixes(built):
    x = np.ascontiguousarray(_sample_major(3))
    with pytest.raises(HawkError) as err:
        rt.run(_vec3(built), x=x, a=2.0, y=hawk.samples_last(np.zeros((3, 3))))
    msg = str(err.value)
    for needle in ("'x'", "(3, 3)", "component-major", "sample-major",
                   'layout="samples_first"', 'layout="samples_last"',
                   "hawk.samples_first(x)", "hawk.samples_last(x)"):
        assert needle in msg, (needle, msg)


def test_layout_samples_first_binds_zero_copy_and_matches_the_n4_run(built):
    sm4, y4 = _reference(built)
    xf = np.asfortranarray(sm4[:3])
    y = np.zeros((3, 3))
    rt.run(_vec3(built), x=xf, a=2.0, y=y.T, layout="samples_first")
    np.testing.assert_array_equal(y, y4[:, :3])          # written in place


def test_layout_samples_first_on_a_c_contiguous_array_is_refused_as_a_copy(built):
    x = np.ascontiguousarray(_sample_major(3))
    with pytest.raises(HawkError, match=r"'x'.*sample-major"):
        rt.run(_vec3(built), x=x, a=2.0, y=np.zeros((3, 3)),
               layout="samples_first")


def test_layout_samples_last_binds_as_native(built):
    x = np.arange(9, dtype=float).reshape(3, 3) + 1.0
    y = np.zeros((3, 3))
    rt.run(_vec3(built), x=x, a=2.0, y=y, layout="samples_last")
    np.testing.assert_array_equal(y, 2.0 * x)


def test_marker_samples_first_is_zero_copy(built):
    sm4, y4 = _reference(built)
    xf = np.asfortranarray(sm4[:3])
    y = np.zeros((3, 3))
    rt.run(_vec3(built), x=hawk.samples_first(xf), a=2.0,
           y=hawk.samples_first(y.T))
    np.testing.assert_array_equal(y, y4[:, :3])


def test_marker_is_honoured_for_an_unambiguous_shape(built):
    sm, y_ref = _reference(built)
    y = np.zeros((3, 4))
    rt.run(_vec3(built), x=hawk.samples_first(np.asfortranarray(sm)), a=2.0,
           y=hawk.samples_first(y.T))
    np.testing.assert_array_equal(y, y_ref)
    x = np.arange(12, dtype=float).reshape(3, 4)
    y = np.zeros((3, 4))
    rt.run(_vec3(built), x=hawk.samples_last(x), a=2.0, y=y)
    np.testing.assert_array_equal(y, 2.0 * x)


@pytest.mark.parametrize("marker,shape", [("samples_first", (3, 8)),
                                          ("samples_last", (8, 3))])
def test_a_marker_that_contradicts_the_shape_is_refused_naming_the_argument(
        built, marker, shape):
    x = getattr(hawk, marker)(np.ones(shape))
    with pytest.raises(HawkError, match=rf"'x'.*{marker}.*{shape}"):
        rt.run(_vec3(built), x=x, a=2.0, y=np.zeros((3, 8)), n_samples=8)


def test_per_array_marker_beats_per_call_layout(built):
    x = np.arange(9, dtype=float).reshape(3, 3) + 1.0
    y = np.zeros((3, 3))
    # per call says samples_first (a C-contiguous array would be a refused
    # copy); the array's own samples_last marker wins: it binds as native
    rt.run(_vec3(built), x=hawk.samples_last(x), a=2.0,
           y=hawk.samples_last(y), layout="samples_first")
    np.testing.assert_array_equal(y, 2.0 * x)


def test_a_bad_layout_value_is_refused(built):
    with pytest.raises(HawkError, match="samples_first.*samples_last"):
        rt.run(_vec3(built), x=np.ones((3, 3)), a=2.0, y=np.zeros((3, 3)),
               layout="rows")


def test_the_host_kernel_bind_path_takes_the_same_choices(built):
    k = _vec3(built)
    x = np.arange(9, dtype=float).reshape(3, 3) + 1.0
    y = np.zeros((3, 3))
    with pytest.raises(HawkError, match="samples_first"):
        k.bind_all({"x": x, "a": 2.0, "y": y}, 3)
    k.bind_all({"x": x, "a": 2.0, "y": y}, 3, layout="samples_last")
    k.launch(0, 3, 3)
    np.testing.assert_array_equal(y, 2.0 * x)


def test_a_square_matrix_width_is_ambiguous_too():
    """A matrix head with N == R == C is a (R*C, N) plane at the binding."""
    a = np.zeros((9, 9))
    assert rt.plane_layout(memoryview(a), 9) == "ambiguous"
    assert rt.plane_layout(memoryview(a), 9, layout="samples_last") == "native"
    assert rt.plane_layout(memoryview(a.T), 9, layout="samples_first") == "view"


def test_the_shared_marker_protocol_is_accepted(built):
    """eagle's markers carry the same two attributes; hawk takes any such
    object, so each package accepts the other's (it imports neither)."""

    class Foreign:
        def __init__(self, array, axis):
            self.array, self.__raptor_samples_axis__ = array, axis

    sm4, y4 = _reference(built)
    xf = np.asfortranarray(sm4[:3])
    y = np.zeros((3, 3))
    rt.run(_vec3(built), x=Foreign(xf, "first"), a=2.0, y=Foreign(y.T, "first"))
    np.testing.assert_array_equal(y, y4[:, :3])
    try:
        import eagle
    except ImportError:
        return
    if hasattr(eagle, "samples_first"):
        y[:] = 0
        rt.run(_vec3(built), x=eagle.samples_first(xf), a=2.0,
               y=eagle.samples_first(y.T))
        np.testing.assert_array_equal(y, y4[:, :3])
