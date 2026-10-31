# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``Param[...]`` is ONE value shared by every lane, whatever type it wraps.

The bare type (``Scalar``, ``Vector[W]``, ``Matrix[R,C]``, ``Table[...]``) is per-lane; ``Param``
in front of it makes it a single value. ``Param[Vector[W]]``, ``Param[Matrix[R,C]]`` and
``Param[Table["row":R, "col":C]]`` are bound as an array of their own shape, held in a ``lookup``
plane of their cells (row-major), and read from memory: ``p[j]`` is one cell, never a register copy
of the whole value. Rows: every type on the host and the device, bit-identity with the four-cell
``Table[Scalar]`` spelling that worked before, the notebook shape (a per-lane vector times one 2x2
matrix), the derivatives (the gradient of a shared parameter is summed over the lanes), and the
refusals.
"""

from __future__ import annotations

import numpy as np
import pytest

import hawk
import hawk.math as m
from hawk import HawkError, Matrix, Mutable, Param, Scalar, Table, Vector
from hawk.diff import jvp, vjp
from hawk.trace.kernel import Kernel

LANES = 5
RNG = np.random.default_rng(7)
A22 = np.array([[1.0, 2.0], [3.0, 4.0]])


# --------------------------------------------------------------------------- #
# The subjects.
# --------------------------------------------------------------------------- #
@hawk.kernel
def rotate(x: Vector[2], A: Param[Matrix[2, 2]], y: Mutable[Vector[2]]):
    """Notebook 23's shape: a per-lane vector times ONE matrix every lane shares."""
    y = A @ x


@hawk.kernel
def rotate_cells(x: Vector[2], A: Table[Scalar], y: Mutable[Vector[2]]):
    """The same by hand, before the feature: four cells of a flat table."""
    y = m.mat([[A.at(0), A.at(1)], [A.at(2), A.at(3)]]) @ x


@hawk.kernel
def entries(x: Vector[2], A: Param[Matrix[2, 2]], y: Mutable[Vector[2]]):
    """A matrix read entry by entry: each ``A[i, j]`` is one cell."""
    y = m.vec(A[0, 0] * x[0] + A[0, 1] * x[1], A[1, 0] * x[0] + A[1, 1] * x[1])


@hawk.kernel
def dot_loop(x: Vector[6], w: Param[Vector[6]], y: Mutable[Scalar]):
    """A shared vector read cell by cell in a loop."""
    acc = 0.0
    for j in range(6):
        acc = acc + w[j] * x[j]
    y = acc


@hawk.kernel
def dot_whole(x: Vector[6], w: Param[Vector[6]], y: Mutable[Scalar]):
    """A shared vector used whole."""
    y = m.dot(w, x)


@hawk.kernel
def dot_cells(x: Vector[6], w: Table[Scalar], y: Mutable[Scalar]):
    acc = 0.0
    for j in range(6):
        acc = acc + w.at(j) * x[j]
    y = acc


@hawk.kernel
def grid_table(s: Scalar, T: Param[Table["row":3, "col":4]], y: Mutable[Scalar]):
    y = T.at(row=1, col=2) * s + T.at(row=2, col=3)


@hawk.kernel
def wide_matvec(x: Scalar, bf: Param[Vector[512]], w: Table["row":24, "col":512],
                y: Mutable[Scalar]):
    """24x512 matvec row, the components of a 512-wide shared vector read from memory."""
    acc = 0.0
    for j in range(512):
        acc = acc + w.at(row=3, col=j) * bf[j]
    y = acc * x


@hawk.kernel
def poly(x: Scalar, w: Param[Vector[2]], y: Mutable[Scalar]):
    y = w[0] * x + w[1] * x * x


def _lanes(n=LANES, w=2):
    return RNG.random((w, n)) + 0.5


def _as_array(out):
    return np.asarray(out)


# --------------------------------------------------------------------------- #
# Host.
# --------------------------------------------------------------------------- #
def test_a_shared_matrix_is_one_value_for_every_lane():
    x = _lanes()
    got = _as_array(rotate(x, A22))
    np.testing.assert_allclose(got, A22 @ x, rtol=1e-14)
    assert _as_array(rotate(A=A22, x=x)).shape == (2, LANES)


def test_a_shared_matrix_is_bit_identical_to_the_four_cell_table():
    x = _lanes()
    np.testing.assert_array_equal(_as_array(rotate(x, A22)),
                                  _as_array(rotate_cells(x, A22.ravel())))
    # entry by entry sums in a different order than the matrix product: close, not bit-equal
    np.testing.assert_allclose(_as_array(entries(x, A22)), _as_array(rotate(x, A22)), rtol=1e-15)


@pytest.mark.parametrize("kernel", [dot_loop, dot_whole], ids=["cell_by_cell", "whole"])
def test_a_shared_vector(kernel):
    x, w = _lanes(w=6), RNG.random(6)
    got = _as_array(kernel(x, w))
    np.testing.assert_array_equal(got, _as_array(dot_cells(x, w)) if kernel is dot_loop else got)
    np.testing.assert_allclose(got, w @ x, rtol=1e-14)


def test_a_shared_named_axis_table():
    s, T = RNG.random(LANES), RNG.random((3, 4))
    np.testing.assert_allclose(_as_array(grid_table(s, T)), T[1, 2] * s + T[2, 3], rtol=1e-15)


def test_a_512_wide_shared_vector_is_read_cell_by_cell():
    x, bf, w = RNG.random(LANES), RNG.random(512), RNG.random((24, 512))
    np.testing.assert_allclose(_as_array(wide_matvec(x, bf, w)), (w[3] @ bf) * x, rtol=1e-12)


def test_a_call_with_the_wrong_shape_is_refused_by_name():
    x = _lanes()
    with pytest.raises(ValueError, match=r"'A' is a shared parameter of shape \(2, 2\)"):
        rotate(x, np.ones(4))
    with pytest.raises(ValueError, match=r"'A' is a shared parameter of shape \(2, 2\)"):
        rotate(x, np.ones((3, 2)))


def test_the_sidecar_records_the_natural_shapes_additively(tmp_path):
    from conftest import sidecar_of
    from hawk.artifact import build_bundle

    bundle = build_bundle([rotate, dot_whole, grid_table], tmp_path, targets=("host",),
                          host_profile="native")
    assert sidecar_of(bundle, rotate.name)["param_shapes"] == {"A": [2, 2]}
    assert sidecar_of(bundle, dot_whole.name)["param_shapes"] == {"w": [6]}
    assert sidecar_of(bundle, grid_table.name)["param_shapes"] == {"T": [3, 4]}
    assert sidecar_of(bundle, rotate.name)["arg_spec"] == [["mutable", "y"], ["vec_in", "x"],
                                                           ["lookup", "A"]]


def test_the_host_runtime_binds_a_shared_value_by_its_shape(tmp_path):
    import hawk.runtime as rt
    from conftest import sidecar_of
    from hawk.artifact import build_bundle

    bundle = build_bundle([rotate], tmp_path, targets=("host",), host_profile="native")
    loaded = rt.load(bundle.directory, rotate.name, sidecar_of(bundle, rotate.name))
    x = np.ascontiguousarray(_lanes())
    y = np.zeros((2, LANES))
    rt.run(loaded, x=x, A=A22, y=y)
    np.testing.assert_allclose(y, A22 @ x, rtol=1e-14)
    with pytest.raises(HawkError, match=r"'A' is a shared parameter of shape \(2, 2\)"):
        rt.run(loaded, x=x, A=np.ones(4), y=y)


# --------------------------------------------------------------------------- #
# The refusals, at declaration.
# --------------------------------------------------------------------------- #
def test_a_scalar_param_is_unchanged():
    assert Param[hawk.Int32].form == "uniform" and Param[hawk.UInt32].ttype == hawk.UInt32
    assert Param.form == "uniform"


def test_unsupported_shared_types_are_refused_with_the_fix():
    from hawk.types import TensorType

    with pytest.raises(HawkError, match=r"rank 1 or 2.*separate Scalar/Int32 Params"):
        Param[TensorType((2, 2, 2), "f64")]
    with pytest.raises(HawkError, match=r"rank 1 or 2.*separate Scalar/Int32 Params"):
        Param[TensorType((3,), "int32")]                       # an integer vector
    with pytest.raises(HawkError, match=r"rank 1 or 2"):
        Param[hawk.Quat]
    with pytest.raises(HawkError, match=r"static extents.*already one plane shared"):
        Param[Table[Scalar]]
    with pytest.raises(HawkError, match=r"static extents"):
        Param[Table["row", "col":3]]                           # a runtime first extent


def test_a_static_index_outside_the_value_is_refused_by_name():
    with pytest.raises(HawkError, match=r"'w': index 6 is outside the component extent 6"):
        @hawk.kernel
        def too_far(x: Scalar, w: Param[Vector[6]], y: Mutable[Scalar]):
            y = w[6] * x


# --------------------------------------------------------------------------- #
# Derivatives: the gradient of a shared parameter is summed over the lanes.
# --------------------------------------------------------------------------- #
def test_the_gradient_of_a_shared_vector_is_summed_over_the_lanes():
    x, w, cot = np.linspace(0.5, 2.0, LANES), np.array([1.5, -0.7]), np.linspace(1.0, 2.0, LANES)
    got = Kernel("poly_vjp", vjp(poly))(x=x, w=w, bar_y=cot)
    h = 1e-6

    def y(wv):
        return wv[0] * x + wv[1] * x * x

    expect = [np.sum(cot * (y(w + h * e) - y(w - h * e)) / (2 * h)) for e in np.eye(2)]
    np.testing.assert_allclose(np.asarray(got["bar_w"]), expect, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(got["bar_w"]), [np.sum(cot * x), np.sum(cot * x * x)],
                               rtol=1e-14)


def test_the_forward_derivative_reads_one_tangent_for_all_lanes():
    x, w, dw = np.linspace(0.5, 2.0, LANES), np.array([1.5, -0.7]), np.array([0.3, 1.1])
    got = Kernel("poly_jvp", jvp(poly))(x=x, w=w, dot_w=dw, dot_x=np.zeros(LANES))
    np.testing.assert_allclose(np.asarray(got), dw[0] * x + dw[1] * x * x, rtol=1e-14)


# --------------------------------------------------------------------------- #
# Device (run through the GPU pool).
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_every_shared_type_on_the_device():
    cp = pytest.importorskip("cupy")
    import eagle

    x = _lanes()
    plan = eagle.deploy(rotate)
    got = plan(cp.asarray(np.ascontiguousarray(x)), cp.asarray(A22))
    np.testing.assert_allclose(cp.asnumpy(got), A22 @ x, rtol=1e-14)
    # the device contracts to FMA, so the host's bits are not its bits: the 4-cell twin is
    cells = eagle.deploy(rotate_cells)(cp.asarray(np.ascontiguousarray(x)), cp.asarray(A22.ravel()))
    np.testing.assert_array_equal(cp.asnumpy(got), cp.asnumpy(cells))

    x6, w = _lanes(w=6), RNG.random(6)
    got = eagle.deploy(dot_loop)(cp.asarray(np.ascontiguousarray(x6)), cp.asarray(w))
    np.testing.assert_allclose(cp.asnumpy(got), w @ x6, rtol=1e-13)
    got = eagle.deploy(dot_whole)(cp.asarray(np.ascontiguousarray(x6)), cp.asarray(w))
    np.testing.assert_allclose(cp.asnumpy(got), w @ x6, rtol=1e-13)

    s, T = RNG.random(LANES), RNG.random((3, 4))
    got = eagle.deploy(grid_table)(cp.asarray(s), cp.asarray(T))
    np.testing.assert_allclose(cp.asnumpy(got), T[1, 2] * s + T[2, 3], rtol=1e-14)

    xs, bf, wm = RNG.random(LANES), RNG.random(512), RNG.random((24, 512))
    got = eagle.deploy(wide_matvec)(cp.asarray(xs), cp.asarray(bf), cp.asarray(wm))
    np.testing.assert_allclose(cp.asnumpy(got), (wm[3] @ bf) * xs, rtol=1e-12)


@pytest.mark.gpu
def test_the_shared_gradient_sums_over_the_lanes_on_the_device():
    cp = pytest.importorskip("cupy")
    x, w, cot = np.linspace(0.5, 2.0, LANES), np.array([1.5, -0.7]), np.linspace(1.0, 2.0, LANES)
    d = Kernel("poly_vjp_dev", vjp(poly))
    got = d._deploy()(x=cp.asarray(x), w=cp.asarray(w), bar_y=cp.asarray(cot))
    np.testing.assert_allclose(cp.asnumpy(got["bar_w"]), [np.sum(cot * x), np.sum(cot * x * x)],
                               rtol=1e-13)
