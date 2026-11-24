# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.reduce``, ``hawk.copy``, ``hawk.matmul`` and ``hawk.fft``: one call path, eager and recorded in a graph builder.

Every value is checked against NumPy written here (``np.sum / min / max / all``, ``a @ b``, ``np.fft.fftn / rfftn / irfftn``,
array equality, and the closed-form transposes for the reverse). A recorded program is checked against the same NumPy
computation and against the eager call, and a replay after the inputs change gives the values of the new inputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import hawk
from hawk import HawkError, Mutable, Scalar, Vector

RNG = np.random.default_rng(20261125)
N = 6


@hawk.kernel
def double_v(x: Vector[3], y: Mutable[Vector[3]]):
    y = x * 2.0


@hawk.kernel
def ten_v(v: Vector[3], w: Mutable[Vector[3]]):
    w = v * 10.0


@hawk.kernel
def double(x: Scalar, y: Mutable[Scalar]):
    y = x * 2.0


@hawk.kernel
def tenfold(r: Scalar, w: Mutable[Scalar]):
    w = r * 10.0


def planes(shape, dtype=np.float64):
    """A random plane of ``shape``: normal for floats, small whole numbers for integers."""
    if np.issubdtype(dtype, np.integer):
        return RNG.integers(-50, 50, size=shape).astype(dtype)
    return RNG.normal(size=shape).astype(dtype)


def complex_plane(z):
    """The float plane (real part, imaginary part last) of the complex NumPy array ``z``."""
    return np.ascontiguousarray(np.stack([z.real, z.imag], axis=-1))


def as_complex(plane):
    return plane[..., 0] + 1j * plane[..., 1]


# ---- reduce ----


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int32])
@pytest.mark.parametrize("op", ["sum", "min", "max"])
def test_an_eager_reduce_equals_numpy_per_column(op, dtype):
    """Oracle: ``np.sum / min / max`` over the last axis; a float sum within the compensated bound of ``math.fsum``, the rest exact."""
    p = planes((3, 50), dtype)
    got = hawk.reduce(p, op)
    assert got.shape == (3,) and got.dtype == p.dtype
    if op == "sum" and dtype != np.int32:
        want = np.array([math.fsum(row.astype(np.float64)) for row in p])
        np.testing.assert_allclose(
            got, want, rtol=1e-6 if dtype == np.float32 else 1e-14
        )
    else:
        np.testing.assert_array_equal(got, getattr(np, op)(p, axis=1))


def test_an_eager_reduce_of_a_flat_plane_is_a_one_element_plane():
    p = planes(50)
    got = hawk.reduce(p, "min")
    assert got.shape == (1,)
    assert got[0] == p.min()


def test_the_fixed_contract_sums_within_the_bound_of_the_exact_sum():
    p = planes((2, 500))
    want = np.array([math.fsum(row) for row in p])
    np.testing.assert_allclose(
        hawk.reduce(p, "sum", contract="fixed"), want, rtol=1e-12
    )


@pytest.mark.parametrize("fraction", [0.0, 0.01, 1.0])
def test_an_eager_all_equals_numpy_all_of_a_bool_plane(fraction):
    p = RNG.random((2, 50)) >= fraction
    np.testing.assert_array_equal(hawk.reduce(p, "all"), p.all(axis=1))


def test_reduce_into_the_callers_plane_writes_there():
    p, out = planes((3, 20)), np.zeros(3)
    assert hawk.reduce(p, "max", out=out) is out
    np.testing.assert_array_equal(out, p.max(axis=1))


def test_an_op_or_a_contract_that_is_not_written_is_refused_by_name():
    with pytest.raises(
        HawkError,
        match=r"hawk.reduce: op='any' is not one of \('sum', 'min', 'max', 'all'\)",
    ):
        hawk.reduce(planes(5), "any")
    with pytest.raises(
        HawkError,
        match=r"hawk.reduce: contract='fast' is not one of \('compensated', 'fixed'\)",
    ):
        hawk.reduce(planes(5), "sum", contract="fast")


# ---- copy ----


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int32])
def test_an_eager_copy_equals_the_source(dtype):
    src, dst = planes((3, 17), dtype), np.zeros((3, 17), dtype)
    assert hawk.copy(src, dst) is dst
    np.testing.assert_array_equal(dst, src)


# ---- matrix product ----


@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize("transpose_a", [False, True])
def test_an_eager_matmul_equals_numpy_with_and_without_the_transposes(
    transpose_a, transpose_b
):
    """Oracle: ``a @ b`` of the operands as ``op`` reads them."""
    a, b = planes((4, 5)), planes((5, 6))
    stored_a = np.ascontiguousarray(a.T) if transpose_a else a
    stored_b = np.ascontiguousarray(b.T) if transpose_b else b
    got = hawk.matmul(
        stored_a, stored_b, transpose_a=transpose_a, transpose_b=transpose_b
    )
    np.testing.assert_allclose(got, a @ b, rtol=1e-13, atol=1e-13)


def test_a_matmul_that_accumulates_adds_to_the_callers_plane():
    a, b, out = planes((4, 5)), planes((5, 6)), planes((4, 6))
    start = out.copy()
    hawk.matmul(a, b, out=out, accumulate=True)
    np.testing.assert_allclose(out, start + a @ b, rtol=1e-13, atol=1e-13)


# ---- transforms ----


@pytest.mark.parametrize("norm", ["backward", "forward", "ortho"])
@pytest.mark.parametrize("axes", [None, (1,), (0, 1)])
def test_an_eager_c2c_equals_numpy_fftn(axes, norm):
    z = RNG.normal(size=(4, 6)) + 1j * RNG.normal(size=(4, 6))
    got = as_complex(hawk.fft.c2c(complex_plane(z), axes=axes, norm=norm))
    np.testing.assert_allclose(got, np.fft.fftn(z, axes=axes, norm=norm), atol=1e-12)


def test_an_eager_inverse_c2c_equals_numpy_ifftn():
    z = RNG.normal(size=(4, 6)) + 1j * RNG.normal(size=(4, 6))
    got = as_complex(hawk.fft.c2c(complex_plane(z), inverse=True))
    np.testing.assert_allclose(got, np.fft.ifftn(z), atol=1e-12)


@pytest.mark.parametrize("norm", ["backward", "ortho"])
@pytest.mark.parametrize("axes", [None, (1,)])
def test_an_eager_r2c_then_c2r_equal_numpy_rfftn_and_irfftn(axes, norm):
    y = RNG.normal(size=(4, 6))
    spectrum = hawk.fft.r2c(y, axes=axes, norm=norm)
    np.testing.assert_allclose(
        as_complex(spectrum), np.fft.rfftn(y, axes=axes, norm=norm), atol=1e-12
    )
    chosen = (0, 1) if axes is None else axes
    inverse = np.fft.irfftn(
        np.fft.rfftn(y, axes=chosen, norm=norm),
        s=[y.shape[a] for a in chosen],
        axes=chosen,
        norm=norm,
    )
    np.testing.assert_allclose(
        hawk.fft.c2r(spectrum, axes=axes, norm=norm), inverse, atol=1e-12
    )


# ---- the door's reasons, unparsed ----


@pytest.mark.parametrize(
    ("call", "stem"),
    [
        (
            lambda: hawk.reduce(planes(5), "sum", out=np.zeros(3)),
            r"eg_reduce: out must have shape \(1,\)",
        ),
        (
            lambda: hawk.reduce(planes(5), "sum", out=np.zeros(1, np.float32)),
            r"eg_reduce: out dtype differs from in",
        ),
        (
            lambda: hawk.reduce(planes((1, 5), np.int32), "all"),
            r"eg_reduce: op ALL takes a bool plane",
        ),
        (
            lambda: hawk.matmul(planes((4, 5)), planes((5, 6)), out=np.zeros((4, 7))),
            r"eg_gemm_desc: ",
        ),
        (
            lambda: hawk.matmul(planes((4, 5)), planes((4, 6))),
            r"eg_gemm_desc: the contracted extents disagree",
        ),
        (lambda: hawk.copy(planes(5), np.zeros(6)), r"eg_copy: "),
        (
            lambda: hawk.fft.c2c(
                complex_plane(planes((4, 6)) + 0j), out=np.zeros((4, 7, 2))
            ),
            r"eg_fft_run out: the plane must have shape \(4, 6, 2\), got \(4, 7, 2\)",
        ),
        (
            lambda: hawk.fft.r2c(planes((4, 6)), inverse=True),
            r"EG_FFT_INVERSE is for C2C: R2C is forward and C2R inverse",
        ),
    ],
    ids=[
        "reduce shape",
        "reduce dtype",
        "all of integers",
        "matmul out shape",
        "matmul contracted extents",
        "copy shape",
        "fft out shape",
        "inverse r2c",
    ],
)
def test_a_wrong_operand_is_refused_by_name_with_the_doors_reason(call, stem):
    with pytest.raises(HawkError, match=stem):
        call()


# ---- recorded in a builder: the same nodes between kernels ----


def run_program(build, x):
    """The builder ``build`` made into a graph over ``x``, built and replayed once; returns the graph."""
    graph = hawk.Graph.builder(build)(x)
    graph.launch()
    return graph


def test_a_recorded_reduce_sum_between_two_kernels_equals_the_eager_one_and_numpy():
    """Oracle: ``np.sum`` of twice the input over the samples, scaled by ten, per column."""

    def build(x) -> hawk.Graph:
        g = hawk.Graph()
        g.total = hawk.reduce(double_v(x), "sum")
        g.result = tenfold(g.total)
        return g

    x = planes((3, N))
    graph = run_program(build, x)
    np.testing.assert_allclose(graph.total, (2.0 * x).sum(axis=1), rtol=1e-14)
    np.testing.assert_allclose(graph.result, 10.0 * (2.0 * x).sum(axis=1), rtol=1e-14)
    np.testing.assert_allclose(graph.total, hawk.reduce(2.0 * x, "sum"), rtol=1e-14)
    assert graph.node_labels() == ["double_v", "reduce", "tenfold"]
    x += 1.0  # a replay reads the planes as they are then
    graph.launch()
    np.testing.assert_allclose(graph.total, (2.0 * x).sum(axis=1), rtol=1e-14)
    graph.close()


@pytest.mark.parametrize("op", ["min", "max"])
def test_a_recorded_reduce_min_or_max_equals_numpy_exactly(op):
    def build(x) -> hawk.Graph:
        g = hawk.Graph()
        g.found = hawk.reduce(double_v(x), op)
        return g

    x = planes((3, N))
    graph = run_program(build, x)
    np.testing.assert_array_equal(graph.found, getattr(np, op)(2.0 * x, axis=1))
    graph.close()


def test_a_recorded_all_equals_numpy_all():
    def build(mask) -> hawk.Graph:
        g = hawk.Graph()
        g.everything = hawk.reduce(mask, "all")
        return g

    mask = RNG.random((2, 30)) >= 0.01
    graph = run_program(build, mask)
    np.testing.assert_array_equal(graph.everything, mask.all(axis=1))
    graph.close()


def test_a_recorded_copy_between_two_kernels_equals_the_eager_copy_and_numpy():
    def build(x) -> hawk.Graph:
        g = hawk.Graph()
        moved = np.zeros((3, N))
        hawk.copy(double_v(x), moved)
        g.result = ten_v(moved)
        return g

    x = planes((3, N))
    graph = run_program(build, x)
    np.testing.assert_allclose(graph.result, 20.0 * x, rtol=1e-14)
    assert graph.node_labels() == ["double_v", "copy", "ten_v"]
    graph.close()


def test_a_recorded_matmul_between_two_kernels_equals_numpy():
    """Oracle: ``(2 x) @ b`` then ten times that."""
    b = planes((N, 2))

    def build(x) -> hawk.Graph:
        g = hawk.Graph()
        g.product = hawk.matmul(double_v(x), b)
        g.result = ten_v(g.product)
        return g

    x = planes((3, N))
    graph = run_program(build, x)
    np.testing.assert_allclose(graph.product, (2.0 * x) @ b, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        graph.result, 10.0 * ((2.0 * x) @ b), rtol=1e-13, atol=1e-13
    )
    assert graph.node_labels() == ["double_v", "gemm", "ten_v"]
    x += 0.5
    graph.launch()
    np.testing.assert_allclose(graph.product, (2.0 * x) @ b, rtol=1e-13, atol=1e-13)
    graph.close()


def test_recorded_transforms_between_two_kernels_equal_numpy_and_replay():
    """Oracle: ``np.fft.rfft`` of twice the input, and ``np.fft.irfft`` of that back, doubled."""

    def build(x) -> hawk.Graph:
        g = hawk.Graph()
        g.spectrum = hawk.fft.r2c(double(x))
        g.back = hawk.fft.c2r(g.spectrum, out=np.zeros(x.shape))
        g.result = double(g.back)
        return g

    x = planes(N)
    graph = run_program(build, x)
    np.testing.assert_allclose(
        as_complex(graph.spectrum), np.fft.rfft(2.0 * x), atol=1e-12
    )
    np.testing.assert_allclose(graph.result, 4.0 * x, atol=1e-12)
    assert graph.node_labels() == ["double", "fft", "fft", "double#2"]
    x += 1.0
    graph.launch()
    np.testing.assert_allclose(graph.result, 4.0 * x, atol=1e-12)
    graph.close()


def test_a_recorded_c2c_equals_numpy_fftn():
    z = RNG.normal(size=(4, 6)) + 1j * RNG.normal(size=(4, 6))
    plane = complex_plane(z)

    def build(plane) -> hawk.Graph:
        g = hawk.Graph()
        g.spectrum = hawk.fft.c2c(plane, norm="ortho")
        return g

    graph = run_program(build, plane)
    np.testing.assert_allclose(
        as_complex(graph.spectrum), np.fft.fftn(z, norm="ortho"), atol=1e-12
    )
    graph.close()


# ---- reverse, as the library has it today ----


def reverse_of(build, x, cotangents):
    """The gradients of ``cotangents`` through the graph ``build`` writes with ``record=True``, replayed once."""
    graph = hawk.Graph.builder(build)(x)
    graph.launch()
    return hawk.reverse(graph.record, cotangents)


def test_the_reverse_of_a_copy_is_the_cotangent_moved_back():
    """Oracle: ``w = 10 d`` and ``d = 2 x``, so ``dJ/dx = 20 * cotangent``."""

    def build(x) -> hawk.Graph:
        g = hawk.Graph(record=True)
        moved = np.zeros((3, N))
        hawk.copy(double_v(x), moved)
        g.result = ten_v(moved)
        return g

    x, seed = planes((3, N)), planes((3, N))
    gradients = reverse_of(build, x, {"w": seed})
    np.testing.assert_allclose(gradients["x"], 20.0 * seed, rtol=1e-14)


@pytest.mark.parametrize("transpose_b", [False, True])
def test_the_reverse_of_a_matmul_is_the_numpy_transpose_product(transpose_b):
    """Oracle: ``m = (2 x) @ b``, ``w = 10 m``: ``dJ/dx = 2 * (10 G) @ b.T`` for the cotangent ``G``."""
    b = planes((N, 2))
    stored = np.ascontiguousarray(b.T) if transpose_b else b

    def build(x) -> hawk.Graph:
        g = hawk.Graph(record=True)
        g.result = ten_v(hawk.matmul(double_v(x), stored, transpose_b=transpose_b))
        return g

    x, seed = planes((3, N)), planes((3, 2))
    gradients = reverse_of(build, x, {"w": seed})
    np.testing.assert_allclose(
        gradients["x"], 2.0 * (10.0 * seed) @ b.T, rtol=1e-13, atol=1e-13
    )


def test_the_reverse_of_a_transform_pair_is_the_numpy_identity_of_the_round_trip():
    """Oracle: ``irfft(rfft(2 x))`` is ``2 x``, doubled again: the gradient is ``4 *`` the cotangent."""

    def build(x) -> hawk.Graph:
        g = hawk.Graph(record=True)
        spectrum = hawk.fft.r2c(double(x))
        g.result = double(hawk.fft.c2r(spectrum, out=np.zeros(x.shape)))
        return g

    x, seed = planes(N), planes(N)
    gradients = reverse_of(build, x, {"double#2.y": seed})
    np.testing.assert_allclose(gradients["double.x"], 4.0 * seed, atol=1e-12)


def test_the_reverse_of_a_complex_transform_pair_is_the_numpy_identity_of_the_round_trip():
    """Oracle: forward then inverse ``c2c`` is the identity (``fftn`` then ``ifftn``), so the gradient is ``4 *`` the cotangent."""

    def build(x) -> hawk.Graph:
        g = hawk.Graph(record=True)
        spectrum = hawk.fft.r2c(double(x))
        there = hawk.fft.c2c(spectrum)
        back = hawk.fft.c2c(there, inverse=True)
        g.result = double(hawk.fft.c2r(back, out=np.zeros(x.shape)))
        return g

    x, seed = planes(N), planes(N)
    gradients = reverse_of(build, x, {"double#2.y": seed})
    np.testing.assert_allclose(gradients["double.x"], 4.0 * seed, atol=1e-12)


def test_the_reverse_of_a_flat_sum_is_the_cotangent_on_every_sample():
    """Oracle: ``r = sum(2 x)`` and ``w = 10 r``: every sample receives ``20 *`` the cotangent."""

    def build(x) -> hawk.Graph:
        g = hawk.Graph(record=True)
        g.result = tenfold(hawk.reduce(double(x), "sum"))
        return g

    x, seed = planes(N), planes(1)
    gradients = reverse_of(build, x, {"w": seed})
    np.testing.assert_allclose(gradients["x"], np.full(N, 20.0 * seed[0]), rtol=1e-14)


@pytest.mark.parametrize(
    ("op", "stem"),
    [
        (
            "min",
            r"eg_record_end: eg_reduce \(leaf 0\) has no gradient eagle can pull back \(a tie has no single answer\); only op='sum' reverses",
        ),
        (
            "max",
            r"eg_record_end: eg_reduce \(leaf 0\) has no gradient eagle can pull back \(a tie has no single answer\); only op='sum' reverses",
        ),
    ],
)
def test_the_reverse_of_a_minimum_or_maximum_is_refused_by_the_library(op, stem):
    def build(x) -> hawk.Graph:
        g = hawk.Graph(record=True)
        g.result = tenfold(hawk.reduce(double_v(x), op))
        return g

    with pytest.raises(HawkError, match=stem):
        hawk.Graph.builder(build)(planes((3, N)))


def test_the_reverse_of_a_sum_over_columns_is_refused_by_the_library():
    def build(x) -> hawk.Graph:
        g = hawk.Graph(record=True)
        g.result = tenfold(hawk.reduce(double_v(x), "sum"))
        return g

    with pytest.raises(
        HawkError,
        match=r"eg_reduce \(leaf 0\): a 2-D or companion reduce: eagle does not pull back yet; only a 1-D op='sum' reduce reverses",
    ):
        hawk.Graph.builder(build)(planes((3, N)))


# ---- device rows (run on a machine with a card) ----


@pytest.mark.gpu
@pytest.mark.parametrize("op", ["sum", "min", "max"])
def test_a_reduce_over_device_memory_equals_numpy(op):
    cupy = pytest.importorskip("cupy")
    p = planes((3, 200))
    got = hawk.reduce(cupy.asarray(p), op)
    want = getattr(np, op)(p, axis=1)
    np.testing.assert_allclose(cupy.asnumpy(got), want, rtol=1e-14)


@pytest.mark.gpu
def test_a_matmul_and_a_transform_over_device_memory_equal_numpy():
    cupy = pytest.importorskip("cupy")
    a, b = planes((4, 5)), planes((5, 6))
    np.testing.assert_allclose(
        cupy.asnumpy(hawk.matmul(cupy.asarray(a), cupy.asarray(b))), a @ b, rtol=1e-13
    )
    y = planes((4, 6))
    got = as_complex(cupy.asnumpy(hawk.fft.r2c(cupy.asarray(y))))
    np.testing.assert_allclose(got, np.fft.rfftn(y), atol=1e-12)


@pytest.mark.gpu
def test_a_copy_over_device_memory_equals_the_source():
    cupy = pytest.importorskip("cupy")
    src = planes((3, 17))
    dst = cupy.zeros((3, 17))
    hawk.copy(cupy.asarray(src), dst)
    np.testing.assert_array_equal(cupy.asnumpy(dst), src)
