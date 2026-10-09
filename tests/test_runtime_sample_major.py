# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.runtime`` and sample-major planes, plus the component-pitch bind.

A per-sample plane of width w >= 2 is component-major, ``(w, L)``. HAWK binds
by address and never copies, so a sample-major ``(L, w)`` plane is accepted
only when its transpose is C-contiguous (its bytes are then the native plane):
it binds zero-copy and the kernel's writes land in the caller's array. Any
other sample-major plane is refused with the fix named.

The pitch row is a regression row: a ``(w, L)`` plane with L > n_samples is
allowed (an explicit ``n_samples=`` may launch a sub-range), but the bind
used to set the component pitch to n_samples instead of L, so every component
after the first was read and written at the wrong offset, with no error.

The single-sample rows: a plane whose shape is its per-sample head (``()``,
``(w,)``, a matrix's ``(r, c)``) is ONE sample, bound at its own address as a
batch of one and written in place; it equals the batch's column bit for bit.
A shape that is also a batch stays a batch, a mix of one sample and a batch is
refused naming both, and a Python number bound to a plane is refused."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import sidecar_of

from hawk import runtime as rt
from hawk.ir import HawkError

N = 8


def _vec3(built):
    bundle = built["vec3"]
    return rt.load(bundle.directory, "vec3_scale", sidecar_of(bundle, "vec3_scale"))


def _x(n=N):
    return np.arange(3 * n, dtype=float).reshape(3, n) + 1.0


def test_native_plane_runs(built):
    x = _x()
    y = np.zeros((3, N))
    rt.run(_vec3(built), x=x, a=2.0, y=y)
    np.testing.assert_allclose(y, 2.0 * x)


def test_a_transposed_view_binds_zero_copy_both_ways(built):
    """``x.T`` of a contiguous (3, N) is an (N, 3) whose transpose is
    C-contiguous: it binds with no copy, and the output view is written in
    place, so the caller's (3, N) storage holds the answer."""
    x = _x()
    y_native = np.zeros((3, N))
    rt.run(_vec3(built), x=x.T, a=2.0, y=y_native.T)
    np.testing.assert_allclose(y_native, 2.0 * x)


def test_n_is_inferred_from_the_leading_axis_of_a_transposed_view(built):
    """``_n_from`` must vote L, not the width 3, for an (L, 3) view."""
    x = _x(5)
    y = np.zeros((3, 5))
    rt.run(_vec3(built), x=x.T, a=1.0, y=y.T)
    np.testing.assert_allclose(y, x)


def test_a_contiguous_sample_major_plane_is_refused_with_the_fix(built):
    """A C-contiguous (N, 3) would need a copy, which HAWK never makes."""
    x = np.ascontiguousarray(_x().T)
    y = np.zeros((3, N))
    with pytest.raises(HawkError, match=r"(?s)'x'.*sample-major.*\(8, 3\).*"
                                        r"\(3, L\).*ascontiguousarray\(x\.T\)"):
        rt.run(_vec3(built), x=x, a=2.0, y=y, n_samples=N)


def test_a_contiguous_sample_major_output_is_refused(built):
    x = _x()
    y = np.zeros((N, 3))
    with pytest.raises(HawkError, match=r"'y'.*sample-major"):
        rt.run(_vec3(built), x=x, a=2.0, y=y, n_samples=N)


def test_a_square_plane_needs_its_axis_said(built):
    """(3, 3) reads both ways at N == 3: it is refused (never guessed) and
    resolved by ``layout="samples_last"`` (see test_runtime_layout_ambiguity)."""
    x = _x(3)
    y = np.zeros((3, 3))
    with pytest.raises(HawkError, match="samples_first"):
        rt.run(_vec3(built), x=x, a=3.0, y=y)
    rt.run(_vec3(built), x=x, a=3.0, y=y, layout="samples_last")
    np.testing.assert_allclose(y, 3.0 * x)


def test_plane_layout_classification():
    c = np.zeros((3, N))
    assert rt.plane_layout(memoryview(c), 3) == "native"
    assert rt.plane_layout(memoryview(c.T), 3) == "view"
    assert rt.plane_layout(memoryview(np.zeros((N, 3))), 3) == "copy"
    assert rt.plane_layout(memoryview(np.zeros((3, 3))), 3) == "ambiguous"
    assert rt.plane_layout(memoryview(np.zeros(N)), 1) == "native"
    assert rt.plane_layout(memoryview(np.zeros((N, 3))), 1) == "native"


def test_plane_layout_single_sample_corpus():
    """The head is one sample; a shape that is also a batch is a batch."""
    single = rt.plane_layout
    assert single(memoryview(np.zeros(())), 1) == "single"
    assert single(memoryview(np.zeros(1)), 1) == "native"        # batch of one
    assert single(memoryview(np.zeros(3)), 3) == "single"
    assert single(memoryview(np.zeros((3, 1))), 3) == "native"   # batch of one
    assert single(memoryview(np.zeros((1, 3))), 3) == "view"     # sample-major one
    assert single(memoryview(np.zeros((3, 3))), 3) == "ambiguous"  # both readings
    assert single(memoryview(np.zeros((2, 3))), 6) == "single"   # matrix (r, c)
    assert single(memoryview(np.zeros(6)), 6) == "single"        # matrix flat
    assert single(memoryview(np.zeros((3, 1))), 3) == "native"   # Matrix[3, 1]


def _scalar(built):
    bundle = built["axpb"]
    return rt.load(bundle.directory, "axpb", sidecar_of(bundle, "axpb"))


def test_one_sample_equals_its_batch_column_bit_for_bit(built):
    """Row i of a pairwise-distinct batch, run alone, equals column i exactly
    and differs from every other column (a mutant returning any row fails)."""
    x = _x(7)
    y = np.zeros((3, 7))
    k = _vec3(built)
    rt.run(k, x=x, a=1.5, y=y)
    assert len({tuple(c) for c in y.T}) == 7
    for i in range(7):
        yi = np.zeros(3)
        rt.run(k, x=np.ascontiguousarray(x[:, i]), a=1.5, y=yi)
        assert np.array_equal(yi, y[:, i])
        assert all(not np.array_equal(yi, y[:, j]) for j in range(7) if j != i)


def test_one_scalar_sample_is_a_zero_d_array_written_in_place(built):
    k = _scalar(built)
    x = np.arange(5, dtype=float) * 0.5 + 1.0
    y = np.zeros(5)
    rt.run(k, x=x, a=2.0, b=-1.0, y=y)
    for i in range(5):
        yi = np.zeros(())
        rt.run(k, x=np.array(x[i]), a=2.0, b=-1.0, y=yi)
        assert yi == y[i]


def test_one_matrix_sample_binds_as_r_by_c(built):
    bundle = built["mat"]
    k = rt.load(bundle.directory, "mat_apply", sidecar_of(bundle, "mat_apply"))
    m = np.arange(9, dtype=float).reshape(3, 3) + 1.0
    v = np.array([1.0, -2.0, 0.5])
    y = np.zeros(3)
    rt.run(k, m=m, v=v, y=y)
    assert np.array_equal(y, m @ v)


def test_a_mix_of_one_sample_and_a_batch_is_refused_naming_both(built):
    with pytest.raises(HawkError, match=(
            r"'x' is one sample \(shape \(3,\)\) but 'y' is a batch of 8 "
            r"\(shape \(3, 8\)\); a call is one sample or one batch, never both")):
        rt.run(_vec3(built), x=np.ones(3), a=1.0, y=np.zeros((3, N)))


def test_a_python_number_bound_to_a_plane_is_refused(built):
    with pytest.raises(HawkError, match=(
            r"argument 'x' was given a Python float, which cannot be bound by "
            r"address or receive a write; pass a writable 0-d array \(np.array\(x\)\)")):
        rt.run(_scalar(built), x=2.0, a=1.0, b=0.0, y=np.zeros(()))


def test_a_longer_plane_binds_its_own_component_pitch(built):
    """Regression: (3, L) with L = 2N and n_samples = N. Each component must be
    read at offset c * L. With the old pitch of n_samples, component 1 was
    read from x[0, N:], and y's component 1 landed in y[0, N:]."""
    L = 2 * N
    x = np.arange(3 * L, dtype=float).reshape(3, L) + 1.0
    y = np.zeros((3, L))
    rt.run(_vec3(built), x=x, a=2.0, y=y, n_samples=N)
    np.testing.assert_allclose(y[:, :N], 2.0 * x[:, :N])
    np.testing.assert_array_equal(y[:, N:], 0.0)
