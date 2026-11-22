# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A kernel with an assigned output reads ``p[i, k]``, built and run on the host through the user path.

The table has a runtime first axis and a static second axis. The built kernel, and its forward and reverse
derivatives, are compared with NumPy indexing and the NumPy formulas of the derivative of ``p[i, k] * x``, which
are independent of the code under test.
"""

from __future__ import annotations

import numpy as np
import pytest

import hawk
import hawk.diff
from hawk import HawkError, Index, Kernel, Mutable, Scalar, Table, kernel, runtime
from hawk.artifact import build_bundle

ROWS, COLS, N = 5, 4, 64


@kernel
def scaled_read(
    p: Table["row", "col":4],  # noqa: F821 axis labels
    i: Index,
    k: Index,
    x: Scalar,
    y: Mutable[Scalar],
):
    y = p[i, k] * x


def _bundle(kernels, tmp, cache_dir):
    return build_bundle(
        kernels,
        tmp,
        mode="float64",
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )


def _data():
    rng = np.random.default_rng(20261122)
    return {
        "p": rng.uniform(-2, 2, (ROWS, COLS)),
        "i": rng.integers(0, ROWS, N).astype(np.int32),
        "k": rng.integers(0, COLS, N).astype(np.int32),
        "x": rng.uniform(0.5, 2, N),
    }


def _run(bundle, name, **planes):
    runtime.run(runtime.load(bundle.directory, name), **planes)


def test_a_two_axis_read_with_an_assigned_output_equals_numpy(tmp_path, cache_dir):
    bundle = _bundle([scaled_read], tmp_path, cache_dir)
    d = _data()
    y = np.zeros(N)
    _run(bundle, "scaled_read", **d, y=y)
    np.testing.assert_allclose(y, d["p"][d["i"], d["k"]] * d["x"], rtol=1e-13)


def test_the_forward_derivative_equals_the_numpy_formula(tmp_path, cache_dir):
    primal = _bundle([scaled_read], tmp_path / "f", cache_dir)
    tangent = Kernel(
        "scaled_read_jvp", hawk.diff.jvp(scaled_read, primal_unit=primal.digest)
    )
    bundle = _bundle([tangent], tmp_path / "t", cache_dir)
    d = _data()
    rng = np.random.default_rng(1)
    dot_p, dot_x = rng.uniform(-1, 1, (ROWS, COLS)), rng.uniform(-1, 1, N)
    dot_y = np.zeros(N)
    _run(bundle, "scaled_read_jvp", **d, dot_p=dot_p, dot_x=dot_x, dot_y=dot_y)
    want = dot_p[d["i"], d["k"]] * d["x"] + d["p"][d["i"], d["k"]] * dot_x
    np.testing.assert_allclose(dot_y, want, rtol=1e-12, atol=1e-14)


def test_the_reverse_derivative_equals_the_numpy_formula(tmp_path, cache_dir):
    primal = _bundle([scaled_read], tmp_path / "f", cache_dir)
    adjoint = Kernel(
        "scaled_read_vjp", hawk.diff.vjp(scaled_read, primal_unit=primal.digest)
    )
    bundle = _bundle([adjoint], tmp_path / "t", cache_dir)
    d = _data()
    bar_y = np.random.default_rng(2).uniform(-1, 1, N)
    bar_p, bar_x = np.zeros((ROWS, COLS)), np.zeros(N)
    _run(bundle, "scaled_read_vjp", **d, bar_y=bar_y, bar_p=bar_p, bar_x=bar_x)
    want_p = np.zeros((ROWS, COLS))
    np.add.at(want_p, (d["i"], d["k"]), bar_y * d["x"])
    np.testing.assert_allclose(bar_p, want_p, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        bar_x, bar_y * d["p"][d["i"], d["k"]], rtol=1e-12, atol=1e-14
    )


def test_a_compact_table_with_one_runtime_axis_is_read_by_its_element_count(
    tmp_path, cache_dir
):
    bundle = _bundle([scaled_read], tmp_path, cache_dir)
    d = _data()
    y = np.zeros(N)
    _run(bundle, "scaled_read", **{**d, "p": d["p"].reshape(-1)}, y=y)
    np.testing.assert_allclose(y, d["p"][d["i"], d["k"]] * d["x"], rtol=1e-13)
    with pytest.raises(HawkError, match=r"'p'.*7 elements"):
        _run(bundle, "scaled_read", **{**d, "p": np.zeros(7)}, y=y)


@kernel
def read_flat(
    p: Table["row", "col":4],  # noqa: F821 axis labels
    k: Index,
    y: Mutable[Scalar],
):
    y = p[k]


def test_a_one_index_read_of_a_table_with_axes_is_the_position_in_storage_order(
    tmp_path, cache_dir
):
    bundle = _bundle([read_flat], tmp_path, cache_dir)
    p = _data()["p"]
    flat = np.array([0, 1, 4, 7, 11, 19, ROWS * COLS - 1], dtype=np.int32)
    y = np.zeros(len(flat))
    _run(bundle, "read_flat", p=p, k=flat, y=y)
    np.testing.assert_array_equal(y, p.reshape(-1)[flat])
    outside = np.array([-3, ROWS * COLS, 10 * ROWS * COLS], dtype=np.int32)
    y = np.zeros(len(outside))
    _run(bundle, "read_flat", p=p, k=outside, y=y)
    np.testing.assert_array_equal(
        y, p.reshape(-1)[np.clip(outside, 0, ROWS * COLS - 1)]
    )
