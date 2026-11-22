# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A value of rank three or more: elementwise operations, element reads and element stores.

``Tensor[A, B, C]`` declares a payload of any rank. The kernels are evaluated in Python and, where the host
library builds them, built and run; both are compared with NumPy, which is independent of the code under test.
Operations defined only for rank one or two keep refusing a higher rank by name.
"""

from __future__ import annotations

import numpy as np
import pytest
from _eval import evaluate

import hawk
import hawk.math as hm
from hawk import HawkError, Mutable, Param, Scalar, Tensor, runtime
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp

SHAPE = (2, 3, 4)
RNG = np.random.default_rng(20261122)
A = RNG.uniform(0.5, 2.0, SHAPE)


@hawk.kernel
def _elementwise(a: Param[Tensor[2, 3, 4]], x: Scalar, y: Mutable[Scalar]):
    t = hm.sqrt(a * x + 1.0)
    t[1, 2, 3] = x
    y = t[0, 1, 2] + t[1, 2, 3] + t[1, 0, 0]


def _oracle(a, x):
    t = np.sqrt(a * x + 1.0)
    t[1, 2, 3] = x
    return t[0, 1, 2] + t[1, 2, 3] + t[1, 0, 0]


def test_the_declaration_has_the_shape_it_names_and_reads_back_as_written():
    assert Tensor[2, 3, 4].shape == SHAPE and Tensor[5, 6, 7, 2].shape == (5, 6, 7, 2)
    assert Tensor[3] == hawk.Vector[3] and Tensor[3, 4] == hawk.Matrix[3, 4]


@pytest.mark.parametrize("bad", [(2, 0, 3), (2, -1), (2.5, 3), ()], ids=str)
def test_the_declaration_refuses_an_extent_that_is_not_a_positive_integer(bad):
    with pytest.raises(HawkError, match="one positive static extent per axis"):
        Tensor[bad]


def test_elementwise_operations_and_element_access_equal_numpy():
    for x in (0.3, 1.7, 4.0):
        got = evaluate(_elementwise.sinks, {"a": A, "x": x})["y"]
        np.testing.assert_allclose(got, _oracle(A, x), rtol=1e-14)


def test_a_rank_three_value_is_built_and_run_on_the_host(tmp_path, cache_dir):
    bundle = build_bundle(
        [_elementwise],
        tmp_path,
        mode="float64",
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )
    x, y = RNG.uniform(0.5, 2, 16), np.zeros(16)
    runtime.run(runtime.load(bundle.directory, "_elementwise"), a=A, x=x, y=y)
    np.testing.assert_allclose(y, [_oracle(A, v) for v in x], rtol=1e-13)


def test_the_reverse_and_forward_derivatives_through_elements_equal_the_numpy_formulas():
    x = 1.7
    t = A * x + 1.0
    # y = sqrt(t[0,1,2]) + x + sqrt(t[1,0,0]); the entry (1, 2, 3) was overwritten by x
    d_a = np.zeros(SHAPE)
    for at in ((0, 1, 2), (1, 0, 0)):
        d_a[at] = 0.5 / np.sqrt(t[at]) * x
    d_x = 1.0 + sum(0.5 / np.sqrt(t[at]) * A[at] for at in ((0, 1, 2), (1, 0, 0)))
    seed = 0.8
    reverse = evaluate(
        vjp(_elementwise.sinks),
        {"a": A, "x": x, "bar_y": seed},
        shapes={"bar_a": SHAPE},
    )
    np.testing.assert_allclose(reverse["bar_a"], seed * d_a, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(reverse["bar_x"], seed * d_x, rtol=1e-12)
    dot_a, dot_x = RNG.normal(size=SHAPE), 0.6
    forward = evaluate(
        jvp(_elementwise.sinks), {"a": A, "x": x, "dot_a": dot_a, "dot_x": dot_x}
    )
    np.testing.assert_allclose(
        forward["dot_y"], np.sum(d_a * dot_a) + d_x * dot_x, rtol=1e-12
    )


@hawk.kernel
def _shared_read(
    a: Param[Tensor[2, 3, 4]],
    i: hawk.Index,
    j: hawk.Index,
    k: hawk.Index,
    y: Mutable[Scalar],
):
    y = a[i, j, k]


def test_a_shared_tensor_is_read_at_traced_indices_from_memory_and_equals_numpy():
    for at in ((0, 0, 0), (1, 2, 3), (1, 0, 2)):
        got = evaluate(_shared_read.sinks, {"a": A, **dict(zip("ijk", at))})["y"]
        assert got == A[at]


def test_a_shared_tensor_read_on_the_host_equals_numpy(tmp_path, cache_dir):
    bundle = build_bundle(
        [_shared_read],
        tmp_path,
        mode="float64",
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )
    i, j, k = (RNG.integers(0, e, 16).astype(np.int32) for e in SHAPE)
    y = np.zeros(16)
    runtime.run(runtime.load(bundle.directory, "_shared_read"), a=A, i=i, j=j, k=k, y=y)
    np.testing.assert_array_equal(y, A[i, j, k])


def _trace_refused(body, match):
    """Trace ``body`` as a kernel and expect the refusal ``match``."""
    with pytest.raises(HawkError, match=match):
        hawk.kernel(body).sinks


def test_a_wrong_index_count_on_a_computed_value_is_refused_naming_the_value():
    def body(a: Param[Tensor[2, 3, 4]], i: hawk.Index, j: hawk.Index, y: Mutable[Scalar]):
        t = a * 2.0
        y = t[i, j]

    _trace_refused(body, r"component_at: the value typed .*\(2, 3, 4\).* has 3 axes")


def test_norm_of_a_rank_zero_value_is_refused_by_name():
    def body(x: Scalar, y: Mutable[Scalar]):
        y = hm.norm(x)

    _trace_refused(body, "norm: expected an operand of rank one or more")


def test_an_index_outside_the_extent_is_refused():
    def body(a: Param[Tensor[2, 3, 4]], y: Mutable[Scalar]):
        t = a * 2.0
        y = t[2, 0, 0]

    _trace_refused(body, "static extent")


def test_an_entry_store_outside_the_extents_is_refused_naming_the_shape():
    def body(a: Param[Tensor[2, 3, 4]], y: Mutable[Scalar]):
        t = a * 2.0
        t[0, 3, 0] = 1.0
        y = t[0, 0, 0]

    _trace_refused(body, r"shape \(2, 3, 4\): index 3 is outside the extent 3 of its axis")


@hawk.kernel
def _plane_io(a: Tensor[2, 3, 4], y: Mutable[Tensor[2, 3, 4]]):
    t = hm.sqrt(a * 2.0 + 1.0)
    t[1, 2, 3] = a[0, 0, 0]
    y = t * a


def _plane_io_oracle(a):
    t = np.sqrt(a * 2.0 + 1.0)
    t[1, 2, 3] = a[0, 0, 0]
    return t * a


def _plane_io_partials(a):
    """The entrywise derivative of the output by the input, and the two cross terms of the overwritten entry."""
    t = np.sqrt(a * 2.0 + 1.0)
    own = a / t + t
    own[1, 2, 3] = a[0, 0, 0]
    return own


def _a_plane_run(tmp_path, cache_dir, a, y):
    bundle = build_bundle(
        [_plane_io],
        tmp_path,
        mode="float64",
        targets=("host",),
        cache_dir=cache_dir,
        host_profile="native",
    )
    runtime.run(runtime.load(bundle.directory, "_plane_io"), a=a, y=y)


def test_a_per_sample_input_and_a_mutable_output_of_rank_three_equal_numpy_on_the_host(
    tmp_path, cache_dir
):
    samples = 8
    a = RNG.uniform(0.5, 2.0, SHAPE + (samples,))
    y = np.zeros(SHAPE + (samples,))
    # a plane of per-sample values is bound as its components flat, then the samples: (24, N)
    _a_plane_run(tmp_path, cache_dir, a.reshape(24, samples), y.reshape(24, samples))
    for k in range(samples):
        np.testing.assert_allclose(y[..., k], _plane_io_oracle(a[..., k]), rtol=1e-13)


def test_the_derivatives_of_a_per_sample_rank_three_plane_equal_the_numpy_formulas():
    own = _plane_io_partials(A)
    seed = RNG.normal(size=SHAPE)
    reverse = evaluate(
        vjp(_plane_io.sinks), {"a": A, "bar_y": seed}, shapes={"bar_a": SHAPE}
    )
    want = seed * own
    want[0, 0, 0] += seed[1, 2, 3] * A[1, 2, 3]
    np.testing.assert_allclose(reverse["bar_a"], want, rtol=1e-12, atol=1e-15)
    dot = RNG.normal(size=SHAPE)
    forward = evaluate(jvp(_plane_io.sinks), {"a": A, "dot_a": dot})
    expected = own * dot
    expected[1, 2, 3] = A[0, 0, 0] * dot[1, 2, 3] + A[1, 2, 3] * dot[0, 0, 0]
    expected[0, 0, 0] += 0.0
    np.testing.assert_allclose(forward["dot_y"], expected, rtol=1e-12, atol=1e-15)


def test_an_array_of_the_wrong_value_shape_is_refused_naming_the_plane(
    tmp_path, cache_dir
):
    wrong = np.ones((30, 4))
    with pytest.raises(
        HawkError, match=r"plane 'a' is declared 24 component\(s\) wide"
    ):
        _a_plane_run(tmp_path, cache_dir, wrong, np.zeros((24, 4)))


# ---- element access of a computed value at traced indices: one index per axis -------------------------------

MATRIX = (3, 4)
B = RNG.uniform(0.5, 2.0, MATRIX)


@hawk.kernel
def _matrix_read(a: Param[Tensor[3, 4]], i: hawk.Index, j: hawk.Index, y: Mutable[Scalar]):
    t = a * 2.0
    y = t[i, j]


@hawk.kernel
def _tensor_read(
    a: Param[Tensor[2, 3, 4]], i: hawk.Index, j: hawk.Index, k: hawk.Index, y: Mutable[Scalar]
):
    t = hm.sqrt(a * 2.0 + 1.0)
    y = t[i, j, k]


@hawk.kernel
def _mixed_read(a: Param[Tensor[2, 3, 4]], j: hawk.Index, k: hawk.Index, y: Mutable[Scalar]):
    t = a * 3.0
    y = t[1, j, k] + t[0, 2, k]


@hawk.kernel
def _tensor_store(
    a: Param[Tensor[2, 3, 4]],
    x: Scalar,
    i: hawk.Index,
    j: hawk.Index,
    k: hawk.Index,
    y: Mutable[Tensor[2, 3, 4]],
):
    t = a * 2.0
    t[i, j, k] = x
    y = t


@hawk.kernel
def _tensor_accumulate(
    a: Param[Tensor[2, 3, 4]],
    x: Scalar,
    i: hawk.Index,
    j: hawk.Index,
    k: hawk.Index,
    y: Mutable[Tensor[2, 3, 4]],
):
    t = a * 2.0
    t[i, j, k] += x * a[0, 0, 0]
    y = t


@hawk.kernel
def _mixed_store(
    a: Param[Tensor[2, 3, 4]],
    x: Scalar,
    j: hawk.Index,
    k: hawk.Index,
    y: Mutable[Tensor[2, 3, 4]],
):
    t = a * 2.0
    t[1, j, k] = x
    y = t


def _host_rows(tmp_path, cache_dir, kernel, name, **arrays):
    """Build ``kernel`` for the host and run it over the per-lane ``arrays``; the plane outputs are bound flat."""
    bundle = build_bundle(
        [kernel], tmp_path, mode="float64", targets=("host",), cache_dir=cache_dir, host_profile="native"
    )
    runtime.run(runtime.load(bundle.directory, name), **arrays)


LANES = 12
IJK = tuple(RNG.integers(0, e, LANES).astype(np.int32) for e in SHAPE)


def test_a_traced_index_read_of_a_computed_matrix_equals_numpy_on_the_host(tmp_path, cache_dir):
    i, j = (RNG.integers(0, e, LANES).astype(np.int32) for e in MATRIX)
    y = np.zeros(LANES)
    _host_rows(tmp_path, cache_dir, _matrix_read, "_matrix_read", a=B, i=i, j=j, y=y)
    np.testing.assert_array_equal(y, (B * 2.0)[i, j])
    got = evaluate(_matrix_read.sinks, {"a": B, "i": 2, "j": 1})["y"]
    assert got == (B * 2.0)[2, 1]


def test_a_traced_index_read_of_a_computed_rank_three_tensor_equals_numpy_on_the_host(
    tmp_path, cache_dir
):
    i, j, k = IJK
    y = np.zeros(LANES)
    _host_rows(tmp_path, cache_dir, _tensor_read, "_tensor_read", a=A, i=i, j=j, k=k, y=y)
    np.testing.assert_allclose(y, np.sqrt(A * 2.0 + 1.0)[i, j, k], rtol=1e-14)


def test_static_and_traced_indices_mix_in_one_read(tmp_path, cache_dir):
    _, j, k = IJK
    y = np.zeros(LANES)
    _host_rows(tmp_path, cache_dir, _mixed_read, "_mixed_read", a=A, j=j, k=k, y=y)
    np.testing.assert_allclose(y, (A * 3.0)[1, j, k] + (A * 3.0)[0, 2, k], rtol=1e-14)


def _stored_tensors(tmp_path, cache_dir, kernel, name, x, indices, mixed=False):
    """The tensors a store kernel returns per lane, bound as (24, lanes)."""
    out = np.zeros((24, LANES))
    given = dict(zip("ijk", indices)) if not mixed else dict(zip("jk", indices))
    _host_rows(tmp_path, cache_dir, kernel, name, a=A, x=x, y=out, **given)
    return out.reshape(SHAPE + (LANES,))


def test_a_traced_index_store_into_a_computed_rank_three_value_equals_numpy_on_the_host(
    tmp_path, cache_dir
):
    i, j, k = IJK
    x = RNG.uniform(0.5, 2, LANES)
    got = _stored_tensors(tmp_path, cache_dir, _tensor_store, "_tensor_store", x, IJK)
    for lane in range(LANES):
        want = A * 2.0
        want[i[lane], j[lane], k[lane]] = x[lane]
        np.testing.assert_array_equal(got[..., lane], want)


def test_a_traced_index_accumulate_into_a_computed_rank_three_value_equals_numpy_on_the_host(
    tmp_path, cache_dir
):
    i, j, k = IJK
    x = RNG.uniform(0.5, 2, LANES)
    got = _stored_tensors(tmp_path, cache_dir, _tensor_accumulate, "_tensor_accumulate", x, IJK)
    for lane in range(LANES):
        want = A * 2.0
        want[i[lane], j[lane], k[lane]] += x[lane] * A[0, 0, 0]
        np.testing.assert_allclose(got[..., lane], want, rtol=1e-14)


def test_a_store_with_static_and_traced_indices_equals_numpy_on_the_host(tmp_path, cache_dir):
    _, j, k = IJK
    x = RNG.uniform(0.5, 2, LANES)
    got = _stored_tensors(tmp_path, cache_dir, _mixed_store, "_mixed_store", x, (j, k), mixed=True)
    for lane in range(LANES):
        want = A * 2.0
        want[1, j[lane], k[lane]] = x[lane]
        np.testing.assert_array_equal(got[..., lane], want)


AT = {"i": 1, "j": 2, "k": 3}


def test_the_derivatives_of_a_traced_index_read_equal_the_numpy_formulas():
    t = np.sqrt(A * 2.0 + 1.0)
    at = (AT["i"], AT["j"], AT["k"])
    want = np.zeros(SHAPE)
    want[at] = 0.5 / t[at] * 2.0
    reverse = evaluate(vjp(_tensor_read.sinks), {"a": A, **AT, "bar_y": 0.7}, shapes={"bar_a": SHAPE})
    np.testing.assert_allclose(reverse["bar_a"], 0.7 * want, rtol=1e-12, atol=1e-15)
    dot = RNG.normal(size=SHAPE)
    forward = evaluate(jvp(_tensor_read.sinks), {"a": A, **AT, "dot_a": dot})
    np.testing.assert_allclose(forward["dot_y"], np.sum(want * dot), rtol=1e-12)


def test_the_derivatives_of_a_traced_index_store_equal_the_numpy_formulas():
    seed, dot, x, dot_x = RNG.normal(size=SHAPE), RNG.normal(size=SHAPE), 1.3, 0.4
    at = (AT["i"], AT["j"], AT["k"])
    env = {"a": A, "x": x, **AT}
    reverse = evaluate(vjp(_tensor_store.sinks), {**env, "bar_y": seed}, shapes={"bar_a": SHAPE})
    want_a = 2.0 * seed
    want_a[at] = 0.0
    np.testing.assert_allclose(reverse["bar_a"], want_a, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(reverse["bar_x"], seed[at], rtol=1e-12)
    forward = evaluate(jvp(_tensor_store.sinks), {**env, "dot_a": dot, "dot_x": dot_x})
    want_y = 2.0 * dot
    want_y[at] = dot_x
    np.testing.assert_allclose(forward["dot_y"], want_y, rtol=1e-12, atol=1e-15)


def test_the_derivatives_of_a_traced_index_accumulate_equal_the_numpy_formulas():
    seed, dot, x, dot_x = RNG.normal(size=SHAPE), RNG.normal(size=SHAPE), 1.3, 0.4
    at = (AT["i"], AT["j"], AT["k"])
    env = {"a": A, "x": x, **AT}
    reverse = evaluate(vjp(_tensor_accumulate.sinks), {**env, "bar_y": seed}, shapes={"bar_a": SHAPE})
    want_a = 2.0 * seed
    want_a[0, 0, 0] += seed[at] * x
    np.testing.assert_allclose(reverse["bar_a"], want_a, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(reverse["bar_x"], seed[at] * A[0, 0, 0], rtol=1e-12)
    forward = evaluate(jvp(_tensor_accumulate.sinks), {**env, "dot_a": dot, "dot_x": dot_x})
    want_y = 2.0 * dot
    want_y[at] += dot_x * A[0, 0, 0] + x * dot[0, 0, 0]
    np.testing.assert_allclose(forward["dot_y"], want_y, rtol=1e-12, atol=1e-15)


def test_the_derivatives_of_a_read_with_static_and_traced_indices_equal_the_numpy_formulas():
    env = {"a": A, "j": 1, "k": 2}
    want = np.zeros(SHAPE)
    want[1, 1, 2] += 3.0
    want[0, 2, 2] += 3.0
    reverse = evaluate(vjp(_mixed_read.sinks), {**env, "bar_y": 1.0}, shapes={"bar_a": SHAPE})
    np.testing.assert_allclose(reverse["bar_a"], want, rtol=1e-12)
    dot = RNG.normal(size=SHAPE)
    forward = evaluate(jvp(_mixed_read.sinks), {**env, "dot_a": dot})
    np.testing.assert_allclose(forward["dot_y"], np.sum(want * dot), rtol=1e-12)


# ---- a Mutable plane holding a value of rank two or more: its sample value is indexed like a computed value ----


@hawk.kernel
def _plane_element(
    x: Scalar,
    i: hawk.Index,
    j: hawk.Index,
    k: hawk.Index,
    y: Mutable[Tensor[2, 3, 4]],
    z: Mutable[Scalar],
):
    z = y[i, j, k] * x
    t = y
    t[i, j, k] += x
    t[1, j, k] = z
    y = t


def test_a_mutable_plane_of_rank_three_is_read_stored_and_accumulated_at_traced_indices(
    tmp_path, cache_dir
):
    i, j, k = IJK
    x = RNG.uniform(0.5, 2, LANES)
    flat = RNG.uniform(0.5, 2, SHAPE + (LANES,))
    y = flat.reshape(24, LANES).copy()
    z = np.zeros(LANES)
    _host_rows(tmp_path, cache_dir, _plane_element, "_plane_element", x=x, i=i, j=j, k=k, y=y, z=z)
    got = y.reshape(SHAPE + (LANES,))
    for lane in range(LANES):
        want = flat[..., lane].copy()
        read = want[i[lane], j[lane], k[lane]] * x[lane]
        want[i[lane], j[lane], k[lane]] += x[lane]
        want[1, j[lane], k[lane]] = read
        np.testing.assert_allclose(z[lane], read, rtol=1e-14)
        np.testing.assert_allclose(got[..., lane], want, rtol=1e-14)


# ---- sum and norm of a value of any rank ---------------------------------------------------------------------


@hawk.kernel
def _fold_all(a: Param[Tensor[2, 3, 4]], x: Scalar, s: Mutable[Scalar], n: Mutable[Scalar]):
    t = a * x + 1.0
    s = hm.sum(t)
    n = hm.norm(t)


@hawk.kernel
def _fold_matrix(a: Param[Tensor[3, 4]], x: Scalar, s: Mutable[Scalar], n: Mutable[Scalar]):
    t = a * x + 1.0
    s = hm.sum(t)
    n = hm.norm(t)


def test_sum_and_norm_of_a_computed_value_of_rank_two_and_three_equal_numpy_on_the_host(
    tmp_path, cache_dir
):
    x = RNG.uniform(0.5, 2, LANES)
    for kernel, name, a in ((_fold_all, "_fold_all", A), (_fold_matrix, "_fold_matrix", B)):
        s, n = np.zeros(LANES), np.zeros(LANES)
        _host_rows(tmp_path, cache_dir, kernel, name, a=a, x=x, s=s, n=n)
        for lane in range(LANES):
            t = a * x[lane] + 1.0
            np.testing.assert_allclose(s[lane], np.sum(t), rtol=1e-13)
            np.testing.assert_allclose(n[lane], np.linalg.norm(t.ravel()), rtol=1e-13)


@pytest.mark.parametrize("kernel,a", [(_fold_all, A), (_fold_matrix, B)], ids=["rank3", "rank2"])
def test_the_derivatives_of_sum_and_norm_of_a_value_of_rank_two_and_three_equal_numpy(kernel, a):
    x = 1.3
    t = a * x + 1.0
    norm = np.linalg.norm(t.ravel())
    seed_s, seed_n = 0.7, -0.4
    reverse = evaluate(
        vjp(kernel.sinks),
        {"a": a, "x": x, "bar_s": seed_s, "bar_n": seed_n},
        shapes={"bar_a": a.shape},
    )
    d_t = seed_s * np.ones_like(t) + seed_n * t / norm
    np.testing.assert_allclose(reverse["bar_a"], d_t * x, rtol=1e-12)
    np.testing.assert_allclose(reverse["bar_x"], np.sum(d_t * a), rtol=1e-12)
    dot_a, dot_x = RNG.normal(size=a.shape), 0.6
    forward = evaluate(jvp(kernel.sinks), {"a": a, "x": x, "dot_a": dot_a, "dot_x": dot_x})
    dot_t = dot_a * x + a * dot_x
    np.testing.assert_allclose(forward["dot_s"], np.sum(dot_t), rtol=1e-12)
    np.testing.assert_allclose(forward["dot_n"], np.sum(t * dot_t) / norm, rtol=1e-12)
