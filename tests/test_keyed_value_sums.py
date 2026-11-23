# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A keyed ``+=`` of a vector or matrix value is ONE update that adds each component at its own element.

The plane a caller binds holds the value's components first, then the declared axes. The oracle of every value test is NumPy's
``np.add.at`` applied to each component plane (``math.fsum`` per element for the compensated sums on an ill-conditioned input,
and for the derivatives the NumPy formulas of the transpose of a scatter-add).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import _launch

import hawk
import hawk.diff
from hawk import Index, Kernel, Matrix, Mutable, Vector, kernel, runtime
from hawk.artifact import build_bundle, emit_source
from hawk.trace import _ir

N, CELLS = 3000, 9
CONTRACTS = ("plain", "auto", "compensated", "ordered")
MATRIX_CONTRACTS = CONTRACTS[:3]


def _vector_kernel(contract: str):
    @kernel
    def vsum(key: Index, v: Vector[3], acc: Mutable[Vector[3], "bin":9]):  # noqa: F821 axis labels
        hawk.scatter(acc, contract=contract)[key] += v

    vsum.name = f"vector_{contract}"
    return vsum


def _matrix_kernel(contract: str):
    @kernel
    def msum(key: Index, m: Matrix[2, 2], acc: Mutable[Matrix[2, 2], "bin":9]):  # noqa: F821 axis labels
        hawk.scatter(acc, contract=contract)[key] += m

    msum.name = f"matrix_{contract}"
    return msum


@kernel
def vector_grid(row: Index, col: Index, v: Vector[3], acc: Mutable[Vector[3], "row":4, "col":5]):  # noqa: F821 axis labels
    hawk.scatter(acc)[row, col] += v


KERNELS = (
    [_vector_kernel(c) for c in CONTRACTS] + [_matrix_kernel(c) for c in MATRIX_CONTRACTS] + [vector_grid]
)


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    return build_bundle(KERNELS, tmp_path_factory.mktemp("keyed_value_sums"), mode="float64", targets=("cuda", "host"),
                        cache_dir=cache_dir)


@pytest.fixture(params=[1, 4], ids=["one_thread", "team"])
def threads(request):
    """The host runs on one thread, then on a team."""
    hawk.set_host_threads(request.param)
    yield request.param
    hawk.set_host_threads(None)


def _crowded(rng, cells, n=N):
    """Keys with many lanes on few cells: half of the lanes hit cell 0."""
    return np.where(rng.random(n) < 0.5, 0, rng.integers(0, cells, n)).astype(np.int32)


def _ill_conditioned(rng, shape):
    """Terms of wildly different magnitude, so that a plain sum loses digits a compensated one keeps."""
    return rng.normal(size=shape) * np.exp(rng.normal(size=shape) * 8)


def _add_at(key, terms, cells=CELLS):
    """The oracle: ``np.add.at`` per component of ``terms`` (components first, then the samples)."""
    want = np.zeros((*terms.shape[:-1], cells))
    for at in np.ndindex(*terms.shape[:-1]):
        np.add.at(want[at], key, terms[at])
    return want


def _exact_sums(key, terms, cells=CELLS):
    """The oracle of a compensated sum: ``math.fsum`` per component and cell."""
    want = np.zeros((*terms.shape[:-1], cells))
    for at in np.ndindex(*terms.shape[:-1]):
        for cell in range(cells):
            want[(*at, cell)] = math.fsum(terms[at][key == cell])
    return want


def _run(bundle, name, key, terms, shape):
    acc = np.zeros(shape)
    _launch(bundle, name, "host", {"key": key, "v" if len(shape) == 2 else "m": terms, "acc": acc}).launch()
    return acc


def test_a_vector_sum_is_one_update_typed_as_the_planes_value():
    (sink,) = [n for n in KERNELS[0].sinks if n.field_text("name") == "acc"]
    assert _ir.is_accumulate(sink) and _ir.type_of(sink).shape == (3,) and len(sink.operands) == 2
    assert _ir.field_pairs(sink, "axes") == (("bin", 9),)


def test_a_matrix_sum_is_one_update_typed_as_the_planes_value():
    (sink,) = [n for n in KERNELS[4].sinks if n.field_text("name") == "acc"]
    assert _ir.is_accumulate(sink) and _ir.type_of(sink).shape == (2, 2) and len(sink.operands) == 2


@pytest.mark.parametrize("contract", ["plain", "auto", "compensated", "ordered"])
def test_a_vector_sum_equals_add_at_per_component(bundle, threads, contract):
    key, v = _crowded(np.random.default_rng(1), CELLS), np.random.default_rng(2).normal(size=(3, N))
    got = _run(bundle, f"vector_{contract}", key, v, (3, CELLS))
    np.testing.assert_allclose(got, _add_at(key, v), rtol=1e-12, atol=1e-12 * np.abs(got).max())


@pytest.mark.parametrize("contract", MATRIX_CONTRACTS)
def test_a_matrix_sum_equals_add_at_per_component(bundle, threads, contract):
    key, m = _crowded(np.random.default_rng(3), CELLS), np.random.default_rng(4).normal(size=(2, 2, N))
    got = _run(bundle, f"matrix_{contract}", key, m.reshape(4, N), (2, 2, CELLS))
    np.testing.assert_allclose(got, _add_at(key, m), rtol=1e-12, atol=1e-12 * np.abs(got).max())


@pytest.mark.parametrize("contract", ["auto", "compensated"])
def test_a_compensated_vector_sum_equals_fsum_per_element(bundle, threads, contract):
    key, v = _crowded(np.random.default_rng(5), CELLS), _ill_conditioned(np.random.default_rng(6), (3, N))
    got = _run(bundle, f"vector_{contract}", key, v, (3, CELLS))
    want = _exact_sums(key, v)
    scale = np.finfo(np.float64).eps * np.array([[math.fsum(np.abs(v[c][key == s])) for s in range(CELLS)] for c in range(3)])
    assert np.all(np.abs(got - want) <= scale)


def test_an_ordered_matrix_sum_is_refused_by_name():
    with pytest.raises(hawk.HawkError, match="contract='ordered' adds a scalar or a vector payload"):
        _matrix_kernel("ordered")


def test_an_ordered_vector_sum_gives_the_same_bits_on_every_run(bundle, threads):
    key, v = _crowded(np.random.default_rng(7), CELLS), _ill_conditioned(np.random.default_rng(8), (3, N))
    runs = [_run(bundle, "vector_ordered", key, v, (3, CELLS)).tobytes() for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_a_vector_sum_over_two_key_axes_equals_add_at(bundle, threads):
    rng = np.random.default_rng(9)
    row, col, v = rng.integers(0, 4, N).astype(np.int32), rng.integers(0, 5, N).astype(np.int32), rng.normal(size=(3, N))
    acc = np.zeros((3, 4, 5))
    _launch(bundle, "vector_grid", "host", {"row": row, "col": col, "v": v, "acc": acc}).launch()
    want = np.zeros((3, 4, 5))
    for c in range(3):
        np.add.at(want[c], (row, col), v[c])
    np.testing.assert_allclose(acc, want, rtol=1e-12, atol=1e-12 * np.abs(want).max())


@pytest.mark.gpu
@pytest.mark.parametrize("contract", ["plain", "compensated", "ordered"])
def test_the_device_vector_sum_equals_add_at_per_component(bundle, contract):
    import cupy as cp

    key, v = _crowded(np.random.default_rng(1), CELLS), np.random.default_rng(2).normal(size=(3, N))
    acc = cp.zeros((3, CELLS))
    _launch(bundle, f"vector_{contract}", "cuda", {"key": cp.asarray(key), "v": cp.asarray(v), "acc": acc}).launch()
    cp.cuda.Device().synchronize()
    np.testing.assert_allclose(acc.get(), _add_at(key, v), rtol=1e-12, atol=1e-12 * np.abs(v).sum())


@pytest.mark.gpu
def test_the_device_compensated_matrix_sum_equals_fsum_per_element(bundle):
    import cupy as cp

    key, m = _crowded(np.random.default_rng(5), CELLS), _ill_conditioned(np.random.default_rng(6), (2, 2, N))
    acc = cp.zeros((2, 2, CELLS))
    _launch(bundle, "matrix_compensated", "cuda", {"key": cp.asarray(key), "m": cp.asarray(m.reshape(4, N)), "acc": acc}).launch()
    cp.cuda.Device().synchronize()
    scale = np.finfo(np.float64).eps * np.abs(m).sum() * 8
    assert np.all(np.abs(acc.get() - _exact_sums(key, m)) <= scale)


@pytest.mark.parametrize("target, helper", [("host", "accum_add_at"), ("cuda", "accum_add_at")])
def test_the_text_of_a_plain_vector_sum_adds_each_component_at_its_own_element(target, helper):
    text = emit_source(KERNELS[0], targets=(target,), device_arch="sm_80" if target == "cuda" else "").sources[target].text
    assert text.count(f"hawk_abi::{helper}(") >= 3
    assert all(f"aout_acc({c}, " in text for c in range(3))


def test_the_text_of_a_compensated_vector_sum_adds_each_component_with_its_companion():
    text = emit_source(KERNELS[2], targets=("cuda",), device_arch="sm_80").sources["cuda"].text
    assert all(f"hawk_abi::accum_add_compensated_at(aout_acc({c}, " in text for c in range(3))
    assert all(f"aout_acc_c({c}, " in text for c in range(3))


def _derivative_bundle(kernels, tmp_path, cache_dir):
    return build_bundle(kernels, tmp_path, mode="float64", targets=("host",), cache_dir=cache_dir, host_profile="native")


def test_the_forward_derivative_of_a_vector_sum_adds_the_tangents_at_the_same_keys(tmp_path, cache_dir):
    primal = KERNELS[1]
    built = _derivative_bundle([primal], tmp_path / "p", cache_dir)
    tangent = Kernel("vector_jvp", hawk.diff.jvp(primal, primal_unit=built.digest))
    bundle = _derivative_bundle([tangent], tmp_path / "t", cache_dir)
    rng = np.random.default_rng(10)
    key, dot_v = _crowded(rng, CELLS), rng.normal(size=(3, N))
    dot_acc = np.zeros((3, CELLS))
    runtime.run(runtime.load(bundle.directory, "vector_jvp"), key=key, dot_v=dot_v, dot_acc=dot_acc)
    np.testing.assert_allclose(dot_acc, _add_at(key, dot_v), rtol=1e-12, atol=1e-12 * np.abs(dot_v).sum())


def test_the_reverse_derivative_of_a_vector_sum_gathers_the_adjoint_at_the_key(tmp_path, cache_dir):
    primal = KERNELS[1]
    built = _derivative_bundle([primal], tmp_path / "p", cache_dir)
    adjoint = Kernel("vector_vjp", hawk.diff.vjp(primal, primal_unit=built.digest))
    bundle = _derivative_bundle([adjoint], tmp_path / "t", cache_dir)
    rng = np.random.default_rng(11)
    key, bar_acc = _crowded(rng, CELLS), rng.normal(size=(3, CELLS))
    bar_v = np.zeros((3, N))
    runtime.run(runtime.load(bundle.directory, "vector_vjp"), key=key, bar_acc=bar_acc, bar_v=bar_v)
    np.testing.assert_allclose(bar_v, bar_acc[:, key], rtol=1e-13)


# ---- a compensated add at several indices that are the same across a warp commits once per warp run, at the element ----

NET_SAMPLES = 64   # two warps per net: a run of 32 lanes shares the element (net, 2)


def _net_sum_kernel(contract: str, column):
    """The sum of each net's samples into a column of the plane ``acc[net, p]``: ``column`` is the second index."""

    @kernel
    def net_sum(lane: hawk.Grid["net", "sample":NET_SAMPLES], v: hawk.Scalar, acc: Mutable[hawk.Scalar, "net", "p":4]):  # noqa: F821
        hawk.scatter(acc, contract=contract)[lane.net, column(lane)] += v

    net_sum.name = f"net_sum_{contract}_{column.__name__}"
    return net_sum


def _fixed_column(lane):
    return 2


def _moving_column(lane):
    return lane.sample % 4


NET_KERNELS = [_net_sum_kernel("compensated", _fixed_column), _net_sum_kernel("compensated", _moving_column)]


@pytest.fixture(scope="module")
def net_bundle(tmp_path_factory, cache_dir):
    return build_bundle(NET_KERNELS[:1], tmp_path_factory.mktemp("net_sum"), mode="float64", targets=("cuda", "host"),
                        cache_dir=cache_dir)


def _net_terms(nets, seed):
    return _ill_conditioned(np.random.default_rng(seed), (nets, NET_SAMPLES))


def _net_sum_want(terms):
    """The oracle: ``math.fsum`` of each net's samples in column 2, zero elsewhere."""
    want = np.zeros((terms.shape[0], 4))
    want[:, 2] = [math.fsum(row) for row in terms]
    return want


def test_a_compensated_net_sum_equals_fsum_per_net_on_the_host(net_bundle, threads):
    """The tolerance is the one of the compensated matrix rows of this file: 8 eps of the sum of magnitudes."""
    terms = _net_terms(5, 7)
    acc = np.zeros((5, 4))
    _launch(net_bundle, NET_KERNELS[0].name, "host", {"net": 5, "v": terms.ravel(), "acc": acc}).launch()
    scale = np.finfo(np.float64).eps * np.abs(terms).sum(axis=1, keepdims=True) * 8
    assert np.all(np.abs(acc - _net_sum_want(terms)) <= scale)


@pytest.mark.gpu
def test_sixteen_thousand_compensated_net_sums_equal_fsum_per_net_on_the_device(net_bundle):
    import cupy as cp

    terms = _net_terms(16384, 8)
    acc = cp.zeros((16384, 4))
    _launch(net_bundle, NET_KERNELS[0].name, "cuda", {"net": 16384, "v": cp.asarray(terms.ravel()), "acc": acc}).launch()
    cp.cuda.Device().synchronize()
    scale = np.finfo(np.float64).eps * np.abs(terms).sum(axis=1, keepdims=True) * 8
    assert np.all(np.abs(acc.get() - _net_sum_want(terms)) <= scale)


def test_the_text_of_a_compensated_net_sum_commits_once_per_warp_run_with_its_companion():
    text = emit_source(NET_KERNELS[0], targets=("cuda",), device_arch="sm_80").sources["cuda"].text
    assert "aether::accum::run::addCompensatedAt<32>(aout_acc(" in text
    assert ", aout_acc_c(" in text
    assert "hawk_abi::accum_add_compensated_at(aout_acc(" not in text


def test_the_text_of_a_compensated_net_sum_whose_column_moves_with_the_sample_commits_per_lane():
    text = emit_source(NET_KERNELS[1], targets=("cuda",), device_arch="sm_80").sources["cuda"].text
    assert "hawk_abi::accum_add_compensated_at(aout_acc(" in text
    assert "addCompensatedAt<" not in text
