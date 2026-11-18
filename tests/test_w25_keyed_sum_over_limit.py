# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A keyed sum whose payload exceeds the wing's ``linalg.team_max_dim`` (4096 elements) is a block sum at trace time; the host build
of it delivers the NumPy value. Independent oracle: NumPy sums per key (outer products; a left-to-right ordered sum)."""

from __future__ import annotations

import numpy as np
import pytest
from _block_kernels import gram_kernel  # noqa: F401  (the same lane layout: key-major rows)
from test_team_linalg import KEYS, ROWS, _build_sum, _run_sum

import hawk
import hawk.math as m
from hawk import Grid, Matrix, Vector, kernel


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("w25_keyed_over_limit")




#: The element count past which a keyed sum is a block sum at trace time.
TEAM_MAX_DIM = 4096


def test_a_keyed_sum_of_an_outer_product_above_the_limit_is_the_numpy_value(cache):
    p, q = 65, 64                      # 4160 elements per key: above team_max_dim
    assert p * q > TEAM_MAX_DIM

    @kernel
    def outer_sum(lane: Grid["key", "row":ROWS], a: Vector[p], b: Vector[q]) -> Matrix[p, q]:
        return hawk.sum(m.outer(a, b), over="row")

    rng = np.random.default_rng(65)
    lanes = KEYS * ROWS
    a, b = rng.standard_normal((lanes, p)), rng.standard_normal((lanes, q))
    bundle = _build_sum(outer_sum, cache / "outer", "host")
    out = {"out": np.zeros((p, q, KEYS))}
    got = _run_sum(bundle, outer_sum, {"a": np.ascontiguousarray(a.T), "b": np.ascontiguousarray(b.T)}, out, "host")["out"]
    want = np.einsum("kri,krj->ijk", a.reshape(KEYS, ROWS, p), b.reshape(KEYS, ROWS, q))
    scale = np.einsum("kri,krj->ijk", np.abs(a).reshape(KEYS, ROWS, p), np.abs(b).reshape(KEYS, ROWS, q))
    assert np.all(np.abs(got - want) <= 8 * np.finfo(np.float64).eps * scale + 1e-300), float(np.max(np.abs(got - want)))


def test_an_ordered_vector_payload_above_the_limit_is_the_left_to_right_sum(cache):
    n = 4100

    @kernel
    def ordered_sum(lane: Grid["key", "row":ROWS], j: Vector[n], r: hawk.Scalar) -> Vector[n]:
        return hawk.sum(j * r, over="row", contract="ordered")

    rng = np.random.default_rng(4100)
    lanes = KEYS * ROWS
    j, r = rng.standard_normal((lanes, n)), rng.standard_normal(lanes)
    bundle = _build_sum(ordered_sum, cache / "ordered", "host")
    out = {"out": np.zeros((n, KEYS))}
    got = _run_sum(bundle, ordered_sum, {"j": np.ascontiguousarray(j.T), "r": r}, out, "host")["out"]
    terms = (j * r[:, None]).reshape(KEYS, ROWS, n)
    want = np.zeros((KEYS, n))
    for row in range(ROWS):            # the oracle: row order, left to right, one rounding per add
        want = want + terms[:, row, :]
    assert np.array_equal(got, want.T), float(np.max(np.abs(got - want.T)))


def _vector_sum(n, contract):
    @kernel
    def vec_sum(lane: Grid["key", "row":ROWS], j: Vector[n], r: hawk.Scalar) -> Vector[n]:
        return hawk.sum(j * r, over="row", contract=contract)

    return vec_sum


@pytest.mark.parametrize("contract", ["ordered", "auto"])
@pytest.mark.parametrize("n", [4096, 4097])
def test_the_boundary_of_the_limit_is_the_numpy_value_under_both_contracts(cache, n, contract):
    k = _vector_sum(n, contract)
    rng = np.random.default_rng(n)
    lanes = KEYS * ROWS
    j, r = rng.standard_normal((lanes, n)), rng.standard_normal(lanes)
    bundle = _build_sum(k, cache / f"b{n}{contract}", "host")
    out = {"out": np.zeros((n, KEYS))}
    got = _run_sum(bundle, k, {"j": np.ascontiguousarray(j.T), "r": r}, out, "host")["out"]
    terms = (j * r[:, None]).reshape(KEYS, ROWS, n)
    want = np.zeros((KEYS, n))
    for row in range(ROWS):
        want = want + terms[:, row, :]
    if contract == "ordered":
        assert np.array_equal(got, want.T)
    else:
        scale = np.abs(terms).sum(axis=1)
        assert np.all(np.abs(got - want.T) <= 8 * np.finfo(np.float64).eps * scale.T + 1e-300)
