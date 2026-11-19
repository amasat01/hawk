# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A loop reads the components of an element-wise value of the kernel at the loop's own index.

The value is computed once before the loop and read at run time; a constant index still computes only that element. The
oracle is NumPy over the same inputs; the bound is a few units in the last place of the sum of the absolute terms (the host
``exp`` may differ from NumPy's by one).
"""

from __future__ import annotations

import numpy as np
import pytest

import hawk
import hawk.math as m
from hawk import runtime as rt
from conftest import sidecar_of
from hawk.artifact import build_bundle

WIDTH, LANES = 4, 64
EPS = np.finfo(np.float64).eps
#: ULPS units in the last place of the sum of the absolute terms: a libm ``exp`` is within one, the product and the sum a few.
ULPS = 8


@hawk.kernel
def loop_sum(x: hawk.Vector[WIDTH], out: hawk.Mutable[hawk.Scalar]):
    bf = m.exp(x) * 2.0
    acc = 0.0
    for j in range(WIDTH):
        acc = acc + bf[j]
    out = acc


@hawk.kernel
def loop_square_sum(x: hawk.Vector[WIDTH], out: hawk.Mutable[hawk.Scalar]):
    bf = m.exp(x) * 2.0
    acc = 0.0
    for j in range(WIDTH):
        acc = acc + bf[j] * bf[j]
    out = acc


@hawk.kernel
def one_component(x: hawk.Vector[WIDTH], out: hawk.Mutable[hawk.Scalar]):
    bf = m.exp(x) * 2.0
    out = bf[2]


def _run(kernel, x, tmp_path, cache_dir):
    bundle = build_bundle([kernel], tmp_path, targets=("host",), cache_dir=cache_dir, host_profile="native")
    out = np.zeros(LANES)
    plane = np.ascontiguousarray(x.T)       # (WIDTH, LANES): one row per component
    rt.run(rt.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name)), x=plane, out=out)
    return out


@pytest.mark.parametrize(
    "kernel,terms",
    [
        (loop_sum, lambda e: e),
        (loop_square_sum, lambda e: e * e),
        (one_component, lambda e: e[:, 2:3]),
    ],
    ids=lambda v: getattr(v, "name", "terms"),
)
def test_a_loop_reads_an_element_wise_value_at_its_own_index_and_matches_numpy(kernel, terms, tmp_path, cache_dir):
    x = np.random.default_rng(5).uniform(-1.0, 1.0, (LANES, WIDTH))
    got = _run(kernel, x, tmp_path, cache_dir)
    each = terms(np.exp(x) * 2.0)
    want, scale = each.sum(axis=1), np.abs(each).sum(axis=1)
    assert np.all(np.abs(got - want) <= ULPS * EPS * scale)
