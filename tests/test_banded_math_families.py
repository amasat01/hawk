# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The banded ops admitted after the first certified facade, one kernel per family (and the literal division), run end to end on
float32-exact inputs inside each op's domain and held against the float64 build of the same kernel: the
bound is the sum of the per-op bounds along the longest input path (0.5 ULP53 each)."""

from __future__ import annotations

import numpy as np
import pytest

import hawk
import hawk.math as m
from test_banded_end_to_end import N, _run_kernel, _ulp53


@hawk.kernel
def exp_log(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    out = m.expm1(x) + m.log1p(y) + m.exp2(x) + m.log2(y) + m.log10(y)


@hawk.kernel
def hyperbolic(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    out = m.sinh(x) + m.cosh(x) + m.tanh(x)


@hawk.kernel
def inverse_hyperbolic(x: hawk.Scalar, y: hawk.Scalar, z: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    out = m.asinh(x) + m.acosh(y) + m.atanh(z)


@hawk.kernel
def over_literal(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    out = x / 3.0 + x / 4.0 + x * 3.0


@hawk.kernel
def stepping(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    acc = x
    for _k in range(16):
        acc = acc * 0.5 + m.sin(acc) * 0.25
    out = acc


@hawk.kernel
def sin_cos(x: hawk.Scalar, a: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    out = m.sin(x) * a + m.cos(x) * a


@hawk.kernel
def fused(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    out = x * y * x + y


@hawk.kernel
def coefficient(x: hawk.Scalar, k: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    acc = x
    for _j in range(8):
        acc = acc * k + m.sin(acc) * 0.25
    out = acc


@hawk.kernel
def raw_twins(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    out = m.sin(x) * y + m.cos(x) * x + m.exp(y) * 0.5 + (x / y) * x - (x / y) * 0.25


@hawk.kernel
def tangent(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    out = m.tan(x)


def _f32(rng, lo, hi):
    return rng.uniform(lo, hi, N).astype(np.float32).astype(np.float64)


def _planes(name):
    rng = np.random.default_rng(29)
    x, y = _f32(rng, 0.1, 2.0), _f32(rng, 1.1, 4.0)
    return {
        "exp_log": {"x": x, "y": y},
        "hyperbolic": {"x": x},
        "inverse_hyperbolic": {"x": x, "y": _f32(rng, 1.0, 4.0), "z": _f32(rng, 0.1, 0.9)},
        "over_literal": {"x": x},
        "raw_twins": {"x": _f32(rng, 0.1, 1.4), "y": y},
        "coefficient": {"x": x, "k": _f32(rng, 0.1, 0.5)},
        "fused": {"x": x, "y": y},
        "sin_cos": {"x": _f32(rng, 0.1, 1.4), "a": y},
        "stepping": {"x": x},
        "tangent": {"x": _f32(rng, 0.1, 1.4)},
    }[name]


#: kernel, bound: each op and each joining add 0.5 ULP53, plus 1 ULP53 for the float64 build's libm (all terms positive,
#: so the sum does not cancel).
FAMILIES = ((exp_log, 5.5), (hyperbolic, 3.5), (inverse_hyperbolic, 3.5), (tangent, 1.5),
            (over_literal, 4.0),
            (stepping, 25.0),
            (sin_cos, 3.5),
            (fused, 2.5),
            (coefficient, 15.0),
            (raw_twins, 8.0))        # a contraction: each of the 16 steps adds its ~1.5 ULP53 and the next halves it


def _check(kernel, bound, target, tmp_path):
    pytest.importorskip("eagle")
    ins = _planes(kernel.name)
    got, _ = _run_kernel(kernel, ins, "banded", target, tmp_path)
    want, _ = _run_kernel(kernel, ins, "float64", target, tmp_path)
    assert np.all(np.isfinite(got))
    assert _ulp53(got, want).max() <= bound


@pytest.mark.gpu
@pytest.mark.parametrize("kernel,bound", FAMILIES, ids=[k.name for k, _ in FAMILIES])
def test_a_banded_family_on_the_gpu_is_within_the_sum_of_its_per_op_bounds(kernel, bound, tmp_path):
    _check(kernel, bound, "cuda", tmp_path)
