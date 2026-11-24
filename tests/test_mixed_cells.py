# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Cells of different folds in one kernel: ``hawk.sum/min/max/mean(x)`` stored together, against NumPy.

Each cell of a kernel folds with its own reduction and starts from its own identity, whatever the other cells of the kernel
fold with: the same term or different terms, a masked term, float32 and float64, sizes that cross the row of 32 samples, one
thread, a team and the default plan.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import _launch

import hawk
from hawk import Kernel, Mutable, Scalar, kernel
from hawk.artifact import build_bundle

SIZES = (1, 31, 32, 33, 1000)


@kernel
def same_term(x: Scalar, s: Mutable[Scalar], lo: Mutable[Scalar], hi: Mutable[Scalar]):
    s = hawk.sum(x)
    lo = hawk.min(x)
    hi = hawk.max(x)


@kernel
def other_terms(
    x: Scalar, s: Mutable[Scalar], lo: Mutable[Scalar], hi: Mutable[Scalar]
):
    s = hawk.sum(x * x)
    lo = hawk.min(x * 2.0)
    hi = hawk.max(x + 1.0)


@kernel
def min_and_max(x: Scalar, lo: Mutable[Scalar], hi: Mutable[Scalar]):
    lo = hawk.min(x)
    hi = hawk.max(x)


@kernel
def mean_and_min(x: Scalar, m: Mutable[Scalar], lo: Mutable[Scalar]):
    m = hawk.mean(x)
    lo = hawk.min(x)


@kernel
def masked(x: Scalar, s: Mutable[Scalar], lo: Mutable[Scalar], hi: Mutable[Scalar]):
    s = hawk.sum(x, where=x < 3.0)
    lo = hawk.min(x * x, where=x < 3.0)
    hi = hawk.max(x, where=x < 3.0)


def _x(n):
    """Whole numbers from -4 to 6 in a scrambled order, so each fold has its winner away from the first sample."""
    return (7 * np.arange(n) % 11 - 4).astype(np.float64)


def _want(name, x):
    """The cells NumPy gives for ``x``, the oracle of every row."""
    keep = x[x < 3.0]
    return {
        "same_term": {"s": x.sum(), "lo": x.min(), "hi": x.max()},
        "other_terms": {"s": (x * x).sum(), "lo": (2 * x).min(), "hi": (x + 1).max()},
        "min_and_max": {"lo": x.min(), "hi": x.max()},
        "mean_and_min": {"m": x.mean(), "lo": x.min()},
        "masked": {"s": keep.sum(), "lo": (keep * keep).min(), "hi": keep.max()},
    }[name]


KERNELS = {
    "same_term": same_term,
    "other_terms": other_terms,
    "min_and_max": min_and_max,
    "mean_and_min": mean_and_min,
    "masked": masked,
}


@pytest.fixture(scope="module", params=["float64", "float32"])
def built(request, tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp(f"mixed_{request.param}")
    kernels = [Kernel(name, k.sinks) for name, k in KERNELS.items()]
    return build_bundle(
        kernels, root, mode=request.param, targets=("host",), cache_dir=cache_dir
    ), request.param


@pytest.fixture(
    params=[1, 4, None], ids=["one_thread", "four_threads", "default_threads"]
)
def threads(request):
    """The host team size of the launch, restored afterwards."""
    wanted = hawk.host_threads()
    hawk.set_host_threads(request.param)
    yield request.param
    hawk.set_host_threads(wanted)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("name", KERNELS)
def test_every_cell_of_a_kernel_of_mixed_folds_is_its_numpy_value(
    built, name, n, threads
):
    bundle, mode = built
    x = _x(n).astype(mode)
    want = _want(name, x.astype(np.float64))
    planes = {"x": x, **{cell: np.zeros(1, dtype=mode) for cell in want}}
    _launch(bundle, name, "host", planes).launch()
    got = {cell: float(planes[cell][0]) for cell in want}
    assert got == pytest.approx(want, rel=1e-5), (name, got, want)
