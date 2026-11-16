# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A kernel whose IR holds a long dependent chain builds, and so do its derivatives: no pass recurses once per node.

An unrolled Python accumulation over a wide vector is a chain of thousands of nodes, deeper than the interpreter's recursion
limit. The kernel below writes a 512-wide Mutable from such a chain, which the warp form slices on a CUDA target; the host,
CUDA, forward and reverse builds must all reach the end of the emit (text only: no compiler, no card).
"""

import _wide_kernels as W
import pytest

import hawk
from hawk import Mutable, Scalar, Vector
import hawk.artifact

DEPTH = 1200


def _chain(a, x):
    """Unrolled at trace time: a plain-python helper loop over a list is run as written."""
    acc = x
    for _ in [0] * DEPTH:
        acc = acc * a + x
    return acc


@hawk.kernel
def deep_chain(a: Scalar, x: Vector[512], y: Mutable[Vector[512]]):
    y = _chain(a, x)


def _scalar(a, x):
    s = a
    for j, _ in enumerate([0] * DEPTH):
        s = s * a + x[j % 512]
    return x * s


@hawk.kernel
def deep_scalar(a: Scalar, x: Vector[512], y: Mutable[Vector[512]]):
    y = _scalar(a, x)


KERNELS = {"chain": deep_chain, "scalar": deep_scalar}


def _derived(kernel, kind):
    from hawk.diff import jvp, vjp

    return W._derived(kernel.name + "_" + kind, kernel, {"jvp": jvp, "vjp": vjp}[kind], ("a",))


def _emit(kernel, target):
    return hawk.artifact.emit_source(kernel, targets=(target,)).sources[target].text


@pytest.mark.parametrize("name", KERNELS)
@pytest.mark.parametrize("which", ["primal", "jvp", "vjp"])
@pytest.mark.parametrize("target", ["host", "cuda"])
def test_a_chain_deeper_than_the_recursion_limit_emits(name, which, target):
    kernel = KERNELS[name] if which == "primal" else _derived(KERNELS[name], which)
    assert "deep_" in _emit(kernel, target)
