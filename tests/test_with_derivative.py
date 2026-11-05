# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""hawk.diff.with_derivative: ONE kernel computing a kernel's outputs and their forward-mode tangents. It is a plain
Kernel (not a derivative), so it is a step of an eagle.Graph builder; its values match the primal and jvp run apart
within rounding (different compiled entries, so ULP, not bits)."""

import numpy as np
import pytest

import hawk
from hawk import Mutable, Param, Scalar
from hawk import math as m
from hawk.diff import with_derivative

eagle = pytest.importorskip("eagle")


@hawk.kernel
def _g(x: Scalar, p: Param, y: Mutable[Scalar]):
    y = m.sin(x) * p


def test_with_derivative_is_one_plain_kernel_with_declared_tangent_planes():
    k = with_derivative(_g, wrt=("x",))
    assert isinstance(k, hawk.Kernel) and k.name == "_g_with_derivative"
    assert list(k.planes) == ["x", "p", "y", "dot_x", "dot_y"]
    assert with_derivative(_g, wrt=("x",), name="g_and_dg").name == "g_and_dg"


def test_with_derivative_values_match_the_function_and_its_derivative():
    k = with_derivative(_g, wrt=("x",))
    x = np.linspace(-2.0, 2.0, 257)
    dx = np.linspace(0.5, 1.5, 257)
    y, dy = np.zeros_like(x), np.zeros_like(x)
    k(x=x, p=3.0, y=y, dot_x=dx, dot_y=dy)
    np.testing.assert_allclose(y, 3.0 * np.sin(x), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(dy, 3.0 * np.cos(x) * dx, rtol=1e-14, atol=1e-15)


def test_with_derivative_is_a_builder_step_and_allocates_what_the_call_omits():
    k = with_derivative(_g, wrt=("x",))
    x = np.linspace(-1.0, 1.0, 64)
    dx = np.ones_like(x)

    @eagle.Graph.builder
    def block(g: eagle.Graph, x, dx):
        g.result = k(x=x, p=2.0, dot_x=dx)

    graph = eagle.Graph()
    block(graph, x, dx)
    graph.build()
    out = graph()
    np.testing.assert_allclose(np.asarray(out["y"]), 2.0 * np.sin(x), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(np.asarray(out["dot_y"]), 2.0 * np.cos(x), rtol=1e-14, atol=1e-15)


def test_with_derivative_refuses_a_steps_kernel_by_name():
    with pytest.raises(hawk.HawkError, match="hawk.steps"):
        with_derivative(hawk.steps(_g, 4), wrt=("x",))


@pytest.mark.gpu
def test_with_derivative_runs_on_the_device():
    cp = pytest.importorskip("cupy")
    k = with_derivative(_g, wrt=("x",))
    x = cp.linspace(-2.0, 2.0, 1000)
    dx = cp.ones_like(x)
    y, dy = cp.zeros_like(x), cp.zeros_like(x)
    k(x=x, p=3.0, y=y, dot_x=dx, dot_y=dy)
    np.testing.assert_allclose(cp.asnumpy(y), 3.0 * np.sin(cp.asnumpy(x)), rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(cp.asnumpy(dy), 3.0 * np.cos(cp.asnumpy(x)), rtol=1e-14, atol=1e-15)
