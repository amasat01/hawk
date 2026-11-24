# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Derivatives of a call of a function, in both directions.

* A traced function is differentiated through its body; the tangent is ONE function and the reverse is ONE function per read,
  however many calls the kernel makes.
* An author's rule (``jvp=``, ``vjp=``) is the derivative; on a traced function it replaces the derived one.
* A native function without the rule for a direction is refused by the function's name (the case with no rule at all is in
  ``test_native_function.py``).

Every value is compared with the closed form in NumPy. A derivative sums at most four rounded terms of the closed form, so
the bound is four units of the machine epsilon times the sum of the magnitudes of the terms.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import hawk
import hawk.diff
import hawk.math as m
from hawk import HawkError
from hawk.artifact import emit_source
from hawk.trace.kernel import Kernel

LANES = 64
USES = 16
EPS = np.finfo(np.float64).eps


@hawk.function
def square_plus(x: hawk.Scalar, y: hawk.Scalar) -> hawk.Scalar:
    return x * x + y


fma_sq_ruled = hawk.native(
    "fma_sq_ruled",
    reads={"x": hawk.Scalar, "y": hawk.Scalar},
    returns=hawk.Scalar,
    text={"host": "return std::fma(x, x, y);"},
    jvp=lambda x, y, dx, dy: 2.0 * x * dx + dy,
    vjp=lambda x, y, bar: (2.0 * x * bar, bar),
)


#: The author's rules claim the derivative of ``x * x + y`` is ``x`` and ``3``: they differ from the derived ones, so they show.
@hawk.function(
    jvp=lambda x, y, dx, dy: x * dx + 3.0 * dy,
    vjp=lambda x, y, bar: (x * bar, 3.0 * bar),
)
def square_plus_ruled(x: hawk.Scalar, y: hawk.Scalar) -> hawk.Scalar:
    return x * x + y


@hawk.function
def outer(a: hawk.Scalar, b: hawk.Scalar) -> hawk.Scalar:
    return square_plus(a, b) * b


STEPS = 5


@hawk.function
def horner(x: hawk.Scalar) -> hawk.Scalar:
    acc = 0.0
    for _ in range(STEPS):
        acc = acc * x + 1.0
    return acc


@hawk.function
def energy(v: hawk.Vector[3], k: hawk.Scalar) -> hawk.Scalar:
    return k * m.dot(v, v)


def _xy():
    rng = np.random.default_rng(11)
    return rng.uniform(0.5, 1.5, LANES), rng.uniform(-1.0, 1.0, LANES)


def _text(kernel, target="host"):
    return emit_source(kernel, targets=(target,)).sources[target].text


def _definitions(text, stem):
    """The names of the functions called ``stem...`` that the unit text defines."""
    return re.findall(rf"^static inline \S+ (hawk_fn_{stem}\w*)\(", text, re.M)


def _tangent(primal, cache_dir, **planes):
    """The tangent of the output ``out`` of ``primal`` along the ``dot_*`` planes given, run on the host."""
    kernel = Kernel(f"{primal.name}_jvp", hawk.diff.jvp(primal))
    shape = planes["dot_out"].shape if "dot_out" in planes else None
    planes.setdefault("dot_out", np.zeros(LANES) if shape is None else np.zeros(shape))
    hawk.deploy(kernel, targets=("host",), cache_dir=cache_dir).run(**planes)
    return planes["dot_out"]


def _gradients(primal, cache_dir, names, **planes):
    """The gradients named ``names`` of the output of ``primal``, with the adjoint of ``out`` set to one."""
    kernel = Kernel(f"{primal.name}_vjp", hawk.diff.vjp(primal))
    for name in names:
        planes[f"bar_{name}"] = np.zeros_like(planes[name])
    hawk.deploy(kernel, targets=("host",), cache_dir=cache_dir).run(**planes)
    return [planes[f"bar_{name}"] for name in names]


def _within(got, terms):
    """Whether ``got`` is within four epsilons of the sum of the magnitudes of ``terms`` of its closed form."""
    scale = sum(np.abs(t) for t in terms)
    return np.all(np.abs(got - sum(terms)) <= 4.0 * EPS * scale)


@hawk.kernel
def _traced_kernel(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    out = square_plus(x, y)


@hawk.kernel
def _native_kernel(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    out = fma_sq_ruled(x, y)


_KERNELS = {"traced": _traced_kernel, "native": _native_kernel}


@pytest.mark.parametrize("label", ["traced", "native"])
def test_the_tangent_of_x_squared_plus_y_equals_its_closed_form(label, cache_dir):
    x, y = _xy()
    dx, dy = np.full(LANES, 0.75), np.full(LANES, -0.5)
    got = _tangent(_KERNELS[label], cache_dir, x=x, y=y, dot_x=dx, dot_y=dy)
    assert _within(got, [2.0 * x * dx, dy])


@pytest.mark.parametrize("label", ["traced", "native"])
def test_the_gradient_of_x_squared_plus_y_equals_its_closed_form(label, cache_dir):
    x, y = _xy()
    bar = np.full(LANES, 1.25)
    gx, gy = _gradients(_KERNELS[label], cache_dir, ("x", "y"), x=x, y=y, bar_out=bar)
    assert _within(gx, [2.0 * x * bar])
    assert np.array_equal(gy, bar)


def _sixteen():
    @hawk.kernel
    def kern(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        acc = 0.0
        acc = acc + square_plus(x * 1.0, y + 0.0)
        acc = acc + square_plus(x * 1.1, y + 0.1)
        acc = acc + square_plus(x * 1.2, y + 0.2)
        acc = acc + square_plus(x * 1.3, y + 0.3)
        acc = acc + square_plus(x * 1.4, y + 0.4)
        acc = acc + square_plus(x * 1.5, y + 0.5)
        acc = acc + square_plus(x * 1.6, y + 0.6)
        acc = acc + square_plus(x * 1.7, y + 0.7)
        acc = acc + square_plus(x * 1.8, y + 0.8)
        acc = acc + square_plus(x * 1.9, y + 0.9)
        acc = acc + square_plus(x * 2.0, y + 1.0)
        acc = acc + square_plus(x * 2.1, y + 1.1)
        acc = acc + square_plus(x * 2.2, y + 1.2)
        acc = acc + square_plus(x * 2.3, y + 1.3)
        acc = acc + square_plus(x * 2.4, y + 1.4)
        acc = acc + square_plus(x * 2.5, y + 1.5)
        out = acc

    return kern


def test_sixteen_calls_make_one_tangent_function_and_one_reverse_function_per_read(
    cache_dir,
):
    kern = _sixteen()
    forward = _text(Kernel("sixteen_jvp", hawk.diff.jvp(kern)))
    assert len(_definitions(forward, "square_plus_jvp")) == 1
    assert len(re.findall(r"hawk_fn_square_plus_jvp_\w+\(", forward)) == USES + 1
    reverse = _text(Kernel("sixteen_vjp", hawk.diff.vjp(kern)))
    assert len(_definitions(reverse, "square_plus_vjp_x")) == 1
    assert len(_definitions(reverse, "square_plus_vjp_y")) == 1
    assert len(re.findall(r"hawk_fn_square_plus_vjp_x_\w+\(", reverse)) == USES + 1
    x, y = _xy()
    scales = [1.0 + 0.1 * k for k in range(USES)]
    got = _tangent(
        kern, cache_dir, x=x, y=y, dot_x=np.ones(LANES), dot_y=np.ones(LANES)
    )
    assert _within(
        got, [2.0 * s * s * x for s in scales] + [np.full(LANES, float(USES))]
    )
    gx, gy = _gradients(kern, cache_dir, ("x", "y"), x=x, y=y, bar_out=np.ones(LANES))
    assert _within(gx, [2.0 * s * s * x for s in scales])
    assert np.array_equal(gy, np.full(LANES, float(USES)))


def test_a_function_calling_a_function_has_the_derivative_of_the_composition(cache_dir):
    @hawk.kernel
    def kern(a: hawk.Scalar, b: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = outer(a, b)

    a, b = _xy()
    da, db = np.full(LANES, 0.5), np.full(LANES, 2.0)
    got = _tangent(kern, cache_dir, a=a, b=b, dot_a=da, dot_b=db)
    assert _within(got, [2.0 * a * b * da, (a * a + 2.0 * b) * db])
    ga, gb = _gradients(kern, cache_dir, ("a", "b"), a=a, b=b, bar_out=np.ones(LANES))
    assert _within(ga, [2.0 * a * b])
    assert _within(gb, [a * a, 2.0 * b])


def test_a_function_whose_body_holds_a_loop_has_the_derivative_of_the_series(cache_dir):
    @hawk.kernel
    def kern(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = horner(x)

    x = np.linspace(0.1, 0.9, LANES)
    slope = [j * x ** (j - 1) for j in range(1, STEPS)]
    got = _tangent(kern, cache_dir, x=x, dot_x=np.ones(LANES))
    assert _within(got, slope)
    (gx,) = _gradients(kern, cache_dir, ("x",), x=x, bar_out=np.ones(LANES))
    assert _within(gx, slope)


def test_a_vector_parameter_has_a_vector_gradient_and_a_tangent_along_it(cache_dir):
    @hawk.kernel
    def kern(v: hawk.Vector[3], k: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = energy(v, k)

    rng = np.random.default_rng(4)
    v, k = rng.uniform(-1.0, 1.0, (3, LANES)), rng.uniform(0.5, 1.5, LANES)
    dv, dk = rng.uniform(-1.0, 1.0, (3, LANES)), np.full(LANES, 0.25)
    got = _tangent(kern, cache_dir, v=v, k=k, dot_v=dv, dot_k=dk)
    assert _within(
        got, [2.0 * k * v[i] * dv[i] for i in range(3)] + [dk * (v * v).sum(axis=0)]
    )
    bar = np.full(LANES, 0.5)
    gv, gk = _gradients(kern, cache_dir, ("v", "k"), v=v, k=k, bar_out=bar)
    for i in range(3):
        assert _within(gv[i], [2.0 * k * v[i] * bar])
    assert _within(gk, [bar * (v * v).sum(axis=0)])


def test_the_authors_rule_on_a_traced_function_replaces_the_derived_one(cache_dir):
    @hawk.kernel
    def kern(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = square_plus_ruled(x, y)

    x, y = _xy()
    got = _tangent(
        kern, cache_dir, x=x, y=y, dot_x=np.ones(LANES), dot_y=np.ones(LANES)
    )
    assert _within(got, [x, np.full(LANES, 3.0)])
    gx, gy = _gradients(kern, cache_dir, ("x", "y"), x=x, y=y, bar_out=np.ones(LANES))
    assert _within(gx, [x])
    assert np.array_equal(gy, np.full(LANES, 3.0))


def test_a_native_function_with_only_one_rule_is_refused_in_the_other_direction():
    forward_only = hawk.native(
        "forward_only",
        reads={"x": hawk.Scalar},
        returns=hawk.Scalar,
        text={"host": "return x * x;"},
        jvp=lambda x, dx: 2.0 * x * dx,
    )

    @hawk.kernel
    def kern(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = forward_only(x)

    assert hawk.diff.jvp(kern)
    with pytest.raises(
        HawkError, match="native function 'forward_only' has no vjp rule; pass vjp="
    ):
        hawk.diff.vjp(kern)


def test_a_vjp_rule_that_returns_the_wrong_number_of_gradients_is_refused_by_name():
    broken = hawk.native(
        "broken_rule",
        reads={"x": hawk.Scalar, "y": hawk.Scalar},
        returns=hawk.Scalar,
        text={"host": "return x * y;"},
        vjp=lambda x, y, bar: (y * bar,),
    )

    with pytest.raises(
        HawkError, match="function 'broken_rule': its vjp rule returned 1 gradient"
    ):

        @hawk.kernel
        def kern(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
            out = broken(x, y)


def test_a_rule_may_be_a_native_function(cache_dir):
    slope = hawk.native(
        "slope_native",
        reads={"x": hawk.Scalar, "dx": hawk.Scalar},
        returns=hawk.Scalar,
        text={"host": "return 3.0 * x * x * dx;"},
    )
    cube = hawk.native(
        "cube_native",
        reads={"x": hawk.Scalar},
        returns=hawk.Scalar,
        text={"host": "return x * x * x;"},
        jvp=slope,
    )

    @hawk.kernel
    def kern(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = cube(x)

    x, _ = _xy()
    got = _tangent(kern, cache_dir, x=x, dot_x=np.ones(LANES))
    assert _within(got, [3.0 * x * x])


def test_the_tangent_of_the_reverse_of_a_traced_function_equals_the_second_derivative(
    cache_dir,
):
    @hawk.kernel
    def kern(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = outer(x, y)

    reverse = Kernel("outer_vjp2", hawk.diff.vjp(kern))
    second = Kernel("outer_jvp_vjp", hawk.diff.jvp(reverse, wrt=["x"]))
    x, y = _xy()
    planes = {
        "x": x, "y": y, "bar_out": np.ones(LANES), "dot_x": np.ones(LANES),
        "dot_bar_x": np.zeros(LANES), "dot_bar_y": np.zeros(LANES),
    }  # fmt: skip
    hawk.deploy(second, targets=("host",), cache_dir=cache_dir).run(**planes)
    # outer = (x*x + y) * y: d/dx = 2 x y, whose derivative along x is 2 y; d/dy = x*x + 2 y, along x it is 2 x
    assert _within(planes["dot_bar_x"], [2.0 * y])
    assert _within(planes["dot_bar_y"], [2.0 * x])


def test_a_call_inside_a_loop_of_the_kernel_has_the_derivative_of_the_sum(cache_dir):
    @hawk.kernel
    def kern(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        acc = 0.0
        for _ in range(4):
            acc = acc + square_plus(x, y)
        out = acc

    x, y = _xy()
    got = _tangent(
        kern, cache_dir, x=x, y=y, dot_x=np.ones(LANES), dot_y=np.ones(LANES)
    )
    assert _within(got, [8.0 * x, np.full(LANES, 4.0)])
    gx, gy = _gradients(kern, cache_dir, ("x", "y"), x=x, y=y, bar_out=np.ones(LANES))
    assert _within(gx, [8.0 * x])
    assert np.array_equal(gy, np.full(LANES, 4.0))


def test_a_function_called_in_and_out_of_a_loop_has_one_tangent_definition():
    @hawk.kernel
    def kern(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        acc = square_plus(x, y)
        for _ in range(4):
            acc = acc + square_plus(x, y)
        out = acc

    text = _text(Kernel("mixed_jvp", hawk.diff.jvp(kern)))
    assert len(_definitions(text, "square_plus_jvp")) == 1
