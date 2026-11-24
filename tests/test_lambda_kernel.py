# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The lambda form of a kernel: ``hawk.kernel(lambda a, b: a + b, name="add", types=(Scalar, Scalar))``.

The returned value is committed to the output ``out`` (a tuple: ``out0``, ``out1``) as a ``def`` that returns does. Every
value is checked against NumPy written here; every refusal states the kernel's name and the way out.
"""

from __future__ import annotations

import numpy as np
import pytest
from _eval import evaluate

import hawk
import hawk.diff
from hawk import HawkError, Scalar, Vector
from hawk import math as m
from hawk.artifact import build_bundle

RNG = np.random.default_rng(20261124)
N = 37

add = hawk.kernel(lambda a, b: a + b, name="add", types=(Scalar, Scalar))
sum_and_product = hawk.kernel(
    lambda a, b: (a + b, a * b), name="sum_and_product", types=(Scalar, Scalar)
)
scaled = hawk.kernel(lambda v, s: v * s, name="scaled", types=(Vector[3], Scalar))
cube = hawk.kernel(lambda x: x * x * x, name="cube", types=(Scalar,))


def plain(a, b):
    return a - b


minus = (
    hawk.kernel(plain, name=None) if False else None
)  # a named function takes no name=; see the refusal test below
minus = hawk.kernel(lambda a, b: a - b, name="minus", types=(Scalar, Scalar))


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    return build_bundle(
        [add, sum_and_product, scaled, minus],
        tmp_path_factory.mktemp("lambda"),
        targets=("host",),
        cache_dir=cache_dir,
    )


def _run(bundle, name, **planes):
    hawk.run(hawk.load(bundle.directory, name), **planes)


def test_a_two_input_add_equals_numpy_add(bundle):
    a, b = RNG.normal(size=N), RNG.normal(size=N)
    out = np.zeros(N)
    _run(bundle, "add", a=a, b=b, out=out)
    np.testing.assert_array_equal(out, np.add(a, b))


def test_a_lambda_returning_a_tuple_commits_out0_and_out1_equal_to_numpy(bundle):
    a, b = RNG.normal(size=N), RNG.normal(size=N)
    out0, out1 = np.zeros(N), np.zeros(N)
    _run(bundle, "sum_and_product", a=a, b=b, out0=out0, out1=out1)
    np.testing.assert_array_equal(out0, a + b)
    np.testing.assert_array_equal(out1, a * b)


def test_a_lambda_with_a_vector_type_equals_numpy_broadcast_product(bundle):
    v, s = (
        RNG.normal(size=(3, N)),
        RNG.normal(size=N),
    )  # a vector plane is component-major
    out = np.zeros((3, N))
    _run(bundle, "scaled", v=v, s=s, out=out)
    np.testing.assert_array_equal(out, v * s)


def test_types_on_a_function_without_annotations_declares_its_parameters(bundle):
    a, b = RNG.normal(size=N), RNG.normal(size=N)
    out = np.zeros(N)
    _run(bundle, "minus", a=a, b=b, out=out)
    np.testing.assert_array_equal(out, np.subtract(a, b))


def test_types_on_a_def_without_annotations_is_the_kernel_of_the_def():
    def difference(a, b):
        return a - b

    kernel = hawk.kernel(difference, types=(Scalar, Scalar))
    assert (kernel.name, kernel.arg_spec) == (
        "difference",
        (("mutable", "out"), ("plane_in", "a"), ("plane_in", "b")),
    )


def test_the_same_lambda_built_twice_has_the_same_cache_name():
    def build():
        return hawk.kernel(lambda a, b: a + b, name="add", types=(Scalar, Scalar))

    first, second = build(), build()
    assert first.name == second.name == "add"
    assert first.body_digest == second.body_digest


def test_the_forward_derivative_of_a_lambda_kernel_equals_the_numpy_closed_form():
    """Oracle: d(x^3) = 3 x^2 dx."""
    x, dx = RNG.normal(size=5), RNG.normal(size=5)
    got = evaluate(hawk.diff.jvp(cube), {"x": x, "dot_x": dx}, lane=0, lanes=5)[
        "dot_out"
    ]
    np.testing.assert_allclose(got, 3.0 * x**2 * dx, rtol=1e-14)


def test_annotations_and_types_together_are_refused_by_name():
    def annotated(a: Scalar, b: Scalar):
        return a + b

    with pytest.raises(
        HawkError,
        match=r"kernel 'annotated': the parameters \['a', 'b'\] are annotated and types= is given too; state each declaration once",
    ):
        hawk.kernel(annotated, types=(Scalar, Scalar))


def test_a_wrong_count_of_types_is_refused_by_name():
    with pytest.raises(
        HawkError,
        match=r"kernel 'pair': types= holds one declaration per parameter, in order: 2 for \['a', 'b'\]",
    ):
        hawk.kernel(lambda a, b: a + b, name="pair", types=(Scalar,))


def test_a_lambda_without_a_name_is_refused_by_name():
    with pytest.raises(
        HawkError,
        match=r"kernel '<lambda>': a lambda has no name of its own; give one: hawk.kernel\(fn, name=",
    ):
        hawk.kernel(lambda a, b: a + b, types=(Scalar, Scalar))


def test_a_name_on_a_named_function_is_refused_by_name():
    def named(a, b):
        return a + b

    with pytest.raises(
        HawkError,
        match=r"kernel 'named': name= is for a function whose own name is not an identifier .*leave name= out",
    ):
        hawk.kernel(named, name="other", types=(Scalar, Scalar))


@pytest.mark.parametrize(
    "branch",
    [lambda a, b: a if a > b else b, lambda a, b: a and b, lambda a, b: a or b],
    ids=["conditional expression", "and", "or"],
)
def test_a_branch_on_a_traced_value_is_refused_by_name_with_the_way_out(branch):
    with pytest.raises(
        HawkError,
        match=r"kernel 'branching': .*hawk\.math\.where\(c, a, b\), or as a `def`",
    ):
        hawk.kernel(branch, name="branching", types=(Scalar, Scalar))


def test_where_is_the_way_out_of_a_branch(bundle):
    chosen = hawk.kernel(
        lambda a, b: m.where(a > b, a, b), name="chosen", types=(Scalar, Scalar)
    )
    assert chosen.name == "chosen"
