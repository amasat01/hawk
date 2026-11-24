# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``@hawk.function`` with parameters that carry no type: each call gives them theirs.

* An untyped parameter takes the type of its argument; one function serves a scalar and a vector, written once per signature
  with a name that is stable and distinct per signature.
* A typed parameter beside an untyped one is checked as written; the result annotation is optional.
* The rules and the derived functions are made per signature.
* A body that cannot be traced for a signature is refused naming the function and the signature.
* ``name=`` names the function; two defs under one name are refused by the function table, which holds one function per name.
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

LANES = 32
EPS = np.finfo(np.float64).eps


@hawk.function
def twice(x):
    return x * 2.0


@hawk.function(vjp=lambda x, bar: bar * 3.0, jvp=lambda x, dx: dx * 3.0)
def ruled(x):
    return x * x * 0.0 + x


@hawk.function
def scaled(k: hawk.Scalar, x):
    return x * k


@hawk.function
def norm_of(v):
    return m.sqrt(m.dot(v, v))


@hawk.kernel
def both(
    s: hawk.Scalar,
    v: hawk.Vector[3],
    a: hawk.Mutable[hawk.Scalar],
    b: hawk.Mutable[hawk.Vector[3]],
):
    a = twice(s)
    b = twice(v)


@hawk.kernel
def ruled_both(
    s: hawk.Scalar,
    v: hawk.Vector[2],
    a: hawk.Mutable[hawk.Scalar],
    b: hawk.Mutable[hawk.Vector[2]],
):
    a = ruled(s)
    b = ruled(v)


@hawk.kernel
def mixed(k: hawk.Scalar, v: hawk.Vector[3], out: hawk.Mutable[hawk.Vector[3]]):
    out = scaled(k, v)


@hawk.kernel
def lengths(v: hawk.Vector[3], out: hawk.Mutable[hawk.Scalar]):
    out = norm_of(v)


def _random(*shape):
    return np.random.default_rng(5).uniform(0.5, 1.5, shape)


def _text(kernel):
    return emit_source(kernel, targets=("host",)).sources["host"].text


def _run(kernel, cache_dir, **planes):
    hawk.deploy(kernel, targets=("host",), cache_dir=cache_dir).run(**planes)
    return planes


def _close(got, want):
    return np.all(np.abs(got - want) <= 8.0 * EPS * np.maximum(1.0, np.abs(want)))


def test_one_function_serves_a_scalar_and_a_vector_and_equals_numpy(cache_dir):
    s, v = _random(LANES), _random(3, LANES)
    planes = _run(both, cache_dir, s=s, v=v, a=np.zeros(LANES), b=np.zeros((3, LANES)))
    assert _close(planes["a"], 2.0 * s) and _close(planes["b"], 2.0 * v)


def test_each_signature_is_one_definition_with_a_distinct_stable_name():
    names = re.findall(r"^static inline .* (hawk_fn_twice_\w+)\(", _text(both), re.M)
    assert len(names) == 2 and len(set(names)) == 2
    assert names == re.findall(
        r"^static inline .* (hawk_fn_twice_\w+)\(", _text(both), re.M
    )


def test_a_typed_parameter_beside_an_untyped_one_is_checked_as_written(cache_dir):
    k, v = _random(LANES), _random(3, LANES)
    planes = _run(mixed, cache_dir, k=k, v=v, out=np.zeros((3, LANES)))
    assert _close(planes["out"], k * v)
    with pytest.raises(HawkError, match="call of function 'scaled__f64_3': argument 0"):

        @hawk.kernel
        def wrong(v: hawk.Vector[3], out: hawk.Mutable[hawk.Vector[3]]):
            out = scaled(v, v)


def test_the_result_annotation_is_optional_and_checked_when_written():
    @hawk.function
    def declared(x) -> hawk.Scalar:
        return x * 2.0

    with pytest.raises(HawkError, match="declared to return"):

        @hawk.kernel
        def wrong(v: hawk.Vector[3], out: hawk.Mutable[hawk.Vector[3]]):
            out = declared(v)


def test_the_rules_are_traced_per_signature(cache_dir):
    s, v = _random(LANES), _random(2, LANES)
    jvp_kernel = Kernel("ruled_jvp", hawk.diff.jvp(ruled_both))
    planes = _run(
        jvp_kernel, cache_dir, s=s, v=v, dot_s=np.ones(LANES), dot_v=np.ones((2, LANES)),
        dot_a=np.zeros(LANES), dot_b=np.zeros((2, LANES)),
    )  # fmt: skip
    assert _close(planes["dot_a"], np.full(LANES, 3.0))
    assert _close(planes["dot_b"], np.full((2, LANES), 3.0))
    vjp_kernel = Kernel("ruled_vjp", hawk.diff.vjp(ruled_both))
    planes = _run(
        vjp_kernel, cache_dir, s=s, v=v, bar_a=np.ones(LANES), bar_b=np.ones((2, LANES)),
        bar_s=np.zeros(LANES), bar_v=np.zeros((2, LANES)),
    )  # fmt: skip
    assert _close(planes["bar_s"], np.full(LANES, 3.0))
    assert _close(planes["bar_v"], np.full((2, LANES), 3.0))


def test_a_derived_function_is_made_per_signature_and_equals_the_closed_form(cache_dir):
    v = _random(3, LANES)
    kernel = Kernel("lengths_vjp", hawk.diff.vjp(lengths))
    planes = _run(
        kernel, cache_dir, v=v, bar_out=np.ones(LANES), bar_v=np.zeros((3, LANES))
    )
    assert _close(planes["bar_v"], v / np.sqrt((v * v).sum(axis=0)))


def test_a_body_that_cannot_be_traced_for_a_signature_is_refused_naming_it():
    @hawk.function
    def first_two(v):
        return v[1] + v[0]

    with pytest.raises(HawkError, match=r"function 'first_two' called with \(.*\): "):

        @hawk.kernel
        def wrong(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
            out = first_two(x)


def test_a_stated_name_is_the_name_in_the_text():
    @hawk.function(name="stated")
    def whatever(x):
        return x + 1.0

    @hawk.kernel
    def uses(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = whatever(x)

    assert re.search(r"^static inline \S+ hawk_fn_stated_\w+\(", _text(uses), re.M)
    with pytest.raises(HawkError, match="name= must be an identifier"):
        hawk.function(name="not an identifier")(lambda x: x)


def test_two_defs_under_one_stated_name_are_refused_by_the_function_table():
    @hawk.function(name="shared")
    def plus_one(x):
        return x + 1.0

    @hawk.function(name="shared")
    def plus_two(x):
        return x + 2.0

    with pytest.raises(
        HawkError, match="'shared__f64' is already declared with another body"
    ):

        @hawk.kernel
        def uses(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
            out = plus_one(x) * plus_two(x)
