# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A real value divided by a trace-time constant is traced as a
multiply by the reciprocal. Every knot-span division in a spline evaluator is
one, and each compiles to an IEEE division check plus a call into the slow
path. The fold moves
a result by at most one ulp, inside the twin bands. It is REAL-only: an index
divided by an integer stays a division, which the contraction recognizer reads
as a stride; a constant divided BY a value, and a value divided by a value, are
not folds at all."""
import hawk
import hawk.math as hm
from hawk.ir import Leaf
from hawk.types import TensorType

S = TensorType((), "f64")
V3 = TensorType((3,), "f64")


def _real(t=S):
    role = {0: "per_sample", 1: "vec_in"}[len(t.shape)]
    return hawk.Value(Leaf("vocab_read", role, "a", t))


def test_real_over_constant_is_a_multiply_by_the_reciprocal():
    node = (_real() / 0.4).node
    assert node.kind == "mul", node.kind
    literals = [getattr(o, "literal", None) for o in node.operands]
    assert 2.5 in literals, literals


def test_vector_over_constant_folds_too():
    assert (_real(V3) / 4.0).node.kind == "mul"


def test_value_over_value_and_constant_over_value_stay_divisions():
    x, y = _real(), _real()
    assert (x / y).node.kind == "div"
    assert (2.0 / x).node.kind == "div"


def test_index_over_integer_stays_a_division_for_the_recognizer():
    node = (hm.sample_index() / 4).node
    assert node.kind == "div", node.kind


def test_zero_and_non_finite_constants_are_not_folded():
    assert (_real() / 0.0).node.kind == "div"
    assert (_real() / float("inf")).node.kind == "div"
