# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Tests for the standalone C target's spelling tables."""

import pytest

from hawk.embedded import spelling as S
from hawk.emit import FLOAT32, FLOAT64
from hawk.ir import HawkError
from hawk.types import TensorType


def test_ctype_by_rank_and_mode():
    assert S.ctype(TensorType((), "f64"), FLOAT64) == "hawk_real_t"
    assert S.ctype(TensorType((3,), "f32"), FLOAT32) == "aether_c_f32_vec3_t"
    assert S.ctype(TensorType((2, 3), "f64"), FLOAT64) == "aether_c_f64_mat2x3_t"
    assert S.ctype(TensorType((), "i32"), FLOAT64) == "hawk_int_t"
    assert S.ctype(TensorType((), "uint32"), FLOAT64) == "uint32_t"
    assert S.ctype(TensorType((), "bool"), FLOAT64) == "bool"


def test_shape_word():
    assert S.shape_word((3,)) == "vec3"
    assert S.shape_word((2, 3)) == "mat2x3"
    assert S.shape_word(()) == "scalar"


def test_libm_names_carry_the_f_suffix_only_in_float32():
    assert S.libm_name("sqrt", FLOAT64) == "sqrt"
    assert S.libm_name("sqrt", FLOAT32) == "sqrtf"
    assert S.libm_name("fmin", FLOAT32) == "fminf"
    assert S.libm_name("nextafter", FLOAT32) == "nextafterf"


def test_libm_unknown_name_lists_known_names():
    with pytest.raises(HawkError, match="sqrt"):
        S.libm_name("isnan", FLOAT64)


def test_real_literals_are_cast_double_literals_in_both_modes():
    assert S.real_literal(1.5, FLOAT64) == "((hawk_real_t)1.5)"
    assert S.real_literal(1.5, FLOAT32) == "((hawk_real_t)1.5)"
    assert S.real_literal(float("-inf"), FLOAT64) == "((hawk_real_t)-INFINITY)"
    assert S.real_literal(float("nan"), FLOAT32) == "((hawk_real_t)NAN)"
    assert S.real_literal(2.0, FLOAT64) == "((hawk_real_t)2.0)"


def test_int_literals_and_helpers():
    assert S.int_literal(-3, "i32") == "INT64_C(-3)"
    assert S.int_literal(7, "uint32") == "UINT32_C(7)"
    assert S.int_literal(5, "int32") == "INT32_C(5)"
    assert S.int_literal(True, "bool") == "true"
    assert S.int_literal(0, "bool") == "false"
    assert S.INT_HELPER[("add", "i32")] == "hawk_i64_add"
    assert S.INT_HELPER[("add", "i64")] == "hawk_i64_add"
    assert S.INT_HELPER[("band", "int32")] == "hawk_i32_and"
    assert S.INT_HELPER[("from_real", "uint32")] == "hawk_u32_from_real"
    assert len(S.INT_HELPER) == 17 * 4


def test_clamps():
    assert S.clamp_index("t4", "lut_t_len") == "hawk_clamp_index(t4, lut_t_len)"
    assert S.clamp_trips("n", 8) == "hawk_clamp_trips(n, 8)"
