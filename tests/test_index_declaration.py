# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.Index`` and the other payload declarations: what an author may write with them."""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import _ir


def test_index_is_the_32_bit_index_type_and_equals_the_plain_record():
    """``Index`` is the ``i32`` scalar type and compares equal to the record of that type."""
    assert hawk.Index == _ir.TensorType((), "i32")
    assert (hawk.Index.shape, hawk.Index.dtype, hawk.Index.tag) == ((), "i32", None)


def test_index_subscripted_by_a_width_gives_the_32_or_the_64_bit_plane():
    """``Index[np.int32]`` is ``Index`` itself and ``Index[np.int64]`` the ``i64`` plane."""
    assert hawk.Index[np.int32] == hawk.Index
    assert hawk.Index[np.int64] == _ir.TensorType((), "i64")
    assert hawk.Index[np.dtype("int64")] == _ir.TensorType((), "i64")


def test_index_refuses_a_width_it_does_not_take_by_name():
    """A float dtype names the two it takes."""
    with pytest.raises(TypeError, match=r"Index\[\.\.\.\] takes int32 \(the default\) or int64"):
        hawk.Index[np.float64]


def test_no_other_payload_type_is_subscriptable_or_callable():
    """``Scalar[...]`` is refused by name, and a type is annotated with, never called."""
    with pytest.raises(TypeError, match="not subscriptable; only Index takes a width"):
        hawk.Scalar[np.int64]
    with pytest.raises(TypeError, match="annotate with it, do not call it"):
        hawk.Scalar()


def test_a_payload_type_still_reads_as_the_tuple_it_is():
    """A position reads a field and a type unpacks, so the IR module's tuple reads keep working."""
    shape, dtype, tag = hawk.Index
    assert (hawk.Index[0], hawk.Index[1], hawk.Index[2]) == (shape, dtype, tag)
    assert repr(hawk.Scalar) == "TensorType(shape=(), dtype='f64', tag=None)"
