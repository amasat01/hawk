# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A missing fold threshold is refused by name (the wing contract), never read as "nothing exceeds it"."""
import pytest
from raptor.ir.fold import exceeds, rolled
from raptor.ir.nodes import Leaf, Op
from hawk import HawkError
from raptor.ir.types import TensorType


def test_a_missing_threshold_is_refused_by_name_where_it_is_read():
    with pytest.raises(HawkError, match="form.rolled_fold_min"):
        exceeds(10, None)
    assert exceeds(10, 4) and not exceeds(4, 4)


def test_the_static_and_rolled_forms_read_no_threshold():
    v = Leaf("vocab_read", "per_sample", "v", TensorType((8,), "f64"))
    node = Op("dot", (v, v), TensorType((), "f64"))
    assert rolled(node, None, "rolled") and not rolled(node, None, "static")
    with pytest.raises(HawkError, match="fold threshold"):
        rolled(node, None, "auto")
