# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A missing fold threshold is refused by name (the wing contract), never read as "nothing exceeds it"."""
import pytest
from hawk import HawkError, _ir


def test_a_missing_threshold_is_refused_by_name_where_it_is_read():
    with pytest.raises(HawkError, match="form.rolled_fold_min"):
        _ir.fold_exceeds(10, None)
    assert _ir.fold_exceeds(10, 4) and not _ir.fold_exceeds(4, 4)


def test_the_static_and_rolled_forms_read_no_threshold():
    v = _ir.leaf(_ir.new_context(), "vocab_read", "per_sample", "v", ((8,), "f64", None))
    node = _ir.make("dot", (v, v))
    assert _ir.fold_rolled(node, None, "rolled") and not _ir.fold_rolled(node, None, "static")
    with pytest.raises(HawkError, match="fold threshold"):
        _ir.fold_rolled(node, None, "auto")
