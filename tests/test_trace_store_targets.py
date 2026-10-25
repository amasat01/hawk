# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Which store targets a kernel body may write through.

A subscript's index is a read: ``acc[M.index(v, 5)] = x`` stores into ``acc``
and is not an attribute store; ``acc.field = x`` is."""

from __future__ import annotations

import ast

import pytest

from hawk.ir import HawkError
from hawk.trace.astpass import _check_targets


def _check(line: str) -> None:
    _check_targets(ast.parse(line).body[0], "<test>")


@pytest.mark.parametrize("line", [
    "acc[M.index(v, 5)] = x",
    "acc[k.real] = x",
    "acc[M.index(v, 5)] += x",
    "a, acc[M.index(v, 5)] = x",
])
def test_an_attribute_in_a_subscripts_index_is_a_read(line):
    _check(line)


@pytest.mark.parametrize("line", ["acc.field = x", "acc[0].field = x", "a, acc.field = x"])
def test_an_attribute_store_is_refused(line):
    with pytest.raises(HawkError, match="attribute store"):
        _check(line)
