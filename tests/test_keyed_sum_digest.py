"""Two keyed sums that differ only in their key are two computations: they must not share a walk digest, as a plain reduction or as a block."""

import pytest

import hawk
from hawk import Grid, Scalar, kernel
from hawk import _ir


def make(over, contract):
    @kernel
    def k(lane: Grid["a":4, "b":4], x: Scalar) -> Scalar:
        return hawk.sum(x, over=over, contract=contract)

    return k


@pytest.mark.parametrize("contract", ["auto", "ordered"])
def test_keyed_sums_that_differ_only_in_their_key_have_two_digests(contract):
    assert _ir.digest(make("a", contract).walk) != _ir.digest(make("b", contract).walk)
