"""Two keyed sums that differ only in their key are two computations: they must not share a walk digest, as a plain reduction or as a block."""
import pytest

import hawk
from hawk import Grid, Scalar, kernel


def make(over, contract):
    @kernel
    def k(lane: Grid["a":4, "b":4], x: Scalar) -> Scalar:
        return hawk.sum(x, over=over, contract=contract)

    return k


# An ordered sum over a leading axis is refused by name until it is built (tests/test_ordered_sum_axes.py holds the red reminder):
# the strict mark turns this row red the day the refusal goes, so the row comes back with the fix.
@pytest.mark.parametrize("contract", ["auto", pytest.param("ordered", marks=pytest.mark.xfail(strict=True, raises=hawk.HawkError))])
def test_keyed_sums_that_differ_only_in_their_key_have_two_digests(contract):
    assert make("a", contract).walk.digest != make("b", contract).walk.digest
