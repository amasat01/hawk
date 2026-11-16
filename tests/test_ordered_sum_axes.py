"""An ordered sum over some lane axes is built only over the trailing axes of the Grid; any other choice is refused by name.

The ordered path does not yet carry WHICH axes are kept into the emitted text, so over a leading axis it returned the sum over the
last axis (DEFECT_ORDERED_KEY). Until it is implemented the build is refused. The last test is the REMINDER, red by ruling: it states
the right values and fails on the refusal; it turns green when the ordered sum over any axes is implemented. Do not skip or xfail it."""
import numpy as np
import pytest

import hawk
from hawk import Grid, Scalar, kernel
from hawk._errors import HawkError

AXES = ("a", "b", "c")


def two_axes(over):
    @kernel
    def two(lane: Grid["a":3, "b":4], x: Scalar) -> Scalar:
        return hawk.sum(x, over=over, contract="ordered")

    return two


def three_axes(over):
    @kernel
    def three(lane: Grid["a":2, "b":3, "c":4], x: Scalar) -> Scalar:
        return hawk.sum(x, over=over, contract="ordered")

    return three


def values(make, over, shape):
    x = np.arange(float(np.prod(shape)))
    return np.asarray(make(over)(x)).ravel(), x.reshape(shape).sum(axis=tuple(AXES.index(a) for a in (over if isinstance(over, tuple) else (over,)))).ravel()


def test_the_refusal_names_the_kernel_the_axes_asked_the_ones_built_and_the_ways_out():
    with pytest.raises(HawkError) as e:
        values(two_axes, "a", (3, 4))
    text = str(e.value)
    assert "'two'" in text and "['a']" in text and "['b']" in text and 'contract="auto"' in text and "summed axes come last" in text


@pytest.mark.parametrize("over", ["a", "b", ("a", "b"), ("a", "c")])
def test_an_ordered_sum_over_axes_that_are_not_the_trailing_ones_is_refused_by_name(over):
    with pytest.raises(HawkError, match="trailing axes of the Grid"):
        values(three_axes, over, (2, 3, 4))


@pytest.mark.parametrize("over", ["c", ("b", "c")])
def test_an_ordered_sum_over_the_trailing_axes_is_built_and_right(over):
    got, want = values(three_axes, over, (2, 3, 4))
    np.testing.assert_allclose(got, want)


def test_an_ordered_sum_over_every_axis_is_not_refused():
    three_axes(("a", "b", "c"))


def test_an_ordered_sum_over_a_leading_axis_has_the_right_values():
    """RED BY RULING, the reminder: implement an ordered sum over any axes; until then the build is refused."""
    got, want = values(two_axes, "a", (3, 4))
    np.testing.assert_allclose(got, want)
