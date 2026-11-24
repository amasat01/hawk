"""An ordered sum over any lane axes of the Grid has the values of the NumPy sum over those axes.

The lanes of one key are found from the strides of the kept and the summed axes, so the summed axes need not be the trailing ones;
each key adds its lanes in lane order. Oracle for every row: ``x.reshape(grid).sum(axis=summed axes)``."""

import numpy as np
import pytest

import hawk
import hawk.math as hm
from hawk import Grid, Matrix, Scalar, Vector, kernel

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
    return np.asarray(make(over)(x)).ravel(), x.reshape(shape).sum(
        axis=tuple(
            AXES.index(a) for a in (over if isinstance(over, tuple) else (over,))
        )
    ).ravel()


@pytest.mark.parametrize("over", ["a", "b", ("a", "b"), ("a", "c")])
def test_an_ordered_sum_over_axes_that_are_not_the_trailing_ones_has_the_right_values(
    over,
):
    got, want = values(three_axes, over, (2, 3, 4))
    np.testing.assert_allclose(got, want)


def test_an_ordered_sum_over_a_leading_axis_of_two_has_the_right_values_and_the_kept_axis_is_the_output():
    """Restates the property of the refusal text: the kept axis ``b`` is the one the result runs along, 4 values."""
    got, want = values(two_axes, "a", (3, 4))
    assert got.shape == (4,)
    np.testing.assert_allclose(got, want)


def test_an_ordered_mean_over_a_leading_axis_is_the_numpy_mean():
    @kernel
    def average(lane: Grid["a":3, "b":4], x: Scalar) -> Scalar:
        return hawk.mean(x, over="a", contract="ordered")

    x = np.arange(12.0)
    np.testing.assert_allclose(
        np.asarray(average(x)).ravel(), x.reshape(3, 4).mean(axis=0)
    )


@pytest.mark.parametrize("over", ["c", ("b", "c")])
def test_an_ordered_sum_over_the_trailing_axes_is_built_and_right(over):
    got, want = values(three_axes, over, (2, 3, 4))
    np.testing.assert_allclose(got, want)


def test_an_ordered_sum_over_every_axis_is_not_refused():
    three_axes(("a", "b", "c"))


def test_an_ordered_sum_over_a_leading_axis_has_the_right_values():
    """Oracle: NumPy's sum over the leading axis of the (3, 4) grid."""
    got, want = values(two_axes, "a", (3, 4))
    np.testing.assert_allclose(got, want)


def test_the_sidecar_states_the_axes_the_library_states_for_a_slot_in_the_form_of_a_declared_sink(
    tmp_path, cache_dir, monkeypatch
):
    """The unit record of the lowering answer names axes by slot, label and extent (-1: a runtime one); the sidecar holds them as a
    declared sink's axes (payload, then ``{label, extent}`` each, ``None`` for a runtime extent), and a declared sink keeps its own."""
    import copy

    import hawk.artifact.bundle as bundle_module
    from hawk import Mutable
    from hawk.artifact import build_bundle

    @kernel
    def stated(x: Scalar, y: Mutable[Scalar]):
        y = x + 1.0

    real = bundle_module.sidecar_of_unit

    def with_stated_axes(k, unit, **rest):
        unit = copy.deepcopy(
            unit
        )  # the compiled unit stays as the library made it: only what hawk reads is stated here
        slot = [s["name"] for s in unit["meta"]["body"]["slots"]].index("y")
        unit["meta"]["launch"]["plane_axes"] = [
            {"slot": slot, "labels": ["a", "b"], "extents": [2, -1]}
        ]
        return real(k, unit, **rest)

    monkeypatch.setattr(bundle_module, "sidecar_of_unit", with_stated_axes)
    built = build_bundle([stated], tmp_path, targets=("host",), cache_dir=cache_dir)
    axes = built.artifacts[0].sidecar["plane_axes"]
    assert axes == {
        "y": {
            "payload": [],
            "axes": [{"label": "a", "extent": 2}, {"label": "b", "extent": None}],
        }
    }


def left_to_right(terms, axis):
    """The ordered sum of ``terms`` along ``axis``: the slices added one after the other in index order, the order the
    ordered contract promises (an independent loop, not a library reduction)."""
    slices = np.moveaxis(terms, axis, 0)
    total = np.zeros_like(slices[0])
    for each in slices:
        total = total + each
    return total


@kernel
def vector_over_leading(lane: Grid["a":2, "b":3, "c":4], j: Vector[3]) -> Vector[3]:
    return hawk.sum(j, over="a", contract="ordered")


@kernel
def vector_over_middle(lane: Grid["a":2, "b":3, "c":4], j: Vector[3]) -> Vector[3]:
    return hawk.sum(j, over="b", contract="ordered")


@kernel
def matrix_over_leading(lane: Grid["a":2, "b":3, "c":4], j: Vector[3]) -> Matrix[3, 3]:
    return hawk.sum(hm.outer(j, j), over="a", contract="ordered")


@kernel
def matrix_over_middle(lane: Grid["a":2, "b":3, "c":4], j: Vector[3]) -> Matrix[3, 3]:
    return hawk.sum(hm.outer(j, j), over="b", contract="ordered")


@pytest.mark.parametrize(
    ("make", "axis"), [(vector_over_leading, 0), (vector_over_middle, 1)]
)
def test_an_ordered_sum_of_a_vector_over_a_leading_or_a_middle_axis_adds_the_lanes_in_order(
    make, axis
):
    """Oracle: the planes' lanes added left to right along the summed axis of the (2, 3, 4) grid."""
    j = np.random.default_rng(3).standard_normal((3, 24))
    want = left_to_right(j.reshape(3, 2, 3, 4), axis + 1)
    np.testing.assert_array_equal(
        np.asarray(make(j)).reshape(3, -1), want.reshape(3, -1)
    )


@pytest.mark.parametrize(
    ("make", "axis"), [(matrix_over_leading, 0), (matrix_over_middle, 1)]
)
def test_an_ordered_sum_of_an_outer_product_over_a_leading_or_a_middle_axis_adds_the_lanes_in_order(
    make, axis
):
    """Oracle: ``j j^T`` of every lane, added left to right along the summed axis of the (2, 3, 4) grid."""
    j = np.random.default_rng(5).standard_normal((3, 24))
    terms = np.einsum("pl,ql->pql", j, j).reshape(3, 3, 2, 3, 4)
    want = left_to_right(terms, axis + 2).reshape(9, -1)
    got = np.zeros_like(want)
    make(j, out=got)
    np.testing.assert_array_equal(got, want)
