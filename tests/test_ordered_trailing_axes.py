# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``contract="ordered"`` over some lane axes sums over the TRAILING axes only. Over other axes it is refused by name when
the kernel is traced; the sums that work keep their values (NumPy is the reference), on the host and on a card."""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import Grid, HawkError, Mutable, Scalar, kernel
from hawk.artifact import build_bundle

A, B = 4, 8
TARGETS = ["host", pytest.param("cuda", marks=pytest.mark.gpu)]


@pytest.mark.parametrize("op", ["sum", "mean"])
def test_an_ordered_reduction_over_a_leading_axis_is_refused_by_name(op):
    with pytest.raises(HawkError, match=r"is built only over the trailing axes of the Grid"):
        @kernel
        def leading(lane: Grid["a":A, "b":B], x: Scalar, y: Mutable["b":B]):
            y = getattr(hawk, op)(x, over="a", contract="ordered")


def test_an_ordered_sum_over_a_middle_axis_is_refused_and_over_the_last_two_is_not():
    with pytest.raises(HawkError, match=r"over=\('b'\)|over='b'"):
        @kernel
        def middle(lane: Grid["a", "b":A, "c":B], x: Scalar, y: Mutable["a"]):
            y = hawk.sum(x, over="b", contract="ordered")

    @kernel
    def last_two(lane: Grid["a", "b":A, "c":B], x: Scalar, y: Mutable["a"]):
        y = hawk.sum(x, over=("b", "c"), contract="ordered")

    assert last_two.sinks


@kernel
def ordered_trailing(lane: Grid["a", "b":B], x: Scalar, y: Mutable["a"]):
    y = hawk.sum(x, over="b", contract="ordered")


@kernel
def compensated_leading(lane: Grid["a":A, "b":B], x: Scalar, y: Mutable["b":B]):
    y = hawk.sum(x, over="a", contract="compensated")


@kernel
def auto_leading(lane: Grid["a":A, "b":B], x: Scalar, y: Mutable["b":B]):
    y = hawk.sum(x, over="a")


def _run(bundle, name, target, ins, outs, **extent):
    """The ``outs`` planes after one launch of ``name``: through hawk's plan on the host, eagle's staged plan on the device
    (an ordered sum runs in stages)."""
    if target == "host":
        planes = {k: np.asarray(v) for k, v in {**ins, **outs}.items()}
        hawk.plan(hawk.load(bundle.directory, name)).bind(**extent, **planes).launch()
        return {k: planes[k] for k in outs}
    import eagle
    import eagle.exec as eexec
    from eagle import _split, arrays
    from eagle import plan as eplan
    from eagle._alloc import synchronize

    from hawk.artifact import plugins

    by = plugins(bundle, device_loader=eagle.registry.load_manifest)
    plan = eplan.plan(_split.linked(by[name], "test_ordered_trailing_axes"), structure=eexec.DeviceKernel)
    bound = {k: arrays.asarray(v, device="cuda:0") for k, v in {**ins, **outs}.items()}
    plan.bind(**extent, **bound).launch()
    synchronize()
    return {k: np.asarray(bound[k].get()) for k in outs}


@pytest.mark.parametrize("target", TARGETS)
def test_the_ordered_sum_over_the_trailing_axis_and_the_other_policies_over_a_leading_axis_equal_numpy(target, tmp_path, cache_dir):
    def bundle(k):
        return build_bundle([k], tmp_path / k.name, mode="float64", targets=(target,), cache_dir=cache_dir, host_profile="native")

    x = np.random.default_rng(7).uniform(-2.0, 2.0, A * B)
    grid = x.reshape(A, B)
    rows = _run(bundle(ordered_trailing), "ordered_trailing", target, {"x": x}, {"y": np.zeros(A)}, a=A)["y"]
    np.testing.assert_allclose(rows, grid.sum(axis=1), rtol=1e-13, atol=1e-13)
    for k in (compensated_leading, auto_leading):
        cols = _run(bundle(k), k.name, target, {"x": x}, {"y": np.zeros(B)})["y"]
        np.testing.assert_allclose(cols, grid.sum(axis=0), rtol=1e-13, atol=1e-13, err_msg=k.name)
