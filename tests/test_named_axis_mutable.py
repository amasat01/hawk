# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``Mutable["bin":256]``: a sink with named axes.

The payload is the value written; the plane is bound ``(*payload, *axes)``,
written by key rather than once per lane. A sink you pass keeps its contents
(the adds land on top of them); one you omit is allocated at zero by the caller,
in the shape ``hawk._lane_bind.output_shape`` gives. A wrong bound shape is
refused naming the axes."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import Grid, HawkError, Mutable, Scalar, Vector
from hawk import _lane_bind as LB
from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.trace.decl import Plane

BINS = 16


@hawk.kernel
def hist(lane: Grid["i"], x: Scalar, bins: Mutable["bin":BINS]):
    hawk.scatter(bins)[m.index(x * BINS, BINS)] += 1.0


@hawk.kernel
def cells(lane: Grid["channel", "input":8], x: Scalar, tb: Mutable["channel":4, "input":8]):
    hawk.scatter(tb)[lane.channel * 8 + lane.input] += x


@pytest.fixture(scope="module")
def loaded(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("hawk_axis_sinks")
    out = {}
    for k in (hist, cells):
        bundle = build_bundle([k], pathlib.Path(root) / k.name, targets=("host",),
                              cache_dir=cache_dir, host_profile="native")
        out[k.name] = rt.load(bundle.directory, k.name, sidecar_of(bundle, k.name))
    return out


def _ref_hist(x):
    return np.bincount((x * BINS).astype(int), minlength=BINS).astype(float)


def test_the_declaration_names_the_axes_and_the_payload():
    one = Mutable["bin":256]
    assert (one.form, one.ttype, [(a, n) for a, n, _s in one.dims]) == (
        "mutable", Scalar, [("bin", 256)])
    two = Mutable["channel":64, "input":32]
    assert [(a, n, s) for a, n, s in two.dims] == [("channel", 64, 32), ("input", 32, 1)]
    vec = Mutable[Vector[3], "bin":4]
    assert (vec.ttype, [(a, n) for a, n, _s in vec.dims]) == (Vector[3], [("bin", 4)])
    assert Mutable[Scalar] == Plane("mutable", Scalar)         # unchanged


@pytest.mark.parametrize("spec,why", [
    (lambda: Mutable[Scalar, 3], r"named axis is spelled <label>:<extent>"),
    (lambda: Mutable["bin":0], r"positive compile-time int"),
    (lambda: Mutable["bin":4, "bin":4], r"duplicate axis label"),
    (lambda: Mutable[3], r"takes a payload type"),
])
def test_a_bad_axis_declaration_is_refused(spec, why):
    with pytest.raises(HawkError, match=why):
        spec()


def test_writing_a_sink_with_axes_per_lane_is_refused():
    with pytest.raises(HawkError, match=r"'bins' is declared with the axes \[\"bin\":4\].*"
                                        r"written by key"):
        @hawk.kernel
        def k(x: Scalar, bins: Mutable["bin":4]):
            bins = x


def test_the_sidecar_declares_the_bound_shape(loaded):
    axes = {"payload": [], "axes": [{"label": "bin", "extent": BINS}]}
    assert loaded["hist"].sidecar["plane_axes"] == {"bins": axes, "bins_c": axes}
    assert loaded["cells"].sidecar["plane_axes"]["tb"]["axes"] == [
        {"label": "channel", "extent": 4}, {"label": "input", "extent": 8}]


def test_an_omitted_sink_is_allocated_zero_and_filled(loaded):
    k = loaded["hist"]
    x = np.random.default_rng(5).random(1000)
    bins = np.zeros(LB.output_shape(k.sidecar, "bins", 1, len(x)))
    assert bins.shape == (BINS,)
    rt.run(k, x=x, bins=bins)
    np.testing.assert_array_equal(bins, _ref_hist(x))


def test_a_passed_sink_keeps_its_contents(loaded):
    k = loaded["hist"]
    x = np.random.default_rng(6).random(500)
    start = np.arange(BINS, dtype=float) * 100.0
    bins = start.copy()
    rt.run(k, x=x, bins=bins)
    np.testing.assert_array_equal(bins, start + _ref_hist(x))
    rt.run(k, x=x, bins=bins)
    np.testing.assert_array_equal(bins, start + 2 * _ref_hist(x))


def test_two_axes_bind_payload_then_axes(loaded):
    k = loaded["cells"]
    x = np.random.default_rng(7).random((4, 8))
    tb = np.zeros(LB.output_shape(k.sidecar, "tb", 1, 32))
    assert tb.shape == (4, 8)
    rt.run(k, x=x, tb=tb)
    np.testing.assert_array_equal(tb, x)


@pytest.mark.parametrize("name,shape,want", [
    ("hist", (BINS - 1,), r"'bins'.*axes \(bin=16\).*bound \(16,\) \(payload first\), got \(15,\)"),
    ("hist", (1, BINS), r"'bins'.*got \(1, 16\)"),
    ("cells", (8, 4), r"'tb'.*axes \(channel=4, input=8\).*bound \(4, 8\).*got \(8, 4\)"),
    ("cells", (32,), r"'tb'.*bound \(4, 8\).*got \(32,\)"),
])
def test_a_wrong_bound_shape_is_refused_naming_the_axes(loaded, name, shape, want):
    k = loaded[name]
    sink = "bins" if name == "hist" else "tb"
    kw = ({"x": np.zeros(8)} if name == "hist" else {"x": np.zeros((4, 8))})
    with pytest.raises(HawkError, match=want):
        rt.run(k, **kw, **{sink: np.zeros(shape)})
