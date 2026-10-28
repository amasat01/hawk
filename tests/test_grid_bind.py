# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Binding a ``Grid`` kernel on the host artifact (``hawk.runtime``).

The extent comes BY LABEL (``channel=64``) or from a lane-shaped plane, never as
a total: ``n_samples=`` is refused naming the label. A lane-shaped plane
``(64, 8192)`` (``(w, 64, 8192)`` for a ``Vector[w]``) is the flat plane seen
with the launch's shape, so it binds as a view; the static extents are the
kernel's and a shape that contradicts them is refused. A kernel with nothing per
lane to read the extent from is refused rather than sized by a Table's length.
Outputs of a ``Grid`` kernel are allocated lane-shaped (``output_shape``)."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import Grid, HawkError, Mutable, Scalar, Table, Vector
from hawk import _lane_bind as LB
from hawk import runtime as rt
from hawk.artifact import build_bundle

CHANNEL, SAMPLE = 64, 8192


@hawk.kernel
def scale(lane: Grid["channel", "sample":8192], x: Scalar, th: Table[Scalar]) -> Scalar:
    return x * th.at(lane.channel) + lane.sample


@hawk.kernel
def rows(lane: Grid["channel", "sample":128], va: Vector[64], vb: Vector[64], x: Scalar) -> Scalar:
    return x + va[3] + 2.0 * vb[5] + lane.channel


@hawk.kernel
def fixed(lane: Grid["a":4, "b":8], x: Scalar) -> Scalar:
    return x + lane.a * 10.0 + lane.b


@hawk.kernel
def fixed_count(lane: Grid["a":4, "b":8], th: Table[Scalar], acc: Mutable["bin":4]):
    hawk.scatter(acc)[lane.a] += th.at(lane.b)


@hawk.kernel
def cube_count(lane: Grid["a":2, "b":3, "c":4], th: Table[Scalar], acc: Mutable["bin":2]):
    hawk.scatter(acc)[lane.a] += th.at(lane.c)


@hawk.kernel
def tables_only(lane: Grid["channel", "s":4], th: Table[Scalar], acc: Mutable["bin":4]):
    hawk.scatter(acc)[lane.s] += th.at(lane.channel)


@hawk.kernel
def tables_flat(th: Table[Scalar], acc: Mutable["bin":4]):
    hawk.scatter(acc)[0] += th.at(hawk.math.sample_index())


@hawk.kernel
def flat(x: Scalar, th: Table[Scalar]) -> Scalar:
    return x * th.at(0)


@hawk.kernel
def lap(lane: Grid["row", "col":16], u: Scalar) -> Scalar:
    i, j = lane.row, lane.col
    if m.land(m.land(i > 0, j > 0), j < 15):
        out = u.at(row=i - 1, col=j) + u.at(i * 16 + j - 1) + u.at(row=i, col=j + 1) - 3.0 * u
    else:
        out = 0.0
    return out


KERNELS = (scale, rows, fixed, fixed_count, cube_count, tables_only, tables_flat, flat, lap)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory, cache_dir):
    root = tmp_path_factory.mktemp("hawk_grid_bind")
    out = {}
    for k in KERNELS:                                 # one execution axis per bundle
        bundle = build_bundle([k], pathlib.Path(root) / k.name, targets=("host",),
                              cache_dir=cache_dir, host_profile="native")
        out[k.name] = rt.load(bundle.directory, k.name, sidecar_of(bundle, k.name))
    return out


def _out(kernel, n, name="out", width=1):
    return np.zeros(LB.output_shape(kernel.sidecar, name, width, n))


def _scale_ref(x, th):
    return x * th[:, None] + np.arange(SAMPLE)[None, :]


@pytest.fixture(scope="module")
def planes():
    rng = np.random.default_rng(7)
    return rng.random((CHANNEL, SAMPLE)), np.arange(CHANNEL) + 1.0


def test_a_lane_shaped_plane_runs_and_returns_lane_shaped(loaded, planes):
    x, th = planes
    k = loaded["scale"]
    out = _out(k, CHANNEL * SAMPLE)
    assert out.shape == (CHANNEL, SAMPLE)
    rt.run(k, x=x, th=th, out=out)
    np.testing.assert_array_equal(out, _scale_ref(x, th))


def test_the_extent_by_label_runs_flat_planes(loaded, planes):
    x, th = planes
    k = loaded["scale"]
    out = np.zeros(CHANNEL * SAMPLE)
    rt.run(k, x=x.reshape(-1), th=th, out=out, channel=CHANNEL)
    np.testing.assert_array_equal(out.reshape(CHANNEL, SAMPLE), _scale_ref(x, th))


def test_a_smaller_runtime_extent_is_valid(loaded, planes):
    x, th = planes
    k = loaded["scale"]
    out = np.zeros((32, SAMPLE))
    rt.run(k, x=x[:32].copy(), th=th, out=out)
    np.testing.assert_array_equal(out, _scale_ref(x, th)[:32])


@pytest.mark.parametrize("shape", [(CHANNEL, 4096), (SAMPLE, CHANNEL)])
def test_a_shape_that_contradicts_the_declaration_is_refused(loaded, planes, shape):
    k = loaded["scale"]
    with pytest.raises(HawkError, match=rf"shape \({shape[0]}, {shape[1]}\).*contradicts the "
                                        r"declaration.*expected \(channel, 8192\)"):
        rt.run(k, x=np.zeros(shape), th=planes[1], out=np.zeros(shape))


def test_a_flat_plane_that_is_not_whole_rows_is_refused(loaded):
    k = loaded["scale"]
    with pytest.raises(HawkError, match=r"100 lanes do not fit Grid\[\"channel\", \"sample\":8192\]"):
        rt.run(k, x=np.zeros(100), th=np.zeros(4), out=np.zeros(100))


def test_the_label_and_a_lane_shaped_plane_must_agree(loaded, planes):
    x, th = planes
    with pytest.raises(HawkError, match=r"channel=32 gives 262144 lanes"):
        rt.run(loaded["scale"], x=x, th=th, out=np.zeros_like(x), channel=32)


@pytest.mark.parametrize("door", ["n_samples", "n"])
def test_a_total_is_refused_naming_the_label(loaded, planes, door):
    x, th = planes
    with pytest.raises(HawkError, match=r"takes its extent by label.*pass channel=<extent of "
                                        r"'channel'> instead of n=/n_samples="):
        rt.run(loaded["scale"], x=x, th=th, out=np.zeros_like(x), **{door: CHANNEL * SAMPLE})


def test_flat_kernels_keep_n_samples_as_the_total(loaded):
    k = loaded["flat"]
    out = np.zeros(10)
    rt.run(k, x=np.arange(10.0), th=np.array([3.0]), out=out, n_samples=10)
    np.testing.assert_array_equal(out, np.arange(10.0) * 3.0)


def test_only_the_runtime_axis_takes_an_extent(loaded, planes):
    x, th = planes
    with pytest.raises(HawkError, match=r"the axis 'sample' is static \(8192\); drop sample="):
        rt.run(loaded["scale"], x=x, th=th, out=np.zeros_like(x), sample=8192)


def test_a_vector_plane_takes_both_forms_in_one_call(loaded):
    k = loaded["rows"]
    rng = np.random.default_rng(3)
    n = 128
    va = rng.random((64, CHANNEL * n))                 # the flat (w, N) plane
    vb = rng.random((64, CHANNEL, n))                  # head + lane shape
    x = rng.random((CHANNEL, n))                       # the Scalar, lane-shaped
    out = _out(k, CHANNEL * n)
    assert out.shape == (CHANNEL, n)
    rt.run(k, va=va, vb=vb, x=x, out=out)
    want = x + va[3].reshape(CHANNEL, n) + 2.0 * vb[5] + np.arange(CHANNEL)[:, None]
    np.testing.assert_array_equal(out, want)


def test_a_static_kernel_needs_no_extent(loaded):
    k = loaded["fixed"]
    x = np.random.default_rng(1).random((4, 8))
    out = _out(k, 32)
    assert out.shape == (4, 8)
    rt.run(k, x=x, out=out)
    np.testing.assert_array_equal(out, x + np.arange(4)[:, None] * 10.0 + np.arange(8)[None, :])


def test_a_static_first_axis_runs_every_lane_with_no_plane_to_read_it_from(loaded):
    acc = np.zeros(4)
    rt.run(loaded["fixed_count"], th=np.arange(8.0), acc=acc)
    np.testing.assert_array_equal(acc, np.full(4, 28.0))          # 4 x 8 lanes, not the last 8


def test_an_all_static_cube_runs_every_lane(loaded):
    acc = np.zeros(2)
    rt.run(loaded["cube_count"], th=np.arange(4.0), acc=acc)
    np.testing.assert_array_equal(acc, np.full(2, 18.0))          # 2 x 3 x 4 lanes


def test_a_static_kernel_given_an_extent_is_refused(loaded):
    x = np.zeros((4, 8))
    with pytest.raises(HawkError, match=r"the axis 'a' is static \(4\); drop a="):
        rt.run(loaded["fixed"], x=x, out=np.zeros((4, 8)), a=4)
    with pytest.raises(HawkError, match=r"no extent \(every axis is static\)"):
        rt.run(loaded["fixed"], x=x, out=np.zeros((4, 8)), n_samples=32)


def test_a_static_kernel_with_the_wrong_shape_is_refused(loaded):
    with pytest.raises(HawkError, match=r"shape \(8, 4\).*expected \(4, 8\)"):
        rt.run(loaded["fixed"], x=np.zeros((8, 4)), out=np.zeros((8, 4)))


def test_a_tables_only_grid_kernel_without_an_extent_is_refused(loaded):
    k = loaded["tables_only"]
    th = np.arange(4.0)
    with pytest.raises(HawkError, match=r"binds no per-sample plane to read it from; pass "
                                        r"channel=<extent of 'channel'>.*never the lane count"):
        rt.run(k, th=th, acc=np.zeros(4))
    acc = np.zeros(4)
    rt.run(k, th=th, acc=acc, channel=4)               # the Table's length (4) is not the count
    np.testing.assert_array_equal(acc, np.full(4, th.sum()))
    acc = np.zeros(4)
    rt.run(k, th=np.arange(8.0), acc=acc, channel=2)   # two rows, whatever the Table holds
    np.testing.assert_array_equal(acc, np.full(4, 1.0))


def test_a_tables_only_flat_kernel_without_an_extent_is_refused(loaded):
    with pytest.raises(HawkError, match=r"no bound PER-SAMPLE plane.*pass n_samples"):
        rt.run(loaded["tables_flat"], th=np.arange(4.0), acc=np.zeros(4))
    acc = np.zeros(4)
    rt.run(loaded["tables_flat"], th=np.arange(4.0), acc=acc, n_samples=4)
    assert acc[0] == 6.0


def test_a_per_lane_plane_is_read_at_another_lane_by_axis_or_flat_index(loaded):
    k = loaded["lap"]
    u = np.random.default_rng(2).random((10, 16))
    out = np.zeros_like(u)
    rt.run(k, u=u, out=out)
    want = np.zeros_like(u)
    want[1:, 1:15] = u[:-1, 1:15] + u[1:, 0:14] + u[1:, 2:16] - 3.0 * u[1:, 1:15]
    np.testing.assert_array_equal(out, want)


def test_the_sidecar_carries_the_declared_shape(loaded):
    sc = loaded["scale"].sidecar
    assert sc["launch"]["lane_space"] == {
        "axes": [{"label": "channel", "extent": None}, {"label": "sample", "extent": SAMPLE}],
        "group": 1}


@pytest.mark.gpu
def test_a_static_first_axis_runs_every_lane_on_the_device(cache_dir):
    cp = pytest.importorskip("cupy")
    import eagle
    plan = eagle.deploy(fixed_count, cache_dir=cache_dir)
    acc = cp.zeros(4)
    plan.bind(th=cp.arange(8.0), acc=acc).launch()
    np.testing.assert_array_equal(cp.asnumpy(acc), np.full(4, 28.0))
