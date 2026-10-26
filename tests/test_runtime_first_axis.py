# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A bare first label on ``Mutable``/``Table`` (``Mutable["node"]``) is a runtime
extent, as in ``Grid``: given as ``node=N`` or by the plane passed, one build for
every size. No stride depends on it, so it is never in the generated code."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from conftest import sidecar_of

import eagle
import hawk
from hawk import Grid, HawkError, Index, Mutable, Scalar, Table
from hawk import runtime as rt
from hawk.artifact import build_bundle, reset_unit_memo, unit_stats


@hawk.kernel
def count(lane: Grid["e", "a":3], conn: Table[Index], b: Mutable["node"]):
    hawk.scatter(b)[conn.at(lane.flat)] += 1.0


@hawk.kernel
def by_row(lane: Grid["e"], w: Table["elem", "c":3]) -> Scalar:
    return w.at(elem=lane.e, c=1) + w.at(elem=lane.e, c=2)


@hawk.kernel
def lane_sink(lane: Grid["e"], x: Scalar, b: Mutable["e"]):
    hawk.scatter(b)[lane.e] += x


def _conn(n):
    return np.arange(6, dtype=np.int32) % n


def _want(n):
    want = np.zeros(n)
    np.add.at(want, _conn(n), 1.0)
    return want


@pytest.fixture
def host(tmp_path, cache_dir):
    def load(kernel):
        bundle = build_bundle([kernel], pathlib.Path(tmp_path) / kernel.name, targets=("host",),
                              cache_dir=cache_dir, host_profile="native")
        return rt.load(bundle.directory, kernel.name, sidecar_of(bundle, kernel.name))
    return load


def test_a_bare_first_axis_declares_a_runtime_extent():
    assert Mutable["node"].dims == (("node", None, 1),)
    assert Table["elem", "c":3].dims == (("elem", None, 3), ("c", 3, 1))
    assert Mutable[hawk.Vector[2], "node", ].dims == (("node", None, 1),)


@pytest.mark.parametrize("bad", [lambda: Mutable["a":2, "b"], lambda: Table["a":2, "b"],
                                 lambda: Mutable["flat"]])
def test_a_bare_label_is_first_only_and_a_plain_name(bad):
    with pytest.raises(HawkError, match=r"only the FIRST axis may be bare|is reserved"):
        bad()


def test_one_unit_serves_every_size(tmp_path, cache_dir):
    def kernel(n):
        @hawk.kernel
        def counted(lane: Grid["e", "a":3], conn: Table[Index], b: Mutable["node"]):
            hawk.scatter(b)[conn.at(lane.flat)] += 1.0
        return counted

    reset_unit_memo()
    digests = {build_bundle([kernel(n)], tmp_path / f"b{n}", targets=("host",),
                            cache_dir=cache_dir).digest for n in (4, 9, 4)}
    assert len(digests) == 1 and unit_stats()["published"] == 1
    assert build_bundle([count], tmp_path / "c", targets=("host",),
                        cache_dir=cache_dir).artifacts[0].sidecar["plane_axes"]["b"]["axes"] \
        == [{"label": "node", "extent": None}]


@pytest.mark.parametrize("n", [4, 9])
def test_the_label_sizes_the_sink_and_the_plane_passed_agrees(n):
    deployed = eagle.deploy(count)
    by_label = deployed.run(conn=_conn(n), e=2, node=n)
    by_plane = deployed.run(conn=_conn(n), e=2, b=np.zeros(n))
    assert by_label.shape == (n,)
    np.testing.assert_array_equal(by_label, _want(n))
    np.testing.assert_array_equal(by_plane, _want(n))


def test_a_label_that_contradicts_the_plane_passed_is_refused(host):
    k = host(count)
    with pytest.raises(HawkError, match=r"declared with the axes \(node\).*got \(5,\)"):
        rt.run(k, conn=_conn(4), b=np.zeros(5), e=2, node=4)
    with pytest.raises(HawkError, match=r"node= is the extent.*positive int"):
        rt.run(k, conn=_conn(4), b=np.zeros(4), e=2, node=0)
    with pytest.raises(ValueError, match=r"pass node=<extent of 'node'>"):
        eagle.deploy(count).run(conn=_conn(4), e=2)


def test_a_table_first_axis_is_the_plane_passed():
    deployed = eagle.deploy(by_row)
    for rows in (3, 7):
        w = np.arange(rows * 3, dtype=float)
        got = deployed.run(w=w, e=rows)
        np.testing.assert_array_equal(got, w.reshape(rows, 3)[:, 1] + w.reshape(rows, 3)[:, 2])


def test_a_label_a_lane_axis_shares_is_the_lane_extent(host):
    assert lane_sink.planes["b"].dims == (("e", None, 1),)
    got = eagle.deploy(lane_sink).run(x=np.ones(5), e=5)
    np.testing.assert_array_equal(got, np.ones(5))


def test_a_label_that_is_also_a_parameter_name_is_refused():
    with pytest.raises(HawkError, match=r"also a parameter name"):
        @hawk.kernel
        def clash(lane: Grid["e"], node: Scalar, b: Mutable["node"]):
            hawk.scatter(b)[0] += node


def test_a_bound_plan_and_its_rebind_take_the_plane_as_its_extent():
    plan = eagle.deploy(count).bind(conn=_conn(9), e=2, b=np.zeros(9))
    plan.launch()
    np.testing.assert_array_equal(plan.planes["b"], _want(9))
    plan.rebind(b=np.zeros(9))


def test_a_direct_call_takes_the_free_label_as_a_keyword(tmp_path):
    import eagle

    conn = _conn(4)
    got = count(conn, e=2, node=4)
    np.testing.assert_array_equal(got, _want(4))
    np.testing.assert_array_equal(eagle.deploy(count).run(conn=conn, e=2, node=4), got)
    with pytest.raises(ValueError, match="node"):
        count(conn, b=np.zeros(5), e=2, node=4)
