# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``Table.at`` clamps every index to its axis: ``[0, extent)``, per axis, in the index's own integer type.

A read outside the table reads its edge element, so a branch that is not taken (both sides of an ``if`` are
evaluated) never reads outside the plane. An in-range read is the plain read it always was.
"""

from __future__ import annotations

import numpy as np
import pytest

import hawk
from hawk import Index, Mutable, Scalar, Table
from hawk import math as hm
from hawk.diff import jvp
from hawk.ir import PlaneExtent
from hawk.trace.kernel import Kernel

eagle = pytest.importorskip("eagle")

FAR = 1 << 40           # an index no table reaches: computed in 64 bits, clamped exactly


@hawk.kernel
def pick(table: Table[Scalar], where: Index, flag: Scalar, y: Mutable[Scalar]):
    if flag > 0.0:
        y = table.at(where)
    else:
        y = table.at(where - 1000) + table.at(where * 1024 * 1024 * 1024 * 1024)



@hawk.kernel
def window(table: Table[Scalar], where: Index, y: Mutable[Scalar]):
    y = (table.at(where - 4) + table.at(where - 3) + table.at(where - 2) + table.at(where - 1) + table.at(where)
         + table.at(where + 1))

def _device() -> bool:
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.fixture
def private_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HAWK_CACHE_DIR", str(tmp_path / "cache"))


def _pick(target: str, dtype: str, length: int, where: np.ndarray, flag: np.ndarray) -> tuple:
    table = (np.arange(length) % 1021).astype(dtype) + np.asarray(0.5, dtype)
    planes = {"table": table, "where": where.astype(np.int32), "flag": flag.astype(dtype),
              "y": np.zeros(where.size, dtype)}
    if target == "cuda":
        planes = {k: eagle.arrays.asarray(v, device="cuda:0") for k, v in planes.items()}
    eagle.deploy(pick, scalar_type=dtype, targets=(target,))(**planes)
    held = planes["y"]
    if not isinstance(held, np.ndarray):
        held = held.get()
        held = np.asarray(held.to("numpy") if hasattr(held, "to") else held)
    edge = table[0] + table[-1]          # where - 1000 below 0 and where * 2**40 past the end, both clamped
    return held, np.where(flag > 0, table[np.clip(where, 0, length - 1)], edge)


def _rows(target: str, dtype: str) -> None:
    rng = np.random.default_rng(2)
    length = 300
    where = rng.integers(1, length, 257)
    flag = rng.uniform(-1.0, 1.0, 257)
    got, want = _pick(target, dtype, length, where, flag)
    np.testing.assert_array_equal(got, want)
    # an index past 2**24 in f32 mode reads exactly its element: the clamp never goes through a real
    big = (1 << 24) + 10
    got, want = _pick(target, dtype, big, np.array([big - 5, big - 7, big - 9]), np.ones(3))
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_reads_outside_a_table_in_a_branch_not_taken_read_its_edge_on_the_host(dtype, private_cache):
    _rows("host", dtype)


@pytest.mark.gpu
@pytest.mark.skipif(not _device(), reason="needs a CUDA device")
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_reads_outside_a_table_in_a_branch_not_taken_read_its_edge_on_the_device(dtype, private_cache):
    _rows("cuda", dtype)



def _window(target: str) -> None:
    """Reads at one runtime base and static offsets: ONE range test picks the plain reads, a lane near or past an
    edge takes the clamped reads. Both give exactly the clipped sum, in the same order."""
    length = 40
    table = np.random.default_rng(3).uniform(-1.0, 1.0, length)
    where = np.arange(-8, length + 8).astype(np.int32)           # in-range lanes and lanes past both edges
    planes = {"table": table, "where": where, "y": np.zeros(where.size)}
    if target == "cuda":
        planes = {k: eagle.arrays.asarray(v, device="cuda:0") for k, v in planes.items()}
    eagle.deploy(window, scalar_type="float64", targets=(target,))(**planes)
    held = planes["y"]
    held = held if isinstance(held, np.ndarray) else np.asarray(held.get())
    reads = [table[np.clip(where + k, 0, length - 1)] for k in (-4, -3, -2, -1, 0, 1)]
    want = reads[0]
    for r in reads[1:]:
        want = want + r
    np.testing.assert_array_equal(held, want)


def test_reads_at_one_base_and_static_offsets_share_one_range_test_on_the_host(private_cache):
    from hawk.emit.render import render_body

    text = render_body(window.sinks, window.walk).text
    assert text.count(">= static_cast<Int>(4)) & (") == 6 == text.count("- static_cast<Int>(1)), 1)")    # one span: -4 .. 1
    assert text.count("int_clamp_index<Int>") == 6                               # the out-of-range reads keep it
    _window("host")


@pytest.mark.gpu
@pytest.mark.skipif(not _device(), reason="needs a CUDA device")
def test_reads_at_one_base_and_static_offsets_share_one_range_test_on_the_device(private_cache):
    _window("cuda")

def test_the_clamp_is_integer_and_a_bounded_literal_is_read_as_it_is():
    from hawk.emit.render import render_body
    from hawk.ext import Kind  # noqa: F401 - the default kind renders the body

    text = render_body(pick.sinks, pick.walk).text
    assert "hawk_abi::int_clamp_index<Int>" in text and "fmin" not in text and "fmax" not in text

    @hawk.kernel
    def grid(t: Table["row":5, "col":2147483651], r: Index, y: Mutable[Scalar]):     # noqa: F821
        y = t.at(row=3, col=r)

    body = render_body(grid.sinks, grid.walk).text
    assert "2147483651" in body                 # the static extent past 2**31, exact in the index type
    assert "static_cast<Int>(3)" in body and body.count("int_clamp_index") == 1     # row 3 < 5: no clamp there


def test_the_derivative_reads_through_the_same_clamp():
    tangent = Kernel("pick_jvp", jvp(pick))
    assert any(isinstance(n, PlaneExtent) for n in tangent.walk.order)


def test_the_hawk_runtime_binds_a_lookup_with_its_own_length(tmp_path, private_cache):
    """``hawk.run`` packs a table's own element count, so the clamp reads the table's last entry, not the run's."""
    from conftest import sidecar_of

    from hawk import runtime as rt
    from hawk.artifact import build_bundle

    bundle = build_bundle([pick], tmp_path / "b", targets=("host",))
    art = rt.load(bundle.directory, pick.name, sidecar_of(bundle, pick.name))
    table = np.arange(300.0) + 0.5
    y = np.zeros(4)
    rt.run(art, table=table, where=np.array([1, 2, 3, 299], np.int32), flag=-np.ones(4), y=y)
    np.testing.assert_array_equal(y, np.full(4, table[0] + table[-1]))


@hawk.kernel
def member_lane(x: Scalar, theta: hawk.Wide[3], y: Mutable[Scalar]):
    y = x + theta.at(edge=hm.sample_index(), component=2)


def _member_lane(target: str, n: int) -> tuple:
    x = np.arange(n, dtype="float64") * 0.25
    theta = np.arange(3 * n, dtype="float64") + 0.5      # member m's lanes at theta[3m : 3m + 3]
    planes = {"x": x, "theta": theta, "y": np.zeros(n)}
    if target == "cuda":
        planes = {k: eagle.arrays.asarray(v, device="cuda:0") for k, v in planes.items()}
    eagle.deploy(member_lane, scalar_type="float64", targets=(target,))(**planes)
    got = planes["y"]
    return (got.get() if hasattr(got, "get") else np.asarray(got)), x + theta.reshape(n, 3)[:, 2]


def test_a_wide_plane_read_at_the_member_index_reaches_every_member_on_the_host(private_cache):
    # the clamp's bound is the plane's element count (n * W), not its sample count: every member past n / W read
    # its own lanes, not the clamped edge
    got, want = _member_lane("host", 512)
    np.testing.assert_array_equal(got, want)


@pytest.mark.skipif(not _device(), reason="needs a CUDA device")
def test_a_wide_plane_read_at_the_member_index_reaches_every_member_on_the_device(private_cache):
    got, want = _member_lane("cuda", 512)
    np.testing.assert_array_equal(got, want)


def test_the_hawk_runtime_reads_a_wide_plane_at_the_member_index_for_every_member(tmp_path, private_cache):
    """A ``Wide[W]`` plane read at ``(edge=sample_index(), component=c)`` through ``hawk.run``: the clamp's bound is
    the plane's element count, so every member reads its own lanes (members past ``n / W`` once read the edge)."""
    from conftest import sidecar_of

    from hawk import runtime as rt
    from hawk.artifact import build_bundle

    bundle = build_bundle([member_lane], tmp_path / "b", targets=("host",))
    art = rt.load(bundle.directory, member_lane.name, sidecar_of(bundle, member_lane.name))
    n = 512
    x = np.arange(n, dtype="float64") * 0.25
    theta = np.arange(3 * n, dtype="float64") + 0.5
    y = np.zeros(n)
    rt.run(art, x=x, theta=theta, y=y)
    np.testing.assert_array_equal(y, x + theta.reshape(n, 3)[:, 2])
