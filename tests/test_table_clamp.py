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
