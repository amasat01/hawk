# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""What a group's scope may hold beyond scratch and reductions, run on the host against NumPy.

Every value is checked against NumPy written here: an output of a group is the same plane an output outside a group is.
"""

from __future__ import annotations

import math
import runpy

import numpy as np
import pytest

import hawk
import hawk.diff
from hawk import HawkError, Index, Matrix, Mutable, Scalar, Vector, _ir
from hawk import math as m
from hawk.artifact import build_bundle

RNG = np.random.default_rng(20261124)
BINS = 5


@hawk.kernel
def vector_out(x: Scalar, y: Mutable[Vector[3]]):
    with hawk.group(lanes=8) as g:
        y = m.vec([x, 2.0 * x, m.real(g.rank)])


@hawk.kernel
def matrix_out(x: Scalar, y: Mutable[Matrix[2, 2]]):
    with hawk.group(lanes=8) as g:
        y = m.outer(m.vec([x, 2.0 * x]), m.vec([m.real(g.rank), m.real(g.size)]))


@pytest.fixture(scope="module")
def wide(tmp_path_factory, cache_dir):
    """The kernels that commit a vector or a matrix per lane, built for the host once."""
    return build_bundle(
        [vector_out, matrix_out],
        tmp_path_factory.mktemp("group_wide"),
        targets=("host",),
        cache_dir=cache_dir,
    )


def test_a_vector_output_committed_in_a_group_is_one_vector_per_lane(wide):
    """Oracle: the three components as NumPy rows, from ``x`` and ``arange % 8`` (19 lanes: the last group has 3)."""
    n = 19
    x = RNG.normal(size=n)
    y = np.zeros((3, n))
    hawk.run(hawk.load(wide.directory, "vector_out"), x=x, y=y)
    np.testing.assert_array_equal(
        y, np.stack([x, 2.0 * x, (np.arange(n) % 8).astype(float)])
    )


def test_a_vector_output_of_a_group_is_written_as_an_ordinary_output_inside_the_phase_loop():
    from hawk.artifact import emit_source

    lines = [
        line.strip()
        for line in emit_source(vector_out, targets=("host",))
        .sources["host"]
        .text.splitlines()
    ]
    loop = lines.index(
        "for (std::int64_t hawk_rank = 0; hawk_rank < hawk_size; ++hawk_rank) {"
    )
    assert any(line.startswith("mut_y(2, i.global()) = ") for line in lines[loop:])
    assert not any("wout_y" in line for line in lines)


def test_a_matrix_output_committed_in_a_group_is_one_matrix_per_lane(wide):
    """Oracle: ``np.einsum`` of the lane's two vectors, from ``x``, ``arange % 8`` and the length of each group."""
    n = 19
    x = RNG.normal(size=n)
    y = np.zeros((4, n))
    hawk.run(hawk.load(wide.directory, "matrix_out"), x=x, y=y)
    size = np.repeat([8.0, 8.0, 3.0], [8, 8, 3])
    left, right = (
        np.stack([x, 2.0 * x]),
        np.stack([(np.arange(n) % 8).astype(float), size]),
    )
    want = np.einsum("il,jl->ijl", left, right).reshape(
        4, n
    )  # a matrix plane holds its entries row by row
    np.testing.assert_array_equal(y, want)


@hawk.kernel
def keyed_add(
    cell: Index, x: Scalar, acc: Mutable[Scalar, "bin":BINS], total: Mutable[Scalar]
):  # noqa: F821 axis labels
    with hawk.group(lanes=8) as g:
        hawk.scatter(acc, contract="plain")[cell] += x
        total = x * m.real(g.size)


@hawk.kernel
def keyed_store(
    cell: Index, x: Scalar, slot: Mutable[Scalar, "bin":32], total: Mutable[Scalar]
):  # noqa: F821 axis labels
    with hawk.group(lanes=8) as g:
        hawk.scatter(slot, conflict="any")[cell] = x
        total = x * m.real(g.size)


@hawk.kernel
def parity(x: Scalar, y: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        t = g.scratch("t", Scalar)
        t[g.rank] = x
        g.barrier()
        if g.rank % 2 == 0:
            y = x * 3.0
        else:
            y = -x


@hawk.kernel
def pick(x: Scalar, y: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        t = g.scratch("t", Scalar)
        t[g.rank] = x
        g.barrier()
        if g.rank % 2 == 0:
            y = t[g.rank]
        else:
            y = t.at(g.rank + 1, edge="wrap")


@hawk.kernel
def unrolled(x: Scalar, y: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        acc = 0.0 * x
        for j in range(3):
            acc = acc + x * m.real(j)
        y = acc + m.real(g.rank)


def _group(kernel):
    (group,) = kernel.sinks
    return group


def test_a_keyed_add_in_the_scope_is_a_sink_of_the_group_in_author_order():
    body = _group(keyed_add).field_nodes("body_sinks")
    assert [(_ir.is_a(k, "update"), k.field_text("name")) for k in body] == [
        (True, "acc"),
        (False, "total"),
    ]
    assert body[0].field_text("op") == "add"
    assert tuple(_group(keyed_add).field_ints("phases")) == (0, 0)


def test_a_keyed_store_with_a_conflict_rule_in_the_scope_is_a_sink_of_the_group():
    body = _group(keyed_store).field_nodes("body_sinks")
    (keyed, ordinary) = body
    assert (
        keyed.field_text("name"),
        keyed.field_text("op"),
        keyed.field_text("conflict"),
    ) == (
        "slot",
        "store",
        "any",
    )
    # the output beside it is the assign an output is outside a group
    assert (ordinary.field_text("name"), _ir.is_a(ordinary, "assign")) == (
        "total",
        True,
    )


@pytest.fixture(scope="module")
def bundle(tmp_path_factory, cache_dir):
    """The kernels that run, built for the host once."""
    return build_bundle(
        [keyed_add, keyed_store, keyed_add_compensated, appended],
        tmp_path_factory.mktemp("group_body"),
        targets=("host",),
        cache_dir=cache_dir,
    )


@pytest.fixture(scope="module")
def sample_local(tmp_path_factory, cache_dir):
    """The kernels that write only their own lane's cells (a bundle holds one kind of access), built for the host."""
    return build_bundle(
        [parity, unrolled, pick, counted, exits],
        tmp_path_factory.mktemp("group_body_local"),
        targets=("host",),
        cache_dir=cache_dir,
    )


def _run(bundle, name, **planes):
    hawk.run(hawk.load(bundle.directory, name), **planes)


def test_a_keyed_add_from_inside_a_group_equals_numpy_add_at_beside_the_group_sum(
    bundle,
):
    """Oracle: ``np.add.at`` of the values at their keys, and the value times the length of the lane's group (19 lanes: two full groups of 8 and one of 3)."""
    n = 19
    cell, x = RNG.integers(0, BINS, size=n).astype(np.int32), RNG.normal(size=n)
    acc, total = np.zeros(BINS), np.zeros(n)
    _run(bundle, "keyed_add", cell=cell, x=x, acc=acc, total=total)
    want = np.zeros(BINS)
    np.add.at(want, cell, x)
    np.testing.assert_allclose(acc, want, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(total, x * np.repeat([8.0, 8.0, 3.0], [8, 8, 3]))


def test_a_keyed_store_with_distinct_keys_from_inside_a_group_equals_numpy_assignment(
    bundle,
):
    """Oracle: ``slot[cell] = x`` of NumPy for distinct keys (a permutation)."""
    n = 19
    cell, x = RNG.permutation(32)[:n].astype(np.int32), RNG.normal(size=n)
    slot, total = np.zeros(32), np.zeros(n)
    _run(bundle, "keyed_store", cell=cell, x=x, slot=slot, total=total)
    want = np.zeros(32)
    want[cell] = x
    np.testing.assert_array_equal(slot, want)


@hawk.kernel
def keyed_add_compensated(
    cell: Index, x: Scalar, acc: Mutable[Scalar, "bin":BINS], total: Mutable[Scalar]
):  # noqa: F821 axis labels
    with hawk.group(lanes=8) as g:
        hawk.scatter(acc)[cell] += x
        total = x * m.real(g.size)


def test_a_compensated_keyed_add_in_a_group_equals_fsum_per_cell_beside_an_output(
    bundle,
):
    """Oracle: ``math.fsum`` of each cell's terms, on terms of wildly different magnitude that a plain sum rounds (19 lanes
    repeated over a few cells); the default contract is compensated and its companion plane is the library's."""
    n = 19
    cell = RNG.integers(0, BINS, size=n).astype(np.int32)
    x = RNG.normal(size=n) * 10.0 ** RNG.integers(-8, 9, size=n)
    acc, total = np.zeros(BINS), np.zeros(n)
    _run(bundle, "keyed_add_compensated", cell=cell, x=x, acc=acc, total=total)
    want = np.array([math.fsum(x[cell == c]) for c in range(BINS)])
    np.testing.assert_allclose(acc, want, rtol=1e-14, atol=1e-14 * np.abs(x).max())
    np.testing.assert_array_equal(total, x * np.repeat([8.0, 8.0, 3.0], [8, 8, 3]))


def test_the_companion_of_a_keyed_add_in_a_group_is_declared_with_its_zero_commit_beside_the_group():
    names = [
        k.field_text("name")
        for k in keyed_add_compensated.sinks
        if _ir.is_a(k, "update")
    ]
    assert names == ["acc_c"]
    assert len(keyed_add_compensated.sinks) == 2
    assert _ir.is_a(keyed_add_compensated.sinks[0], "group")


def test_an_if_in_the_scope_chooses_by_a_select_on_the_rank_of_the_lane(sample_local):
    """Oracle: ``np.where`` on the parity of ``arange % 8`` (19 lanes: the last group has 3)."""
    x = RNG.normal(size=19)
    y = np.zeros(19)
    _run(sample_local, "parity", x=x, y=y)
    np.testing.assert_array_equal(y, np.where(np.arange(19) % 8 % 2 == 0, 3.0 * x, -x))


def test_an_if_whose_arms_read_scratch_chooses_between_the_own_and_the_next_lane(
    sample_local,
):
    """Oracle: even ranks keep their own value, odd ranks take the next lane of their group (``np.roll`` of each slice, 19 lanes)."""
    x = RNG.normal(size=19)
    y = np.zeros(19)
    _run(sample_local, "pick", x=x, y=y)
    rolled = np.concatenate([np.roll(x[k : k + 8], -1) for k in range(0, 19, 8)])
    np.testing.assert_array_equal(y, np.where(np.arange(19) % 8 % 2 == 0, x, rolled))


def test_a_loop_inside_a_phase_runs_for_every_lane_inside_the_lane_loop(sample_local):
    """Oracle: the loop adds ``x * j`` for j = 0, 1, 2, so each lane holds ``3 x`` plus its rank (``arange % 8``)."""
    x = RNG.normal(size=19)
    y = np.zeros(19)
    _run(sample_local, "unrolled", x=x, y=y)
    np.testing.assert_allclose(y, 3.0 * x + np.arange(19) % 8, rtol=1e-15, atol=1e-15)


def test_the_branch_and_the_loop_stay_inside_one_phase_of_the_group():
    assert tuple(_group(parity).field_ints("phases")) == (0, 0, 1)
    assert tuple(_group(unrolled).field_ints("phases")) == (0,)


def test_the_output_of_a_branch_is_a_select_sink_of_the_group_body():
    (sink,) = _group(unrolled).field_nodes("body_sinks")
    assert (sink.field_text("name"), _ir.is_a(sink, "assign")) == ("y", True)
    (chosen,) = _group(parity).field_nodes("body_sinks")[2:]
    assert _ir.is_a(chosen.operands[0], "select")


@pytest.mark.parametrize(
    ("what", "stem"),
    [
        (
            "g.barrier()",
            r"g\.barrier\(\.\.\.\) inside an `if` or a `for`: a barrier ends a phase for every lane",
        ),
        (
            "y = g.sum(x)",
            r"g\.sum\(\.\.\.\) inside an `if` or a `for`: a barrier ends a phase for every lane",
        ),
        (
            "t2 = g.scratch('u', Scalar)",
            r"g\.scratch\(\.\.\.\) inside an `if` or a `for`",
        ),
        (
            "t[g.rank] = x",
            r"a store into the scratch plane 't' inside an `if` or a `for` would run for every lane",
        ),
    ],
    ids=["barrier", "reduction", "scratch declaration", "scratch store"],
)
@pytest.mark.parametrize("shape", ["if", "for"])
def test_what_every_lane_must_reach_is_refused_by_name_inside_an_if_or_a_for(
    tmp_path, what, stem, shape
):
    head = "if x > 0.0:" if shape == "if" else "for j in range(2):"
    source = (
        "import hawk\n"
        "from hawk import Mutable, Scalar\n"
        "@hawk.kernel\n"
        "def refused(x: Scalar, y: Mutable[Scalar]):\n"
        "    with hawk.group(lanes=8) as g:\n"
        "        t = g.scratch('t', Scalar)\n"
        "        y = x\n"
        f"        {head}\n"
        f"            {what}\n"
    )
    file = tmp_path / "refused.py"
    file.write_text(source)
    with pytest.raises(HawkError, match=stem):
        runpy.run_path(str(file))


@pytest.mark.parametrize("kernel", [keyed_add, parity], ids=lambda k: k.name)
@pytest.mark.parametrize(
    "entry",
    [
        hawk.diff.vjp,
        hawk.diff.jvp,
        hawk.diff.with_derivative,
        lambda k: hawk.diff.tape(k, 4),
        hawk.diff.step_vjp,
        hawk.diff.step_jvp,
        hawk.diff.step_function,
        hawk.diff.refine_entry,
    ],
    ids=[
        "vjp",
        "jvp",
        "with_derivative",
        "tape",
        "step_vjp",
        "step_jvp",
        "step_function",
        "refine_entry",
    ],
)
def test_a_kernel_with_a_group_has_no_derivative_yet_at_every_transform_entry(
    kernel, entry
):
    with pytest.raises(
        HawkError,
        match=rf"kernel '{kernel.name}' holds a group; a kernel with a group has no derivative yet",
    ):
        entry(kernel)


@hawk.kernel
def appended(x: Scalar, out: Mutable[Scalar, "item":64], y: Mutable[Scalar]):  # noqa: F821 axis labels
    with hawk.group(lanes=8) as g:
        hawk.append(out, value=x, when=x > 0.0)
        y = x * m.real(g.size)


@hawk.kernel
def counted(x: Scalar, n: Index, y: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        acc = 0.0 * x
        for k in range(n, max=5):
            acc = acc + x
        y = acc + 0.0 * m.real(g.rank)


@hawk.kernel
def exits(x: Scalar, n: Index, y: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        acc = 0.0 * x
        for k in range(5):
            if m.lnot(k < n):
                break
            acc = acc + x
        y = acc + 0.0 * m.real(g.rank)


def test_an_append_in_the_scope_is_one_group_with_the_count_the_list_and_the_output_as_its_sinks():
    names = [k.field_text("name") for k in _group(appended).field_nodes("body_sinks")]
    assert names == ["out_count", "out", "y"]
    assert [_ir.is_a(s, "group") for s in appended.sinks].count(True) == 1


def test_a_loop_whose_trips_are_run_time_reads_back_after_the_loop_in_the_same_phase():
    assert tuple(_group(counted).field_ints("phases")) == (0,)
    assert tuple(_group(exits).field_ints("phases")) == (0,)


@pytest.mark.parametrize("head", ["for k in range(n, max=3):", "for k in range(3):"])
@pytest.mark.parametrize(
    ("what", "stem"),
    [
        ("g.barrier()", r"g\.barrier\(\.\.\.\) inside an `if` or a `for`"),
        (
            "t[g.rank] = x",
            r"a store into the scratch plane 't' inside an `if` or a `for`",
        ),
    ],
    ids=["barrier", "scratch store"],
)
def test_a_loop_that_holds_a_barrier_or_a_scratch_store_is_refused_by_name(
    tmp_path, head, what, stem
):
    file = tmp_path / "looped.py"
    file.write_text(
        "import hawk\n"
        "from hawk import Index, Mutable, Scalar\n"
        "@hawk.kernel\n"
        "def looped(x: Scalar, n: Index, y: Mutable[Scalar]):\n"
        "    with hawk.group(lanes=8) as g:\n"
        "        t = g.scratch('t', Scalar)\n"
        "        y = x\n"
        f"        {head}\n"
        f"            {what}\n"
    )
    with pytest.raises(HawkError, match=stem):
        runpy.run_path(str(file))


def test_an_append_in_the_scope_stores_the_positive_values_beside_an_output(bundle):
    """Oracle: the set ``x[x > 0]`` (no order is promised) and the value times the length of the lane's group."""
    x = RNG.normal(size=19)
    out, y = np.zeros(64), np.zeros(19)
    _run(bundle, "appended", x=x, out=out, y=y)
    np.testing.assert_array_equal(np.sort(out[: (x > 0).sum()]), np.sort(x[x > 0]))
    np.testing.assert_array_equal(y, x * np.repeat([8.0, 8.0, 3.0], [8, 8, 3]))


@pytest.mark.parametrize("name", ["counted", "exits"])
def test_a_loop_in_the_scope_runs_its_trips_and_reads_back_after_it(sample_local, name):
    """Oracle: ``x * n`` for ``n`` between 0 and 4 trips of an addition of ``x``."""
    x = RNG.uniform(1.0, 3.0, size=19)
    n = (np.arange(19) % 5).astype(np.int32)
    y = np.zeros(19)
    _run(sample_local, name, x=x, n=n, y=y)
    np.testing.assert_allclose(y, x * n, rtol=1e-14)
