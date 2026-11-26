# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``with hawk.group(...) as g:`` builds the library's group elements and states nothing the library computes.

Nothing runs here. The oracle of every equality is the stored document of the same kernel (a copy of the library's, under ``data/groups``) (loaded through the
binding, in a context of its own): the traced group and the stored one must have the same canonical digest, the same phases and
the same slots. Every refusal is the library's text (asserted by its stem) with the kernel's name in front, except the few
that state the form of the author's call.
"""

from __future__ import annotations

import pathlib

import pytest

import hawk
from hawk import Grid, HawkError, Index, Mutable, Scalar, Vector, _ir
from hawk import math as m

#: The reference group kernels, as the IR library stores them (copies of its `ring`, `tile` and `rows` documents in the dump format,
#: kept here so that no test reads another repository; a change of the IR's reference documents is made here by copying them again).
STORED = pathlib.Path(__file__).resolve().parent / "data" / "groups"


@hawk.kernel
def ring(x: Scalar, y: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        t = g.scratch("t", Scalar)
        t[g.rank] = x
        g.barrier()
        y = t[g.rank] + t.at(g.rank + 1, edge="wrap")


@hawk.kernel
def tile(x: Scalar, y: Mutable[Scalar]):
    with hawk.group(lanes=8) as g:
        t = g.scratch("t", Scalar)
        t[g.rank] = m.sqrt(x)
        own = t[g.rank]
        y = own / g.sum(own, order="tree")


@hawk.kernel
def rows(
    lane: Grid["row", "col":4],  # noqa: F821 axis labels
    x: Scalar,
    total: Mutable[Scalar],
    least: Mutable[Scalar],
    place: Mutable[Index],
):
    with hawk.group(over=("col",)) as g:
        total = g.sum(x, order="lane")
        least = g.min(x)
        place = g.rank


def _stored(name):
    """The stored document of the reference kernel ``name``, loaded into a context of its own: its group node."""
    context = _ir.new_context()
    return context.load((STORED / f"{name}.dump").read_text())[0]


def _digest(*sinks):
    return _ir.digest(_ir.cells_walk(list(sinks)))


def _group_of(kernel):
    (group,) = kernel.sinks
    return group


@pytest.mark.parametrize("kernel", [ring, tile, rows], ids=lambda k: k.name)
def test_a_kernel_authored_in_hawk_has_the_digest_of_the_stored_document(kernel):
    """Oracle: the library's stored document of the same kernel."""
    assert _digest(_group_of(kernel)) == _digest(_stored(kernel.name))


@pytest.mark.parametrize(
    ("kernel", "phases"),
    [(ring, (0, 0, 1)), (tile, (0, 1)), (rows, (1, 1, 1))],
    ids=lambda p: str(p),
)
def test_the_phases_of_a_traced_group_are_the_library_s(kernel, phases):
    """Oracle: the phases the library's document of the same kernel states (barriers and reductions end a phase)."""
    group = _group_of(kernel)
    assert (
        tuple(group.field_ints("phases"))
        == phases
        == tuple(_stored(kernel.name).field_ints("phases"))
    )


@pytest.mark.parametrize(
    ("kernel", "slots"),
    [
        (ring, (("mutable", "y"), ("plane_in", "x"))),
        (tile, (("mutable", "y"), ("plane_in", "x"))),
        (
            rows,
            (
                ("mutable", "least"),
                ("mutable", "place"),
                ("mutable", "total"),
                ("plane_in", "x"),
            ),
        ),
    ],
    ids=lambda p: p.name if hasattr(p, "name") else "slots",
)
def test_the_slots_of_a_traced_group_are_its_planes_and_no_scratch(kernel, slots):
    """Oracle: the slots the library's walk of the stored document lists."""
    stored = _ir.cells_walk([_stored(kernel.name)])
    assert kernel.arg_spec == slots == _ir.arg_spec(stored)


def test_the_membership_of_each_form_is_stated_to_the_library_as_written():
    assert (
        _group_of(ring).field_text("membership"),
        _group_of(ring).field_int("count"),
    ) == ("count", 8)
    assert _group_of(rows).field_text("membership") == "axes"
    assert tuple(_group_of(rows).field_pairs("grid")) == (("row", None), ("col", 4))
    assert "('axes', ('col',))" in _ir.dump_text([_group_of(rows)])


def test_a_group_by_key_with_a_bound_declares_scratch_of_that_size():
    @hawk.kernel
    def by_key(cell: Index, x: Scalar, y: Mutable[Scalar]):
        with hawk.group(key=cell, max_size=16) as g:
            t = g.scratch("t", Scalar)
            t[g.rank] = x
            g.barrier()
            y = t[g.rank]

    group = _group_of(by_key)
    assert (group.field_text("membership"), group.field_int("max_size")) == ("key", 16)
    assert group.field_nodes("key")[0] is not None


def test_a_group_by_key_without_scratch_needs_no_bound():
    @hawk.kernel
    def by_key(cell: Index, x: Scalar, y: Mutable[Scalar]):
        with hawk.group(key=cell) as g:
            y = g.sum(x)

    assert _group_of(by_key).field_text("membership") == "key"


def test_a_group_size_is_a_value_of_the_body():
    @hawk.kernel
    def sized(x: Scalar, y: Mutable[Scalar]):
        with hawk.group(lanes=8) as g:
            y = x / m.real(g.size)

    assert tuple(_group_of(sized).field_ints("phases")) == (0,)


def test_a_statement_outside_the_scope_stays_an_ordinary_sink_beside_the_group():
    @hawk.kernel
    def beside(x: Scalar, y: Mutable[Scalar], z: Mutable[Scalar]):
        with hawk.group(lanes=8) as g:
            y = g.sum(x)
        z = x + 1.0

    assert [_ir.is_a(s, "group") for s in beside.sinks] == [True, False]


# ---- two groups that differ in one fact are two identities ----


def _with_spare(kind, lanes):
    @hawk.kernel
    def spare(x: Scalar, y: Mutable[Scalar]):
        with hawk.group(lanes=lanes) as g:
            t = g.scratch("t", Scalar)
            g.scratch("spare", kind)
            t[g.rank] = x
            g.barrier()
            y = t[g.rank]

    return spare


@pytest.mark.parametrize(
    ("kind", "lanes"),
    [(Index, 8), (Vector[2], 8), (Scalar, 4)],
    ids=["dtype", "shape", "extent"],
)
def test_groups_that_differ_only_in_a_scratch_declaration_have_different_digests(
    kind, lanes
):
    """Oracle: the same group twice gives one digest, so a difference is the declaration's."""
    assert _digest(_group_of(_with_spare(Scalar, 8))) == _digest(
        _group_of(_with_spare(Scalar, 8))
    )
    assert _digest(_group_of(_with_spare(Scalar, 8))) != _digest(
        _group_of(_with_spare(kind, lanes))
    )


# ---- refusals ----


def test_a_neighbour_read_without_the_barrier_is_refused_by_the_library():
    with pytest.raises(
        HawkError,
        match=r"kernel 'ring': Group: scratch 't' is read at another lane's element in the phase that writes it",
    ):

        @hawk.kernel
        def ring(x: Scalar, y: Mutable[Scalar]):
            with hawk.group(lanes=8) as g:
                t = g.scratch("t", Scalar)
                t[g.rank] = x
                y = t[g.rank] + t.at(g.rank + 1, edge="wrap")


def test_a_scratch_used_after_its_scope_is_refused_by_name():
    with pytest.raises(
        HawkError, match=r"scratch 't' is read: its group's scope has ended"
    ):

        @hawk.kernel
        def after(x: Scalar, y: Mutable[Scalar]):
            with hawk.group(lanes=8) as g:
                t = g.scratch("t", Scalar)
                t[g.rank] = x
            y = t[0]


def test_a_value_read_in_the_scope_and_used_outside_it_is_refused_by_the_library():
    with pytest.raises(
        HawkError,
        match=r"kernel 'leaked': canonical\(\): GroupRank is reached outside a Group",
    ):

        @hawk.kernel
        def leaked(x: Scalar, y: Mutable[Scalar], z: Mutable[Scalar]):
            with hawk.group(lanes=8) as g:
                place = g.rank
                z = x
            y = place + z


def test_a_second_group_in_one_kernel_is_refused_by_the_library():
    with pytest.raises(
        HawkError, match=r"kernel 'twice': canonical\(\): a kernel holds one Group"
    ):

        @hawk.kernel
        def twice(x: Scalar, y: Mutable[Scalar], z: Mutable[Scalar]):
            with hawk.group(lanes=8) as g:
                y = g.sum(x)
            with hawk.group(lanes=4) as h:
                z = h.sum(x)


def test_a_lane_count_of_zero_is_refused_by_the_library():
    with pytest.raises(
        HawkError,
        match=r"kernel 'zero': Group: membership 'count' takes a positive whole number of lanes",
    ):

        @hawk.kernel
        def zero(x: Scalar, y: Mutable[Scalar]):
            with hawk.group(lanes=0) as g:
                y = g.sum(x)


def test_over_an_axis_the_kernel_does_not_have_is_refused_by_the_library():
    with pytest.raises(
        HawkError,
        match=r"membership 'axes' names the axis 'depth', which the lanes do not have",
    ):

        @hawk.kernel
        def wrong(lane: Grid["row", "col":4], x: Scalar, y: Mutable[Scalar]):  # noqa: F821 axis labels
            with hawk.group(over=("depth",)) as g:
                y = g.sum(x)


def test_a_key_group_with_scratch_and_no_bound_is_refused_by_the_library():
    with pytest.raises(
        HawkError, match=r"membership 'key' with scratch needs max_size"
    ):

        @hawk.kernel
        def unbounded(cell: Index, x: Scalar, y: Mutable[Scalar]):
            with hawk.group(key=cell) as g:
                t = g.scratch("t", Scalar)
                t[g.rank] = x
                y = t[g.rank]


@pytest.mark.parametrize(
    "stated",
    [
        {"lanes": 8, "over": ("col",)},
        {"lanes": 8, "key": 1},
        {"over": ("col",), "key": 1},
        {},
    ],
    ids=["lanes+over", "lanes+key", "over+key", "none"],
)
def test_not_exactly_one_membership_argument_is_refused_by_name(stated):
    with pytest.raises(
        HawkError, match=r"hawk.group takes exactly one of lanes=, over= or key="
    ):
        hawk.group(**stated)


def test_a_branch_inside_the_scope_is_refused_by_name():
    with pytest.raises(HawkError, match=r"g\.sum\(\.\.\.\) inside an `if` or a `for`"):

        @hawk.kernel
        def branching(x: Scalar, y: Mutable[Scalar]):
            with hawk.group(lanes=8) as g:
                if x > 0.0:
                    y = g.sum(x)
                else:
                    y = x


def test_an_output_stored_inside_and_outside_the_scope_is_refused_by_name():
    with pytest.raises(
        HawkError, match=r"stored inside a group's scope and outside it"
    ):

        @hawk.kernel
        def both(x: Scalar, y: Mutable[Scalar]):
            with hawk.group(lanes=8) as g:
                y = g.sum(x)
            y = y + 1.0


def test_a_kernel_with_a_count_group_is_built_for_the_host(tmp_path, cache_dir):
    """The host text of a group body is written: the build gives a bundle whose kernel loads."""
    bundle = hawk.artifact.build_bundle(
        [ring], tmp_path, targets=("host",), cache_dir=cache_dir
    )
    assert hawk.load(bundle.directory, "ring") is not None


def test_an_exclusive_scan_is_refused_by_name():
    with pytest.raises(
        HawkError,
        match=r"g\.scan\(exclusive=True\): only the inclusive running sum is built",
    ):

        @hawk.kernel
        def exclusive(x: Scalar, y: Mutable[Scalar]):
            with hawk.group(lanes=8) as g:
                y = g.scan(x, exclusive=True)
