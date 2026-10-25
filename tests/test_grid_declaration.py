# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``Grid[...]`` declares the launch shape; ``lane.<label>``, ``lane[k]`` and
``lane.flat`` read it.

The reads are the SAME Index values ``split_index`` builds, so a ``Grid`` kernel's
emitted text equals the hand-split kernel's, byte for byte, on every wrapper. The
declaration refuses what it cannot honour (a bare label past the first, a reserved
or shadowing label, a ``group`` that names nothing) saying what to write instead.
``@hawk.kernel(group=...)`` stores the compaction group, a lane count."""

from __future__ import annotations

import hashlib

import pytest

import hawk
from hawk import Grid, HawkError, Mutable, Scalar
from hawk.emit import FLOAT64, render_source
from hawk.math import split_index
from hawk.targets import TARGETS

ROWS = ("cuda", "host")


def _md5(kernel, row: str) -> str:
    source = render_source(kernel.name, kernel.sinks, kernel.walk, TARGETS[row], mode=FLOAT64)
    return hashlib.md5(source.text.encode()).hexdigest()


def _by_label():
    @hawk.kernel
    def k(lane: Grid["channel", "sample":8], x: Scalar) -> Scalar:
        return x * lane.channel + lane.sample

    return k


def _by_position():
    @hawk.kernel
    def k(lane: Grid["channel", "sample":8], x: Scalar) -> Scalar:
        return x * lane[0] + lane[-1]

    return k


def _by_split():
    @hawk.kernel
    def k(x: Scalar) -> Scalar:
        channel, sample = split_index(8)
        return x * channel + sample

    return k


def _flat_grid():
    @hawk.kernel
    def k(lane: Grid["channel", "sample":8], x: Scalar) -> Scalar:
        return x * lane.flat

    return k


def _flat_index():
    @hawk.kernel
    def k(x: Scalar) -> Scalar:
        return x * hawk.math.sample_index()

    return k


def _three_axes():
    @hawk.kernel
    def k(lane: Grid["a", "b":4, "c":3], x: Scalar) -> Scalar:
        return x + lane.a * 100.0 + lane.b * 10.0 + lane.c

    return k


def _three_axes_by_hand():
    @hawk.kernel
    def k(x: Scalar) -> Scalar:
        lane = hawk.math.sample_index()
        a = lane / 12
        b = lane / 3 - a * 4
        c = lane - (lane / 3) * 3
        return x + a * 100.0 + b * 10.0 + c

    return k


@pytest.mark.parametrize("row", ROWS)
def test_label_and_position_emit_the_text_split_index_does(row):
    want = _md5(_by_split(), row)
    assert _md5(_by_label(), row) == want
    assert _md5(_by_position(), row) == want


@pytest.mark.parametrize("row", ROWS)
def test_lane_flat_is_the_sample_index(row):
    assert _md5(_flat_grid(), row) == _md5(_flat_index(), row)


@pytest.mark.parametrize("row", ROWS)
def test_three_axes_read_each_coordinate_as_the_hand_split(row):
    k = _three_axes()
    assert k.lane_space == (("a", None), ("b", 4), ("c", 3))
    assert _md5(k, row) == _md5(_three_axes_by_hand(), row)


def test_a_grid_kernel_has_the_lane_space_and_group_it_declared():
    k = _by_label()
    assert (k.lane_space, k.group) == ((("channel", None), ("sample", 8)), 1)

    @hawk.kernel
    def flat(x: Scalar) -> Scalar:
        return x + 1.0

    assert (flat.lane_space, flat.group) == (((None, None),), 1)


def test_grid_is_not_a_slot_of_the_kernel():
    assert _by_label().arg_spec == (("mutable", "out"), ("per_sample", "x"))


def test_the_last_declared_axis_is_the_fastest():
    doc = Grid.__doc__
    assert "LAST axis is the fastest" in doc


# -- refusals ---------------------------------------------------------------
@pytest.mark.parametrize("spec,why", [
    (lambda: Grid["a", "b"], r"'b' at position 1 has no extent.*\"b\":<extent>"),
    (lambda: Grid["a":8, "b"], r"'b' at position 1 has no extent"),
    (lambda: Grid["a", "a":4], r"duplicate axis label"),
    (lambda: Grid["flat", "b":4], r"'flat'.*reserved"),
    (lambda: Grid["n", "b":4], r"'n'.*reserved"),
    (lambda: Grid["class", "b":4], r"'class'.*identifier"),
    (lambda: Grid["my axis", "b":4], r"'my axis'.*identifier"),
    (lambda: Grid["a", "b":0], r"positive compile-time int"),
    (lambda: Grid["a", "b":2.5], r"positive compile-time int"),
    (lambda: Grid["a", 4], r"an axis is"),
    (lambda: Grid[()], r"at least one axis"),
])
def test_a_bad_declaration_is_refused_with_the_fix(spec, why):
    with pytest.raises(HawkError, match=why):
        spec()


def test_a_label_equal_to_a_parameter_name_is_refused():
    with pytest.raises(HawkError, match=r"\['x'\].*parameter names.*rename the label"):
        @hawk.kernel
        def k(lane: Grid["x", "s":4], x: Scalar) -> Scalar:
            return x + 1.0


def test_one_grid_per_kernel():
    with pytest.raises(HawkError, match=r"ONE launch shape"):
        @hawk.kernel
        def k(a: Grid["i"], b: Grid["j"], x: Scalar) -> Scalar:
            return x + 1.0


def test_a_label_the_grid_does_not_declare_is_refused_by_name():
    with pytest.raises(HawkError, match=r"lane\.nope.*\('channel', 'sample'\).*lane\.flat"):
        @hawk.kernel
        def k(lane: Grid["channel", "sample":8], x: Scalar) -> Scalar:
            return x * lane.nope


@pytest.mark.parametrize("index,why", [("5", r"has 2 axes"), ("'channel'", r"takes an int position")])
def test_a_bad_positional_read_is_refused(index, why):
    src = (
        "import hawk\nfrom hawk import Grid, Scalar\n"
        "@hawk.kernel\n"
        "def k(lane: Grid['channel', 'sample':8], x: Scalar) -> Scalar:\n"
        f"    return x * lane[{index}]\n")
    with pytest.raises(HawkError, match=why):
        _exec(src)


def _exec(src: str):
    from raptor import flow

    ns: dict = {}
    flow.exec_source(src, ns)
    return ns


# -- group ------------------------------------------------------------------
def _grouped(group):
    @hawk.kernel(group=group)
    def k(lane: Grid["channel", "sample":8, "mc":4], x: Scalar) -> Scalar:
        return x + 1.0

    return k


def test_group_is_a_lane_count_or_an_axis_label():
    assert _grouped(32).group == 32
    assert _grouped("mc").group == 4
    assert _grouped("sample").group == 32      # that axis and everything after it


@pytest.mark.parametrize("group,why", [
    (0, r"lane count \(an int >= 1\)"),
    (-4, r"lane count \(an int >= 1\)"),
    (True, r"lane count"),
    (2.5, r"lane count"),
    ("nope", r"names no axis.*declares \['channel', 'sample', 'mc'\]"),
    ("channel", r"includes the runtime axis 'channel'"),
])
def test_a_bad_group_is_refused(group, why):
    with pytest.raises(HawkError, match=why):
        _grouped(group)


def test_a_group_label_on_a_kernel_without_a_grid_is_refused():
    with pytest.raises(HawkError, match=r"names no axis; the kernel declares no Grid"):
        @hawk.kernel(group="sample")
        def k(x: Scalar) -> Scalar:
            return x + 1.0


def test_group_survives_the_default_step_kernels():
    @hawk.kernel(group=16)
    def step(x: Scalar, y: Mutable[Scalar]):
        y = x + 1.0

    assert step.group == 16


# -- .at on a per-lane plane --------------------------------------------------
@pytest.mark.parametrize("read,why", [
    ("u.at(row=1)", r"must name every Grid axis exactly once \(declared \['row', 'col'\]\); missing \['col'\]"),
    ("u.at(row=1, col=2, depth=3)", r"unknown \['depth'\]"),
    ("u.at(1, 2)", r"takes ONE flat lane index, got 2; by axis: at\(row=…, col=…\)"),
    ("u.at(1, col=2)", r"flat lane index OR every Grid axis by name, never both"),
])
def test_a_bad_read_at_another_lane_is_refused(read, why):
    src = (
        "import hawk\nfrom hawk import Grid, Scalar\n"
        "@hawk.kernel\n"
        "def k(lane: Grid['row', 'col':4], u: Scalar) -> Scalar:\n"
        f"    return {read}\n")
    with pytest.raises(HawkError, match=why):
        _exec(src)


def test_an_axis_read_needs_a_grid():
    src = (
        "import hawk\nfrom hawk import Scalar\n"
        "@hawk.kernel\n"
        "def k(u: Scalar) -> Scalar:\n"
        "    return u.at(row=1)\n")
    with pytest.raises(HawkError, match=r"names Grid axes, but the kernel declares no Grid"):
        _exec(src)


def test_the_axis_read_is_the_flat_read_of_its_index():
    def build(read):
        @hawk.kernel
        def k(lane: Grid["row", "col":4], u: Scalar) -> Scalar:
            return read(lane, u)

        return k

    by_axis = build(lambda lane, u: u.at(row=lane.row - 1, col=lane.col))
    by_flat = build(lambda lane, u: u.at((lane.row - 1) * 4 + lane.col))
    assert _md5(by_axis, "host") != ""
    assert by_axis.walk.access.cls == by_flat.walk.access.cls == "cross_sample_read"


def test_reading_an_input_keeps_a_comparison_in_the_order_it_was_written():
    from hawk.emit import render_body

    @hawk.kernel
    def k(a: Scalar, b: Scalar) -> Scalar:
        return hawk.math.where(a >= b, a, b)

    text = render_body(k.sinks, k.walk).text
    assert ">=" in text and "<=" not in text


def test_a_value_computed_in_the_launch_cannot_be_read_at_another_lane():
    src = (
        "import hawk\nfrom hawk import Grid, Scalar\n"
        "@hawk.kernel\n"
        "def k(lane: Grid['i'], x: Scalar) -> Scalar:\n"
        "    y = x * 2.0\n"
        "    return y.at(lane.i - 1)\n")
    with pytest.raises(HawkError, match=r"only a plane that existed before the launch can be "
                                        r"read at another lane"):
        _exec(src)
