# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``t[e, a]``: a named-axis ``Table``/``Staged``/``Wide`` read by position.

One index per declared axis, in declared order, beside ``t.at(e=..., a=...)``:
the same node, so the emitted text is identical. ``t[i]`` stays the flat read. A
wrong number of indices is refused naming the axes. The per-lane planes are read
at another lane with ``.at`` only."""

from __future__ import annotations

import hashlib

import pytest

import hawk
from hawk import Grid, HawkError, Scalar, Staged, Table, Wide
from hawk.emit import FLOAT64, render_source
from hawk.targets import TARGETS


def _md5(kernel, row: str) -> str:
    source = render_source(kernel.name, kernel.sinks, kernel.walk, TARGETS[row], mode=FLOAT64)
    return hashlib.md5(source.text.encode()).hexdigest()


def _named():
    @hawk.kernel
    def k(lane: Grid["e", "a":4], t: Table["e":8, "a":4], s: Staged[hawk.Vector[3], 16],
          w: Wide[2]) -> Scalar:
        return (t.at(e=lane.e, a=lane.a) + s.at(plane=1, sample=lane.e)
                + w.at(edge=lane.e, component=1))

    return k


def _positional():
    @hawk.kernel
    def k(lane: Grid["e", "a":4], t: Table["e":8, "a":4], s: Staged[hawk.Vector[3], 16],
          w: Wide[2]) -> Scalar:
        return t[lane.e, lane.a] + s[1, lane.e] + w[lane.e, 1]

    return k


@pytest.mark.parametrize("row", ["cuda", "host"])
def test_the_bracket_emits_the_text_of_at(row):
    assert _md5(_positional(), row) == _md5(_named(), row)


def _reads(read):
    """A kernel named ``k`` returning ``read(lane, t)`` (the name is part of the text)."""
    @hawk.kernel
    def k(lane: Grid["e", "a":4], t: Table["e":8, "a":4]) -> Scalar:
        return read(lane, t)

    return k


def test_the_bracket_takes_the_axes_in_declared_order():
    swapped = _reads(lambda lane, t: t[lane.a, lane.e])
    named = _reads(lambda lane, t: t.at(e=lane.a, a=lane.e))
    assert _md5(swapped, "host") == _md5(named, "host")
    assert _md5(swapped, "host") != _md5(_reads(lambda lane, t: t[lane.e, lane.a]), "host")


def test_a_single_index_stays_the_flat_read():
    bracket = _reads(lambda lane, t: t[lane.flat])
    flat = _reads(lambda lane, t: t.at(lane.flat))
    assert _md5(bracket, "host") == _md5(flat, "host")


@pytest.mark.parametrize("index,why", [
    ("lane.e", None),
    ("lane.e, lane.a, 0", r"'t': t\[\.\.\.\] takes one index per declared axis \['e', 'a'\], got 3"),
    ("(lane.e, lane.a, 0, 1)", r"got 4"),
])
def test_the_wrong_arity_is_refused_naming_the_axes(index, why):
    src = (
        "import hawk\nfrom hawk import Grid, Scalar, Table\n"
        "@hawk.kernel\n"
        "def k(lane: Grid['e', 'a':4], t: Table['e':8, 'a':4]) -> Scalar:\n"
        f"    return t[{index}]\n")
    if why is None:
        _exec(src)                                 # one index: the flat read, accepted
        return
    with pytest.raises(HawkError, match=why):
        _exec(src)


def test_a_flat_table_refuses_a_bracket_of_two():
    src = (
        "import hawk\nfrom hawk import Grid, Scalar, Table\n"
        "@hawk.kernel\n"
        "def k(lane: Grid['i'], t: Table[Scalar]) -> Scalar:\n"
        "    return t[lane.i, 1]\n")
    with pytest.raises(HawkError, match=r"one index per declared axis \(none: this plane is "
                                        r"flat, read t\[i\]\), got 2"):
        _exec(src)


def _exec(src: str):
    from raptor import flow

    ns: dict = {}
    flow.exec_source(src, ns)
    return ns
