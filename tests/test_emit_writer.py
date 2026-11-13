# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The spelling seam: one renderer, N writers.

``aether.wings.base.writer.Writer`` names every text the body renderer produces;
``aether.wings.base.writer_aether.AetherWriter()`` is today's aether C++ spelling moved
behind it. These rows pin that the protocol is complete and that the
renderer really delegates (a writer that spells declarations differently
changes the body), while the aether writer's own text stays what it was.
"""

from __future__ import annotations
import pytest
from _specs import HOST, default_kind
from _specs import aether_writer

import re

from aether.wings.base.writer import WRITER_MEMBERS, Writer


def test_the_writer_protocol_names_every_spelling_seam():
    expected = {"declare", "carry_type", "tape_decl", "counter_decl", "for_header",
                "tail_break", "sample_index_value", "plane_extent_value", "hoist", "selector_decl", "mix_count",
                "predicated_rank0", "predicated_tensor", "switch_default_tail", "typed_zero",
                "leaf", "const", "at", "select", "op", "guard", "store", "scatter_store",
                "accum_scattered", "or_scattered", "store_compensated", "sample_sum", "tape_write",
                "finish", "exit_if", "component_commit", "in_place_accumulate", "zero_row"}
    assert set(WRITER_MEMBERS) == expected
    assert len(WRITER_MEMBERS) == 33
    assert all(callable(getattr(Writer, m)) for m in WRITER_MEMBERS)
    assert "id" in Writer.__annotations__
    assert "names_every_tensor" in Writer.__annotations__
    assert "commits_per_component" in Writer.__annotations__


# -- the seam is live -------------------------------------------------------

import _kernels as K  # noqa: E402

from aether.wings.base.render import render_body  # noqa: E402
from aether.wings.base.writer_aether import AetherWriter  # noqa: E402


class _Probe(AetherWriter):
    """The aether writer with ONE spelling changed: declarations drop ``const``."""

    id = "probe"

    def declare(self, name, ttype, text, ordered=True, held=False):
        return f"auto {name} = {text};"


def test_a_writer_that_spells_declarations_differently_changes_the_body():
    k = K.chain6
    plain = render_body(k.sinks, k.walk, spec=HOST, kind=default_kind(), writer=aether_writer()).text
    probed = render_body(k.sinks, k.walk, writer=_Probe(), spec=HOST, kind=default_kind()).text
    assert plain != probed
    assert "const auto " in plain and "const auto " not in probed
    assert re.sub(r"const (auto|aether::Item<[^>]*>) ", "auto ", plain) == probed


def test_the_aether_writer_implements_every_member_and_for_header_takes_the_renderer():
    import inspect

    assert all(callable(getattr(aether_writer(), m)) for m in WRITER_MEMBERS)
    # a runtime-count header is TWO statements for a C writer (the clamped
    # count, then the loop), so the member gets the renderer like `hoist` does
    assert list(inspect.signature(aether_writer().for_header).parameters)[:2] == ["r", "index"]


def test_param_cell_binding_still_re_exports_from_hawk_emit_render():
    from aether.wings.base.render import param_cell_binding

    assert param_cell_binding("k") == "lut_k_cell"


# -- a table read twice is never loaded at the sample index ----------------

import hawk  # noqa: E402
from hawk import Index, Mutable, Scalar, Table, Vector  # noqa: E402


@hawk.kernel
def _two_reads(table: Table[Scalar], where: Index, y: Mutable[Scalar]):
    y = table.at(where) + table.at(where + 1)


def test_a_table_read_twice_is_not_loaded_at_the_sample_index():
    """A ``lookup`` plane is read BY NAME at computed indices (``_at``); naming
    the leaf for its fan-out would emit a dead ``lut_table[i].eval()`` at the
    sample index, an out-of-bounds read on a table shorter than the batch."""
    text = render_body(_two_reads.sinks, _two_reads.walk, spec=HOST, kind=default_kind(), writer=aether_writer()).text
    assert "lut_table_i" not in text
    assert text.count("lut_table[aether::SampleIndex::make(") == 2


@hawk.kernel
def _vector_read_once(table: Table[Vector[3]], where: Index, y: Mutable[Vector[3]]):
    y = table.at(where)


def test_a_rank1_table_read_once_is_not_loaded_at_the_sample_index():
    """A rank>=1 leaf is named for correctness, but a ``lookup`` plane is read
    by name at its computed index (``_at``), so even ONE read of a
    ``Table[Vector[3]]`` emits no dead ``lut_table_i = lut_table[i].get()``."""
    text = render_body(_vector_read_once.sinks, _vector_read_once.walk, spec=HOST, kind=default_kind(), writer=aether_writer()).text
    assert "lut_table_i" not in text
    assert "lut_table[i]" not in text
    assert text.count("lut_table[aether::SampleIndex::make(") == 1


def test_a_head_break_test_is_spelled_by_the_writer():
    """The ``if (cond) {`` that opens a head/middle-form ``break`` is writer
    text: a C writer must not wrap an already parenthesised ``(a == b)`` in a
    second pair (clang's ``-Wparentheses-equality`` under ``-Werror``)."""
    import inspect

    assert list(inspect.signature(aether_writer().exit_if).parameters) == ["cond"]
    assert aether_writer().exit_if("(a == b)") == "if ((a == b)) {"


def test_the_writer_refuses_a_missing_kind_by_name():
    from aether.wings.base.render import render_body
    from raptor.errors import RaptorError as HawkError

    k = K.chain6
    with pytest.raises(HawkError, match="needs the kernel kind"):
        render_body(k.sinks, k.walk, spec=HOST, kind=None, writer=aether_writer())


def test_the_renderer_refuses_a_missing_writer_by_name():
    from aether.wings.base.render import render_body
    from raptor.errors import RaptorError as HawkError

    k = K.chain6
    with pytest.raises(HawkError, match="needs the writer"):
        render_body(k.sinks, k.walk, spec=HOST, kind=default_kind(), writer=None)
