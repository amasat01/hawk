# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The spelling seam: one renderer, N dialects.

``hawk.emit.dialect.Dialect`` names every text the body renderer produces;
``hawk.emit.dialect_aether.AETHER`` is today's aether C++ spelling moved
behind it. These rows pin that the protocol is complete and that the
renderer really delegates (a dialect that spells declarations differently
changes the body), while the aether dialect's own text stays what it was.
"""

from __future__ import annotations

from hawk.emit.dialect import DIALECT_MEMBERS, Dialect


def test_the_dialect_protocol_names_every_spelling_seam():
    expected = {"declare", "carry_type", "tape_decl", "counter_decl", "for_header",
                "tail_break", "sample_index_value", "hoist", "selector_decl", "mix_count",
                "predicated_rank0", "predicated_tensor", "switch_default_tail", "typed_zero",
                "leaf", "const", "at", "select", "op", "guard", "store", "scatter_store",
                "accum_scattered", "store_compensated", "sample_sum", "tape_write",
                "finish", "exit_if"}
    assert set(DIALECT_MEMBERS) == expected
    assert len(DIALECT_MEMBERS) == 28
    assert all(callable(getattr(Dialect, m)) for m in DIALECT_MEMBERS)
    assert "id" in Dialect.__annotations__
    assert "names_every_tensor" in Dialect.__annotations__


# -- the seam is live -------------------------------------------------------

import _kernels as K  # noqa: E402

from hawk.emit import render_body  # noqa: E402
from hawk.emit.dialect_aether import AETHER, AetherDialect  # noqa: E402


class _Probe(AetherDialect):
    """The aether dialect with ONE spelling changed: declarations drop ``const``."""

    id = "probe"

    def declare(self, name, ttype, text):
        return f"auto {name} = {text};"


def test_render_body_defaults_to_the_aether_dialect():
    k = K.chain6
    assert render_body(k.sinks, k.walk).text == render_body(
        k.sinks, k.walk, dialect=AETHER).text


def test_a_dialect_that_spells_declarations_differently_changes_the_body():
    k = K.chain6
    plain = render_body(k.sinks, k.walk).text
    probed = render_body(k.sinks, k.walk, dialect=_Probe()).text
    assert plain != probed
    assert "const auto " in plain and "const auto " not in probed
    assert plain.replace("const auto ", "auto ") == probed


def test_the_aether_dialect_implements_every_member_and_for_header_takes_the_renderer():
    import inspect

    assert all(callable(getattr(AETHER, m)) for m in DIALECT_MEMBERS)
    # a runtime-count header is TWO statements for a C dialect (the clamped
    # count, then the loop), so the member gets the renderer like `hoist` does
    assert list(inspect.signature(AETHER.for_header).parameters)[:2] == ["r", "index"]


def test_param_cell_binding_still_re_exports_from_hawk_emit_aether():
    from hawk.emit.aether import param_cell_binding

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
    text = render_body(_two_reads.sinks, _two_reads.walk).text
    assert "lut_table_i" not in text
    assert text.count("lut_table[aether::SampleIndex::make(") == 2


@hawk.kernel
def _vector_read_once(table: Table[Vector[3]], where: Index, y: Mutable[Vector[3]]):
    y = table.at(where)


def test_a_rank1_table_read_once_is_not_loaded_at_the_sample_index():
    """A rank>=1 leaf is named for correctness, but a ``lookup`` plane is read
    by name at its computed index (``_at``), so even ONE read of a
    ``Table[Vector[3]]`` emits no dead ``lut_table_i = lut_table[i].get()``."""
    text = render_body(_vector_read_once.sinks, _vector_read_once.walk).text
    assert "lut_table_i" not in text
    assert "lut_table[i]" not in text
    assert text.count("lut_table[aether::SampleIndex::make(") == 1


def test_a_head_break_test_is_spelled_by_the_dialect():
    """The ``if (cond) {`` that opens a head/middle-form ``break`` is dialect
    text: a C dialect must not wrap an already parenthesised ``(a == b)`` in a
    second pair (clang's ``-Wparentheses-equality`` under ``-Werror``)."""
    import inspect

    assert list(inspect.signature(AETHER.exit_if).parameters) == ["cond"]
    assert AETHER.exit_if("(a == b)") == "if ((a == b)) {"
