# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``@raw_device`` without ``access=`` is refused at trace time.

Spliced text is opaque to the walk, so its access class cannot be inferred from
the DAG's forms; an unannotated block would therefore be declared
``sample_local`` by default and, under a partitioning that class does not
permit, return a wrong answer rather than a placement cost. The
refusal fires where the author is -- at the declaration -- and a second time in
the classifier for any raw op that reaches it without one. The last two tests
keep the annotation from being decorative: the declared class actually folds
into the walk's inferred class, and an opaque block cannot be
differentiated.

This test previously failed when ``raw_device``'s signature
defaulted to ``access="sample_local"`` instead of refusing -- the unannotated
block traced, walked and classified ``sample_local``.
"""

from __future__ import annotations

from _specs import default_kind
from _specs import spec_for
import pytest

from hawk import Scalar, raw_device
from hawk.diff import vjp
from hawk import _ir
from hawk import HawkError

TEXT = "acc[i] += rho * pos[i];"
F64 = _ir.TensorType((), "f64")
CTX = _ir.new_context()


def _x():
    return _ir.leaf(CTX, "vocab_read", "per_sample", "x", F64)


def _out(value):
    return (_ir.make_node("Assign", F64, [value], name="out"),)


def test_raw_device_without_access_is_refused_at_trace_time():
    with pytest.raises(HawkError, match=r"access="):
        raw_device(TEXT)


def test_raw_device_with_an_unknown_access_class_is_refused():
    with pytest.raises(HawkError, match="not an access class"):
        raw_device(TEXT, access="whenever")


def test_raw_device_with_an_explicit_class_is_accepted():
    block = raw_device(TEXT, access="cross_sample_write", reads=("pos",),
                       writes=("acc",))
    assert block.access == "cross_sample_write"
    assert block.reads == ("pos",) and block.writes == ("acc",)


def test_the_decorator_spelling_adopts_the_stub_name():
    @raw_device(TEXT, access="sample_local", returns=Scalar)
    def wobble():
        """The `@raw_device(...)` spelling: a stub naming the block."""

    assert wobble.name == "wobble"
    assert wobble(_x()).node.field_text("kind") == _ir.RAW_KIND


def test_the_declared_class_folds_into_the_inferred_one():
    block = raw_device(TEXT, access="cross_sample_write", returns=Scalar)
    walk = _ir.canonical(_out(block(_x()).node))
    assert _ir.access_class(walk) == "cross_sample_write", (
        "a raw block's DECLARED class folds into the walk's inferred "
        "class -- otherwise the annotation is decorative"
    )


def test_an_unannotated_raw_op_reaching_the_classifier_is_refused():
    opaque = _ir.make_node("Op", F64, [_x()], kind=_ir.RAW_KIND, literal="just some text")
    with pytest.raises(HawkError, match="explicit access="):
        _ir.canonical(_out(opaque))


def test_a_raw_block_cannot_be_differentiated():
    block = raw_device(TEXT, access="sample_local", returns=Scalar)
    sinks = _out(block(_x()).node)
    with pytest.raises(HawkError, match="raw_device"):
        vjp(sinks)


def test_a_raw_block_is_written_in_aether_unless_it_says_otherwise():
    assert raw_device(TEXT, access="sample_local").language == "aether"
    assert raw_device(TEXT, access="sample_local", language="c").language == "c"


