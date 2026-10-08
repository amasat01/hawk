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

import pytest

from hawk import Scalar, raw_device
from hawk.diff import vjp
from hawk.ir import Assign, HawkError, Leaf, Op, canonical
from hawk.ir.nodes import RAW_KIND
from hawk.types import TensorType

TEXT = "acc[i] += rho * pos[i];"
F64 = TensorType((), "f64")


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
    x = Leaf("vocab_read", "per_sample", "x", F64)
    assert wobble(x).node.kind == RAW_KIND


def test_the_declared_class_folds_into_the_inferred_one():
    block = raw_device(TEXT, access="cross_sample_write", returns=Scalar)
    x = Leaf("vocab_read", "per_sample", "x", F64)
    value = block(x)
    walk = canonical((Assign("out", value.node, F64),))
    assert walk.access.cls == "cross_sample_write", (
        "a raw block's DECLARED class folds into the walk's inferred "
        "class -- otherwise the annotation is decorative"
    )


def test_an_unannotated_raw_op_reaching_the_classifier_is_refused():
    x = Leaf("vocab_read", "per_sample", "x", F64)
    opaque = Op(RAW_KIND, (x,), F64, literal="just some text")
    with pytest.raises(HawkError, match="explicit access="):
        canonical((Assign("out", opaque, F64),))


def test_a_raw_block_cannot_be_differentiated():
    block = raw_device(TEXT, access="sample_local", returns=Scalar)
    x = Leaf("vocab_read", "per_sample", "x", F64)
    sinks = (Assign("out", block(x).node, F64),)
    with pytest.raises(HawkError, match="raw_device"):
        vjp(sinks)
