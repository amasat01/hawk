# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Tests hawk's own refusals: cases where a plausible wrong answer would be
worse than raising, so each is a loud, named error a caller can pin on.

A ``banded`` build that reached an op with no certified banded row would
report a precision the artifact does not carry, which is the one failure a
banded consumer cannot detect from the answer, so those ops are refused by
name. A ``max`` over samples is NOT refused: it reverses through its winning
lane (the lowest lane at a tie), and its row here pins that it does.
"""

from __future__ import annotations

import pytest

import hawk
import hawk.math as m
from hawk.diff import vjp
from hawk.ext import sink_policy
from hawk import HawkError

ROWS = 4


@hawk.kernel
def row_max(scores: hawk.Table["row":ROWS], peak: hawk.Mutable[hawk.Scalar]):
    """softmax's first pass as a max over samples: the row's largest score."""
    peak = hawk.max(scores.at(row=m.sample_index()))


@hawk.kernel
def row_sum(scores: hawk.Table["row":ROWS], total: hawk.Mutable[hawk.Scalar]):
    """The pass that DOES carry the gradient — the control beside the refusal."""
    total = hawk.sum(scores.at(row=m.sample_index()))


@hawk.kernel
def plain(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    y = x * 2.0


# --------------------------------------------------------------------------- #
# (a) a max over samples reverses through its winner, as a sum reverses to a broadcast.
# --------------------------------------------------------------------------- #

def test_the_reverse_of_a_max_reduction_derives_through_its_winner():
    """The adjoint of a ``max`` over samples goes to the winning lane alone, so
    the reverse kernel exists and writes the adjoint of the table it read (the
    values are pinned against NumPy in test_reduce_index.py)."""
    reverse = hawk.Kernel("row_max_reverse", vjp(row_max))
    assert [s.field_text("name") for s in reverse.sinks] == ["bar_scores"]


def test_the_sum_pass_beside_it_still_derives():
    """The control beside the max: a rank-0 ``sum`` reverses to a broadcast."""
    assert vjp(row_sum), "a rank-0 'sum' reverses to a broadcast"


# --------------------------------------------------------------------------- #
# (a2) The retired Accum / Reduce / .add / .contribute / .write forms refuse,
# each naming its replacement.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name, needle", [
    ("Accum", "hawk.scatter(bins)[key] += term"),
    ("Reduce", "total = hawk.sum(term)"),
])
def test_a_retired_declaration_refuses_naming_its_replacement(name, needle):
    with pytest.raises(HawkError, match=f"{name} is gone") as excinfo:
        getattr(hawk, name)
    assert needle in str(excinfo.value)


def test_a_retired_add_refuses_naming_its_replacement():
    with pytest.raises(HawkError, match=r"'out'.*hawk.scatter\(out\)\[key\] \+= v"):
        @hawk.kernel
        def bad(x: hawk.Scalar, lane: hawk.Index, out: hawk.Mutable[hawk.Scalar]):
            out.add(x, at=lane)


def test_a_retired_write_refuses_naming_its_replacement():
    with pytest.raises(HawkError, match=r"'out'.*hawk.scatter\(out\)\[key\] = v"):
        @hawk.kernel
        def bad(x: hawk.Scalar, lane: hawk.Index, out: hawk.WideOut[hawk.Scalar]):
            out.write(x, at=lane)


def test_a_retired_contribute_refuses_naming_its_replacement():
    with pytest.raises(HawkError, match=r"'out'.*out = hawk.sum\(v\)"):
        @hawk.kernel
        def bad(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
            out.contribute(x)


# --------------------------------------------------------------------------- #
# (b) The banded scalar mode refuses BY NAME what it does not build yet.
# --------------------------------------------------------------------------- #


@hawk.kernel
def scattered(x: hawk.Scalar, lane: hawk.Index, acc: hawk.Mutable[hawk.Scalar]):
    hawk.scatter(acc, contract="plain")[lane] += x * x


def test_the_banded_SINK_policy_is_the_same_named_slot_on_its_own_seam():
    """ carries the same unbuilt ``banded`` slot, and it refuses the same
    way: a seam value is never silently absent from the vocabulary."""
    with pytest.raises(HawkError) as excinfo:
        sink_policy("banded")
    message = str(excinfo.value)
    assert "emulated double precision" in message and "not built" in message, message
    assert "aether/banded/" in message


def test_an_unknown_mode_still_shows_banded():
    """Non-vacuity for the row above: the vocabulary a refusal prints must
    include ``banded``, or "named rather than absent" means nothing."""
    with pytest.raises(HawkError, match="banded"):
        sink_policy("float128")
