# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Tests hawk's own refusals: two cases where a plausible wrong answer would
be worse than raising, so each is a loud, named error a caller can pin on.

Reverse-differentiating a ``max`` fold has an ambiguous subgradient at a
tie, so silently differentiating it would let a consumer's gradient be
wrong only on the ties; softmax's max pass stays FORWARD-ONLY and the
gradient rides the sum pass instead. A ``banded`` scalar mode that fell
back to ``float64`` would report a precision the artifact does not carry,
which is the one failure a SoftDouble consumer cannot detect from the
answer, so no shim is built for it yet.
"""

from __future__ import annotations

import pytest

import hawk
import hawk.math as m
from hawk.artifact import build_bundle
from hawk.diff import jvp, vjp
from hawk.emit import SCALAR_MODES, compose, render_source, scalar_mode
from hawk.emit.backend import _UNBUILT_MODES
from hawk.ext import sink_policy
from hawk.ir import HawkError

ROWS = 4


@hawk.kernel
def row_max(scores: hawk.Table["row":ROWS], peak: hawk.Reduce("max")):
    """softmax's first pass as a mapreduce sink: the row's largest score."""
    peak.contribute(scores.at(row=m.sample_index()))


@hawk.kernel
def row_sum(scores: hawk.Table["row":ROWS], total: hawk.Reduce("sum")):
    """The pass that DOES carry the gradient — the control beside the refusal."""
    total.contribute(scores.at(row=m.sample_index()))


@hawk.kernel
def plain(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    y = x * 2.0


# --------------------------------------------------------------------------- #
# (a) Reduce(max) has no reverse rule, and the refusal says which rule.
# --------------------------------------------------------------------------- #
def test_the_reverse_of_a_max_reduction_is_refused_naming_the_rule():
    """Needs the ``sink.op != "sum"`` guard in ``hawk/diff/transform._seed``,
    or a ``max`` partial is seeded with a broadcast adjoint exactly like a
    ``sum``::

        Failed: DID NOT RAISE <class 'hawk.ir.nodes.HawkError'>

    and ``vjp(row_max)`` returned a gradient — a number, for a fold whose
    subgradient is ambiguous at every tie."""
    with pytest.raises(HawkError) as excinfo:
        vjp(row_max)
    message = str(excinfo.value)
    for needle in ("mapreduce_partial", "op='max'", "'sum'", "subgradient",
                   "FORWARD-ONLY"):
        assert needle in message, (
            f"the refusal must name the rule that is missing and why; {needle!r} "
            f"is absent from: {message}")


def test_the_sum_pass_beside_it_still_derives():
    """Non-vacuity: the refusal is about the ``max`` FOLD, not about mapreduce
    sinks. A row that refused both would be an implementation gap wearing a
    principle's clothes."""
    assert vjp(row_sum), "a rank-0 'sum' partial reverses to a broadcast"
    assert jvp(row_max), (
        "forward mode is unaffected: a directional derivative of a max fold is "
        "its own value's, and softmax's max pass is FORWARD-only, not undefined")


def test_a_wide_max_reduction_is_refused_for_its_shape_too():
    """The rule HAWK defines is the RANK-0 'sum' one; both halves are named."""
    from hawk.ir import Leaf, MapreducePartial
    from hawk.types import TensorType

    v3 = TensorType((3,), "f64")
    wide = MapreducePartial("total", Leaf("vocab_read", "vec_in", "v", v3),
                            "sum", v3)
    with pytest.raises(HawkError, match="no reverse rule"):
        vjp((wide,))


# --------------------------------------------------------------------------- #
# (b) The banded scalar mode is seam NAMED unbuilt slot.
# --------------------------------------------------------------------------- #
def test_the_banded_scalar_mode_is_refused_naming_the_seam_and_the_wave():
    """``_UNBUILT_MODES`` given a ``"banded"`` entry pointing at
    ``FLOAT64`` (a shim), so a softdouble consumer got a float64 artifact that
    called itself banded::

        Failed: DID NOT RAISE <class 'hawk.ir.nodes.HawkError'>

    which is the one failure a SoftDouble consumer cannot see in the answer."""
    assert "banded" in _UNBUILT_MODES and "banded" not in SCALAR_MODES
    with pytest.raises(HawkError) as excinfo:
        scalar_mode("banded")
    message = str(excinfo.value)
    for needle in ("declared but not built yet", "emulated double precision",
                   "aether/banded/", "no softdouble shim"):
        assert needle in message, (
            f"the banded refusal must name what it is and where it lands; "
            f"{needle!r} is absent from: {message}")


@pytest.mark.parametrize("door", ("build_bundle", "render_source", "compose"))
def test_every_door_into_the_scalar_mode_refuses_banded(tmp_path, door):
    """One seam, and every entrance to it. A mode resolved in three places is a
    mode that can be built in two of them."""
    from hawk.emit import BACKENDS

    if door == "build_bundle":
        with pytest.raises(HawkError, match="declared but not built yet"):
            build_bundle([plain], tmp_path / "banded", mode="banded",
                         targets=("host",))
    elif door == "render_source":
        with pytest.raises(HawkError, match="declared but not built yet"):
            render_source("plain", plain.sinks, plain.walk, BACKENDS["host"],
                          mode=scalar_mode("banded"))
    else:
        with pytest.raises(HawkError, match="declared but not built yet"):
            compose([plain], "banded_group", "banded")


def test_the_banded_SINK_policy_is_the_same_named_slot_on_its_own_seam():
    """ carries the same unbuilt ``banded`` slot, and it refuses the same
    way: a seam value is never silently absent from the vocabulary."""
    with pytest.raises(HawkError) as excinfo:
        sink_policy("banded")
    message = str(excinfo.value)
    assert "emulated double precision" in message and "not built" in message, message
    assert "aether/banded/" in message


def test_an_unknown_mode_still_shows_the_third_slot():
    """Non-vacuity for the pair above: the vocabulary a refusal prints must
    include the unbuilt slot, or "named rather than absent" means nothing."""
    with pytest.raises(HawkError, match="banded"):
        scalar_mode("softdouble")
    with pytest.raises(HawkError, match="banded"):
        sink_policy("softdouble")
