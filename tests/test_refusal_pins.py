# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Tests hawk's own refusals: two cases where a plausible wrong answer would
be worse than raising, so each is a loud, named error a caller can pin on.

Reverse-differentiating a ``max`` fold has an ambiguous subgradient at a
tie, so silently differentiating it would let a consumer's gradient be
wrong only on the ties; softmax's max pass stays FORWARD-ONLY and the
gradient rides the sum pass instead. A ``banded`` build that reached an op
with no certified banded row would report a precision the artifact does not
carry, which is the one failure a banded consumer cannot detect from the
answer, so those ops are refused by name.
"""

from __future__ import annotations

from _specs import spec_for, default_kind
import pytest

import hawk
import hawk.math as m
from hawk.artifact import build_bundle
from hawk.diff import vjp
from aether.wings.base.wrapper import render_source, scalar_mode
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
# (a) a max over samples has no reverse rule, and the refusal says which rule.
# --------------------------------------------------------------------------- #
def test_the_reverse_of_a_max_reduction_is_refused_naming_the_rule():
    """Needs the ``sink.op != "sum"`` guard in ``hawk/diff/transform._seed``,
    or a ``max`` is seeded with a broadcast adjoint exactly like a ``sum``::

        Failed: DID NOT RAISE <class 'hawk.HawkError'>

    and ``vjp(row_max)`` returned a gradient — a number, for a max whose
    subgradient is ambiguous at every tie."""
    with pytest.raises(HawkError) as excinfo:
        vjp(row_max)
    message = str(excinfo.value)
    for needle in ("max over samples", "subgradient", "reduce with a sum"):
        assert needle in message, (
            f"the refusal must name the rule that is missing and why; {needle!r} "
            f"is absent from: {message}")


def test_the_sum_pass_beside_it_still_derives():
    """Non-vacuity: the refusal is about the ``max``, not about reductions
    over samples. A row that refused both would be an implementation gap wearing a
    principle's clothes."""
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
def test_every_row_of_the_banded_table_refuses_naming_the_op_and_its_remedy():
    """``refuse_uncertified`` walks the kernel's nodes; each row of the table, reached, raises with the name the
    author wrote and the remedy. With the lookup removed every row fails with ``DID NOT RAISE``."""
    from types import SimpleNamespace

    from aether.wings.base.wrapper import BANDED, FLOAT64
    from aether.wings.simt.wrapper_text import _BANDED_UNCERTIFIED, refuse_uncertified

    assert {"accum_write", "erf", "rint", "remainder"} <= set(_BANDED_UNCERTIFIED)
    for kind, (name, remedy) in _BANDED_UNCERTIFIED.items():
        node = SimpleNamespace(kind=kind)
        with pytest.raises(HawkError) as excinfo:
            refuse_uncertified((node,), SimpleNamespace(kinds=frozenset()), BANDED)
        assert name in str(excinfo.value) and remedy in str(excinfo.value), kind
        refuse_uncertified((node,), SimpleNamespace(kinds=frozenset()), FLOAT64)     # the float modes build it


@hawk.kernel
def scattered(x: hawk.Scalar, lane: hawk.Index, acc: hawk.Mutable[hawk.Scalar]):
    hawk.scatter(acc, contract="plain")[lane] += x * x


@pytest.mark.parametrize("door", ("build_bundle", "render_source"))
def test_every_door_into_the_banded_mode_refuses_an_atomic_scatter_by_name(tmp_path, door):
    """One seam, and every entrance to it: a reduction refused in one place and built in another is the
    float64-shaped answer a banded consumer cannot see."""
    from hawk.targets import TARGETS

    with pytest.raises(HawkError, match=r"atomic accumulate into a plane .*not built in scalar mode 'banded'"):
        if door == "build_bundle":
            build_bundle([scattered], tmp_path / "banded", mode="banded", targets=("cuda",))
        else:
            render_source("scattered", scattered.sinks, scattered.walk, TARGETS["cuda"], mode=scalar_mode("banded"), spec=spec_for(TARGETS["cuda"]), kind=default_kind())


def test_the_banded_mode_admits_the_certified_facade_and_refuses_the_missing_math():
    from hawk.targets import TARGETS

    @hawk.kernel
    def facade(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        a = m.sqrt(x) + m.rsqrt(y) + m.cbrt(x) + m.hypot(x, y) + m.exp(x) + m.log(y) + m.pow(x, y)
        b = m.sin(x) + m.cos(y) + m.asin(x) + m.acos(x) + m.atan(y) + m.atan2(x, y) + m.floor(x) + m.ceil(y)
        c = m.round(x) + m.trunc(y) + m.fmod(x, y) + m.fma(x, y, x) + m.abs(x) + m.maximum(x, y) + m.minimum(x, y)
        out = m.where(x < y, a + b, c + m.copysign(x, y) / y)

    @hawk.kernel
    def special(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
        out = m.erf(x)

    banded = scalar_mode("banded")
    assert "aether::math::atan2(" in render_source("facade", facade.sinks, facade.walk, TARGETS["cuda"],
                                                   mode=banded, spec=spec_for(TARGETS["cuda"]), kind=default_kind()).text
    with pytest.raises(HawkError, match=r"hawk\.math\.erf .*until its banded row lands"):
        render_source("special", special.sinks, special.walk, TARGETS["cuda"], mode=banded, spec=spec_for(TARGETS["cuda"]), kind=default_kind())


@pytest.mark.parametrize("door", ("build_bundle", "render_source"))
@pytest.mark.parametrize("targets", (("host",), ("cuda", "host")))
def test_the_banded_mode_with_a_host_target_is_refused_as_a_whole_by_name(door, targets, tmp_path):
    """Banded is a device feature: a bundle with any target that is not a device is refused, not built device-only."""
    from hawk.targets import TARGETS

    with pytest.raises(HawkError, match=r"banded runs on device execution paths; on a CPU use float64"):
        if door == "build_bundle":
            build_bundle([plain], tmp_path / "banded", mode="banded", targets=targets)
        else:
            render_source("plain", plain.sinks, plain.walk, TARGETS["host"], mode=scalar_mode("banded"), spec=spec_for(TARGETS["host"]), kind=default_kind())


def test_the_c_target_refuses_the_banded_mode_by_name():
    from ospreys.c import render

    with pytest.raises(HawkError, match="the c target has no banded spelling"):
        render(plain, mode="banded")


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
    include ``banded`` on both seams, or "named rather than absent" means nothing."""
    with pytest.raises(HawkError, match="banded"):
        scalar_mode("float128")
    with pytest.raises(HawkError, match="banded"):
        sink_policy("float128")
