# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``Output.returned`` may coexist with ordinary named sinks.

The refusal that used to fire the instant a returned-output kernel also
declared a Mutable/Accum parameter is LIFTED: the returned slot is an
ordinary synthesised ``Assign``, so a body may commit through it AND through
whatever else it declares — a diagnostic ``Mutable`` beside a returned
acceleration, say. Sink order is every named sink in PARAMETER order, then
the synthesised slot, then the kind's own companion (if any); a
``compensated`` sink's target is resolved from the kind's OUTPUT SET alone,
so a scattered diagnostic beside the returned target is never mistaken for
it.

This test previously failed: restoring the old "commits through the synthesised
slot alone" refusal makes every kernel below refuse to trace; reverting the
scattered-commit fix so ``_commit_scattered`` reads the sink policy
unconditionally makes the scattered-diagnostic row refuse (or silently
compensate the wrong plane). Every plant was then removed."""

from __future__ import annotations

import _deploy as L
import numpy as np
import pytest
from conftest import sidecar_of

from hawk import Accum, Mutable, Scalar, Terminated, Vector
from hawk.artifact import build_bundle as _build_bundle
from hawk.emit import render_body
from hawk.ext import Kind, Output, compensated
from hawk.ir import HawkError
from hawk.math import norm

N = 8

RETURNED = Kind("returned_with_diagnostic", output=Output.returned(Vector[3], slot="acc"),
               sink=compensated(into="acc_c"))
NAMED = Kind("named_with_diagnostic", output=Output.named("acc"),
            sink=compensated(into="acc_c"))


@RETURNED
def returned_with_diagnostic(term: Vector[3], speed: Mutable[Scalar]):
    """a named sink (``speed``) beside the returned+compensated slot."""
    speed = norm(term)
    return term


@NAMED
def named_with_diagnostic(term: Vector[3], speed: Mutable[Scalar],
                          acc: Mutable[Vector[3]]):
    """The ``Output.named`` twin: the target ``acc`` declared LAST."""
    speed = norm(term)
    acc = term


RETURNED_GUARDED = Kind("returned_with_diagnostic_guarded",
                        output=Output.returned(Vector[3], slot="acc"),
                        sink=compensated(into="acc_c"))


@RETURNED_GUARDED
def returned_with_diagnostic_guarded(term: Vector[3], terminated: Terminated,
                                     speed: Mutable[Scalar]):
    speed = norm(term)
    return term


@RETURNED
def returned_with_scatter_diagnostic(term: Vector[3], hits: Accum[Scalar]):
    """a SCATTERED diagnostic Accum beside the returned compensated
    target — the scatter is not the target, so it commits plain."""
    hits.add(1.0, at=0)
    return term


#: The SAME cancelling-sum shape ``test_ext_seams.py`` uses, as Vector[3]
#: terms parallel to one axis so the compensated arithmetic stays exact.
_TERMS = [np.array(t) for t in
         ([1e16, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
          [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1e16, 0.0, 0.0])]


def _sequential_launch(bundle, name, terms, *, mask=None, extra=()):
    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = L.host_plugin(bundle.directory, name, sidecar_of(bundle, name))
    plan = eplan.plan(plugin, structure=eexec.HostTeam)
    kw = {"term": np.tile(terms[0][:, None], (1, N))}
    if mask is not None:
        kw["terminated"] = mask
    for out_name, out in extra:
        kw[out_name] = out
    bound = plan.bind(**kw)
    bound.launch()
    for t in terms[1:]:
        bound = bound.rebind(term=np.tile(t[:, None], (1, N)))
        bound.launch()
    return {out_name: out for out_name, out in extra}


# --------------------------------------------------------------------------- #
# byte identity with the ``Output.named`` twin.
# --------------------------------------------------------------------------- #
def test_returned_output_matches_the_named_twin_with_a_diagnostic_sink():
    assert returned_with_diagnostic.arg_spec == named_with_diagnostic.arg_spec
    assert returned_with_diagnostic.walk.digest == named_with_diagnostic.walk.digest
    a = render_body(returned_with_diagnostic.sinks, returned_with_diagnostic.walk,
                    kind=returned_with_diagnostic.kind).text
    b = render_body(named_with_diagnostic.sinks, named_with_diagnostic.walk,
                    kind=named_with_diagnostic.kind).text
    assert a == b


# --------------------------------------------------------------------------- #
# the named sink commits PLAIN (last launch wins); the returned slot
# accumulates the exact compensated sum.
# --------------------------------------------------------------------------- #
def test_named_sink_commits_plain_while_the_returned_slot_accumulates(tmp_path, cache_dir):
    bundle = _build_bundle([returned_with_diagnostic], tmp_path / "b2",
                           targets=("host",), cache_dir=cache_dir)
    acc, acc_c = np.zeros((3, N)), np.zeros((3, N))
    speed = np.zeros(N)
    got = _sequential_launch(bundle, "returned_with_diagnostic", _TERMS,
                             extra=(("acc", acc), ("acc_c", acc_c), ("speed", speed)))
    total = got["acc"][0] + got["acc_c"][0]
    np.testing.assert_array_equal(total, np.full(N, 6.0))
    np.testing.assert_array_equal(got["speed"], np.full(N, 1e16), (
        "the named sink is a PLAIN store -- the last launch's norm wins, no "
        "accumulate"))


# --------------------------------------------------------------------------- #
# a scattered diagnostic beside the returned compensated target commits
# through the target's plain atomic add, never through the compensated
# helper -- the policy applies to the KIND's resolved target alone.
# --------------------------------------------------------------------------- #
def test_a_scattered_diagnostic_beside_a_compensated_returned_target_commits_plain(
        tmp_path, cache_dir):
    bundle = _build_bundle([returned_with_scatter_diagnostic], tmp_path / "b3",
                           targets=("host", "cuda"), cache_dir=cache_dir)
    for src in ("returned_with_scatter_diagnostic.cpp", "returned_with_scatter_diagnostic.cu"):
        text = (bundle.directory / src).read_text()
        assert text.count("hawk_abi::accum_add(") == 1, text
        assert text.count("hawk_abi::store_compensated(") == 1, text
        assert "accum_add_compensated" not in text


# --------------------------------------------------------------------------- #
# a terminated sample keeps BOTH the named sink's and the returned
# slot's prior bytes.
# --------------------------------------------------------------------------- #
def test_a_terminated_sample_keeps_both_named_and_returned_bytes(tmp_path, cache_dir):
    bundle = _build_bundle([returned_with_diagnostic_guarded], tmp_path / "b4",
                           targets=("host",), cache_dir=cache_dir)
    mask = np.zeros(N, dtype=bool)
    mask[0] = True
    acc, acc_c = np.zeros((3, N)), np.zeros((3, N))
    speed = np.zeros(N)
    got = _sequential_launch(bundle, "returned_with_diagnostic_guarded", _TERMS, mask=mask,
                             extra=(("acc", acc), ("acc_c", acc_c), ("speed", speed)))
    assert got["speed"][0] == 0.0 and got["acc"][0, 0] == 0.0 and got["acc_c"][0, 0] == 0.0
    assert got["speed"][1] == 1e16
    np.testing.assert_array_equal(got["acc"][0, 1:] + got["acc_c"][0, 1:], np.full(N - 1, 6.0))


# --------------------------------------------------------------------------- #
# the refusals that remain, and the one that is new (a named sink
# colliding with the compensated companion's name).
# --------------------------------------------------------------------------- #
def test_a_parameter_named_like_the_synthesised_slot_still_refuses():
    K = Kind("returned_slot_collision_b5", output=Output.returned(Scalar, slot="acc"))
    with pytest.raises(HawkError, match="may not also be a parameter name"):
        @K
        def bad(acc: Scalar):
            return acc


def test_a_returned_value_of_the_wrong_rank_still_refuses():
    K = Kind("returned_wrong_rank_b5", output=Output.returned(Vector[3], slot="acc"))
    with pytest.raises(HawkError):
        @K
        def bad(e: Scalar):
            return e


def test_a_named_sink_declared_but_never_committed_still_refuses():
    K = Kind("returned_uncommitted_named_b5", output=Output.returned(Scalar, slot="acc"))
    with pytest.raises(HawkError, match="declared but never committed"):
        @K
        def bad(velocity: Vector[3], speed: Mutable[Scalar]):  # noqa: F841
            return norm(velocity)


def test_a_named_sink_colliding_with_the_compensated_companion_name_refuses():
    K = Kind("returned_companion_collision_b5", output=Output.returned(Vector[3], slot="acc"),
            sink=compensated(into="speed"))
    with pytest.raises(HawkError, match="collides"):
        @K
        def bad(term: Vector[3], speed: Mutable[Scalar]):
            speed = norm(term)
            return term


# --------------------------------------------------------------------------- #
# the derivative differentiates EVERY sink, named and returned alike,
# and never re-runs the compensated commit.
# --------------------------------------------------------------------------- #
def test_a_derivative_of_returned_plus_named_sinks_differentiates_every_sink():
    from hawk import Kernel
    from hawk.diff import jvp

    d = jvp(returned_with_diagnostic, wrt=("term",))
    assert [s.name for s in d] == ["dot_speed", "dot_acc", "dot_acc_c"]
    text = render_body(tuple(d), Kernel("d", tuple(d)).walk).text
    assert "store_compensated" not in text
