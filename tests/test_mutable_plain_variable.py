# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A Mutable/Terminated/WideOut own-column name is an ordinary local.

A read BEFORE the first store is the plane's launch-start value, lazily (a
plane never read before its own store mints no leaf at all); the FINAL value
commits once at the end. Rows this file enumerates that no other file in the
corpus already covers through a migrated kernel (``tests/test_steps.py`` and
``tests/test_finish_own_sample.py`` already exercise ``terminated = cond``,
its refusals, and the ``steps=K``/``"auto"`` digest-equality rows end to end):

* **tuple vs sequential** -- Python's own assignment rules decide which
  value a read sees; a tuple target reads every name's value BEFORE any of
  them is reassigned, a sequential pair of statements reads the first
  statement's NEW value in the second.
* **one-sided if keeps the old value** -- the existing select-merge, now
  reached with the plane's own bare name on both sides.
* **write-only mints no leaf** -- the digest-level guarantee behind "a
  plane never read before assignment creates no leaf".
* **component update on the plane's own name** -- `x[k] = x[k] + e`
  directly on a Mutable's own name, no longer requiring a separate local
  bound from a prior read first.
* **a derivative refuses a state read, naming the plane**.
"""

from __future__ import annotations

import pytest

from hawk import Mutable, Param, Scalar, Terminated, Vector, WideOut, kernel
from hawk.diff import jvp, vjp
from hawk.ir import HawkError
from hawk.math import dot


# --------------------------------------------------------------------------- #
# tuple vs sequential: Python's own rules, not a hawk-specific one.
# --------------------------------------------------------------------------- #
def test_a_tuple_target_reads_every_name_s_old_value():
    """``x, v = x + dt * v, v - dt * w2 * x`` -- both right-hand sides read
    the values `x`/`v` held BEFORE this statement (a symplectic step);
    unlike the sequential pair below, `v`'s own term never sees the new
    `x`."""
    @kernel(steps=1)
    def tupled(dt: Param, w2: Param, x: Mutable[Scalar], v: Mutable[Scalar]):
        x, v = x + dt * v, v - dt * w2 * x

    assert sorted(tupled.walk.prior_reads) == ["v", "x"]


def test_a_sequential_pair_reads_the_first_statement_s_new_value():
    """``x = x + dt * v; v = v - dt * w2 * x`` -- the second statement's
    `x` is the line above's NEW value, so this digests DIFFERENTLY from the
    tuple form above (a different DAG: `v`'s term depends on the updated
    `x`, not the launch-start one)."""
    @kernel(steps=1)
    def sequential(dt: Param, w2: Param, x: Mutable[Scalar], v: Mutable[Scalar]):
        x = x + dt * v
        v = v - dt * w2 * x

    @kernel(steps=1)
    def tupled(dt: Param, w2: Param, x: Mutable[Scalar], v: Mutable[Scalar]):
        x, v = x + dt * v, v - dt * w2 * x

    assert sequential.walk.digest != tupled.walk.digest
    # `v`'s term in the sequential form reads `x`'s NEW value -- a single
    # `prior_read` leaf for `x` (the FIRST statement's read), not two.
    assert sorted(sequential.walk.prior_reads) == ["v", "x"]


def test_an_augmented_assignment_reads_then_stores():
    """``t += dt`` on a Mutable's own name: an ordinary Python augmented
    store once the front end has seeded the launch-start read."""
    @kernel(steps=1)
    def ticked(dt: Param, t: Mutable[Scalar]):
        t += dt

    assert sorted(ticked.walk.prior_reads) == ["t"]


# --------------------------------------------------------------------------- #
# one-sided if keeps the old value -- the existing select-merge.
# --------------------------------------------------------------------------- #
def test_a_one_sided_if_keeps_the_launch_start_value_on_the_other_path():
    @kernel(steps=1)
    def clamp_up(c: Scalar, x: Mutable[Scalar]):
        if c > 0.0:
            x = c

    assert sorted(clamp_up.walk.prior_reads) == ["x"]
    from hawk.ir import Select
    assert any(isinstance(n, Select) for n in clamp_up.walk.order), (
        "a one-sided `if` on a Mutable's own name must still fold to a select "
        "against its launch-start value on the path that does not store it")


# --------------------------------------------------------------------------- #
# write-only mints no leaf at all.
# --------------------------------------------------------------------------- #
def test_a_write_only_plane_has_no_prior_read_leaf():
    @kernel(steps=1)
    def energy(v: Vector[3], out: Mutable[Scalar]):
        out = 0.5 * dot(v, v)

    assert energy.walk.prior_reads == frozenset()
    assert not any(n.kind == "prior_read" for n in energy.walk.order)


def test_a_mutable_read_before_it_is_ever_committed_refuses():
    with pytest.raises(HawkError, match="declared but never committed"):
        @kernel(steps=1)
        def bad(a: Scalar, r: Mutable[Scalar]):  # noqa: F841
            b = a + r  # noqa: F841


# --------------------------------------------------------------------------- #
# a component update directly on the plane's own name -- no separate local.
# --------------------------------------------------------------------------- #
def test_a_component_store_on_the_plane_s_own_name_updates_one_entry():
    @kernel(steps=1)
    def bump_z(dt: Param, x: Mutable[Vector[3]]):
        x[2] = x[2] + dt

    assert sorted(bump_z.walk.prior_reads) == ["x"]
    from hawk.ir.nodes import Assign
    sinks = [s for s in bump_z.sinks if isinstance(s, Assign) and s.name == "x"]
    assert len(sinks) == 1 and sinks[0].value.kind == "set_component_at"


# --------------------------------------------------------------------------- #
# a derivative refuses a launch-start read, naming the plane.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("transform", [vjp, jvp], ids=["vjp", "jvp"])
def test_a_derivative_refuses_a_launch_start_read_naming_the_plane(transform):
    @kernel(steps=1)
    def running_max(a: Scalar, out: Mutable[Scalar]):
        out = a if a > out else out

    with pytest.raises(HawkError, match=r"'out'.*is read before it is assigned"):
        transform(running_max, wrt=("a",))


# --------------------------------------------------------------------------- #
# the old spellings raise with the rewrite, naming the line.
# --------------------------------------------------------------------------- #
def test_an_old_style_prior_read_refuses_with_the_rewrite():
    with pytest.raises(HawkError, match=r"`\.prior` is no longer a HAWK form"):
        @kernel(steps=1)
        def bad(dt: Param, x: Mutable[Scalar]):
            x = x.prior + dt  # noqa: F821 - the refused spelling itself


def test_an_old_style_subscript_store_refuses_with_the_rewrite():
    with pytest.raises(HawkError, match=r"`x\[\.\.\.\] = …` is no longer a HAWK form"):
        @kernel(steps=1)
        def bad(dt: Param, x: Mutable[Scalar]):
            x[...] = x + dt


def test_an_old_style_terminated_store_refuses_with_the_rewrite():
    with pytest.raises(HawkError, match=r"`terminated\[\.\.\.\] = …` is no longer a HAWK form"):
        @kernel(steps=1)
        def bad(t: Scalar, terminated: Terminated):
            terminated[...] = t > 1.0


# --------------------------------------------------------------------------- #
# WideOut own-column: a write-only plane inside nested control flow.
# --------------------------------------------------------------------------- #
# A WideOut own column has no launch-start value at all -- its `.prior`
# always refuses by name (a scattered/reduced plane has no single value one
# sample owns). The seed-or-not scan for a bare read before a store is
# conservative: "any mention of the name inside this `if`/`for` needs the
# seed", without telling a READ apart from a WRITE-only mention. For a
# Mutable that is harmless (an unused seed never reaches a sink); for a
# WideOut it crashes trying to mint a seed the plane cannot provide, even
# though the body below never reads the plane at all. Both rows are
# `xfail` until that scan tells a write-only mention apart from a read.
def test_a_write_only_wideout_plane_traces_inside_an_if():
    @kernel(steps=1)
    def wo_if(cond: Scalar, y: WideOut[Scalar]):
        if cond > 0.0:
            y = 1.0
        else:
            y = 2.0

    assert wo_if.walk.prior_reads == frozenset()


def test_a_write_only_wideout_plane_assigned_only_inside_a_for_is_refused():
    """A WideOut column has no launch-start value to carry through a loop, so
    the refusal asks for the write after the loop instead."""
    with pytest.raises(HawkError, match="after a loop"):
        @kernel(steps=1)
        def wo_for(v: Vector[3], y: WideOut[Scalar]):
            for c in range(3):
                y = v[c]
