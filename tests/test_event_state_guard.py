# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.diff.event.state_guard``: the stop rule's guard as a function of
the state AFTER the step (every committed value read back from its plane), so
its vjp is ``g_x`` over the extended state; and the closed stop test of a
guard that does not read the time (its ``dg/dt`` is zero)."""

from __future__ import annotations

from hawk import Mutable, Param, Scalar, Terminated, kernel
from hawk.diff import tape, vjp
from hawk.diff.event import state_guard


@kernel(steps=1, time="t", time_step="dt")
def surf(dt: Param, a: Param, terminated: Terminated,
         x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    v_start = v
    v = v_start - dt * x
    x = x + dt * v_start
    tn = t + dt
    t = tn
    terminated = x <= a * tn


@kernel(steps=1, time="t", time_step="dt")
def drop(dt: Param, terminated: Terminated, y: Mutable[Scalar], t: Mutable[Scalar]):
    yn = y - dt
    y = yn
    t = t + dt
    terminated = yn < 0.0


def test_the_guard_reads_the_state_after_the_step():
    sg = state_guard(surf)
    assert sg.name == "surf_state_guard"
    reads = {name for _role, name in sg.walk.arg_spec}
    assert reads == {"event_g0", "x", "t", "a"}       # no dt: x and t are read back
    grads = {s.name for s in vjp(sg)}
    assert {"bar_x", "bar_t", "bar_a"} <= grads


def test_a_time_free_guard_gets_the_plain_closed_test():
    twin = tape(drop, 4, events="refine")
    assert twin.name == "drop_tape"
