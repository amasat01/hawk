# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.diff.tape(kernel, window)`` — the TAPING TWIN of a step kernel.

A windowed checkpoint–recompute reverse needs, for every step ``j`` of a
window, the state the step READ: the one-step derivative's primal inputs.
The twin is the step kernel itself plus three reserved planes:

* ``applied_steps`` (``Mutable`` i32, per sample): the steps this sample has
  applied; +1 per applied step, the finishing step included (the tail-break
  convention of a loop's per-sample count);
* ``tape`` (an ``out`` plane of ``(W*w,)`` per sample, sample-last, i.e.
  ``[W, w, N]`` in memory): row ``applied_steps mod W`` holds the PRE-step
  state, written before any commit and before ``applied_steps`` moves;
* ``tape_on`` (an i32 uniform): the stores happen only when it is non-zero,
  so a forward that records no tape (bound to a one-row dummy) runs the SAME
  compiled entry as the recompute that does — their state results are
  bitwise equal by construction, in every compiler profile.

The tape WIDTH is derived, not chosen: every state plane the step's
one-step ``vjp`` takes as a primal input (a floating-point ``Mutable`` read
before its store) AND the step overwrites, in ``arg_spec`` order, flattened
component by component. A plane written without being read is not taped.

The twin is named ``<name>_tape`` and takes the kernel's own form: an
automatic kernel (``@hawk.kernel`` without ``steps=``, or ``steps="auto"``)
gets the fused twin (one tape row per loop trip, the trip count read from
the ``fused_steps`` word), carrying its single-step twin as ``.step`` and
``.one_step``; a single-step kernel (``steps=1``) gets the single-step twin.
A fixed ``steps=K`` kernel gets the same fused twin from its single step
``.step`` (the runtime keeps the trip count at most K). The plain kernel is
never changed.
"""

from __future__ import annotations

from typing import Any

from ..ir import Const, HawkError, Leaf, Node
from ..ir import make as _mk
from ..ir.loop_nodes import LoopCarry, LoopIndex, LoopValue
from ..ir.loops import build_loop
from ..ir.nodes import (
    AccumWrite,
    Assign,
    Finish,
    SampleSum,
    Select,
    TapeWrite,
    element_count,
    is_param_cell,
)
from ..ir.team import TeamStore
from ..ir.walk import substitute
from ..types import TensorType

#: The reserved plane names the twin adds (refused as a kernel's own).
TAPE_PLANE = "tape"
APPLIED_STEPS = "applied_steps"
TAPE_ON = "tape_on"
RESERVED = (TAPE_PLANE, APPLIED_STEPS, TAPE_ON)

#: The twin's name suffix.
SUFFIX = "_tape"

_I32 = TensorType((), "i32")
_FLAG = TensorType((), "f64")
_FLOAT = ("f64", "f32")


def tape(kernel: Any, window: int, *, events: str = "step") -> Any:
    """The taping twin of ``kernel`` for windows of ``window`` steps (see
    the module docstring). A default automatic kernel, or a ``steps=K``
    kernel, is read through its single step ``.step`` and gets the fused
    twin (a launching runtime sets its trip count, at most ``K``).

    ``events="refine"`` (a kernel declaring ``time_step=``) makes the twin's
    stop test CLOSED and tolerant: a sample stops where a guard ``g`` (see
    :mod:`hawk.diff.event`) reaches ``g <= 4 * spacing(|t|) * |dg/dt|``, the
    rounding of the index-form time a crossing exactly on the step grid
    can carry — so a stop that lands on a grid point is taken at that step,
    never one step late. ``events="step"`` keeps the kernel's own test."""
    step = (getattr(kernel, "step", None) or getattr(kernel, "one_step", None)
            or kernel)
    fused = step is not kernel
    name = step.name
    finish = getattr(step.walk, "finish", None)
    if finish is not None and finish[1] != 1:
        raise HawkError(
            f"hawk.diff.tape({name!r}): the kernel takes {finish[1]} steps per "
            "launch; tape the single step (steps=1) or the automatic kernel")
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise HawkError(
            f"hawk.diff.tape({name!r}): the window is a positive int of steps, "
            f"got {window!r}")
    if events not in ("step", "refine"):
        raise HawkError(
            f"hawk.diff.tape({name!r}): events is 'step' (the kernel's own stop "
            f"test) or 'refine' (the closed test refinement needs), got {events!r}")
    if events == "refine" and _time(step) is None:
        raise HawkError(
            f"hawk.diff.tape({name!r}, events='refine'): refinement steps by a "
            "partial step size; declare it with @hawk.kernel(time_step=...)")
    taken = [n for _, n in step.walk.arg_spec if n in RESERVED]
    if taken:
        raise HawkError(
            f"hawk.diff.tape({name!r}): {taken} are reserved for the taping twin "
            f"({list(RESERVED)}); rename those parameters")
    step, t_value = _promoted(step)
    if events == "refine":
        step = _closed_stop(step, t_value)
    planes = taped_planes(step)
    if not planes:
        raise HawkError(
            f"hawk.diff.tape({name!r}): the step reads and overwrites no "
            "floating-point state plane, so there is nothing to tape")
    single = _single_twin(step, planes, window)
    if not fused:
        return single
    return _fused_twin(step, single, planes, window)


def taped_planes(step: Any) -> tuple:
    """``((name, ttype), ...)``: the planes the twin tapes, in ``arg_spec``
    order — floating-point ``Mutable`` planes read before their store and
    overwritten by the step."""
    walk = step.walk
    committed = {s.name for s in step.sinks if isinstance(s, Assign)}
    time = getattr(step, "time", None) if getattr(step, "time_step", None) else None
    out = []
    for role, name in walk.arg_spec:
        ttype = walk.slot_types[(role, name)]
        if (role == "mutable" and name in committed and ttype.dtype in _FLOAT
                and (name in walk.prior_reads or name == time)):
            out.append((name, ttype))
    return tuple(out)


def tape_width(step: Any) -> int:
    """``w``: the flattened component count of one tape row."""
    return sum(element_count(t) for _, t in taped_planes(step))


def _tape_ttype(name: str, planes: tuple, window: int) -> TensorType:
    dtypes = {t.dtype for _, t in planes}
    if len(dtypes) != 1:
        raise HawkError(
            f"hawk.diff.tape({name!r}): the taped planes mix {sorted(dtypes)}; "
            "one tape holds one floating-point type")
    width = sum(element_count(t) for _, t in planes)
    return TensorType((window * width,), dtypes.pop())


def _row(applied: Node, window: int) -> Node:
    """``applied mod window`` for a non-negative ``applied`` (integer ops)."""
    w = Const(window, _I32)
    return _mk("sub", (applied, _mk("mul", (_mk("div", (applied, w)), w))))


def _on() -> Leaf:
    return Leaf("uniform", "uniform", TAPE_ON, _I32)


def _single_twin(step: Any, planes: tuple, window: int) -> Any:
    from ..trace.kernel import Kernel

    applied = Leaf("prior_read", "mutable", APPLIED_STEPS, _I32)
    values = tuple(Leaf("prior_read", "mutable", n, t) for n, t in planes)
    store = TapeWrite(TAPE_PLANE, _row(applied, window), _on(), values,
                      _tape_ttype(step.name, planes, window))
    finishes = tuple(s for s in step.sinks if isinstance(s, Finish))
    others = tuple(s for s in step.sinks if not isinstance(s, Finish))
    count = Assign(APPLIED_STEPS, _mk("add", (applied, Const(1, _I32))), _I32)
    # The store FIRST: it reads the pre-step state before any commit moves it.
    sinks = (store, *others, count, *finishes)
    return Kernel(step.name + SUFFIX, sinks, step.planes, kind=step.kind)


def _fused_twin(step: Any, single: Any, planes: tuple, window: int) -> Any:
    """The ``steps="auto"`` twin: the transform of :mod:`hawk.trace.steps`
    with ``applied_steps`` one more carry and the tape store a per-trip
    commit of the loop."""
    from ..trace.kernel import Kernel
    from ..trace.steps import AUTO_K_MAX, _clamped_word, _prior_reads

    sinks = tuple(step.sinks)
    finish = next(s for s in sinks if isinstance(s, Finish))
    commits = [s for s in sinks if s is not finish]
    depth = 0
    carries = [LoopCarry(depth, j, s.name, s.ttype) for j, s in enumerate(commits)]
    m = len(commits)
    applied = LoopCarry(depth, m, APPLIED_STEPS, _I32)
    flag = LoopCarry(depth, m + 1, "finished", _FLAG)
    by_name = {s.name: c for s, c in zip(commits, carries)}
    by_name[APPLIED_STEPS] = applied
    mapping = {id(leaf): by_name[leaf.name] for leaf in _prior_reads(sinks)}
    roots = tuple(s.value for s in commits) + (finish.value,)
    new = substitute(roots, mapping)
    cond = new[-1]
    nexts = (tuple(new[:-1]) + (_mk("add", (applied, Const(1, _I32))),)
             + (Select(cond, Const(1.0, _FLAG), Const(0.0, _FLAG), _FLAG),))
    inits = (tuple(Leaf("prior_read", "mutable", s.name, s.ttype) for s in commits)
             + (Leaf("prior_read", "mutable", APPLIED_STEPS, _I32), Const(0.0, _FLAG)))
    store = TapeWrite(TAPE_PLANE, _row(applied, window), _on(),
                      tuple(by_name[n] for n, _ in planes),
                      _tape_ttype(step.name, planes, window))
    loop = build_loop(LoopIndex(depth, "step"), (*carries, applied, flag), inits,
                      nexts, start=0, stop=AUTO_K_MAX, step=1, body_sinks=(store,),
                      exit_cond=cond, exit_values=nexts, count=_clamped_word())
    out: list = [loop]
    out += [Assign(s.name, LoopValue(loop, j), s.ttype) for j, s in enumerate(commits)]
    out.append(Assign(APPLIED_STEPS, LoopValue(loop, m), _I32))
    done = _mk("gt", (LoopValue(loop, m + 1), Const(0.5, _FLAG)))
    out.append(Finish(finish.name, done, steps="auto"))
    return Kernel(step.name + SUFFIX, tuple(out), step.planes, kind=step.kind,
                  step=single, one_step=single)


# --------------------------------------------------------------------------
# the time plane: ``@hawk.kernel(time="t", time_step="dt")``

#: The suffix of the runner-owned start-time plane (``<t>_0``).
T0_SUFFIX = "_0"
#: The suffix of the per-sample partial step a refine entry takes (``<dt>_tau``).
TAU_SUFFIX = "_tau"
#: The suffix of the planes a refine entry commits (``<plane>_refined``).
REFINED_SUFFIX = "_refined"


def _reachable(roots) -> list:
    out, seen, stack = [], set(), list(roots)
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        out.append(node)
        stack.extend(node.operands)
    return out


def check_time(kernel: Any, time: str, time_step: str) -> None:
    """Refuse a time declaration the body contradicts: ``time_step`` must be
    a floating-point rank-0 input the body reads, and a time plane the body
    commits must be updated exactly as ``time = time + time_step`` (either
    operand order), its launch-start value plus the step. A time plane the
    body does not declare is legal (the twin mints it)."""
    step = (getattr(kernel, "step", None) or getattr(kernel, "one_step", None)
            or kernel)
    name = step.name
    nodes = _reachable(step.sinks)
    dts = [n for n in nodes if isinstance(n, Leaf) and n.name == time_step]
    if not dts or any(d.role not in ("uniform", "per_sample") or d.ttype.shape
                      or d.ttype.dtype not in _FLOAT for d in dts):
        raise HawkError(
            f"kernel {name!r}: time_step={time_step!r} must name a scalar "
            "floating-point input the body reads (a Param or a Scalar)")
    sink = next((s for s in step.sinks
                 if isinstance(s, Assign) and s.name == time), None)
    if sink is None:
        if any(isinstance(n, Leaf) and n.name == time for n in nodes):
            raise HawkError(
                f"kernel {name!r}: time={time!r} is read but never updated; a "
                "time plane is a Mutable the step advances by time_step")
        return
    ops = sink.value.operands if getattr(sink.value, "kind", None) == "add" else ()
    ok = (len(ops) == 2
          and any(isinstance(o, Leaf) and o.kind == "prior_read" and o.name == time
                  for o in ops)
          and any(isinstance(o, Leaf) and o.name == time_step for o in ops))
    if not ok:
        raise HawkError(
            f"kernel {name!r}: the time plane {time!r} must advance as "
            f"`{time} = {time} + {time_step}` (its launch-start value plus the "
            "step), so the twin can write it in index form; got a different "
            "update")


def _time(step: Any) -> tuple | None:
    dt = getattr(step, "time_step", None)
    return None if dt is None else (step.time or "t", dt)


def _resink(sink: Node, value: Node) -> Node:
    if isinstance(sink, Finish):
        return Finish(sink.name, value, steps=sink.steps)
    if isinstance(sink, (SampleSum, AccumWrite)):
        return sink.rebuilt(value)
    if isinstance(sink, Assign):
        return Assign(sink.name, value, sink.ttype)
    if isinstance(sink, TeamStore):
        return TeamStore(sink.name, value)
    raise HawkError(
        f"a promotion rebuilds Mutable commits, sample sums, keyed accumulates and a finish; "
        f"{sink.kind} {getattr(sink, 'name', '?')!r} is none of them")


def _index_time(t: str, h: Node, k: Node) -> Node:
    """``t_0 + k*h``: ONE multiply, ONE add (the index form)."""
    t0 = Leaf("vocab_read", "per_sample", t + T0_SUFFIX, h.ttype)
    return _mk("add", (t0, _mk("mul", (k, h))))


def _promoted(step: Any, *, tau: bool = False) -> Any:
    """``step`` with its time declaration applied (unchanged without one):
    the step size read as a per-sample plane, the time plane committed in
    index form ``t_0 + (k+1)*dt`` with ``k = applied_steps`` (minted if the
    body declares none). With ``tau=True``, the refine form instead: the
    body steps by the per-sample ``<dt>_tau`` and the time plane is
    ``(t_0 + k*dt) + tau``."""
    from ..trace.kernel import Kernel

    got = _time(step)
    if got is None:
        return step, None
    t, dt = got
    nodes = _reachable(step.sinks)
    if any(getattr(n, "kind", None) == "loop" for n in nodes):
        raise HawkError(
            f"hawk.diff.tape({step.name!r}): promoting the time plane through a "
            "loop in the step body is not built")
    dt_leaves = [n for n in nodes if isinstance(n, Leaf) and n.name == dt]
    ttype = dt_leaves[0].ttype
    h = Leaf("vocab_read", "per_sample", dt, ttype)
    k = Leaf("prior_read", "mutable", APPLIED_STEPS, _I32)
    if tau:
        step_size = Leaf("vocab_read", "per_sample", dt + TAU_SUFFIX, ttype)
        t_value = _mk("add", (_index_time(t, h, k), step_size))
    else:
        step_size = h
        t_value = _index_time(t, h, _mk("add", (k, Const(1, _I32))))
    mapping = {id(leaf): step_size for leaf in dt_leaves}
    sink = next((s for s in step.sinks if isinstance(s, Assign) and s.name == t),
                None)
    if sink is not None:
        mapping[id(sink.value)] = t_value
    values = substitute(tuple(s.value for s in step.sinks), mapping)
    sinks = tuple(_resink(s, v) for s, v in zip(step.sinks, values))
    if sink is None:
        finishes = tuple(s for s in sinks if isinstance(s, Finish))
        others = tuple(s for s in sinks if not isinstance(s, Finish))
        sinks = (*others, Assign(t, t_value, ttype), *finishes)
    out = Kernel(step.name, sinks, step.planes, kind=step.kind)
    out.time, out.time_step = t, dt
    out.jump = step.jump
    return out, t_value


#: The closed stop test's tolerance, in ulps of the time plane times |dg/dt|.
EVENT_ULPS = 4.0


def _closed_stop(step: Any, t_value: Node) -> Any:
    """``step`` with its finish replaced by the closed, tolerant test
    ``any_i g_i <= EVENT_ULPS * spacing(|t|) * |dg_i/dt|`` (``t`` the
    post-step time, ``t_value``)."""
    from ..trace.kernel import Kernel
    from .event import guards_of

    finish = next(s for s in step.sinks if isinstance(s, Finish))
    spacing = _mk("spacing", (_mk("abs", (t_value,)),))
    cond = None
    for g in guards_of(step.name, finish.value):
        tol = _mk("mul", (Const(EVENT_ULPS, g.ttype),
                          _mk("mul", (spacing, _mk("abs", (_d_dt(g, t_value),))))))
        test = _mk("le", (g, tol))
        cond = test if cond is None else _mk("lor", (cond, test))
    sinks = tuple(Finish(s.name, cond, steps=s.steps) if s is finish else s
                  for s in step.sinks)
    out = Kernel(step.name, sinks, step.planes, kind=step.kind)
    out.time, out.time_step = step.time, step.time_step
    out.jump = step.jump
    return out


def _d_dt(g: Node, t_value: Node) -> Node:
    """``dg/dt`` as an IR expression of the same inputs: the forward-mode
    rule of ``g`` with the time value as the one direction."""
    from .event import as_plain_reads
    from .transform import jvp

    probe = Leaf("vocab_read", "per_sample", "hawk_event_time", t_value.ttype)
    prior = {n.name: n for n in _reachable((g,))
             if isinstance(n, Leaf) and n.kind == "prior_read"}
    (plain,) = as_plain_reads(substitute((g,), {id(t_value): probe}))
    if not any(isinstance(n, Leaf) and n.name == probe.name for n in _reachable((plain,))):
        return Const(0.0, g.ttype)      # a guard that does not read the time
    (sink,) = jvp((Assign("hawk_event_g", plain, plain.ttype),), wrt=(probe.name,))
    back = {}
    for n in _reachable((sink.value,)):
        if not isinstance(n, Leaf):
            continue
        if n.name == probe.name:
            back[id(n)] = t_value
        elif n.name.endswith(probe.name):
            back[id(n)] = Const(1.0, n.ttype)
        elif n.name in prior:
            back[id(n)] = prior[n.name]
    return substitute((sink.value,), back)[0]


def refine_entry(kernel: Any) -> Any:
    """The REFINE entry ``<name>_refine`` of a kernel declaring
    ``time_step=``: one step from the state a sample held before its stop
    step, of size ``<dt>_tau`` (a per-sample plane), with the time plane in
    the split index form ``(t_0 + k*dt) + tau`` — so ``d t / d tau == 1``.

    Reads every state plane (and ``applied_steps`` = ``k``, ``<t>_0``, the
    grid step ``dt``) as plain inputs and commits ``<plane>_refined`` per
    committed state plane plus ``event_g<i>`` per stop guard
    (:func:`hawk.diff.event.event_guard`'s convention, ``g <= 0`` stops): a
    pure function of its inputs, so :func:`~hawk.diff.jvp` w.r.t.
    ``<dt>_tau`` gives the step's and the guard's derivative in the partial
    step. No finish and no mask: it runs on finished samples."""
    from ..trace.kernel import Kernel
    from .event import GUARD_PREFIX, as_plain_reads, guards_of

    step = (getattr(kernel, "step", None) or getattr(kernel, "one_step", None)
            or kernel)
    if _time(step) is None:
        raise HawkError(
            f"hawk.diff.refine_entry({step.name!r}): the kernel declares no "
            "time_step=; a refine entry steps by a per-sample partial step")
    refined, _ = _promoted(step, tau=True)
    finish = next((s for s in refined.sinks if isinstance(s, Finish)), None)
    if finish is None:
        raise HawkError(
            f"hawk.diff.refine_entry({step.name!r}): the kernel never finishes "
            "its sample, so there is no stop to refine")
    guards = guards_of(step.name, finish.value)
    commits = [s for s in refined.sinks if isinstance(s, Assign)]
    new = as_plain_reads(tuple(s.value for s in commits) + guards)
    sinks = tuple(Assign(s.name + REFINED_SUFFIX, v, s.ttype)
                  for s, v in zip(commits, new[:len(commits)]))
    sinks += tuple(Assign(f"{GUARD_PREFIX}{i}", g, g.ttype)
                   for i, g in enumerate(new[len(commits):]))
    out = Kernel(step.name + "_refine", sinks, {}, kind=step.kind)
    out.time, out.time_step = refined.time, refined.time_step
    out.jump = refined.jump
    return out


#: The suffix naming a plane's value AFTER the step in :func:`step_vjp`.
NEXT_SUFFIX = "_next"


def step_vjp(kernel: Any, *, wrt=None, primal_unit: str | None = None,
             steps_reverse: str = "per_step") -> Any:
    """The vjp of ONE step of an in-place step kernel (one that updates
    ``Mutable`` planes and finishes its samples), for a reverse through a
    rollout. :func:`~hawk.diff.vjp` refuses such a kernel, because a plane
    read before it is assigned holds the previous step's value; here the
    step is read as a pure function of the values its planes hold BEFORE it.

    Naming contract (the planes of the result):

    * a plane the derivative needs is read under its own name and holds its
      value before the step (for a taped run, a row of :func:`tape`'s tape);
    * ``bar_<plane>_next`` (adjoint in) is ∂J/∂(``<plane>`` after the step),
      one per plane the step commits;
    * ``bar_<plane>`` (gradient out) is ∂J/∂(``<plane>`` before the step),
      one per differentiable input (``wrt`` narrows them, as for ``vjp``).

    The stop rule and the step count are not differentiated (a sample's stop
    step is fixed by the recorded run). A kernel declaring ``time_step=`` is
    read in the taping twin's index form, so the result also reads
    ``applied_steps`` (the step's index), ``<t>_0`` and the step size as a
    per-sample plane — the same planes the twin binds. A default automatic or
    ``steps=K`` kernel is differentiated through its single step.

    The result is tagged as the derivative of that single step (its name):
    publish it beside that step, or pass ``primal_unit=`` naming the unit it
    was published in.

    ``steps_reverse="fused"`` on ``hawk.steps(k, K)`` (a fixed integer ``K``)
    instead gives the reverse of ALL ``K`` steps in ONE launch: the same planes
    (``bar_<plane>_next`` in, ``bar_<plane>`` out, the state before the block
    read under its own name), tagged as the derivative of the fused kernel and
    recording ``steps_reverse`` in the sidecar. It re-runs the forward from the
    block's start, so it needs no tape row. State adjoints equal ``K`` calls of
    the per-step form bit for bit; a ``Param`` gradient sums the ``K`` terms in
    another order (ULP-level). ``"per_step"`` is the default.

    Example::

        @hawk.kernel(steps=1)
        def fall(dt: Param, terminated: Terminated, y: Mutable[Scalar],
                 v: Mutable[Scalar]):
            v_start = v
            v = v_start - 9.81 * dt
            y = y + dt * v_start
            terminated = y < 0.0

        back = hawk.Kernel("fall_step_vjp", hawk.diff.step_vjp(fall))
        # reads y, v, dt, bar_y_next, bar_v_next; writes bar_y, bar_v, bar_dt
    """
    from ..trace.steps import fused_steps_of
    from .transform import _rederivable

    if steps_reverse not in ("per_step", "fused"):
        raise HawkError(
            f"step_vjp: steps_reverse is 'per_step' or 'fused', got {steps_reverse!r}")
    if steps_reverse == "fused" and fused_steps_of(kernel) is None:
        raise HawkError(
            f"step_vjp({kernel.name!r}, steps_reverse='fused'): needs hawk.steps(k, K) "
            "with a fixed integer K > 1 (an 'auto' kernel's trip count is a launch-time word)")
    return _rederivable(_step_vjp)(kernel, wrt=wrt, primal_unit=primal_unit,
                                   steps_reverse=steps_reverse)


def _step_vjp(kernel: Any, *, wrt, primal_unit, steps_reverse) -> Any:
    from ..trace.kernel import Kernel
    from ..trace.steps import fused_function, fused_steps_of
    from .event import as_plain_reads
    from .transform import Derived, vjp

    fused = fused_steps_of(kernel) if steps_reverse == "fused" else None
    step = (getattr(kernel, "step", None) or getattr(kernel, "one_step", None)
            or kernel)
    if fused is not None:
        function = fused_function(*fused)
    else:
        twin = tape(kernel, 1)
        single = getattr(twin, "one_step", None) or twin
        commits = [s for s in single.sinks if isinstance(s, Assign) and s.name != APPLIED_STEPS]
        values = as_plain_reads(tuple(s.value for s in commits))
        function = Kernel(f"{step.name}_function",
                          tuple(Assign(s.name + NEXT_SUFFIX, v, s.ttype)
                                for s, v in zip(commits, values)), {})
    function.jump = getattr(step, "jump", None)
    out = vjp(function, wrt=wrt)
    if (out.jump or {}).get("engaged"):
        raise HawkError(
            f"step_vjp({step.name!r}): a jump= switch whose value reaches control or a "
            "nonlinear op is differentiated by an alternate path of ONE launch; "
            "carrying it across launches (steps=, time_step, eagle stepping) is not "
            "built. Differentiate the whole rollout in one kernel with hawk.diff.vjp")
    result = Derived(tuple(out), primal=kernel.name if fused else step.name, kind="vjp",
                     wrt=out.wrt, primal_unit=primal_unit, jump=out.jump)
    if fused:
        result.fused_steps = fused[1]
        result.kernel_kind = function.kind     # the stop flag is read in the body: no second guard
    return result


def step_jvp(kernel: Any, *, wrt=None) -> Any:
    """The forward-mode twin of ONE step of an in-place step kernel, for a
    tangent carried across launches: a kernel that commits each state plane
    ``<plane>`` AND ``dot_<plane>`` (its tangent), both advanced by the same
    step. Where :func:`step_vjp` runs backwards over a taped rollout, this
    runs forward beside the primal and needs no tape, so it launches like
    any step kernel: ``hawk.steps(result, K)``, or paced by ``eagle.until_done``.

    The tangent of a state plane is carried like the plane (``dot_<plane>``
    is read at the launch start and written at its end); ``wrt`` names the
    other inputs differentiated (a ``Param`` by name, ``None`` for all), and
    every state plane is always differentiated. Seed ``dot_<input>`` before
    the first launch. A sample that terminates keeps its tangent, as it keeps
    its state. A kernel declaring ``jump=`` or reading parameter cells is
    refused: their tangent is not carried here.

    Example::

        @hawk.kernel(steps=1)
        def decay(a: Param, terminated: Terminated, x: Mutable[Scalar]):
            x = x * a
            terminated = x < 0.5

        fwd = hawk.diff.step_jvp(decay)
        # reads a, dot_a, x, dot_x; commits x and dot_x
    """
    from ..trace.kernel import Kernel
    from .event import as_plain_reads
    from .transform import TANGENT_PREFIX, jvp

    step = (getattr(kernel, "step", None) or getattr(kernel, "one_step", None)
            or kernel)
    if getattr(step, "jump", None) is not None:
        raise HawkError(
            f"step_jvp({step.name!r}): a kernel declaring jump= is not carried forward; "
            "differentiate its per-launch term with hawk.diff.jvp")
    if any(is_param_cell(n) for n in _reachable(tuple(s.value for s in step.sinks))):
        raise HawkError(
            f"step_jvp({step.name!r}): the kernel reads parameter cells, whose tangent "
            "is not carried across launches")
    state = tuple(dict.fromkeys(s.name for s in step.sinks if isinstance(s, Assign)))
    plain = Kernel(f"{step.name}_plain", tuple(
        _resink(s, v) for s, v in zip(step.sinks, as_plain_reads(tuple(s.value for s in step.sinks)))),
        step.planes, kind=step.kind)
    tangent = jvp(plain, wrt=None if wrt is None else tuple(dict.fromkeys((*wrt, *state))))
    carried = set(state) | {f"{TANGENT_PREFIX}_{name}" for name in state}
    values = tuple(s.value for s in tangent)
    start = {id(n): Leaf("prior_read", "mutable", n.name, n.ttype)
             for n in _reachable(values)
             if isinstance(n, Leaf) and n.kind != "prior_read" and n.name in carried}
    values = substitute(values, start) if start else values
    out = Kernel(f"{step.name}_jvp", tuple(step.sinks) + tuple(
        _resink(s, v) for s, v in zip(tangent, values)), step.planes, kind=step.kind)
    out.time, out.time_step = step.time, step.time_step
    return out
