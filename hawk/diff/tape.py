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

from .. import _ir
from ..errors import HawkError

#: The reserved plane names the twin adds (refused as a kernel's own).
TAPE_PLANE = "tape"
APPLIED_STEPS = "applied_steps"
TAPE_ON = "tape_on"
RESERVED = (TAPE_PLANE, APPLIED_STEPS, TAPE_ON)

#: The twin's name suffix.
SUFFIX = "_tape"

_I32 = _ir.TensorType((), "i32")
_FLAG = _ir.TensorType((), "f64")
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
    step = getattr(kernel, "step", None) or getattr(kernel, "one_step", None) or kernel
    fused = step is not kernel
    name = step.name
    if any(_ir.is_a(s, "finish") for s in step.sinks) and step.steps != 1:
        raise HawkError(
            f"hawk.diff.tape({name!r}): the kernel takes {step.steps} steps per "
            "launch; tape the single step (steps=1) or the automatic kernel"
        )
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise HawkError(
            f"hawk.diff.tape({name!r}): the window is a positive int of steps, got {window!r}"
        )
    if events not in ("step", "refine"):
        raise HawkError(
            f"hawk.diff.tape({name!r}): events is 'step' (the kernel's own stop "
            f"test) or 'refine' (the closed test refinement needs), got {events!r}"
        )
    if events == "refine" and _time(step) is None:
        raise HawkError(
            f"hawk.diff.tape({name!r}, events='refine'): refinement steps by a "
            "partial step size; declare it with @hawk.kernel(time_step=...)"
        )
    taken = [n for _, n in _ir.arg_spec(step.walk) if n in RESERVED]
    if taken:
        raise HawkError(
            f"hawk.diff.tape({name!r}): {taken} are reserved for the taping twin "
            f"({list(RESERVED)}); rename those parameters"
        )
    step, t_value = _promoted(step)
    if events == "refine":
        step = _closed_stop(step, t_value)
    planes = taped_planes(step)
    if not planes:
        raise HawkError(
            f"hawk.diff.tape({name!r}): the step reads and overwrites no "
            "floating-point state plane, so there is nothing to tape"
        )
    single = _single_twin(step, planes, window)
    if not fused:
        return single
    return _fused_twin(step, single, planes, window)


def taped_planes(step: Any) -> tuple:
    """``((name, ttype), ...)``: the planes the twin tapes, in ``arg_spec``
    order — floating-point ``Mutable`` planes read before their store and
    overwritten by the step."""
    walk = step.walk
    committed = {s.field_text("name") for s in step.sinks if _ir.is_a(s, "assign")}
    time = getattr(step, "time", None) if getattr(step, "time_step", None) else None
    out = []
    prior = _ir.prior_reads(walk)
    for role, name in _ir.arg_spec(walk):
        ttype = _ir.slot_type(walk, role, name)
        if (
            role == "mutable"
            and name in committed
            and ttype.dtype in _FLOAT
            and (name in prior or name == time)
        ):
            out.append((name, ttype))
    return tuple(out)


def tape_width(step: Any) -> int:
    """``w``: the flattened component count of one tape row."""
    return sum(_ir.element_count(t) for _, t in taped_planes(step))


def _tape_ttype(name: str, planes: tuple, window: int) -> Any:
    dtypes = {t.dtype for _, t in planes}
    if len(dtypes) != 1:
        raise HawkError(
            f"hawk.diff.tape({name!r}): the taped planes mix {sorted(dtypes)}; "
            "one tape holds one floating-point type"
        )
    width = sum(_ir.element_count(t) for _, t in planes)
    return _ir.TensorType((window * width,), dtypes.pop())


def _row(applied: Any, window: int) -> Any:
    """``applied mod window`` for a non-negative ``applied`` (integer ops)."""
    w = _ir.const(applied.ctx, window, _I32)
    return _ir.make(
        "sub", (applied, _ir.make("mul", (_ir.make("div", (applied, w)), w)))
    )


def _on(context: Any) -> Any:
    """The ``tape_on`` flag, a uniform read in ``context``."""
    return _ir.leaf(context, "uniform", "uniform", TAPE_ON, _I32)


def _tape_write(row: Any, on: Any, values: tuple, ttype: Any) -> Any:
    """The tape store of ``values`` at ``row`` (a whole row of their components), when ``on``."""
    width = sum(_ir.element_count(_ir.type_of(v)) for v in values)
    return _ir.make_node(
        "TapeWrite", ttype, [row, on, *values], name=TAPE_PLANE, width=width
    )


def _carry(context: Any, depth: int, slot: int, name: str, ttype: Any) -> Any:
    """The loop-carried value ``name`` in ``slot`` of the loop at nesting ``depth``."""
    return _ir.make_node(
        "LoopCarry", ttype, context=context, depth=depth, slot=slot, name=name
    )


def _assign(name: str, value: Any, ttype: Any) -> Any:
    """The commit of ``value`` to the plane ``name``."""
    return _ir.make_node("Assign", ttype, [value], name=name)


def _single_twin(step: Any, planes: tuple, window: int) -> Any:
    from ..trace.kernel import Kernel

    context = step.context
    applied = _ir.leaf(context, "prior_read", "mutable", APPLIED_STEPS, _I32)
    values = tuple(_ir.leaf(context, "prior_read", "mutable", n, t) for n, t in planes)
    store = _tape_write(
        _row(applied, window),
        _on(context),
        values,
        _tape_ttype(step.name, planes, window),
    )
    finishes = tuple(s for s in step.sinks if _ir.is_a(s, "finish"))
    others = tuple(s for s in step.sinks if not _ir.is_a(s, "finish"))
    count = _assign(
        APPLIED_STEPS, _ir.make("add", (applied, _ir.const(context, 1, _I32))), _I32
    )
    # The store FIRST: it reads the pre-step state before any commit moves it.
    sinks = (store, *others, count, *finishes)
    return Kernel(step.name + SUFFIX, sinks, step.planes, kind=step.kind)


def _fused_twin(step: Any, single: Any, planes: tuple, window: int) -> Any:
    """The ``steps="auto"`` twin: the transform of :mod:`hawk.trace.steps`
    with ``applied_steps`` one more carry and the tape store a per-trip
    commit of the loop."""
    from ..trace.kernel import Kernel
    from ..trace.steps import AUTO_K_MAX, _clamped_word, _prior_reads

    context = step.context
    sinks = tuple(step.sinks)
    finish = next(s for s in sinks if _ir.is_a(s, "finish"))
    commits = [s for s in sinks if s != finish]
    depth = 0
    carries = [
        _carry(context, depth, j, s.field_text("name"), _ir.type_of(s))
        for j, s in enumerate(commits)
    ]
    m = len(commits)
    applied = _carry(context, depth, m, APPLIED_STEPS, _I32)
    flag = _carry(context, depth, m + 1, "finished", _FLAG)
    by_name = {s.field_text("name"): c for s, c in zip(commits, carries)}
    by_name[APPLIED_STEPS] = applied
    mapping = {leaf: by_name[leaf.field_text("name")] for leaf in _prior_reads(sinks)}
    roots = tuple(s.operands[0] for s in commits) + (finish.operands[0],)
    new = _ir.substitute(roots, mapping)
    cond = new[-1]
    nexts = (
        tuple(new[:-1])
        + (_ir.make("add", (applied, _ir.const(context, 1, _I32))),)
        + (
            _ir.make_node(
                "Select",
                _FLAG,
                [cond, _ir.const(context, 1.0, _FLAG), _ir.const(context, 0.0, _FLAG)],
            ),
        )
    )
    inits = tuple(
        _ir.leaf(context, "prior_read", "mutable", s.field_text("name"), _ir.type_of(s))
        for s in commits
    ) + (
        _ir.leaf(context, "prior_read", "mutable", APPLIED_STEPS, _I32),
        _ir.const(context, 0.0, _FLAG),
    )
    store = _tape_write(
        _row(applied, window),
        _on(context),
        tuple(by_name[n] for n, _ in planes),
        _tape_ttype(step.name, planes, window),
    )
    index = _ir.make_node(
        "LoopIndex", _I32, context=context, depth=depth, name="step", bounds=None
    )
    loop = _ir.loop(
        index,
        (*carries, applied, flag),
        inits,
        nexts,
        start=0,
        stop=AUTO_K_MAX,
        step=1,
        body_sinks=(store,),
        exit_cond=cond,
        exit_values=nexts,
        count=_clamped_word(context),
    )
    out: list = [loop]
    out += [
        _assign(s.field_text("name"), _ir.loop_value(loop, j), _ir.type_of(s))
        for j, s in enumerate(commits)
    ]
    out.append(_assign(APPLIED_STEPS, _ir.loop_value(loop, m), _I32))
    done = _ir.make("gt", (_ir.loop_value(loop, m + 1), _ir.const(context, 0.5, _FLAG)))
    out.append(
        _ir.make_node(
            "Finish", _ir.type_of(done), [done], name=finish.field_text("name")
        )
    )
    return Kernel(
        step.name + SUFFIX,
        tuple(out),
        step.planes,
        kind=step.kind,
        step=single,
        one_step=single,
        steps="auto",
    )


# --------------------------------------------------------------------------
# the time plane: ``@hawk.kernel(time="t", time_step="dt")``

#: The suffix of the runner-owned start-time plane (``<t>_0``).
T0_SUFFIX = "_0"
#: The suffix of the per-sample partial step a refine entry takes (``<dt>_tau``).
TAU_SUFFIX = "_tau"
#: The suffix of the planes a refine entry commits (``<plane>_refined``).
REFINED_SUFFIX = "_refined"


def check_time(kernel: Any, time: str, time_step: str) -> None:
    """Refuse a time declaration the body contradicts: ``time_step`` must be
    a floating-point rank-0 input the body reads, and a time plane the body
    commits must be updated exactly as ``time = time + time_step`` (either
    operand order), its launch-start value plus the step. A time plane the
    body does not declare is legal (the twin mints it)."""
    step = getattr(kernel, "step", None) or getattr(kernel, "one_step", None) or kernel
    name = step.name
    nodes = _ir.post_order(step.sinks)
    dts = [
        n for n in nodes if _ir.is_a(n, "leaf") and n.field_text("name") == time_step
    ]
    if not dts or any(
        d.field_text("role") not in ("uniform", "per_sample")
        or _ir.type_of(d).shape
        or _ir.type_of(d).dtype not in _FLOAT
        for d in dts
    ):
        raise HawkError(
            f"kernel {name!r}: time_step={time_step!r} must name a scalar "
            "floating-point input the body reads (a Param or a Scalar)"
        )
    sink = next(
        (
            s
            for s in step.sinks
            if _ir.is_a(s, "assign") and s.field_text("name") == time
        ),
        None,
    )
    if sink is None:
        if any(_ir.is_a(n, "leaf") and n.field_text("name") == time for n in nodes):
            raise HawkError(
                f"kernel {name!r}: time={time!r} is read but never updated; a "
                "time plane is a Mutable the step advances by time_step"
            )
        return
    value = sink.operands[0]
    ops = value.operands if value.field_text("kind") == "add" else ()
    ok = (
        len(ops) == 2
        and any(
            _ir.is_a(o, "leaf")
            and o.field_text("kind") == "prior_read"
            and o.field_text("name") == time
            for o in ops
        )
        and any(_ir.is_a(o, "leaf") and o.field_text("name") == time_step for o in ops)
    )
    if not ok:
        raise HawkError(
            f"kernel {name!r}: the time plane {time!r} must advance as "
            f"`{time} = {time} + {time_step}` (its launch-start value plus the "
            "step), so the twin can write it in index form; got a different "
            "update"
        )


def _time(step: Any) -> tuple | None:
    dt = getattr(step, "time_step", None)
    return None if dt is None else (step.time or "t", dt)


def refine_plane_names(kernel: Any) -> dict | None:
    """The extra per-sample planes the refine entry of ``kernel`` reads, ``{"t0": <time plane at launch start>, "tau":
    <partial step>}``, or ``None`` for a kernel that declares no ``time_step=``. The sidecar states them, so a consumer
    never spells the suffixes."""
    got = _time(kernel)
    if got is None:
        return None
    t, dt = got
    return {"t0": t + T0_SUFFIX, "tau": dt + TAU_SUFFIX}


def _index_time(t: str, h: Any, k: Any) -> Any:
    """``t_0 + k*h``: ONE multiply, ONE add (the index form)."""
    t0 = _ir.leaf(h.ctx, "vocab_read", "per_sample", t + T0_SUFFIX, _ir.type_of(h))
    return _ir.make("add", (t0, _ir.make("mul", (k, h))))


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
    context = step.context
    nodes = _ir.post_order(step.sinks)
    if any(_ir.is_a(n, "loop") for n in nodes):
        raise HawkError(
            f"hawk.diff.tape({step.name!r}): promoting the time plane through a "
            "loop in the step body is not built"
        )
    dt_leaves = [n for n in nodes if _ir.is_a(n, "leaf") and n.field_text("name") == dt]
    ttype = _ir.type_of(dt_leaves[0])
    h = _ir.leaf(context, "vocab_read", "per_sample", dt, ttype)
    k = _ir.leaf(context, "prior_read", "mutable", APPLIED_STEPS, _I32)
    if tau:
        step_size = _ir.leaf(
            context, "vocab_read", "per_sample", dt + TAU_SUFFIX, ttype
        )
        t_value = _ir.make("add", (_index_time(t, h, k), step_size))
    else:
        step_size = h
        t_value = _index_time(t, h, _ir.make("add", (k, _ir.const(context, 1, _I32))))
    mapping = {leaf: step_size for leaf in dt_leaves}
    sink = next(
        (s for s in step.sinks if _ir.is_a(s, "assign") and s.field_text("name") == t),
        None,
    )
    if sink is not None:
        mapping[sink.operands[0]] = t_value
    sinks = _ir.swap(step.sinks, mapping)
    if sink is None:
        finishes = tuple(s for s in sinks if _ir.is_a(s, "finish"))
        others = tuple(s for s in sinks if not _ir.is_a(s, "finish"))
        sinks = (*others, _assign(t, t_value, ttype), *finishes)
    out = Kernel(step.name, sinks, step.planes, kind=step.kind)
    out.time, out.time_step = t, dt
    out.jump = step.jump
    return out, t_value


#: The closed stop test's tolerance, in ulps of the time plane times |dg/dt|.
EVENT_ULPS = 4.0


def _closed_stop(step: Any, t_value: Any) -> Any:
    """``step`` with its finish replaced by the closed, tolerant test
    ``any_i g_i <= EVENT_ULPS * spacing(|t|) * |dg_i/dt|`` (``t`` the
    post-step time, ``t_value``)."""
    from ..trace.kernel import Kernel
    from .event import guards_of

    finish = next(s for s in step.sinks if _ir.is_a(s, "finish"))
    spacing = _ir.make("spacing", (_ir.make("abs", (t_value,)),))
    cond = None
    for g in guards_of(step.name, finish.operands[0]):
        tol = _ir.make(
            "mul",
            (
                _ir.const(g.ctx, EVENT_ULPS, _ir.type_of(g)),
                _ir.make("mul", (spacing, _ir.make("abs", (_d_dt(g, t_value),)))),
            ),
        )
        test = _ir.make("le", (g, tol))
        cond = test if cond is None else _ir.make("lor", (cond, test))
    sinks = tuple(
        _ir.make_node("Finish", _ir.type_of(cond), [cond], name=s.field_text("name"))
        if s == finish
        else s
        for s in step.sinks
    )
    out = Kernel(step.name, sinks, step.planes, kind=step.kind)
    out.time, out.time_step = step.time, step.time_step
    out.jump = step.jump
    return out


def _d_dt(g: Any, t_value: Any) -> Any:
    """``dg/dt`` as an IR expression of the same inputs: the forward-mode
    rule of ``g`` with the time value as the one direction."""
    from .event import as_plain_reads
    from .transform import jvp

    probe_name = "hawk_event_time"
    probe = _ir.leaf(
        g.ctx, "vocab_read", "per_sample", probe_name, _ir.type_of(t_value)
    )
    prior = {
        n.field_text("name"): n
        for n in _ir.post_order((g,))
        if _ir.is_a(n, "leaf") and n.field_text("kind") == "prior_read"
    }
    (plain,) = as_plain_reads(_ir.substitute((g,), {t_value: probe}))
    if not any(
        _ir.is_a(n, "leaf") and n.field_text("name") == probe_name
        for n in _ir.post_order((plain,))
    ):
        return _ir.const(
            g.ctx, 0.0, _ir.type_of(g)
        )  # a guard that does not read the time
    (sink,) = jvp(
        (_assign("hawk_event_g", plain, _ir.type_of(plain)),), wrt=(probe_name,)
    )
    back = {}
    for n in _ir.post_order((sink.operands[0],)):
        if not _ir.is_a(n, "leaf"):
            continue
        name = n.field_text("name")
        if name == probe_name:
            back[n] = t_value
        elif name.endswith(probe_name):
            back[n] = _ir.const(n.ctx, 1.0, _ir.type_of(n))
        elif name in prior:
            back[n] = prior[name]
    return _ir.substitute((sink.operands[0],), back)[0]


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

    step = getattr(kernel, "step", None) or getattr(kernel, "one_step", None) or kernel
    if _time(step) is None:
        raise HawkError(
            f"hawk.diff.refine_entry({step.name!r}): the kernel declares no "
            "time_step=; a refine entry steps by a per-sample partial step"
        )
    refined, _ = _promoted(step, tau=True)
    finish = next((s for s in refined.sinks if _ir.is_a(s, "finish")), None)
    if finish is None:
        raise HawkError(
            f"hawk.diff.refine_entry({step.name!r}): the kernel never finishes "
            "its sample, so there is no stop to refine"
        )
    guards = guards_of(step.name, finish.operands[0])
    commits = [s for s in refined.sinks if _ir.is_a(s, "assign")]
    new = as_plain_reads(tuple(s.operands[0] for s in commits) + guards)
    sinks = tuple(
        _assign(s.field_text("name") + REFINED_SUFFIX, v, _ir.type_of(s))
        for s, v in zip(commits, new[: len(commits)])
    )
    sinks += tuple(
        _assign(f"{GUARD_PREFIX}{i}", g, _ir.type_of(g))
        for i, g in enumerate(new[len(commits) :])
    )
    out = Kernel(step.name + "_refine", sinks, {}, kind=step.kind)
    out.time, out.time_step = refined.time, refined.time_step
    out.jump = refined.jump
    return out


#: The suffix naming a plane's value AFTER the step in :func:`step_vjp`.
NEXT_SUFFIX = "_next"


def step_vjp(
    kernel: Any,
    *,
    wrt=None,
    primal_unit: str | None = None,
    steps_reverse: str = "per_step",
) -> Any:
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
    from ..trace.steps import _fused_steps_of
    from .transform import _rederivable

    if steps_reverse not in ("per_step", "fused"):
        raise HawkError(
            f"step_vjp: steps_reverse is 'per_step' or 'fused', got {steps_reverse!r}"
        )
    if steps_reverse == "fused" and _fused_steps_of(kernel) is None:
        raise HawkError(
            f"step_vjp({kernel.name!r}, steps_reverse='fused'): needs hawk.steps(k, K) "
            "with a fixed integer K > 1 (an 'auto' kernel's trip count is a launch-time word)"
        )
    if steps_reverse == "fused" and wrt is None:
        # the same inputs as one step (the loop's stop rule reads more, none differentiable)
        wrt = _step_vjp(
            kernel, wrt=None, primal_unit=None, steps_reverse="per_step"
        ).wrt
    return _rederivable(_step_vjp)(
        kernel, wrt=wrt, primal_unit=primal_unit, steps_reverse=steps_reverse
    )


def step_function(kernel: Any) -> Any:
    """ONE step of an in-place step kernel as a pure map: a kernel that reads every carried (``Mutable``) plane as it was
    before the step, writes each new value to the separate plane ``<plane>_next`` and leaves its inputs untouched. The
    kernel is named ``<kernel>_step`` and keeps the jump of the step. It is an ordinary kernel, so :func:`~hawk.diff.jvp`
    and :func:`~hawk.diff.vjp` differentiate it, and :func:`step_vjp` is the vjp of it.

    A kernel that carries no floating-point state plane is refused by name (the taping of its single step refuses it).

    Example::

        @hawk.kernel(steps=1)
        def decay(a: Param, terminated: Terminated, x: Mutable[Scalar]):
            x = x * a
            terminated = x < 0.5

        pure = hawk.diff.step_function(decay)    # reads a, x; writes x_next
    """
    from ..trace.kernel import Kernel
    from .event import as_plain_reads

    step = getattr(kernel, "step", None) or getattr(kernel, "one_step", None) or kernel
    twin = tape(kernel, 1)
    single = getattr(twin, "one_step", None) or twin
    commits = [s for s in single.sinks if _ir.is_a(s, "assign") and s.field_text("name") != APPLIED_STEPS]
    values = as_plain_reads(tuple(s.operands[0] for s in commits))
    function = Kernel(
        f"{step.name}_step",
        tuple(_assign(s.field_text("name") + NEXT_SUFFIX, v, _ir.type_of(s)) for s, v in zip(commits, values)),
        {},
    )
    function.jump = getattr(step, "jump", None)
    function.next_values = {s.field_text("name") + NEXT_SUFFIX: s.field_text("name") for s in commits}
    return function


def _step_vjp(kernel: Any, *, wrt, primal_unit, steps_reverse) -> Any:
    from ..trace.steps import _fused_function, _fused_steps_of
    from .transform import Derived, vjp

    fused = _fused_steps_of(kernel) if steps_reverse == "fused" else None
    step = getattr(kernel, "step", None) or getattr(kernel, "one_step", None) or kernel
    if fused is not None:
        function = _fused_function(*fused)
    else:
        function = step_function(kernel)
    function.jump = getattr(step, "jump", None)
    out = vjp(function, wrt=wrt)
    if (out.jump or {}).get("engaged"):
        raise HawkError(
            f"step_vjp({step.name!r}): a jump= switch whose value reaches control or a "
            "nonlinear op is differentiated by an alternate path of ONE launch; "
            "carrying it across launches (steps=, time_step, eagle stepping) is not "
            "built. Differentiate the whole rollout in one kernel with hawk.diff.vjp"
        )
    result = Derived(
        tuple(out),
        primal=kernel.name if fused else step.name,
        kind="vjp",
        wrt=out.wrt,
        primal_unit=primal_unit,
        jump=out.jump,
        gradients=out.gradients,
        cotangents=out.cotangents,
        next_values=function.next_values,
    )
    if fused:
        result._fused_steps = fused[1]
        result._kernel_kind = (
            function.kind
        )  # the stop flag is read in the body: no second guard
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
    from .transform import jvp

    step = getattr(kernel, "step", None) or getattr(kernel, "one_step", None) or kernel
    if getattr(step, "jump", None) is not None:
        raise HawkError(
            f"step_jvp({step.name!r}): a kernel declaring jump= is not carried forward; "
            "differentiate its per-launch term with hawk.diff.jvp"
        )
    if any(_ir.is_param_cell(n) for n in _ir.post_order(step.sinks)):
        raise HawkError(
            f"step_jvp({step.name!r}): the kernel reads parameter cells, whose tangent "
            "is not carried across launches"
        )
    state = tuple(
        dict.fromkeys(s.field_text("name") for s in step.sinks if _ir.is_a(s, "assign"))
    )
    plain = Kernel(
        f"{step.name}_plain",
        tuple(
            _ir.with_value(s, v)
            for s, v in zip(
                step.sinks, as_plain_reads(tuple(s.operands[0] for s in step.sinks))
            )
        ),
        step.planes,
        kind=step.kind,
    )
    tangent = jvp(
        plain, wrt=None if wrt is None else tuple(dict.fromkeys((*wrt, *state)))
    )
    carried = set(state) | {t for t, plane in tangent.tangents.items() if plane in state}
    values = tuple(s.operands[0] for s in tangent)
    start = {
        n: _ir.leaf(
            n.ctx, "prior_read", "mutable", n.field_text("name"), _ir.type_of(n)
        )
        for n in _ir.post_order(values)
        if _ir.is_a(n, "leaf")
        and n.field_text("kind") != "prior_read"
        and n.field_text("name") in carried
    }
    values = _ir.substitute(values, start) if start else values
    out = Kernel(
        f"{step.name}_jvp",
        tuple(step.sinks)
        + tuple(_ir.with_value(s, v) for s, v in zip(tangent, values)),
        step.planes,
        kind=step.kind,
    )
    out.time, out.time_step = step.time, step.time_step
    out.tangents = tangent.tangents
    return out
