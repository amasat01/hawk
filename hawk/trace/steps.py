# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.steps(kernel, K)`` — K steps of a finishing step kernel per launch.

An IR transform, no build option and no new user code: the step body
becomes ONE lowered loop of ``K`` trips
(:func:`hawk._ir.loop`) whose carries are the kernel's Mutable planes — every
read of a plane's launch-start value becomes the carry, every commit the
carry's next value — and whose ``break`` is the finish condition, evaluated
AFTER the step's
values (a TAIL break: a sample finishing at trip ``j`` leaves holding step
``j``'s values). After the loop the carries commit to their planes and the
finish epilogue fires with the loop's exit flag, carried as one more
``Real`` slot.

Bit identity with ``K = 1`` holds by construction: each trip evaluates the
SAME DAG, with register carries in place of plane loads/stores (device
results may differ by an ULP where the compiler contracts a multiply-add
differently). A terminated-on-entry sample skips the whole loop; a sample
may overshoot a run's step cap by fewer than ``K`` steps, so exact caps
use ``K = 1``.

Admissible shape: sinks are Mutable commits plus ONE finish; reads are any
leaf role. Refused, naming the kernel: an out/wide_out/accum_out
sink, a kernel without a finish, and a body loop depending on the carried
state. A ``jvp`` of the result is the fused ``step_jvp`` of the single step; its
``vjp`` refuses through the launch-start-read rule: differentiate the single
step, or ``step_vjp(..., steps_reverse="fused")`` for all K in one launch.

``steps="auto"`` derives ONE kernel, ``<name>_xauto``, by the same
transform: the loop's static range is :data:`AUTO_K_MAX` trips, and its
runtime trip count is read once per launch from the reserved ``lookup``
word :data:`~hawk._ir.FUSED_STEPS_PLANE` (set by the runner, never
bound by the host, clamped below at 0, any word above ``AUTO_K_MAX``
running that many steps). The sidecar records
the sidecar's ``finish.steps == "auto"`` and ``finish.steps_max == AUTO_K_MAX``. The
derived kernel carries the single step as ``.one_step``: its entry
branches once per launch on the word, a word of ``1`` running that step's
straight-line body and any other word the fused loop.

``@hawk.kernel`` WITHOUT ``steps=`` builds that automatic kernel by
default for every kernel :func:`admits_auto` (:func:`default_steps`),
under the author's own name, carrying its single step as ``.step``;
``steps=1`` is the opt-out. :func:`steps` and the derivatives of a
default automatic kernel start from its ``.step``.
"""

from __future__ import annotations

from typing import Any

from .. import _ir
from .._errors import HawkError

#: The static bound of a ``steps="auto"`` loop: the most steps any launch
#: can take (the sidecar's ``finish.steps_max``).
AUTO_K_MAX = 64

#: The type of the ``fused_steps`` word and of the loop's runtime trip count.
_COUNT = _ir.TensorType((), "i32")

#: The carried exit flag's type: a ``Real`` register (the loop emitter's rule).
_FLAG = _ir.TensorType((), "f64")


def _loop_index(context: Any, depth: int, name: str) -> Any:
    """The index variable of a lowered loop at nesting ``depth``."""
    return _ir.make_node(
        "LoopIndex", _COUNT, context=context, depth=depth, name=name, bounds=None
    )


def _loop_carry(context: Any, depth: int, slot: int, name: str, ttype: Any) -> Any:
    """The placeholder of carry ``slot`` inside a loop body at nesting ``depth``."""
    return _ir.make_node("LoopCarry", ttype, context=context, depth=depth, slot=slot, name=name)


def _assign(name: str, value: Any, ttype: Any) -> Any:
    """The plain commit of ``value`` to the plane ``name``."""
    return _ir.make_node("Assign", ttype, (value,), name=name)


def _prior_read(context: Any, name: str, ttype: Any) -> Any:
    """The launch-start value of the ``Mutable`` plane ``name``."""
    return _ir.leaf(context, "prior_read", "mutable", name, ttype)


def steps(kernel: Any, K: int | str) -> Any:
    """``kernel`` stepping ``K`` times per launch, named ``<name>_x<K>``.

    ``kernel`` is a traced :class:`hawk.Kernel` that finishes its own
    sample; the result shares its declarations, kind and slots, with
    ``finish.steps == K``. ``K`` is an int ``>= 1`` or exactly ``"auto"``
    (named ``<name>_xauto`` instead — see the module docstring for its
    trip-count mechanics).

    The decorator form ``@hawk.kernel(steps=K)`` is
    ``hawk.steps(hawk.kernel(fn), K)``, with ``steps=1`` the plain kernel.
    A default automatic kernel (``@hawk.kernel`` without ``steps=``) is
    derived from its single step ``.step``, so ``hawk.steps(k, 16)`` means
    the same with or without the default."""
    kernel = default_step(kernel)
    # the single step rides along for hawk.diff.tape; see default_step
    return _derive(kernel, K, f"{kernel.name}_x{K}", step=kernel)


def default_step(kernel: Any) -> Any:
    """The single step a default automatic kernel stands for (its ``.step``,
    under the same name), else ``kernel``. A ``steps=K`` kernel also carries
    its single step, under another name; a derivative tagged with that name
    would point at a kernel its bundle does not hold, so it is not used
    here."""
    step = getattr(kernel, "step", None)
    return step if step is not None and step.name == kernel.name else kernel


def _fused_steps_of(kernel: Any) -> tuple | None:
    """``(single step, K)`` of a ``hawk.steps(k, K)`` kernel with a fixed integer
    ``K``, else ``None`` (a single step, or an ``"auto"`` kernel whose trip
    count is a launch-time word)."""
    step = getattr(kernel, "step", None)
    finishes = [s for s in getattr(kernel, "sinks", ()) if _ir.is_a(s, "finish")]
    if step is None or step is kernel or not finishes:
        return None
    k = getattr(kernel, "steps", 1)
    return (step, k) if isinstance(k, int) and k > 1 else None


def admits_auto(kernel: Any) -> bool:
    """Whether :func:`steps` admits ``kernel`` for ``"auto"``: a single
    step finishing its own sample, committing at least one ``Mutable``
    plane and nothing else, with no plane named ``fused_steps``."""
    sinks = tuple(kernel.sinks)
    finishes = [s for s in sinks if _ir.is_a(s, "finish")]
    if len(finishes) != 1 or kernel.steps != 1:
        return False
    commits = [s for s in sinks if s != finishes[0]]
    if not commits or not all(_ir.is_a(s, "assign") for s in commits):
        return False
    return not any(n == _ir.FUSED_STEPS_PLANE for _, n in _ir.arg_spec(kernel.walk))


def default_steps(kernel: Any) -> Any:
    """The kernel ``@hawk.kernel`` builds with no ``steps=``: the automatic
    kernel of ``kernel`` under its OWN name (``.step = kernel``) when
    :func:`admits_auto`, else ``kernel`` unchanged."""
    if not admits_auto(kernel):
        return kernel
    return _derive(kernel, "auto", kernel.name, step=kernel)


def _derive(kernel: Any, K: int | str, out_name: str, *, step: Any = None) -> Any:
    """The transform of :func:`steps`, naming the result ``out_name``."""
    from .kernel import Kernel

    name = kernel.name
    check_steps(name, K, where="hawk.steps")
    sinks = tuple(kernel.sinks)
    finishes = [s for s in sinks if _ir.is_a(s, "finish")]
    if not finishes:
        raise HawkError(
            f"hawk.steps({name!r}): the kernel never finishes its sample — K "
            "steps with no exit is a plain `for` the author can write; finish "
            "with `terminated = cond`")
    finish = finishes[0]
    if kernel.steps != 1:
        raise HawkError(
            f"hawk.steps({name!r}): the kernel already takes {kernel.steps} "
            "steps per launch; derive from the single step")
    commits = [s for s in sinks if s != finish]
    for sink in commits:
        if not _ir.is_a(sink, "assign"):
            what = ("a loop committing per iteration" if _ir.is_a(sink, "loop")
                    else f"a {sink.field_text('role')} sink "
                         f"({sink.field_text('kind')} {sink.field_text('name')!r})")
            raise HawkError(
                f"hawk.steps({name!r}): {what} is not a step kernel's commit; a "
                "step kernel commits Mutable planes and finishes — an "
                "out/wide_out/accum_out sink has no per-sample state to "
                "carry from one step to the next")
    if not commits:
        raise HawkError(
            f"hawk.steps({name!r}): the kernel commits no Mutable plane, so "
            "there is no state to step")

    context = kernel.context
    depth = 0
    names = [s.field_text("name") for s in commits]
    types = [_ir.type_of(s) for s in commits]
    carries = [
        _loop_carry(context, depth, j, name, ttype)
        for j, (name, ttype) in enumerate(zip(names, types))
    ]
    flag = _loop_carry(context, depth, len(commits), "finished", _FLAG)
    by_name = dict(zip(names, carries))
    mapping = {leaf: by_name[leaf.field_text("name")] for leaf in _prior_reads(sinks)}

    auto = K == "auto"
    if auto and any(n == _ir.FUSED_STEPS_PLANE for _, n in _ir.arg_spec(kernel.walk)):
        raise HawkError(
            f"hawk.steps({name!r}, 'auto'): {_ir.FUSED_STEPS_PLANE!r} is reserved "
            "for the trip count of an automatic kernel; rename that parameter")

    roots = tuple(s.operands[0] for s in commits) + (finish.operands[0],)
    try:
        new = _ir.substitute(roots, mapping)
    except HawkError as exc:
        raise HawkError(
            f"hawk.steps({name!r}): the step body cannot be carried — {exc}"
        ) from None
    cond = new[-1]
    flagged = _ir.make_node(
        "Select",
        _FLAG,
        (cond, _ir.const(context, 1.0, _FLAG), _ir.const(context, 0.0, _FLAG)),
    )
    nexts = tuple(new[:-1]) + (flagged,)
    inits = tuple(_prior_read(context, name, ttype) for name, ttype in zip(names, types)) + (
        _ir.const(context, 0.0, _FLAG),
    )
    count = _clamped_word(context) if auto else None
    loop = _ir.loop(
        _loop_index(context, depth, "step"), (*carries, flag), inits, nexts,
        start=0, stop=AUTO_K_MAX if auto else K, step=1,
        exit_cond=cond, exit_values=nexts, count=count,
    )  # fmt: skip
    out = [
        _assign(name, _ir.loop_value(loop, j), ttype)
        for j, (name, ttype) in enumerate(zip(names, types))
    ]
    done = _ir.make("gt", (_ir.loop_value(loop, len(commits)), _ir.const(context, 0.5, _FLAG)))
    out.append(
        _ir.make_node(
            "Finish", _ir.TensorType((), "bool"), (done,), name=finish.field_text("name")
        )
    )
    return Kernel(out_name, tuple(out), kernel.planes, kind=kernel.kind,
                  step=step, steps=K, one_step=kernel if auto else None,
                  source_digest=getattr(kernel, "source_digest", None),
                  returns=getattr(kernel, "returns", None))


def _fused_function(step: Any, K: int) -> Any:
    """``K`` steps of ``step`` as a pure function: the launch-start state is read
    as plain planes, ``<plane>_next`` is the K-trip loop's result. What the
    derivatives of a fused kernel differentiate (``step_vjp(..., steps_reverse=
    "fused")``): the reverse loop reads the forward loop's carries, so it needs
    no recorded tape.

    A sample terminated on entry keeps its state (a ``Select`` on the stop flag,
    so its adjoint passes straight through); a sample stopping at trip ``j`` leaves the loop there, as in
    :func:`steps`. A ``time_step=`` kernel is read in the taping twin's index
    form, trip ``i`` taking step index ``applied_steps + i`` (a float counter,
    since an integer carry has no adjoint)."""
    from ..diff.event import as_plain_reads
    from ..diff.tape import APPLIED_STEPS, NEXT_SUFFIX, _promoted
    from ..ext import DATA_ONLY, Kind
    from .kernel import Kernel

    timed = bool(getattr(step, "time_step", None))
    if timed:
        step, _ = _promoted(step)
    sinks = tuple(step.sinks)
    context = step.context
    finish = next(s for s in sinks if _ir.is_a(s, "finish"))
    commits = [s for s in sinks if s != finish]
    names = [s.field_text("name") for s in commits]
    types = [_ir.type_of(s) for s in commits]
    carries = [
        _loop_carry(context, 0, j, name, ttype)
        for j, (name, ttype) in enumerate(zip(names, types))
    ]
    by_name = dict(zip(names, carries))
    counter = _loop_carry(context, 0, len(commits), "trip", _FLAG)
    if timed:
        by_name[APPLIED_STEPS] = _ir.make("cast", (counter,), "i32")
    mapping = {leaf: by_name[leaf.field_text("name")] for leaf in _prior_reads(sinks)}
    new = _ir.substitute(tuple(s.operands[0] for s in commits) + (finish.operands[0],), mapping)
    nexts = tuple(new[:-1])
    inits = as_plain_reads(
        tuple(_prior_read(context, name, ttype) for name, ttype in zip(names, types))
    )
    if timed:
        start = as_plain_reads((_prior_read(context, APPLIED_STEPS, _COUNT),))[0]
        carries.append(counter)
        inits += (_ir.make("cast", (start,), "f64"),)
        nexts += (_ir.make("add", (counter, _ir.const(context, 1.0, _FLAG))),)
    loop = _ir.loop(
        _loop_index(context, 0, "step"), tuple(carries), inits, nexts,
        start=0, stop=K, step=1, exit_cond=new[-1], exit_values=nexts,
    )  # fmt: skip
    entry = _ir.leaf(
        context, "terminated", "terminated", finish.field_text("name"),
        _ir.type_of(finish.operands[0]),
    )  # fmt: skip
    out = [
        _assign(
            name + NEXT_SUFFIX,
            _ir.make_node("Select", ttype, (entry, inits[j], _ir.loop_value(loop, j))),
            ttype,
        )
        for j, (name, ttype) in enumerate(zip(names, types))
    ]
    # the Select above is the skip: no guard of the kind, so a derivative neither
    # masks nor wraps it a second time (a standalone C unit would flag the repeat)
    return Kernel(f"{step.name}_x{K}_function", tuple(out), {}, kind=Kind("fused_steps", guard=DATA_ONLY))


def _clamped_word(context: Any) -> Any:
    """The trip count of a ``steps="auto"`` loop: the ``fused_steps`` word,
    clamped below at 0 so a negative word runs no step. The bound above is
    the word itself, a runtime value: a runner may ask one launch for every
    step left in its run (``AUTO_K_MAX`` stays the static range a host
    tile is sized for and the policy's largest band word)."""
    lookup = _ir.leaf(context, "table_read", "lookup", _ir.FUSED_STEPS_PLANE, _COUNT)
    zero = _ir.const(context, 0, _COUNT)
    word = _ir.make_node("At", _COUNT, (lookup, zero))  # the gather class: a plain op named "at" is not a table read
    return _ir.make_node("Select", _COUNT, (_ir.make("lt", (word, zero)), zero, word))


def check_steps(name: str, K: Any, *, where: str = "@hawk.kernel(steps=...)") -> None:
    """Refuse a step count ``K`` that is neither an int ``>= 1`` nor exactly
    ``"auto"``, naming the kernel ``name`` and the spelling ``where`` it came
    through."""
    if isinstance(K, str) and K == "auto":
        return
    if isinstance(K, bool) or not isinstance(K, int) or K < 1:
        raise HawkError(
            f"{where} on {name!r}: the number of steps per launch is a "
            f"positive int or exactly 'auto', got {K!r}")


def _prior_reads(sinks: tuple) -> list:
    """Every launch-start-read leaf the sinks reach (each read before a plane's
    first store is one, and every one becomes the carry)."""
    return [
        node
        for node in _ir.post_order(sinks)
        if _ir.is_a(node, "leaf") and node.field_text("kind") == "prior_read"
    ]
