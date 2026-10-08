# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.steps(kernel, K)`` — K steps of a finishing step kernel per launch.

An IR transform, no build option and no new user code: the step body
becomes ONE lowered :class:`~hawk.ir.loop_nodes.Loop` of ``K`` trips
(:func:`build_loop`) whose carries are the kernel's Mutable planes — every
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
leaf role. Refused, naming the kernel: an out/wide_out/accum_out/Reduce
sink, a kernel without a finish, and a body loop depending on the carried
state. A derived kernel of the result refuses through the launch-start-read
rule: differentiate the single step instead.

``steps="auto"`` derives ONE kernel, ``<name>_xauto``, by the same
transform: the loop's static range is :data:`AUTO_K_MAX` trips, and its
runtime trip count is read once per launch from the reserved ``lookup``
word :data:`~hawk.ir.nodes.FUSED_STEPS_PLANE` (set by the runner, never
bound by the host, clamped below at 0, any word above ``AUTO_K_MAX``
running that many steps). The sidecar records
``finish.steps == "auto"`` and ``finish.steps_max == AUTO_K_MAX``. The
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

from ..ir.loop_nodes import Loop, LoopCarry, LoopIndex, LoopValue
from ..ir.loops import build_loop
from ..ir.nodes import (
    FUSED_STEPS_PLANE,
    Assign,
    At,
    Const,
    Finish,
    HawkError,
    Leaf,
    Node,
    Select,
)
from ..ir.ops import make
from ..ir.walk import substitute
from ..types import TensorType

#: The static bound of a ``steps="auto"`` loop: the most steps any launch
#: can take (the sidecar's ``finish.steps_max``).
AUTO_K_MAX = 64

#: The type of the ``fused_steps`` word and of the loop's runtime trip count.
_COUNT = TensorType((), "i32")

#: The carried exit flag's type: a ``Real`` register (the loop emitter's rule).
_FLAG = TensorType((), "f64")


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
    if getattr(kernel, "step", None) is not None:
        kernel = kernel.step
    return _derive(kernel, K, f"{kernel.name}_x{K}")


def admits_auto(kernel: Any) -> bool:
    """Whether :func:`steps` admits ``kernel`` for ``"auto"``: a single
    step finishing its own sample, committing at least one ``Mutable``
    plane and nothing else, with no plane named ``fused_steps``."""
    sinks = tuple(kernel.sinks)
    finishes = [s for s in sinks if isinstance(s, Finish)]
    if len(finishes) != 1 or finishes[0].steps != 1:
        return False
    commits = [s for s in sinks if s is not finishes[0]]
    if not commits or not all(isinstance(s, Assign) for s in commits):
        return False
    return not any(n == FUSED_STEPS_PLANE for _, n in kernel.walk.arg_spec)


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
    finishes = [s for s in sinks if isinstance(s, Finish)]
    if not finishes:
        raise HawkError(
            f"hawk.steps({name!r}): the kernel never finishes its sample — K "
            "steps with no exit is a plain `for` the author can write; finish "
            "with `terminated = cond`")
    finish = finishes[0]
    if finish.steps != 1:
        raise HawkError(
            f"hawk.steps({name!r}): the kernel already takes {finish.steps} "
            "steps per launch; derive from the single step")
    commits = [s for s in sinks if s is not finish]
    for sink in commits:
        if not isinstance(sink, Assign):
            what = ("a loop committing per iteration" if isinstance(sink, Loop)
                    else f"a {getattr(sink, 'role', '?')} sink "
                         f"({sink.kind} {getattr(sink, 'name', '?')!r})")
            raise HawkError(
                f"hawk.steps({name!r}): {what} is not a step kernel's commit; a "
                "step kernel commits Mutable planes and finishes — an "
                "out/wide_out/accum_out/Reduce sink has no per-sample state to "
                "carry from one step to the next")
    if not commits:
        raise HawkError(
            f"hawk.steps({name!r}): the kernel commits no Mutable plane, so "
            "there is no state to step")

    depth = 0
    carries = [LoopCarry(depth, j, s.name, s.ttype) for j, s in enumerate(commits)]
    flag = LoopCarry(depth, len(commits), "finished", _FLAG)
    by_name = {s.name: c for s, c in zip(commits, carries)}
    mapping = {id(leaf): by_name[leaf.name] for leaf in _prior_reads(sinks)}

    auto = K == "auto"
    if auto and any(n == FUSED_STEPS_PLANE for _, n in kernel.walk.arg_spec):
        raise HawkError(
            f"hawk.steps({name!r}, 'auto'): {FUSED_STEPS_PLANE!r} is reserved "
            "for the trip count of an automatic kernel; rename that parameter")

    roots = tuple(s.value for s in commits) + (finish.value,)
    try:
        new = substitute(roots, mapping)
    except HawkError as exc:
        raise HawkError(
            f"hawk.steps({name!r}): the step body cannot be carried — {exc}"
        ) from None
    cond = new[-1]
    nexts = tuple(new[:-1]) + (
        Select(cond, Const(1.0, _FLAG), Const(0.0, _FLAG), _FLAG),)
    inits = tuple(Leaf("prior_read", "mutable", s.name, s.ttype)
                  for s in commits) + (Const(0.0, _FLAG),)
    count = None
    if auto:
        count = _clamped_word()
    loop = build_loop(LoopIndex(depth, "step"), (*carries, flag), inits, nexts,
                      start=0, stop=AUTO_K_MAX if auto else K, step=1,
                      exit_cond=cond, exit_values=nexts, count=count)
    out: list[Node] = [Assign(s.name, LoopValue(loop, j), s.ttype)
                       for j, s in enumerate(commits)]
    done = make("gt", (LoopValue(loop, len(commits)), Const(0.5, _FLAG)))
    out.append(Finish(finish.name, done, steps=K))
    return Kernel(out_name, tuple(out), kernel.planes, kind=kernel.kind,
                  step=step, one_step=kernel if auto else None)


def _clamped_word() -> Node:
    """The trip count of a ``steps="auto"`` loop: the ``fused_steps`` word,
    clamped below at 0 so a negative word runs no step. The bound above is
    the word itself, a runtime value: a runner may ask one launch for every
    step left in its run (``AUTO_K_MAX`` stays the static range a host
    tile is sized for and the policy's largest band word)."""
    word = At(Leaf("table_read", "lookup", FUSED_STEPS_PLANE, _COUNT),
              Const(0, _COUNT), _COUNT)
    zero = Const(0, _COUNT)
    return Select(make("lt", (word, zero)), zero, word, _COUNT)


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
    """Every launch-start-read leaf OBJECT the sinks reach (by identity:
    each read before a plane's first store mints its own leaf, and every
    one becomes the carry)."""
    out, seen, stack = [], set(), list(sinks)
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Leaf) and node.kind == "prior_read":
            out.append(node)
        stack.extend(node.operands)
    return out
