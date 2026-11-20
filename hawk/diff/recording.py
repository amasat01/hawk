# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The kernels a recorded run takes beside its primal, and what hawk states of each.

A step kernel that finishes its samples is recorded with its taping twin (:func:`~hawk.diff.tape.tape`) and the vjp of its
step (:func:`~hawk.diff.tape.step_vjp`); a kernel that takes a fixed ``K`` steps per launch also has the vjp of all ``K`` in one
launch. :func:`recording` makes them and states, for each, the facts only the differentiation knows: which of its slots is the
tape, the step counter and the switch of the tape, which primal slot every other slot is, and in which form (a value, a cotangent
arriving, a cotangent leaving, a gradient, a row of the tape). The build hands them to the library with the primal, which checks
them and writes the derived record of each unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .. import _ir
from .tape import APPLIED_STEPS, T0_SUFFIX, TAPE_ON, TAPE_PLANE, step_vjp, tape, taped_planes, tape_width

__all__ = ["DerivedUnit", "recording"]

#: The planes a derived unit holds that no link names: the tape and its switch, the step counter, the word of fused steps and the
#: finished counter the library owns.
_UNLINKED = (TAPE_PLANE, TAPE_ON, APPLIED_STEPS, _ir.FUSED_STEPS_PLANE, _ir.FINISHED_PLANE)


@dataclass(frozen=True)
class DerivedUnit:
    """One kernel of a recording: ``role`` is ``"tape"``, ``"tape_fused"``, ``"step_vjp"`` or ``"fused_vjp"``, ``kernel`` the
    kernel to build and ``facts`` what is stated of it (the members of the request's derived entry, by their request names).
    ``window`` is the steps a taping twin keeps, ``None`` for a vjp."""

    role: str
    kernel: Any
    facts: dict
    window: int | None = None


def _link(slot: str, what: str, form: str, of: str | None, reduced: bool = False) -> dict:
    """One link of a derived unit's slot, in the request's words."""
    return {"slot": slot, "what": what, "form": form, "of": of, "reduced": reduced}


def _time_of(step: Any) -> str | None:
    """The name of the time plane of a kernel declaring ``time_step=``, else ``None``."""
    return step.time if getattr(step, "time_step", None) else None


def _twin_facts(step: Any, twin: Any, fused: bool) -> dict:
    """What is stated of a taping twin: its tape, its taped planes, the start copy of the time plane when the kernel has one, and
    one value link for every other slot."""
    taped = [{"plane": name, "rows": _ir.element_count(ttype)} for name, ttype in taped_planes(step)]
    names = [name for _, name in _ir.arg_spec(twin.walk)]
    facts = {"tape": TAPE_PLANE, "tape_rows": tape_width(step), "taped": taped, "steps_slot": APPLIED_STEPS, "tape_on": TAPE_ON,
             "links": []}
    time = _time_of(step)
    for name in names:
        if name in _UNLINKED:
            continue
        if time is not None and name == time + T0_SUFFIX:
            facts["time0"] = name
            facts["links"].append(_link(name, "Time0", "Value", time))
        else:
            facts["links"].append(_link(name, "PrimalSlot", "Value", name))
    if fused:
        facts["fused_steps"] = _ir.FUSED_STEPS_PLANE
    return facts


def _vjp_facts(primal: Any, step: Any, vjp: Any) -> dict:
    """What is stated of the vjp of a step: a cotangent arriving at a state plane, a cotangent leaving it, a gradient of any
    other differentiated input (summed over samples when the input is one uniform), a row of the tape for a taped state read, and
    a value for the rest. The step's ``next_values`` and the vjp's ``cotangents`` and ``gradients`` carry the pairing."""
    role_of = {name: role for role, name in _ir.arg_spec(primal.walk)}
    taped = {name for name, _ in taped_planes(step)}
    states = set(vjp.next_values.values())
    time = _time_of(step)
    links = []
    for _, name in _ir.arg_spec(vjp.walk):
        if name in vjp.cotangents:
            links.append(_link(name, "PrimalSlot", "AdjointIn", vjp.next_values[vjp.cotangents[name]]))
        elif name in vjp.gradients:
            of = vjp.gradients[name]
            if time is not None and of == time + T0_SUFFIX:
                links.append(_link(name, "Time0", "Grad", time))
            elif of in states:
                links.append(_link(name, "PrimalSlot", "AdjointOut", of))
            else:
                links.append(_link(name, "PrimalSlot", "Grad", of, reduced=role_of[of] == "uniform"))
        elif name == APPLIED_STEPS:
            links.append(_link(name, "Steps", "Value", APPLIED_STEPS))
        elif name in taped:
            links.append(_link(name, "TapeRow", "Value", name))
        else:
            links.append(_link(name, "PrimalSlot", "Value", name))
    return {"links": links}


def recording(kernel: Any, window: int) -> tuple:
    """The kernels of a recorded run of ``kernel`` and what is stated of each, as a tuple of :class:`DerivedUnit`: the taping
    twin that keeps ``window`` steps, the vjp of one step, and, for a kernel that takes a fixed ``K`` steps per launch, the vjp of
    ``K`` steps in one launch.

    A kernel declaring ``time_step=`` has the start copy of its time plane in the twin (a ``Time0`` value link) and its gradient
    in the vjp (a ``Time0`` gradient link, which the library adds to the gradient of the time plane). The refusals are
    :func:`~hawk.diff.tape.tape`'s and :func:`~hawk.diff.tape.step_vjp`'s."""
    from ..trace.kernel import Kernel
    from ..trace.steps import _fused_steps_of

    step = getattr(kernel, "step", None) or getattr(kernel, "one_step", None) or kernel
    twin = tape(kernel, window)
    fused = step is not kernel
    units = [DerivedUnit("tape_fused" if fused else "tape", twin, _twin_facts(step, twin, fused), window)]
    vjp = Kernel(f"{kernel.name}_step_vjp", step_vjp(kernel))
    units.append(DerivedUnit("step_vjp", vjp, _vjp_facts(kernel, step, vjp)))
    if _fused_steps_of(kernel) is not None:
        many = Kernel(f"{kernel.name}_fused_vjp", step_vjp(kernel, steps_reverse="fused"))
        units.append(DerivedUnit("fused_vjp", many, _vjp_facts(kernel, step, many)))
    return tuple(units)
