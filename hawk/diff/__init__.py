# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""IR -> IR autodiff, symbolic and in both directions. :func:`~hawk.diff.transform.vjp`
and :func:`~hawk.diff.transform.jvp` synthesise
a NEW IR from a primal one, driven by the per-node-KIND rule table
(the library's rules, :func:`hawk._ir.rule_vjp` and :func:`hawk._ir.rule_jvp`); the result is an ordinary HAWK IR that goes through
the same canonical walk, emitter and cache as any primal.

:func:`~hawk.diff.params.param_planes` reads a kernel's uniforms from one-cell
planes, so a recorded launch can change them without re-capture;
:func:`~hawk.diff.tape.tape`, :func:`~hawk.diff.tape.step_vjp`, :func:`~hawk.diff.tape.step_jvp`, :func:`~hawk.diff.tape.step_function` and
:func:`~hawk.diff.tape.refine_entry` build
the kernels a launching runtime needs to reverse a whole rollout of a step
kernel and to refine its stop time; :mod:`hawk.diff.event` reads a stop rule
as a smooth guard.
"""

from .params import CELL_PLANE, param_planes
from .tape import refine_entry, step_function, step_jvp, step_vjp, tape
from .transform import ADJOINT_PREFIX, TANGENT_PREFIX, Derived, jvp, vjp, with_derivative

__all__ = ["vjp", "jvp", "with_derivative", "tape", "step_vjp", "step_jvp", "step_function", "refine_entry", "param_planes", "ADJOINT_PREFIX",
           "CELL_PLANE", "TANGENT_PREFIX", "Derived"]
