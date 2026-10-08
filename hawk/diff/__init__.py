# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""IR -> IR autodiff, symbolic and in both directions. :func:`~hawk.diff.transform.vjp`
and :func:`~hawk.diff.transform.jvp` synthesise
a NEW IR from a primal one, driven by the per-node-KIND rule table
(:mod:`hawk.diff.rules`); the result is an ordinary HAWK IR that goes through
the same canonical walk, emitter and cache as any primal.
"""

from .transform import ADJOINT_PREFIX, TANGENT_PREFIX, Derived, jvp, vjp

__all__ = ["vjp", "jvp", "ADJOINT_PREFIX", "TANGENT_PREFIX", "Derived"]
