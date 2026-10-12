# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Whether the host compiler vectorises the loop over samples of an emitted kernel.

ONE predicate, :func:`vectorises`, answers it for the translation unit the emitter
writes (:func:`hawk.emit.backend.render_source` records it on the
:class:`~hawk.emit.backend.Source`) and for the cost counts a launching runtime
reads (:mod:`hawk.ir.cost`, per dispatch branch), so the two cannot drift apart.

Today it holds when the kernel is sample-local, no library call stays scalar under an
exact host profile, and no ``Loop`` has more trips than the compiler fully unrolls.
The scalar type does not matter: a float32 loop's guards and tallies are float32 too.
"""

from __future__ import annotations

from collections.abc import Iterable

from .loop_nodes import Loop
from .nodes import Node
from .walk import Walk

__all__ = ["UNROLL_LIMIT", "vectorises"]

#: The most trips of a ``Loop`` the host compiler fully unrolls (GCC's complete-peel
#: limit); a longer one stays a rolled loop and the sample loop around it is scalar.
UNROLL_LIMIT = 16


def _flat(nodes: Iterable[Node]):
    """``nodes`` and everything inside their loop bodies."""
    for node in nodes:
        yield node
        if isinstance(node, Loop):
            yield from _flat(node.body_nodes())


def vectorises(walk: Walk, profile: str | None = None, scalar_type: str | None = None,
               nodes: Iterable[Node] | None = None) -> bool:
    """Whether the host loop over samples running ``nodes`` (default: the whole
    kernel) is vectorised under host ``profile`` (default: the one in effect) and
    ``scalar_type`` (accepted for the callers' sake; the verdict is the same for every type)."""
    from ..compile.toolchain import EXACT_HOST_PROFILES, host_profile
    from .cost import is_library_call

    if walk.access.cls != "sample_local":
        return False
    exact = host_profile(profile) in EXACT_HOST_PROFILES
    for node in _flat(walk.order if nodes is None else nodes):
        if exact and is_library_call(node):
            return False
        if isinstance(node, Loop) and node.trip > UNROLL_LIMIT:
            return False
    return True
