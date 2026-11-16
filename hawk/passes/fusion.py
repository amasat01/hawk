# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The register estimate of a kernel and the fence a compiled chain never ships past (:func:`hawk.chain`).

A kernel's registers are estimated from its IR: ``est = R0 + r x maxlive``, ``maxlive`` the widest set of values live
at one point of the canonical order (a plane read is live from the kernel start, as the compiler hoists
the loads; a value the placement pass puts in a scratch plane counts nothing) and ``r`` the registers one element takes
(2 for f64 and 1 for f32 on a device; 1 vector register on a host). After compile the estimate fences the kernel
(:func:`fence`): a chain never ships past it.
"""

from __future__ import annotations

from .chain import bind        # noqa: F401  (the device backend's report reads the chain's crossings through this module)
from raptor.ir.nodes import Const, Leaf, Sink

#: The fence: registers plus frame words may pass the estimate by this factor plus :data:`FENCE_SLACK` registers.
FENCE_RATIO = 1.6
FENCE_SLACK = 16


def _elements(ttype) -> int:
    count = 1
    for d in ttype.shape:
        count *= d
    return count


def maxlive(walk, wide_min: int | None = None) -> int:
    """The most elements per sample live at one point of ``walk``'s canonical order. A value is live from its node (a
    plane read from the start) to its last reader; a sink ends nothing of its own and a literal holds no register. A
    value of ``wide_min`` elements or more is placed in a scratch plane and counts nothing."""
    order = walk.order
    last = {}
    for i, node in enumerate(order):
        for o in node.operands:
            last[id(o)] = i
    delta = [0] * (len(order) + 1)
    for i, node in enumerate(order):
        if isinstance(node, (Sink, Const)) or id(node) not in last:
            continue
        width = _elements(node.ttype)
        if wide_min is not None and width >= wide_min:
            continue
        delta[0 if isinstance(node, Leaf) else i] += width
        delta[last[id(node)] + 1] -= width
    best = live = 0
    for d in delta:
        live += d
        best = max(best, live)
    return best


def row(scalar_type: str, spec, host: bool = False) -> dict:
    """The fusion constants of the build's device (``hawk/targets/cards/fusion.json``), or this host's."""
    return spec.fusion(scalar_type, host)


def estimate(walk, card: dict, scalar_type: str, wide_min: int | None = None) -> int:
    """Predicted registers of the kernel ``walk``: ``R0 + r x maxlive``."""
    per = 1 if card["R0"] == 0 or scalar_type in ("float32", "f32", "banded") else 2
    return card["R0"] + per * maxlive(walk, wide_min)


def fence(regs: int, stack_bytes: int, est: int) -> bool:
    """Whether a compiled kernel stays inside its estimate: ``regs + stack / 4 <= 1.6 x est + 16``."""
    return regs + stack_bytes / 4 <= FENCE_RATIO * est + FENCE_SLACK
