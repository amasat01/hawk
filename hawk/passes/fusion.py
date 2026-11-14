# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Where a chain is cut (:func:`hawk.chain`): one cost comparison per member boundary, keyed on a device row.

A segment's registers are estimated from its IR: ``est = R0 + r x maxlive``, ``maxlive`` the widest set of values live
at one point of the composed body's canonical order (a plane read is live from the kernel start, as the compiler hoists
the loads; a value the placement pass puts in a scratch plane counts nothing) and ``r`` the registers one element takes
(2 for f64 and 1 for f32 on a device; 1 vector register on a host). A boundary stays fused while fusing costs no more than
the cut: ``N x c_local x 2 x frame`` (the frame is ``4 x max(0, est - R_file)`` bytes, past what its parts hold alone) against ``L + N x c_plane x (2 x
crossing + reloaded)`` bytes. On a CUDA device the frame is cached and every boundary fuses; on a host the rule fuses up to
the register line. After compile the estimate fences the kernel (:func:`fence`): a chain never ships past it.
"""

from __future__ import annotations

import platform
from collections.abc import Sequence
from pathlib import Path

from .chain import bind
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


def _segment_walk(members: Sequence, a: int, b: int):

    from .. import wings
    later = set().union(*({lf.name for lf in _leaves(k)} for k in members[b:]))
    sinks = bind(members[a:b], keep=tuple(later)).sinks
    return wings.view("raptor", "base", "cells").walk(sinks)


def _leaves(kernel) -> list:
    return [n for n in kernel.walk.order if isinstance(n, Leaf)]


def cuts(members: Sequence, card: dict, scalar_type: str, samples: int) -> list[int]:
    """The boundaries to cut, greedy left to right over maximal segments: boundary ``b`` (before member ``b``) is cut
    when fusing it costs more than the cut. A row whose local-frame cost is not measured fuses up to ``R_file``."""
    real = 4 if scalar_type in ("float32", "f32", "banded") else 8
    crossing = bind(members).crossing
    out, a = [], 0
    shapes = {s.name: s.ttype for k in members for s in k.sinks}
    for b in range(1, len(members)):
        reads = {lf.name for k in members[b:] for lf in _leaves(k)}
        if any(len(shapes[n].shape) > 1 for k in members[:b] for n in (s.name for s in k.sinks) if n in reads):
            continue                            # a matrix does not cross a cut
        # the frame fusing adds: a part already past the file costs the same in every arrangement
        alone = max(estimate(_segment_walk(members, a, b), card, scalar_type),
                    estimate(_segment_walk(members, b, b + 1), card, scalar_type))
        frame = 4 * max(0, estimate(_segment_walk(members, a, b + 1), card, scalar_type) - max(card["R_file"], alone))
        if card["c_local_over_plane"] is None:
            fused = frame == 0
        else:
            reloaded = len({lf.name for k in members[a:b] for lf in _leaves(k)}
                           & {lf.name for k in members[b:] for lf in _leaves(k)}) * real
            cost_cut = card["L_us"] + samples * card["c_plane_us_per_B"] * (2 * crossing[b - 1] * real + reloaded)
            fused = samples * card["c_plane_us_per_B"] * card["c_local_over_plane"] * 2 * frame <= cost_cut
        if not fused:
            out.append(b)
            a = b
    return out


def fence(regs: int, stack_bytes: int, est: int) -> bool:
    """Whether a compiled kernel stays inside its estimate: ``regs + stack / 4 <= 1.6 x est + 16``."""
    return regs + stack_bytes / 4 <= FENCE_RATIO * est + FENCE_SLACK
