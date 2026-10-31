# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The device feature classes the placement pass's thresholds are keyed by (:mod:`hawk.ir.wide`).

Where a per-lane value is cheaper in registers than in a plane depends on the card: the FP64:FP32 rate, the register file and
the threads an SM holds (how many lanes share the L1 and the local-memory path), and whether local memory is cached. A table
is MEASURED on one class (``tests/_probe_wide.py --dump-table``) and never written by hand; today there is one, ``sm61``.

A build's device arch maps to a feature tuple (:data:`ARCH_FEATURES`: facts of the hardware), and the build uses the table of
the NEAREST class by feature distance: an exact match when the arch is a class of its own, else the closest measured one. The
class used is recorded in the sidecar (``launch.wide.class``) and in the unit key, so a table measured later for a nearer class
changes the unit, and a unit built from the sm_61 table on another card says so."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Features:
    """What the thresholds depend on: ``fp64_ratio`` FP32 lanes per FP64 lane, ``regs_per_sm`` 32-bit registers,
    ``threads_per_sm`` resident threads, ``local_cached`` whether local memory goes through the L1."""

    fp64_ratio: int
    regs_per_sm: int
    threads_per_sm: int
    local_cached: bool


@dataclass(frozen=True)
class Tables:
    """The measured thresholds of one class: ``runtime_min`` ``{pattern: {scalar bytes: widest width kept in registers}}``
    (patterns ``read``, ``slot_read``, ``rotate``, ``write``, ``accum``) and ``wide_min`` ``{scalar bytes: widest value with only
    compile-time indices kept in registers}``."""

    runtime_min: dict
    wide_min: dict


#: The hardware facts of the archs a build may target (compute capability -> features).
ARCH_FEATURES = {
    61: Features(32, 65536, 2048, True),
    70: Features(2, 65536, 2048, True),
    75: Features(32, 65536, 1024, True),
    80: Features(2, 65536, 2048, True),
    86: Features(64, 65536, 1536, True),
    89: Features(64, 65536, 1536, True),
    90: Features(2, 65536, 2048, True),
}

#: The measured classes: ``{name: (features, tables)}``. ``sm61``: Quadro P2000, widths 2 to 128 x read / fill-then-read / rotate / write,
#: three problem sizes (tests/_probe_wide.py); the faster form wins and a spill-free one only on a tie.
CLASSES = {
    "sm61": (ARCH_FEATURES[61], Tables(
        runtime_min={
            "read": {4: 16, 8: 2},
            "slot_read": {4: 16, 8: 4},
            "rotate": {4: 16, 8: 0},
            "write": {4: 96, 8: 8},
            "accum": {4: 4096, 8: 4096},       # an in-place accumulation (reverse sum): slower resident at 512, no width found
        },
        wide_min={4: 128, 8: 64})),
}

#: The class a build with no device arch (a direct call of the pass) uses.
DEFAULT_ARCH = "sm_61"


def capability(arch: str) -> int:
    """The compute capability of an arch string such as ``sm_80`` or ``sm_90a``."""
    found = re.search(r"(\d+)", arch)
    if not found:
        raise ValueError(f"not a CUDA arch: {arch!r}")
    return int(found.group(1))


def features_of(arch: str) -> Features:
    """The features of ``arch``: its own row, else the row of the nearest capability."""
    cc = capability(arch)
    return ARCH_FEATURES[min(ARCH_FEATURES, key=lambda known: (abs(known - cc), known))]


def distance(a: Features, b: Features) -> float:
    """How far two feature tuples are apart: the log2 gaps of the rate, registers and threads, and 1 for a local memory
    one caches and the other does not."""
    return (abs(math.log2(a.fp64_ratio / b.fp64_ratio)) + abs(math.log2(a.regs_per_sm / b.regs_per_sm))
            + abs(math.log2(a.threads_per_sm / b.threads_per_sm)) + (a.local_cached != b.local_cached))


def select(arch: str = "", features: Features | None = None) -> str:
    """The name of the nearest measured class to ``arch`` (or to ``features``): ties go to the lower name."""
    want = features if features is not None else features_of(arch or DEFAULT_ARCH)
    return min(CLASSES, key=lambda name: (distance(want, CLASSES[name][0]), name))


def tables(name: str) -> Tables:
    """The thresholds of class ``name``."""
    return CLASSES[name][1]
