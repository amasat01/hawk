# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The device feature classes the placement pass's thresholds are keyed by (:mod:`hawk.passes.wide`).

Where a per-lane value is cheaper in registers than in a plane depends on the card: the FP64:FP32 rate, the register file and
the threads an SM holds (how many lanes share the L1 and the local-memory path), and whether local memory is cached. A table
is MEASURED on one class (``hawk/targets/cards/_probe_wide.py --dump-table``) and never written by hand; today there is one, ``sm61``. The tables are variant DATA (``variants/cuda/data.py``); this module builds them.

A build's device arch maps to a feature tuple (the variant's ``arch_features``: facts of the hardware), and the build uses the table of
the NEAREST class by feature distance: an exact match when the arch is a class of its own, else the closest measured one. The
class used is recorded in the sidecar (``launch.wide.class``) and in the unit key, so a table measured later for a nearer class
changes the unit, and a unit built from the sm_61 table on another card says so. The vocabulary (:class:`Features`,
:class:`Tables`) and the pure selection rule live here, in the SIMT component's variant loader; the tables are the variant's."""

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


def distance(a: Features, b: Features) -> float:
    """How far two feature tuples are apart: the log2 gaps of the rate, registers and threads, and 1 for a local memory
    one caches and the other does not."""
    return (abs(math.log2(a.fp64_ratio / b.fp64_ratio)) + abs(math.log2(a.regs_per_sm / b.regs_per_sm))
            + abs(math.log2(a.threads_per_sm / b.threads_per_sm)) + (a.local_cached != b.local_cached))


def nearest(want: Features, classes: dict) -> str:
    """The name of the class whose features (``{name: Features}``) are nearest ``want``: ties go to the lower name."""
    return min(classes, key=lambda name: (distance(want, classes[name]), name))


def arch_features(variant: dict) -> dict:
    """The hardware facts of the archs a build may target in ``variant`` (compute capability -> :class:`Features`)."""
    return {cc: Features(*row) for cc, row in variant["arch_features"].items()}


def classes_of(variant: dict) -> dict:
    """The measured classes of ``variant``: ``{name: (features, tables)}``."""
    feats = arch_features(variant)
    return {name: (feats[row["capability"]], Tables(runtime_min=row["runtime_min"], wide_min=row["wide_min"]))
            for name, row in variant["wide_classes"].items()}


def capability(arch: str) -> int:
    """The compute capability of an arch string such as ``sm_80`` or ``sm_90a``."""
    found = re.search(r"(\d+)", arch)
    if not found:
        raise ValueError(f"not a CUDA arch: {arch!r}")
    return int(found.group(1))


def features_of(arch: str, variant: dict) -> Features:
    """The features of ``arch``: its own row, else the row of the nearest capability, in ``variant``'s table."""
    known_rows = arch_features(variant)
    cc = capability(arch)
    return known_rows[min(known_rows, key=lambda known: (abs(known - cc), known))]


def select(arch: str, variant: dict, features: Features | None = None) -> str:
    """The name of the nearest measured class of ``variant`` to ``arch`` (or to ``features``): ties go to the lower name."""
    rows = classes_of(variant)
    want = features if features is not None else features_of(arch or variant["default_arch"], variant)
    return nearest(want, {name: row[0] for name, row in rows.items()})


def tables(name: str, variant: dict) -> Tables:
    """The thresholds of class ``name`` of ``variant``."""
    return classes_of(variant)[name][1]
