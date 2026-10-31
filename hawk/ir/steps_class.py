# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""When a fused K-step reverse beats K step-by-step launches (``steps_reverse="auto"``).

A measured table, never a constant: the rows of a feature class are what a probe
timed (``tests/_probe_steps_reverse.py --dump-table``), and a class with no row
answers ``per_step``. A device build takes the class of :func:`hawk.ir.wide_class.select`
for its arch; the host has its own class. A row is ``(state width, K, widest batch N)``
the fused launch won at: a regime is matched to the nearest measured width and K, and
fused wins up to that batch."""

from __future__ import annotations

import math

from .wide_class import select

#: The host's feature class: the machine the host rows were measured on (the IR names no target).
_HOST = "x86_64"

#: ``{class: ((width, K, n_max), ...)}``. ``x86_64``: 8 cores, compiled hawk.run, fused
#: 8x / 4x faster at N=1, 5x / 3.3x at N=64 and level or slower from N=1024 (width 4, K 8 / 4).
#: The device classes are empty: the card has not been measured (GPU fusion won 1.0-1.3x at most).
_ROWS = {
    _HOST: ((4, 8, 64), (4, 4, 64)),
}


def _class_of(device: bool, arch: str = "") -> str:
    """The row class of a target: the host's, or the nearest measured device class."""
    return select(arch) if device else _HOST


def _choose(device: bool, width: int, K: int, n: int, arch: str = "") -> str:
    """``"fused"`` when the class's nearest measured regime won at ``n`` samples, else ``"per_step"``."""
    rows = _ROWS.get(_class_of(device, arch), ())
    if not rows:
        return "per_step"
    w, k, n_max = min(rows, key=lambda r: (abs(math.log2(width / r[0])) + abs(math.log2(K / r[1])), r))
    return "fused" if n <= n_max else "per_step"
