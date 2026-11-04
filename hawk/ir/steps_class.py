# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""When a fused K-step reverse beats K step-by-step launches (``steps_reverse="auto"``).

A measured table, never a constant: the rows of a feature class are what a probe
timed (``tests/_probe_steps_reverse.py --dump-table``), and a class with no row
answers ``per_step``. A device build takes the class of :func:`hawk.ir.wide_class.select`
for its arch; the host has its own class. A row is ``(state width, K, narrowest N, widest N)``
the fused launch won across: a regime is matched to the nearest measured width and K, and
fused is chosen inside that batch range."""

from __future__ import annotations

import math

from .wide_class import select

#: The host's feature class: the machine the host rows were measured on (the IR names no target).
_HOST = "x86_64"

#: ``{class: ((width, K, n_min, n_max), ...)}``, fused measured to win for ``n_min <= N <= n_max``.
#: ``x86_64`` (8 Xeon cores, compiled hawk.run): the 4-scalar oscillator probe, fused 7x / 5x at N=1 / 64, 1.2x at 1024 and
#: slower from 16384 (K 8); K 4 unchanged. Width 2, K 8 is the wide matvec of the feature card (a 24 by 512 table, ``y``
#: overwritten each step): fused 30-250x at every N from 1 to 65536. ``sm61`` (P2000, the class sm_61..sm_89 share): the same
#: matvec, fused 0.43x at N=64, 2.2x at 1024, 2.8x at 16384 and 65536 (L4 sm_89: 1.8x at 65536).
_ROWS = {
    _HOST: ((4, 8, 1, 1024), (4, 4, 1, 64), (2, 8, 1, 65536)),
    "sm61": ((2, 8, 1024, 65536),),
}


def _class_of(device: bool, arch: str = "") -> str:
    """The row class of a target: the host's, or the nearest measured device class."""
    return select(arch) if device else _HOST


def _choose(device: bool, width: int, K: int, n: int, arch: str = "") -> str:
    """``"fused"`` when the class's nearest measured regime won at ``n`` samples, else ``"per_step"``."""
    rows = _ROWS.get(_class_of(device, arch), ())
    if not rows:
        return "per_step"
    w, k, n_min, n_max = min(rows, key=lambda r: (abs(math.log2(width / r[0])) + abs(math.log2(K / r[1])), r))
    return "fused" if n_min <= n <= n_max else "per_step"
