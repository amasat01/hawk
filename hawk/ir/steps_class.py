# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""When a fused K-step reverse beats K step-by-step launches (``steps_reverse="auto"``).

A measured table, never a constant: the rows are what a probe timed (``tests/_probe_steps_reverse.py --dump-table``), and a
target with no row answers ``per_step``. A row is ``(state width, K, narrowest regime, widest regime)`` the fused launch won
across, and the regime is the device-normalised batch: lanes per resident thread (``N / (SMs x resident threads per SM)``, the
caller reads both from the device), or the plain ``N`` on the host (whose rows were measured on one 8-core machine). The nearest
measured width and K (log2 distance) is matched and fused is chosen inside its regime."""

from __future__ import annotations

import math

#: ``{target: ((width, K, regime_min, regime_max), ...)}``, fused measured to win for ``regime_min <= regime <= regime_max``.
#: ``host`` (8 Xeon cores, compiled hawk.run, regime = N): the 4-scalar oscillator probe, fused 7x / 5x at N=1 / 64, 1.2x at 1024
#: and slower from 16384 (K 8); K 4 unchanged. Width 2, K 8 is the wide matvec of the feature card (a 24 by 512 table, ``y``
#: overwritten each step): fused 30-250x at every N from 1 to 65536. ``device`` (P2000, 5 SMs x 2048 threads, regime = N / 10240):
#: the same matvec, fused 0.43x at N=64, 2.2x at 1024, 2.8x at 16384 and 65536 (L4: 1.8x at 65536).
_ROWS = {
    "host": ((4, 8, 1, 1024), (4, 4, 1, 64), (2, 8, 1, 65536)),
    "device": ((2, 8, 0.1, 6.4),),
}


def _choose(device: bool, width: int, K: int, n: int, resident: int = 1) -> str:
    """``"fused"`` when the nearest measured regime won at ``n`` samples on a target holding ``resident`` threads
    (``SM count x resident threads per SM``; ignored on the host), else ``"per_step"``."""
    rows = _ROWS["device" if device else "host"]
    w, k, lo, hi = min(rows, key=lambda r: (abs(math.log2(width / r[0])) + abs(math.log2(K / r[1])), r))
    return "fused" if lo <= n / (resident if device else 1) <= hi else "per_step"
