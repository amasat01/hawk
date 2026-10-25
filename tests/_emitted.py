# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Every emitted string the rows sweep, built once.

The rows differ in what they assert, not in what they look at: (no legacy headers),
 (no geometry), (one body, two wrappers) and the compile smoke all read
the SAME corpus x both wrappers x both built scalar modes, so a spelling that
escapes one row is still in front of the others.
"""

from __future__ import annotations

import _kernels as K

from hawk.emit import FLOAT32, FLOAT64, render_body, render_source
from hawk.ir import canonical
from hawk.targets import TARGETS

MODES = (FLOAT64, FLOAT32)


def cases():
    """``(name, sinks, walk)`` for every corpus kernel plus one derivative IR."""
    out = [(name, k.sinks, k.walk) for name, k in K.CORPUS.items()]
    sinks = K.drag_vjp()
    out.append(("drag_vjp", sinks, canonical(sinks)))
    return out


def sources():
    """Every emitted translation unit: corpus x target x scalar mode."""
    for name, sinks, walk in cases():
        for target in TARGETS.values():
            if target.wrapper is None:
                continue
            for mode in MODES:
                yield name, target, mode, render_source(name, sinks, walk, target,
                                                        mode=mode)


def bodies():
    """Every emitted BODY string (wrapper-independent by construction)."""
    for name, sinks, walk in cases():
        yield name, render_body(sinks, walk)
