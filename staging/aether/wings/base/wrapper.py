# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The scalar-mode seam: the ``Real`` a translation unit resolves to, by mode id."""

from __future__ import annotations

from dataclasses import dataclass

from raptor.errors import RaptorError


@dataclass(frozen=True)
class ScalarMode:
    """One compiled scalar mode (seam): its id, the storage ``Real`` a
    kernel resolves to and the headers that spell it. ``cells``: a ``Real``
    in storage is an 8-byte cell and a value in registers the three-limb
    ``Band``, so the body takes the cell form."""

    id: str
    real_spelling: str
    includes: str = ""
    cells: bool = False


#: The BUILT modes.
FLOAT64 = ScalarMode("float64", "double")
FLOAT32 = ScalarMode("float32", "float")
BANDED = ScalarMode("banded", "aether::banded::BandedReal",
                    "#include <aether/banded/banded.h>\n#include <aether/banded/BandedRealOps.h>\n",
                    cells=True)

SCALAR_MODES = {m.id: m for m in (FLOAT64, FLOAT32, BANDED)}


def scalar_mode(name: str) -> ScalarMode:
    """Resolve a scalar mode by id."""
    if name in SCALAR_MODES:
        return SCALAR_MODES[name]
    raise RaptorError(
        f"unknown scalar mode {name!r}; the seam declares {tuple(SCALAR_MODES)}"
    )
