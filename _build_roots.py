# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The two header roots the build of ``hawk._core`` compiles against, for the build alone.

``hawk._core`` is one translation unit over aether's index typedef and eagle's ABI header.
``CMakeLists.txt`` runs this file as a script and ``_build_backend.py`` loads it by path: neither
can import the hawk package, whose import checks the very extension being built. Nothing in the
installed package reads it.

An unresolvable root raises and names the variable that fixes it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
#: The folder sibling checkouts of the family hang off.
WORKSPACE = _HERE.parent
#: The header copy a source distribution carries: the last candidate for both roots.
SDIST_INCLUDE = _HERE / "_sdist_include"
#: The source distribution carries only what the extension includes, so aether is probed there
#: by that one header.
_SDIST_PROBES = {"aether/aether.h": "aether/typedefs.h"}


class RootError(Exception):
    """An include root that cannot be resolved."""


def aether_include() -> str:
    """Return the aether header root.

    ``$HAWK_AETHER_INCLUDE``, else an environment prefix, else the sibling checkout, else the
    copy a source distribution carries.
    """
    candidates = [Path(p) / "include" for p in _prefixes()] + [WORKSPACE / "aether", SDIST_INCLUDE]
    return _root("HAWK_AETHER_INCLUDE", candidates, "aether/aether.h")


def eagle_include() -> str:
    """Return the root ``plugin/gref_abi.h`` hangs off.

    ``$HAWK_EAGLE_INCLUDE``, else an environment prefix, else the sibling checkout, else the copy
    a source distribution carries. A header that does not define the current ABI macro is passed
    over: an environment's prefix can hold an older eagle.
    """
    prefixes = _prefixes()
    candidates = ([Path(p) / "include" / "eagle" for p in prefixes]
                  + [Path(p) / "include" for p in prefixes]
                  + [WORKSPACE / "eagle", SDIST_INCLUDE])
    return _root("HAWK_EAGLE_INCLUDE", candidates, "plugin/gref_abi.h", "EAGLE_AETHER_ABI_V2")


def roots() -> tuple:
    """Return both roots, aether first.

    ``EAGLE_ABI_INDEX_T`` binds to ``AETHER_INDEX_T`` only when aether's typedef is visible as
    ``gref_abi.h`` is preprocessed; the other order narrows the index width silently.
    """
    return (aether_include(), eagle_include())


def _prefixes() -> list:
    """The prefixes headers are installed under, the running interpreter's first."""
    found = (sys.prefix, os.environ.get("CONDA_PREFIX"), os.environ.get("PREFIX"))
    return [p for p in found if p]


def _root(var: str, candidates: list, probe: str, must_contain: str | None = None) -> str:
    """Return ``$var`` when set, else the first candidate holding ``probe`` (with ``must_contain``)."""
    override = os.environ.get(var)
    if override:
        return override
    for candidate in candidates:
        name = _SDIST_PROBES.get(probe, probe) if candidate == SDIST_INCLUDE else probe
        header = candidate / name
        if not header.is_file():
            continue
        if must_contain is not None and must_contain not in header.read_text():
            continue
        return str(candidate)
    detail = "" if must_contain is None else f" defining {must_contain!r}"
    raise RootError(f"cannot resolve the include root holding {probe!r}{detail}: tried "
                    f"{[str(c) for c in candidates]}. Set ${var}.")


if __name__ == "__main__":  # the CMake entry point: one root per line
    sys.stdout.write("\n".join(roots()) + "\n")
