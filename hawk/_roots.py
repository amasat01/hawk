# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""THE header-root resolver ((a)), in a module nothing else has to import.

Two consumers resolve the aether and eagle include roots and must agree
to the character: :mod:`hawk.compile.toolchain` (every emitted TU's
command line, the content-closure cache's key) and
``hawk/CMakeLists.txt`` (``hawk/_core``'s own command line).

The second consumer is why this file is a leaf module with no hawk
imports: CMake cannot ``import hawk.compile.toolchain``, since that
would run ``hawk/__init__``'s self-check, which imports ``hawk._core`` —
the extension the CMake run is about to build. So CMake loads this file
by path and calls the same two functions the Python driver calls.

Nothing here is skipped when a root is missing: an unresolvable root
raises and names the variable that would fix it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: The workspace root the four ``*-abi`` worktrees and ``aether`` hang off.
WORKSPACE = Path(__file__).resolve().parents[2]


class RootError(Exception):
    """An include root that cannot be resolved.

    Its own type, not :class:`hawk.ir.HawkError`: this module is loadable
    without the hawk package, so it cannot import HAWK's exception.
    :mod:`hawk.compile.toolchain` re-raises it as ``HawkError`` for every
    in-package caller."""


def aether_include() -> str:
    """The aether header root — ``$HAWK_AETHER_INCLUDE``, else a prefix, else
    the workspace checkout ((a)'s shape, applied to aether)."""
    return _root("HAWK_AETHER_INCLUDE",
                 [Path(p) / "include" for p in _prefixes()] + [WORKSPACE / "aether"],
                 "aether/aether.h")


def eagle_include() -> str:
    """The eagle header root: the directory ``plugin/gref_abi.h`` hangs
    off ((a)) — ``$HAWK_EAGLE_INCLUDE``, else ``<prefix>/include/eagle``,
    else the ``eagle`` checkout.

    The candidate must be an **aether-abi/2** header: a dev env's prefix
    can easily hold a pre-v2 eagle, so one not defining the v2 macro is
    skipped, not used."""
    return _root("HAWK_EAGLE_INCLUDE",
                 [Path(p) / "include" / "eagle" for p in _prefixes()]
                 + [Path(p) / "include" for p in _prefixes()]
                 + [WORKSPACE / "eagle", WORKSPACE / "eagle-abi"],
                 "plugin/gref_abi.h", must_contain="EAGLE_AETHER_ABI_V2")


def roots() -> tuple:
    """Both roots in (b)'s mandated order — aether first, eagle second:
    ``EAGLE_ABI_INDEX_T`` binds to ``AETHER_INDEX_T`` only if visible
    when ``gref_abi.h`` is preprocessed; backwards silently narrows it."""
    return (aether_include(), eagle_include())


def _prefixes() -> list:
    """The RUNNING interpreter's prefix first: the env whose python drives the
    build is the env whose headers the artifact is built against."""
    return [p for p in (sys.prefix, os.environ.get("CONDA_PREFIX"),
                        os.environ.get("PREFIX")) if p]


def _root(var: str, candidates: list, probe: str,
          must_contain: str | None = None) -> str:
    override = os.environ.get(var)
    if override:
        return override
    for candidate in candidates:
        header = candidate / probe
        if not header.is_file():
            continue
        if must_contain is not None and must_contain not in header.read_text():
            continue
        return str(candidate)
    detail = "" if must_contain is None else f" defining {must_contain!r}"
    raise RootError(
        f"cannot resolve the include root holding {probe!r}{detail}: tried "
        f"{[str(c) for c in candidates]}. Set ${var}."
    )


if __name__ == "__main__":       # the CMake entry point: one root per line
    sys.stdout.write("\n".join(roots()) + "\n")
