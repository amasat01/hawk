# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Build backend: scikit-build-core, plus the headers a source distribution needs.

``hawk._core`` compiles against two headers from other repositories, aether's
``aether/typedefs.h`` and eagle's ``plugin/gref_abi.h``. A source distribution
carries nothing from outside itself, so the sdist hook copies those two headers,
and every header they include, into ``_sdist_include/`` (git-ignored, listed in
``sdist.include``) and removes the copy again afterwards. The two roots are found
by ``hawk/_roots.py``, the same resolver the build uses, which also tries
``_sdist_include/`` as its last resort, so a checkout keeps building against
its live roots. Every other hook runs scikit-build-core unchanged.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
from pathlib import Path

from scikit_build_core.build import *  # noqa: F403  (every other hook, unchanged)
from scikit_build_core.build import __all__  # noqa: F401
from scikit_build_core.build import build_sdist as _build_sdist

_HERE = Path(__file__).resolve().parent
_INCLUDE = _HERE / "_sdist_include"
_INCLUDE_LINE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.MULTILINE)


def _load_roots():
    # By path, not `import hawk`: hawk/__init__ imports the extension being built.
    spec = importlib.util.spec_from_file_location(
        "_hawk_sdist_roots", _HERE / "hawk" / "_roots.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_closure(roots: list, start: list) -> None:
    """Copy each header in ``start`` and, transitively, every header it
    includes that resolves under one of ``roots``, keeping paths relative to
    the root it was found in."""
    pending = list(start)
    seen = set()
    while pending:
        rel, base = pending.pop()
        for root in [base, *roots]:
            source = root / rel
            if source.is_file():
                break
        else:
            continue  # a system or standard header
        source = source.resolve()
        if source in seen:
            continue
        seen.add(source)
        # the innermost root holding it: one root may sit inside another
        owner = max((r.resolve() for r in roots if source.is_relative_to(r.resolve())),
                    key=lambda r: len(r.parts))
        target = _INCLUDE / source.relative_to(owner)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        for name in _INCLUDE_LINE.findall(source.read_text()):
            pending.append((name, source.parent))


def build_sdist(sdist_directory, config_settings=None):
    """Write the sdist with the aether and eagle headers ``_core`` needs."""
    roots_module = _load_roots()
    aether, eagle = (Path(r) for r in roots_module.roots())
    shutil.rmtree(_INCLUDE, ignore_errors=True)
    _copy_closure([aether, eagle], [("aether/typedefs.h", aether),
                                    ("plugin/gref_abi.h", eagle)])
    try:
        return _build_sdist(sdist_directory, config_settings)
    finally:
        shutil.rmtree(_INCLUDE, ignore_errors=True)
