# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""HAWK — Hardware Agnostic Writing of Kernels.

A from-scratch kernel-authoring toolchain: Python authoring surface,
tensor-typed IR, IR-level autodiff, aether-vocabulary codegen, and a C++
host execution path — emitting ``aether-abi/2`` artifacts eagle deploys
under the heterogeneous execution contract eagle enforces.

This module is the authoring surface. Importing it never imports
:mod:`hawk.ir`, :mod:`hawk.emit` or :mod:`hawk.compile`.

Runtime severance: no module under ``hawk/hawk/`` imports ``raptor`` or
``eagle`` at runtime, mechanically checked by
``hawk/tests/test_severance.py`` and by raptor's own conformance reader.

This module also adds the authoring names (``kernel``, the declaration
vocabulary, ``raw_device``, ``Quantity``), bound lazily through
:pep:`562`'s module ``__getattr__`` because :mod:`hawk.trace` imports
:mod:`hawk.ir` and forbids ``import hawk`` from pulling that in —
touching one authoring name imports the tracer, but importing the
package alone costs nothing beyond this module.

This module also runs the import-time build-digest/layout self-check
against :mod:`hawk._core` (below), here rather than at first launch
because the failure it catches — a stale, wrong-arch or
editable-shadowed binding — imports cleanly and then computes wrong
answers. ``hawk._core`` stamps its own sources' digest at build; this
module recomputes it from the tree when beside the package (a dev
checkout), or reads the same stamp from a generated module,
:mod:`hawk._build_digest`, when it is not (a wheel install).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from . import _contracts

#: The authoring names put in front of a kernel author — the one
#: canonical top-level surface, resolved lazily from :mod:`hawk.trace`
#: (and, for :data:`_FROM_IR`, from :mod:`hawk.ir`). Math functions
#: (``dot``, ``norm``, ``sqrt``, ...) live at ``hawk.math`` only.
#: ``TensorType``/``Wire`` stay out on purpose: their canonical path is
#: :mod:`hawk.ir`, not the top level.
_AUTHORING = (
    "kernel", "Kernel", "raw_device", "RawBlock", "Value", "Quantity",
    "Scalar", "Index", "Quat", "Vector", "Matrix", "Param",
    "Mutable", "Table", "Staged", "Wide", "WideOut", "Accum", "Reduce",
    "Terminated", "HawkError", "steps",
)

#: The subset of :data:`_AUTHORING` resolved from :mod:`hawk.ir` rather
#: than :mod:`hawk.trace` — today just ``HawkError``, an IR-tier name
#: :mod:`hawk.trace` never re-exports.
_FROM_IR = ("HawkError",)

#: The one authoring transform at the top level: ``hawk.steps(kernel, K)``.
_FROM_STEPS = ("steps",)

#: The three top-level convenience doors (plus the two per-array layout
#: markers, ``hawk.samples_first(x)`` / ``hawk.samples_last(x)``: which axis of
#: a per-sample plane holds the samples; see :mod:`hawk._plane_layout`) (:pep:`562`, lazy, pure
#: re-exports): ``{name: (submodule, attribute)}``. ``import hawk`` stays
#: light and the submodule spellings keep working -- these only ever add a
#: second spelling, never replace the first.
#:
#: ``build`` builds one or more traced kernels into a deployment unit on
#: disk (:func:`hawk.artifact.build_bundle`; the single-kernel PRIMITIVE
#: stays at :func:`hawk.artifact.build` under its own name, not re-exported
#: here). ``load`` opens a built unit's kernel back into a runnable object
#: (:func:`hawk.runtime.load`). ``run`` binds arguments and runs a loaded
#: kernel once, in place (:func:`hawk.runtime.run`).
_TOP_LEVEL_DOORS = {"build": ("artifact", "build_bundle"),
                    "load": ("runtime", "load"),
                    "run": ("runtime", "run"),
                    "samples_first": ("_plane_layout", "samples_first"),
                    "samples_last": ("_plane_layout", "samples_last")}

__all__ = ["__version__", *_AUTHORING, *_TOP_LEVEL_DOORS]

__version__ = "0.3.1"


# --------------------------------------------------------------------------- #
# The import-time self-check.
# --------------------------------------------------------------------------- #
#: The layout fields with a constant twin in :mod:`hawk._contracts`, by
#: their position in the exported ``eagle_layout_sizes`` array and by the
#: name a refusal has to say instead of an index.
#:
#: Field 3, ``sizeof(aether::idx_t)``, is deliberately absent: it is a
#: build axis (``-DAETHER_INDEX_T``), so pinning a number here would
#: refuse a legitimately-built 64-bit-index binding. Its agreement is
#: checked where it can actually disagree — the artifact's exported
#: value against this host's, at load.
_LAYOUT_TWINS = (
    (0, "GRefMirror", _contracts.GREF_MIRROR_SIZE),
    (1, "ScalarHandle", _contracts.SCALAR_HANDLE_SIZE),
    (2, "IntHandle", _contracts.INT_HANDLE_SIZE),
    (4, "PartitionTriple", _contracts.PARTITION_TRIPLE_SIZE),
)

#: The sources the digest covers, relative to the repo root — the same set
#: ``hawk/CMakeLists.txt`` globs and hashes at configure.
_DIGEST_SOURCE_GLOBS = ("src/*",)
_DIGEST_EXTRA_FILES = ("CMakeLists.txt",)


def _source_digest(root: Path) -> str:
    """Recompute ``_core``'s build digest from the tree at ``root``: for
    each covered file, in sorted relative-name order, append the name, a
    newline, the file's sha256, a newline; the digest is one sha256 over
    that whole string (the same rule ``hawk/CMakeLists.txt`` states)."""
    files = []
    for pattern in _DIGEST_SOURCE_GLOBS:
        files += [p for p in root.glob(pattern) if p.is_file()]
    files += [root / name for name in _DIGEST_EXTRA_FILES]
    stream = []
    for path in sorted(files, key=lambda p: p.relative_to(root).as_posix()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        stream.append(f"{path.relative_to(root).as_posix()}\n{digest}\n")
    return hashlib.sha256("".join(stream).encode()).hexdigest()


def _self_check() -> None:
    """Refuse a ``_core`` that is not the one these sources describe.

    Two independent readings, both refused with the field named: the
    build digest (which sources the binding was compiled from) and the
    layout table (how wide the by-value mirrors it marshals are, against
    :mod:`hawk._contracts`' verbatim twins). Compiler identity and flags
    are read back but deliberately not compared.

    The build-digest reading has two non-vacuous paths, tried in order:
    sources beside the package (a dev checkout, recompute
    :func:`_source_digest` and compare), or no sources but
    :mod:`hawk._build_digest` imports (a wheel install, compare against
    its shipped ``SOURCE_DIGEST`` — the editable-shadowing trap caught
    between a wheel's Python files and somebody else's ``_core``).
    Neither path available refuses outright.
    """
    from . import _core

    root = Path(__file__).resolve().parent.parent
    sources_present = (root / "CMakeLists.txt").is_file() and (root / "src").is_dir()
    stamped = _core.build_digest()
    if sources_present:
        recomputed = _source_digest(root)
        if stamped != recomputed:
            raise ImportError(
                f"hawk: hawk._core's build_digest is {stamped!r} but the sources "
                f"beside it hash to {recomputed!r} — this binding was NOT built "
                f"from these sources. Loaded binding: "
                f"{getattr(_core, '__file__', '?')}. Rebuild it (`pip install -e . "
                "--no-build-isolation --no-deps`), or check whether an older hawk "
                "earlier on sys.path is shadowing this one."
            )
    else:
        try:
            from . import _build_digest
        except ImportError:
            raise ImportError(
                "hawk: cannot verify hawk._core's build_digest — neither the "
                f"sources (looked under {root}) nor a shipped build stamp "
                "(hawk._build_digest) are beside the package; a self-check "
                "that silently disabled itself where nothing to compare against "
                "exists would certify nothing exactly where it matters most."
            ) from None
        if stamped != _build_digest.SOURCE_DIGEST:
            raise ImportError(
                f"hawk: hawk._core's build_digest is {stamped!r} but the shipped "
                f"build stamp is {_build_digest.SOURCE_DIGEST!r} "
                "(hawk._build_digest.SOURCE_DIGEST) — this binding was NOT built "
                "from the sources this wheel was built from (the editable-"
                "shadowing trap: a wheel's Python files paired with somebody "
                f"else's _core). Loaded binding: {getattr(_core, '__file__', '?')}."
            )

    sizes = _core.layout_sizes()
    for index, field, expected in _LAYOUT_TWINS:
        if index >= len(sizes) or sizes[index] != expected:
            got = sizes[index] if index < len(sizes) else "<absent>"
            raise ImportError(
                f"hawk: hawk._core's layout table disagrees with hawk._contracts in "
                f"sizeof({field}): the binding was built with {got} bytes, this "
                f"source tree declares {expected}. Loaded binding: "
                f"{getattr(_core, '__file__', '?')}."
            )


_self_check()


def __getattr__(name: str) -> Any:
    """Bind an authoring name, or a top-level door, lazily (importing hawk
    imports no heavy tier)."""
    if name in _TOP_LEVEL_DOORS:
        import importlib

        module_name, attr = _TOP_LEVEL_DOORS[name]
        value = getattr(importlib.import_module(f".{module_name}", __name__), attr)
        globals()[name] = value
        return value
    if name in _AUTHORING:
        if name in _FROM_IR:
            from . import ir
            value = getattr(ir, name)
        elif name in _FROM_STEPS:
            from .trace import steps as _steps_module
            value = getattr(_steps_module, name)
        else:
            from . import trace
            value = getattr(trace, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
