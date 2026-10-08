# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Toolchain + include-root resolution for the compile smoke.

(a): the eagle header root is ``$HAWK_EAGLE_INCLUDE``, else the active
prefix's ``include/eagle``; the aether root follows the same shape under
``$HAWK_AETHER_INCLUDE`` / the active prefix's ``include``. Both fall back to
the workspace checkouts, which is where they live in the dev env. Nothing here
is skipped when a tool is missing: a skipped compile row is RC=0 over an
unexercised emitter, so an unresolvable toolchain FAILS and names the variable
that would fix it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]

#: Flags every emitted host TU is checked with. ``AETHER_CPP_MODE`` is the
#: standalone-artifact mode: a HAWK host object is one self-contained TU that
#: eagle ``dlopen``s, never linked against a CUDA-language aether TU, so the
#: build-mode axis is free to be the CPU one (``aether/macros.h``'s two-axis
#: note). Without it ``aether/dtype/Fetch.h`` pulls CUDA's ``vector_types.h``.
HOST_FLAGS = ("-std=c++23", "-fsyntax-only", "-DAETHER_CPP_MODE")
#: nvcc caps the device TU at C++20; the arch is :func:`real_arch`'s.
DEVICE_FLAGS = ("-std=c++20", "-c", "-o", "/dev/null")


def real_arch(kind: str | None = None) -> str:
    """A real ``sm_<N>`` the device toolkit (``kind``: ``"nvcc"``,
    ``"nvrtc"`` or the active one) can compile a CUBIN for: the running
    GPU's own arch when the toolkit supports it, else the toolkit's minimum
    at or above ``PORTABLE_MIN_ARCH``. Never a hard-coded box arch."""
    from hawk.compile import toolchain as tc

    probed = tc._probed_device_arch()
    toolkit = tc.toolkit_archs(kind)
    if toolkit is None:
        return probed or tc.FALLBACK_ARCH.replace("compute_", "sm_")
    archs = toolkit[1]
    if probed:
        number = int("".join(ch for ch in probed[3:] if ch.isdigit()))
        if archs[0] <= number <= archs[-1]:
            return probed
    portable = [a for a in archs if a >= tc.PORTABLE_MIN_ARCH]
    return f"sm_{portable[0] if portable else archs[-1]}"


def toolkit_supports(arch: str, kind: str | None = None) -> bool:
    """Whether the device toolkit lists ``arch`` (``sm_61``) at all --
    ``True`` when its list cannot be read (the compile then decides)."""
    from hawk.compile import toolchain as tc

    toolkit = tc.toolkit_archs(kind)
    return toolkit is None or int(arch.split("_", 1)[1]) in toolkit[1]


def aether_include() -> str:
    """The aether header root ((a)'s shape)."""
    return _root("HAWK_AETHER_INCLUDE",
                 [Path(p) / "include" for p in _prefixes()] + [WORKSPACE / "aether"],
                 "aether/aether.h")


def eagle_include() -> str:
    """The eagle header root: the directory ``plugin/gref_abi.h`` hangs off."""
    return _root("HAWK_EAGLE_INCLUDE",
                 [Path(p) / "include" for p in _prefixes()]
                 + [WORKSPACE / "eagle", WORKSPACE / "eagle-abi"],
                 "plugin/gref_abi.h")


def _prefixes() -> list[str]:
    """The RUNNING interpreter's prefix first: the env whose python collected
    these tests is the env whose headers the emitter is being checked against
    (an activated shell may sit in a different one)."""
    return [p for p in (sys.prefix, os.environ.get("CONDA_PREFIX"),
                        os.environ.get("PREFIX")) if p]


def _root(var: str, candidates: list[Path], probe: str) -> str:
    override = os.environ.get(var)
    if override:
        return override
    for candidate in candidates:
        if (candidate / probe).is_file():
            return str(candidate)
    raise AssertionError(
        f"cannot resolve the include root holding {probe!r}: tried "
        f"{[str(c) for c in candidates]}. Set ${var}."
    )


def host_compiler() -> str:
    """``g++`` (the host compiler)."""
    found = shutil.which(os.environ.get("HAWK_CXX", "g++"))
    if not found:
        raise AssertionError("no g++ on PATH; set $HAWK_CXX. A skipped compile row "
                             "certifies nothing (RC=0 over an unexercised emitter).")
    return found


def device_compiler() -> str:
    """``nvcc`` (AOT, never NVRTC)."""
    for candidate in (os.environ.get("HAWK_NVCC"), "nvcc"):
        if candidate and (Path(candidate).is_file() or shutil.which(candidate)):
            return candidate
    raise AssertionError("no nvcc found; set $HAWK_NVCC. A skipped compile row "
                         "certifies nothing.")


def _env() -> dict:
    env = dict(os.environ)
    cuda = Path(device_compiler()).resolve().parent
    env["PATH"] = f"{cuda}:{env.get('PATH', '')}"
    # The nvcc host compiler follows hawk's own rule
    # (hawk.compile.toolchain.subprocess_env): $HAWK_NVCC_CCBIN, else the
    # caller's NVCC_PREPEND_FLAGS/NVCC_CCBIN untouched, else nvcc's default.
    ccbin = os.environ.get("HAWK_NVCC_CCBIN")
    if ccbin:
        env["NVCC_PREPEND_FLAGS"] = f"-ccbin {ccbin}"
    env["CUDA_PATH"] = str(cuda.parent)
    env.setdefault("TMPDIR", os.environ.get("TMPDIR", "/tmp"))
    return env


def compile_check(path: Path, target: str) -> tuple[int, str, list[str]]:
    """Syntax/object-check ONE emitted translation unit. Returns
    ``(returncode, stderr, argv)`` — the argv so a failing row can print the
    exact command line that reproduces it."""
    includes = ["-I", aether_include(), "-I", eagle_include()]
    if target == "host":
        argv = [host_compiler(), *HOST_FLAGS, *includes, str(path)]
    else:
        argv = [device_compiler(), *DEVICE_FLAGS, f"-arch={real_arch('nvcc')}",
                *includes, str(path)]
    done = subprocess.run(argv, capture_output=True, text=True, env=_env())
    return done.returncode, done.stderr, argv
