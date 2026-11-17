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
from contextlib import contextmanager
from pathlib import Path

from raptor.wings.base.exchange import ToolchainAbsent

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
    from ospreys.cuda import toolchain as dtc

    probed = dtc._probed_device_arch()
    toolkit = dtc.toolkit_archs(kind)
    if toolkit is None:
        return probed or dtc.FALLBACK_ARCH.replace("compute_", "sm_")
    archs = toolkit[1]
    if probed:
        number = int("".join(ch for ch in probed[3:] if ch.isdigit()))
        if archs[0] <= number <= archs[-1]:
            return probed
    portable = [a for a in archs if a >= dtc.PORTABLE_MIN_ARCH]
    return f"sm_{portable[0] if portable else archs[-1]}"


def toolkit_supports(arch: str, kind: str | None = None) -> bool:
    """Whether the device toolkit lists ``arch`` (``sm_61``) at all --
    ``True`` when its list cannot be read (the compile then decides)."""
    from hawk.compile import toolchain as tc
    from ospreys.cuda import toolchain as dtc

    toolkit = dtc.toolkit_archs(kind)
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
    raise ToolchainAbsent("no nvcc found; set $HAWK_NVCC. A skipped compile row "
                          "certifies nothing.", toolchain="nvcc")


def _env() -> dict:
    env = dict(os.environ)
    cuda = Path(device_compiler()).resolve().parent
    env["PATH"] = f"{cuda}:{env.get('PATH', '')}"
    # The fixtures' own nvcc argv carries no -ccbin (a build's does, from the library's template): $HAWK_NVCC_CCBIN is given
    # to nvcc through its environment here.
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


def host_library(text: str, directory: Path, name: str, *, mode: str = "float64", profile: str | None = None, defines=(), extra=()):
    """The shared library of the host text ``text``, built the way a build builds it (:func:`hawk.compile.compile_source`: the
    library's own commands) and copied to ``<directory>/<name>.so``. ``defines`` and ``extra`` are the caller's own flags. Returns the
    :class:`~hawk.compile.CompileResult`; its ``source`` is the text as compiled."""
    from hawk.compile import CompileOptions, compile_source

    opts = CompileOptions(target="host", mode=mode, cache_dir=str(directory / "hawk_cache"), host_profile=profile or "",
                          defines=tuple(defines), extra=tuple(extra))
    result = compile_source(text, name, opts)
    shutil.copyfile(result.artifact, directory / f"{name}.so")
    return result


def host_template(profile: str | None = None, opt: str | None = None) -> list[str]:
    """The argv template of a unit's host compile that the lowering library gives this machine for ``profile`` and optimisation level
    ``opt`` (``None``: the one in effect)."""
    from hawk.artifact.request import compiling_host, host_build

    facts, line = compiling_host()
    return host_build(facts, line, {"host_profile": profile, "opt_level": opt})["compile_unit"]


@contextmanager
def host_command(source: Path, output: Path, *, profile: str | None = None, defines=(), extra=()):
    """The one host compile command of a build, for a fixture that measures it (stderr reports, peak memory, wall time): the library's
    ``compile_unit`` template for this machine with its places filled by the product's own filler, as the argv of ``source`` to the
    object ``output``. ``extra`` are the fixture's own MEASUREMENT flags, appended after the template. The command is valid inside the
    ``with`` only: the headers may be served from a directory that lives that long."""
    from contextlib import nullcontext

    from hawk.compile import drivers, host
    from hawk.compile import payload as _payload
    from hawk.compile.drivers import CompileOptions
    from hawk.targets import TARGETS, build_spec

    row = TARGETS["host"]
    opts = CompileOptions(target="host", host_profile=profile or "", defines=tuple(defines), extra=tuple(extra))
    build = row.host_build(build_spec(("host",)).for_target(row).primary, {"host_profile": profile}, opts)
    served = drivers.header_shape() == "served"
    with (_payload.current_payload().serve() if served else nullcontext()) as root:
        places = {**host._places(build, door=False), "{compiler}": lambda: build.compiler, "{source}": lambda: source,
                  "{object}": lambda: output, "{depfile}": lambda: output.with_suffix(".d")}
        if served:
            places["{include:served}"] = lambda: root
        yield [*host.fill_argv(build.instructions["compile_unit"], places, {"{defines}": [f"-D{d}" for d in opts.defines]}), *extra]


def device_command(source: Path, output: Path, *, arch: str = "", cubin: bool = False, extra=()) -> list[str]:
    """The one device compile command of a build, for a fixture that measures it: the library's ``compile`` template for this machine
    and compiler, filled by the product's own filler, with the host compiler of :func:`ospreys.cuda.toolchain.device_host_compiler`
    (the ``-ccbin`` a build gives). ``cubin`` asks the library for the form it states when no device driver is known (a CUBIN, with
    the ptxas report), whatever driver this machine has. ``extra`` are the fixture's own MEASUREMENT flags, appended after the template. Refuses by name
    when the device compiler is not nvcc."""
    from hawk.compile import drivers
    from hawk.compile.drivers import CompileOptions
    from hawk.compile.host import fill_argv
    from hawk.targets import TARGETS

    plan = TARGETS["cuda"].compile_plan(CompileOptions(target="cuda", arch=arch, extra=tuple(extra)), drivers._Inputs())
    if not plan["compiler"]:
        raise ToolchainAbsent("the device command of a fixture is the nvcc one; this machine compiles in process", toolchain="nvcc")
    template = plan["template"]
    if cubin:
        from hawk.artifact import request
        from ospreys.cuda.toolchain import resolve_arch

        template = request.device_build(plan["identity"], arch=resolve_arch(arch), release=plan["release"], driver=None,
                                        level=drivers._Inputs.opt_level())["compile"]
    places = {**plan["places"], "{compiler}": lambda: plan["compiler"], "{depfile}": lambda: output.with_suffix(".d"),
              "{source}": lambda: source, "{output}": lambda: output}
    return [*fill_argv(template, places, {"{defines}": plan["defines"]}), *plan["extra"]]
