# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""C compiler and cppcheck discovery, and the checks run on emitted C.

The C compiler is found in this order: ``$HAWK_CC`` (an explicit override;
unresolvable raises), the C sibling of ``$HAWK_CXX`` (``g++-13`` -> ``gcc-13``,
``clang++`` -> ``clang``, ``c++`` -> ``cc``) so the C and C++ checks agree on
one toolchain, the first word of ``$CC``, ``cc``, ``clang`` and ``gcc`` on
``PATH``, and last a ``zig cc`` wrapper from the ``ziglang`` wheel. cppcheck
is ``$HAWK_CPPCHECK`` or ``cppcheck`` on ``PATH``.

Nothing is skipped silently. A missing C compiler raises and names every
candidate tried, and a missing cppcheck is recorded as skipped in the result
(or raises when the profile requires it): a skipped check certifies nothing,
so the caller must be able to see that it did not run.

Rule deviations are recorded in :data:`DEVIATIONS` and in the task report,
and applied as cppcheck ``--suppress`` flags; they are never written as
suppression comments in the emitted C.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from ..compile.cache import default_cache_dir
from ..ir import HawkError

_STDS = ("c99", "c11")
_TIMEOUT_S = 600

DEVIATIONS: tuple[dict, ...] = (
    {"rule": "misra-c2012-8.7", "category": "advisory",
     "reason": "cppcheck analyses one translation unit; the emitted entry "
               "function and its args struct are the unit's exported API, "
               "referenced from the consumer's translation unit, so the "
               "single-TU heuristic cannot see the reference"},
    {"rule": "misra-c2012-9.3", "category": "required",
     "reason": "a checker false positive, not a violation: cppcheck's misra "
               "addon parses every non-name initialiser of a struct with an "
               "array member as if it were a brace initialiser, so "
               "`const aether_c_f64_vec3_t t = f(x);` (a function call result) "
               "is reported as a partially initialised array while "
               "`const pair_t p = f(x);` (no array member) is not; the emitted "
               "C contains no brace initialiser at all, so Rule 9.3 has "
               "nothing to apply to; the emitted-text audit's audit.init rule "
               "(hawk.embedded.audit) refuses any `= {` in the .c, so this "
               "unit-wide suppression can never hide a real partial brace "
               "initialiser"},
    {"rule": "misra-c2012-2.3", "category": "advisory",
     "reason": "the HAWK_EMBEDDED_TYPES block (hawk_real_t, hawk_int_t, "
               "hawk_status_t) is the same text in every hawk header of a "
               "build, so that the first header defines the shared types and "
               "the width guard checks every later one; a unit that reads no "
               "integer still declares hawk_int_t for the units beside it, and "
               "the C11 size assert that references it is compiled out under "
               "the C99 the checker analyses"},
)
_FINDING = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+): (?P<sev>\w+): "
    r"(?P<text>.*) \[(?P<id>[^\]\s]+)\]\s*$")
_SIBLINGS = (("g++", "gcc"), ("clang++", "clang"), ("c++", "cc"))


def _c_sibling(cxx: str) -> str:
    """``cxx`` with its basename rewritten to the C driver's (``g++-13`` ->
    ``gcc-13``); unchanged when the basename is not a known C++ driver."""
    head, base = os.path.split(cxx)
    for cxx_name, c_name in _SIBLINGS:
        if base == cxx_name or base.startswith(cxx_name + "-"):
            return os.path.join(head, c_name + base[len(cxx_name):])
    return cxx


def _run(argv: list[str], cwd: str) -> subprocess.CompletedProcess:
    """``argv`` in ``cwd`` with captured text output and a timeout; a timeout
    raises :class:`HawkError` naming the command."""
    try:
        return subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                              timeout=_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        raise HawkError(f"timed out after {_TIMEOUT_S}s.\n$ "
                        f"{' '.join(argv)}") from None


def c_compiler() -> str:
    """The C compiler, in the discovery order the module docstring gives."""
    tried: list[str] = []
    cc = os.environ.get("HAWK_CC")
    if cc:
        found = shutil.which(cc)
        if not found:
            raise HawkError(f"$HAWK_CC={cc!r} not found on PATH.")
        return found
    tried.append("$HAWK_CC (unset)")
    cxx = os.environ.get("HAWK_CXX")
    if cxx:
        sibling = _c_sibling(cxx)
        found = shutil.which(sibling) if sibling != cxx else None
        if found:
            return found
        tried.append(f"C sibling of $HAWK_CXX={cxx!r} ({sibling!r}, "
                     "did not resolve)")
    else:
        tried.append("C sibling of $HAWK_CXX (unset)")
    words = os.environ.get("CC", "").split()
    if words:
        found = shutil.which(words[0])
        if found:
            return found
        tried.append(f"$CC ({words[0]!r}, did not resolve)")
    else:
        tried.append("$CC (unset)")
    for name in ("cc", "clang", "gcc"):
        found = shutil.which(name)
        if found:
            return found
        tried.append(f"{name} on PATH (absent)")
    return _zig_cc_wrapper(tried)


def _zig_cc_wrapper(tried: list[str]) -> str:
    """A stable ``exec <zig> cc "$@"`` script for the ``ziglang`` fallback,
    written under the compile cache dir (see
    :func:`hawk.compile.toolchain._zig_cxx_wrapper`)."""
    try:
        import ziglang
    except ImportError:
        raise HawkError(
            "no C compiler found. Tried, in order: " + ", ".join(tried)
            + ", the ziglang wheel (not installed). Set $HAWK_CC or $CC, "
            "install cc/clang/gcc, or `pip install ziglang`. "
            "A skipped check certifies nothing.") from None
    zig_bin = Path(ziglang.__file__).resolve().parent / "zig"
    if not zig_bin.is_file():
        raise HawkError(
            f"the ziglang wheel is installed but its zig binary is missing "
            f"at {zig_bin}.")
    script = f"#!/bin/sh\nexec {shlex.quote(str(zig_bin))} cc \"$@\"\n"
    wrapper = default_cache_dir() / "zig-cc-wrapper.sh"
    try:
        current = wrapper.read_text()
    except OSError:
        current = None
    if current != script:
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text(script)
        wrapper.chmod(0o755)
    return str(wrapper)


def compiler_identity(path: str) -> str:
    """``<path>||<first line of --version>``; the second part is empty when
    the compiler cannot be asked."""
    first = ""
    try:
        out = subprocess.run([path, "--version"], capture_output=True,
                             text=True, timeout=60).stdout
        lines = out.splitlines()
        first = lines[0].strip() if lines else ""
    except (OSError, subprocess.SubprocessError):
        pass
    return f"{path}||{first}"


def c_check_flags(std: str) -> list[str]:
    """The strict syntax-only flags for ``std`` (``c99`` or ``c11``)."""
    if std not in _STDS:
        raise HawkError(f"unsupported C standard {std!r}; expected one of "
                        f"{', '.join(_STDS)}.")
    return [f"-std={std}", "-pedantic", "-Wall", "-Wextra", "-Wconversion",
            "-Wsign-conversion", "-Wdouble-promotion", "-Werror",
            "-fsyntax-only"]


def cppcheck() -> str | None:
    """``$HAWK_CPPCHECK`` (must resolve), else ``cppcheck`` on ``PATH``, else
    ``None``."""
    override = os.environ.get("HAWK_CPPCHECK")
    if override:
        found = shutil.which(override)
        if not found:
            raise HawkError(f"$HAWK_CPPCHECK={override!r} not found on PATH.")
        return found
    return shutil.which("cppcheck")


#: The file name :func:`run_checks` writes :data:`STDINT_LIBRARY` to, beside the unit.
STDINT_LIBRARY_NAME = "hawk_stdint.cfg"

#: <stdint.h> constant macros (INT64_C, ...) as glibc defines them. cppcheck
#: does not read system headers here, and older releases (2.13, Ubuntu 24.04)
#: then report each use as an undeclared function (Rule 17.3). Not a suppression.
STDINT_LIBRARY = """<?xml version="1.0"?>
<def format="2">
  <define name="INT8_C(c)" value="c"/>
  <define name="INT16_C(c)" value="c"/>
  <define name="INT32_C(c)" value="c"/>
  <define name="INT64_C(c)" value="c ## L"/>
  <define name="UINT8_C(c)" value="c"/>
  <define name="UINT16_C(c)" value="c"/>
  <define name="UINT32_C(c)" value="c ## U"/>
  <define name="UINT64_C(c)" value="c ## UL"/>
  <define name="INTMAX_C(c)" value="c ## L"/>
  <define name="UINTMAX_C(c)" value="c ## UL"/>
</def>
"""


def cppcheck_argv(exe: str, c_path: str, include_dir: str) -> list[str]:
    """The cppcheck command line with the MISRA addon for ``c_path``. The
    :data:`STDINT_LIBRARY` file is expected in ``include_dir`` (:func:`run_checks`
    writes it there)."""
    return [exe, "--addon=misra", "--std=c99", "--language=c",
            "--enable=warning,style", "--error-exitcode=2",
            "--suppress=missingIncludeSystem",
            *[f"--suppress={d['rule']}" for d in DEVIATIONS],
            f"--library={include_dir}/{STDINT_LIBRARY_NAME}",
            f"-I{include_dir}", c_path]


def parse_cppcheck(stderr: str) -> list[dict]:
    """One ``{file, line, id, text}`` per default-template finding line;
    other lines are ignored."""
    findings = []
    for raw in stderr.splitlines():
        m = _FINDING.match(raw)
        if m:
            findings.append({"file": m["file"], "line": int(m["line"]),
                             "id": m["id"], "text": m["text"]})
    return findings


def run_checks(c_text: str, h_text: str, name: str, profile=None) -> dict:
    """Write ``<name>.c``/``<name>.h`` to a temp dir and check them: the C
    compiler under :func:`c_check_flags` for c99 then c11, then cppcheck with
    the MISRA addon. Any failure or finding raises :class:`HawkError`.

    ``profile`` is duck-typed: only ``.require_cppcheck`` is read, deciding
    whether an absent cppcheck raises or is recorded as skipped.
    """
    start = time.perf_counter()
    cc = c_compiler()
    with tempfile.TemporaryDirectory(prefix="hawk-embedded-") as tmp:
        c_path = Path(tmp) / f"{name}.c"
        c_path.write_text(c_text)
        (Path(tmp) / f"{name}.h").write_text(h_text)
        (Path(tmp) / STDINT_LIBRARY_NAME).write_text(STDINT_LIBRARY)
        cc_runs = []
        for std in _STDS:
            argv = [cc, *c_check_flags(std), f"-I{tmp}", str(c_path)]
            done = _run(argv, tmp)
            if done.returncode != 0:
                raise HawkError(
                    f"C check of {name!r} failed under -std={std} "
                    f"(rc={done.returncode}).\n$ {' '.join(argv)}\n"
                    f"{done.stderr[:4000]}")
            cc_runs.append({"std": std, "argv": argv, "rc": done.returncode})
        exe = cppcheck()
        if exe is None:
            if profile is not None and profile.require_cppcheck:
                raise HawkError(
                    "cppcheck not found but the profile requires it "
                    "(set $HAWK_CPPCHECK or install cppcheck). "
                    "A skipped check certifies nothing.")
            cpp: dict = {"skipped": "cppcheck not found (set $HAWK_CPPCHECK)"}
        else:
            argv = cppcheck_argv(exe, str(c_path), tmp)
            done = _run(argv, tmp)
            findings = parse_cppcheck(done.stderr)
            if done.returncode != 0 or findings:
                listing = "\n".join(
                    f"{f['file']}:{f['line']} {f['id']} {f['text']}"
                    for f in findings)
                raise HawkError(
                    f"cppcheck of {name!r} reported {len(findings)} "
                    f"finding(s) (rc={done.returncode}).\n"
                    f"$ {' '.join(argv)}\n{listing}\n{done.stderr[:2000]}")
            cpp = {"argv": argv, "rc": done.returncode, "findings": findings,
                   "suppressed": [d["rule"] for d in DEVIATIONS]}
    cxx = os.environ.get("HAWK_CXX")
    return {"cc": cc_runs, "cc_identity": compiler_identity(cc),
            "cxx_identity": compiler_identity(cxx) if cxx else None,
            "cppcheck": cpp, "seconds": time.perf_counter() - start}


def compile_shared(c_path, out_path, mode=None) -> Path:
    """Build ``c_path`` into the shared library ``out_path`` (c99, -O2, no FP
    contraction) for the test harness; ``mode`` is reserved for the float32
    flag set."""
    c_path = Path(c_path)
    argv = [c_compiler(), "-std=c99", "-O2", "-ffp-contract=off", "-fPIC",
            "-shared", str(c_path), "-o", str(out_path), "-lm"]
    done = _run(argv, str(c_path.parent))
    if done.returncode != 0:
        raise HawkError(
            f"shared build of {c_path.name!r} failed (rc={done.returncode})."
            f"\n$ {' '.join(argv)}\n{done.stderr[:4000]}")
    return Path(out_path)
