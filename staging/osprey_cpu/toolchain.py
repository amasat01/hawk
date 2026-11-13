# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The host toolchain of the cpu backend: compiler discovery, the code-generation profiles and flags, the C library names the
host unit's headers declare, and the GCC masked-call witness. hawk's shared compile pieces (include roots, compiler identity,
optimisation level, the cache directory) are read from hawk."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import threading
from pathlib import Path

from raptor.errors import RaptorError
from lowering.exchange import ToolchainAbsent

from hawk.compile import toolchain as _shared
from hawk.compile.cache import default_cache_dir

#: ``g++`` compiles the host TU at C++23.
HOST_STD = "c++23"


#: The way out of a missing or unsuitable host compiler.
_NO_COMPILER_FIX = ("install g++ 11+ (apt install g++ / brew install llvm) or set "
                    "HAWK_CXX=/path/to/c++")


def host_compiler() -> str:
    """The host compiler, in discovery order: ``$HAWK_CXX`` when set (an
    explicit override: unresolvable raises), else the standard ``$CXX`` (its
    first word, resolved through ``PATH``; unresolvable falls through, its
    arguments are not kept), else ``g++`` on ``PATH``, else the ``ziglang``
    wheel's ``zig c++`` — a pip-only host tier for a box with no system C++
    toolchain at all.

    ``zig c++`` is a subcommand of the single ``zig`` binary, not its own
    executable — it cannot stand as one ``argv[0]`` the way ``g++``'s path
    can, and every other caller of this function already assumes it can.
    :func:`_zig_cxx_wrapper` closes that gap with a tiny generated
    ``exec zig c++ "$@"`` script, so nothing downstream has to know a third
    compiler exists.
    """
    cxx = os.environ.get("HAWK_CXX")
    if cxx:
        found = shutil.which(cxx)
        if not found:
            raise RaptorError(f"$HAWK_CXX={cxx!r} not found on PATH; "
                            + _NO_COMPILER_FIX)
        return found
    std_cxx = os.environ.get("CXX", "")
    words = std_cxx.split()
    if words:
        found = shutil.which(words[0])
        if found:
            return found
    found = shutil.which("g++")
    if found:
        return found
    return _zig_cxx_wrapper(tried_cxx=std_cxx)


def _zig_cxx_wrapper(tried_cxx: str = "") -> str:
    """The stable ``argv[0]`` :func:`host_compiler` hands back for the
    ``ziglang`` fallback: a one-line ``exec <zig binary> c++ "$@"`` script,
    written once under :func:`~hawk.compile.cache.default_cache_dir` and
    reused — content-checked, not merely existence-checked, so a wrapper left
    from a different ``ziglang`` install is rewritten rather than kept stale.

    A stable path matters because :func:`compiler_identity`'s per-process
    memo keys on the compiler's ``(path, size, mtime_ns)``: a wrapper
    rewritten on every call would cost a ``--version`` subprocess on every
    host compile instead of once per process.
    """
    try:
        import ziglang
    except ImportError:
        raise RaptorError(
            "no host compiler found. Tried, in order: $HAWK_CXX (unset), "
            f"$CXX ({tried_cxx!r}, "
            f"{'did not resolve on PATH' if tried_cxx else 'unset'}), "
            "g++ on PATH (absent), the ziglang wheel (not installed): "
            + _NO_COMPILER_FIX + " ($CXX and `pip install ziglang` also work). "
            "A skipped compile certifies nothing.") from None
    zig_bin = Path(ziglang.__file__).resolve().parent / "zig"
    if not zig_bin.is_file():
        raise RaptorError(
            f"the ziglang wheel is installed but its zig binary is missing "
            f"at {zig_bin}.")
    script = f"#!/bin/sh\nexec {shlex.quote(str(zig_bin))} c++ \"$@\"\n"
    wrapper = default_cache_dir() / "zig-cxx-wrapper.sh"
    try:
        current = wrapper.read_text()
    except OSError:
        current = None
    if current != script:
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text(script)
        wrapper.chmod(0o755)
    return str(wrapper)


#: The host code-generation profiles (:func:`host_codegen_flags`): two EXACT
#: profiles that keep results bit-identical to the scalar build, and one FAST
#: profile (the x86-64 default) that trades bit identity for speed within a
#: documented ULP bound.
#:
#: ``native`` (exact) — for every build compiled at run time on the machine
#: that will run it: the opt level (``-O3`` by default, see :func:`opt_level`)
#: and ``-march=native``, the default off x86-64. On x86-64 it also asks for
#: 512-bit vectors (``-mprefer-vector-width=512``): at GCC's 256-bit default,
#: a per-sample loop with a 1-byte ``terminated`` guard and 8-byte
#: ``float64`` state has no vector type and stays scalar.
#:
#: ``portable`` (exact) — for prebuilt, shipped artifacts, which must load on
#: CPUs other than the build machine's: the opt level at the ``x86-64-v2``
#: baseline (SSE4.2, SSSE3, POPCNT, CMPXCHG16B) — the oldest level any x86-64 CPU
#: still in service supports, and RHEL/OEL 9's own baseline; v3 (AVX2) would
#: refuse some still-sold low-end parts and VMs.
#:
#: The two exact profiles are bit-identical to the scalar build: FMA
#: contraction is off (``-ffp-contract=off``) and nothing enables fast-math or
#: reassociation, so vector lanes perform the same IEEE operations in the
#: same order per sample. Request one of them wherever results must match
#: bit for bit (a reference, a regression baseline, host against host).
#:
#: ``native-vector-math`` (fast) — the default on x86-64 with a GCC host
#: compiler and on arm64 with any host compiler. On arm64 with clang it is
#: ``native`` plus (2) below only: no vector math define, no
#: ``-fno-trapping-math``, and a library call stays scalar. On arm64 with GCC
#: it is that plus (1) for ``pow`` alone: every other library call stays
#: scalar. On x86-64 it is ``native`` plus three trades. (1) ``-DAETHER_HOST_VECTOR_MATH`` and
#: ``-fno-trapping-math`` route aether's transcendental math to vector-ABI
#: functions, so a calling loop is no longer kept scalar by the libm call;
#: those functions are aether's packet math, faithfully rounded (1-3 ULP of
#: glibc), not glibc's. (2) ``-ffp-contract=fast``: the compiler fuses
#: ``a*b + c`` into one FMA instruction (``-march=native`` enables FMA on
#: every x86-64 CPU that has it; on one without, nothing changes). An FMA
#: rounds once where the plain pair rounds twice, so each fused pair is
#: within half an ULP of the exact value instead of one ULP: results are as
#: accurate or more, but no longer bit-identical to the exact profiles; the
#: difference is bounded by the operation count of the kernel (half an ULP
#: per fused pair, propagated by its condition). (3) Code that depends on
#: exact rounding is protected from (2) whatever the profile: aether's packet
#: math is compiled contraction-free (``aether/math/detail/
#: HostVectorMath.h``), and the error-free transformations of the
#: ``compensated`` sink and of ``aether/accum`` pin their operands
#: (``AETHER_FP_BARRIER``), so compensation stays exact.
HOST_PROFILES = ("native", "portable", "native-vector-math", "native-vector-math-nofma")


#: The profiles that route aether's math to the vector ABI (GCC: all of it on x86-64, ``pow`` on arm64). ``native-vector-math-nofma`` is
#: ``native-vector-math`` with ``-ffp-contract=off``: same vector code, so scalar and vector builds of one
#: kernel agree bit for bit (the property the default gives up for speed, about 25 % on the vector kernel).
VECTOR_MATH_PROFILES = ("native-vector-math", "native-vector-math-nofma")


#: The profiles whose builds are bit-identical to the scalar build.
EXACT_HOST_PROFILES = ("native", "portable")


#: The default host profile for a GCC host compiler on an x86-64 host:
#: ``native-vector-math`` there and on arm64 (any compiler), ``native`` on any
#: other architecture (where ``native-vector-math`` is refused). With a non-GCC
#: host compiler (clang, the ``zig c++`` fallback) the default on x86-64 is
#: ``native``, because the vector profile rides on GCC's vector function ABI; use
#: :func:`default_host_profile` for the default that applies to the compiler
#: actually in use.
DEFAULT_HOST_PROFILE = ("native-vector-math"
                        if platform.machine().lower() in ("x86_64", "amd64", "aarch64", "arm64")
                        else "native")


#: Per-process memo of :func:`compiler_is_gcc`, keyed on the resolved compiler
#: path. Guarded by a lock so concurrent first callers (free-threaded CPython)
#: probe at most a few times and always agree.
_GCC_FAMILY_MEMO: dict = {}


_GCC_FAMILY_LOCK = threading.Lock()


def compiler_is_gcc(compiler: str) -> bool:
    """Whether ``compiler`` is GCC proper (``__GNUC__`` defined and
    ``__clang__`` not), read from its predefined macros (:func:`compiler_macros`),
    never from ``--version`` text. A compiler that cannot be probed counts as
    not GCC."""
    with _GCC_FAMILY_LOCK:
        known = _GCC_FAMILY_MEMO.get(compiler)
    if known is not None:
        return known
    macros = compiler_macros(compiler)
    gcc = "__GNUC__" in macros and "__clang__" not in macros
    with _GCC_FAMILY_LOCK:
        return _GCC_FAMILY_MEMO.setdefault(compiler, gcc)


#: Per-process memo of each compiler's predefined macros (name -> value),
#: behind :func:`compiler_macros`; guarded like :data:`_GCC_FAMILY_MEMO`.
_MACRO_MEMO: dict = {}


_MACRO_LOCK = threading.Lock()


def compiler_macros(compiler: str) -> dict:
    """``compiler``'s predefined macros (``-dM -E -x c++``) as a name -> value
    dict, memoised per compiler path for the process; empty when the compiler
    cannot be probed."""
    with _MACRO_LOCK:
        known = _MACRO_MEMO.get(compiler)
    if known is not None:
        return known
    macros: dict = {}
    try:
        rc, text = _shared.probe_stdout(compiler, ("-dM", "-E", "-x", "c++", os.devnull), timeout=60)
        if rc == 0:
            for ln in text.splitlines():
                parts = ln.split(None, 2)
                if len(parts) >= 2 and parts[0] == "#define":
                    macros[parts[1]] = parts[2] if len(parts) > 2 else ""
    except (OSError, subprocess.SubprocessError):
        macros = {}
    with _MACRO_LOCK:
        return _MACRO_MEMO.setdefault(compiler, macros)


#: The system headers whose C declarations an emitted entry must not collide with,
#: added to the emitted unit's own includes when the list is derived.
_SYSTEM_HEADERS = ("cstdlib", "cstdio", "cstring", "cmath", "unistd.h")


_C_CALL_NAME = re.compile(r"[A-Za-z_]\w*(?=\s*\()")


_C_NOISE = re.compile(r"\b(?:__attribute__|__asm__|__asm|asm|__extension__|__THROW)\b")


def _c_declared_names(text: str) -> set[str]:
    """The function names declared inside the ``extern "C" { ... }`` blocks of
    preprocessed C++ ``text``."""
    names: set[str] = set()
    for block in re.finditer(r'extern\s+"C"\s*\{', text):
        depth, start, i = 1, block.end(), block.end()
        while i < len(text) and depth:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            elif ch == ";" and depth == 1:
                names.update(_declared_name(text[start:i]))
                start = i + 1
            i += 1
    return names


def _declared_name(stmt: str) -> list[str]:
    """The function a one-statement declaration declares, as a 0- or 1-item list."""
    if "{" in stmt or re.search(r"\b(?:typedef|struct|union|enum|class)\b", stmt):
        return []
    stmt = re.sub(r"\(\s*([A-Za-z_]\w*)\s*\)(?=\s*\()", r"\1", stmt)    # (name)(args)
    while True:                              # drop attribute / asm label groups, innermost first
        cut = _C_NOISE.search(stmt)
        if cut is None:
            break
        depth, j = 0, stmt.find("(", cut.end())
        if j < 0:
            stmt = stmt[:cut.start()] + stmt[cut.end():]
            continue
        k = j
        while k < len(stmt):
            depth += (stmt[k] == "(") - (stmt[k] == ")")
            k += 1
            if depth == 0:
                break
        stmt = stmt[:cut.start()] + stmt[k:]
    call = _C_CALL_NAME.search(stmt)
    return [call.group()] if call is not None else []


_LIBC_MEMO: dict = {}


def libc_symbols(includes: str) -> frozenset[str]:
    """The C function names the emitted unit's own headers (and the system C
    headers) declare with C linkage: names a kernel's entry must not take, since
    its ``extern "C"`` entry would redeclare them. Derived by preprocessing the
    unit's includes (``includes``, the writer's include block, handed in as data) with the host compiler, once per
    compiler and header set,
    and kept as ``libc-symbols-<key>.json`` under :func:`default_cache_dir`.
    Empty when no host compiler can preprocess them."""
    if includes in _LIBC_MEMO:
        return _LIBC_MEMO[includes]

    names: frozenset[str] = frozenset()
    try:
        compiler = host_compiler()
        probe = includes + "".join(f"#include <{h}>\n" for h in _SYSTEM_HEADERS)
        flags = _shared.include_flags()
        key = hashlib.sha256("\0".join(
            [_shared.compiler_identity(compiler), probe, *flags]).encode()).hexdigest()[:16]
        path = default_cache_dir() / f"libc-symbols-{key}.json"
        try:
            names = frozenset(json.loads(path.read_text()))
        except (OSError, ValueError):
            run = subprocess.run([compiler, "-E", "-P", "-x", "c++", f"-std={HOST_STD}", "-DAETHER_CPP_MODE", *flags, "-"],
                                 input=probe, capture_output=True, text=True, timeout=120,
                                 check=False)
            if run.returncode == 0:
                names = frozenset(_c_declared_names(run.stdout))
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}.tmp")
                tmp.write_text(json.dumps(sorted(names)))
                os.replace(tmp, path)
    except (RaptorError, OSError, subprocess.SubprocessError):
        return names
    _LIBC_MEMO[includes] = names
    return names


def gcc_maskcall_affected(macros: dict) -> bool:
    """Whether a compiler with predefined ``macros`` is a GCC that miscompiles
    a conditional call of a function with masked vector variants.

    GCC 13.1-13.3 and 14.1-14.2 if-convert such a call with the mask of the
    negated condition inverted, so the masked variant evaluates the lanes that
    should be inactive. aether-dsc drops its masked variants on exactly these
    compilers. Fixed upstream by "middle-end: Fix ifcvt predicate generation
    for masked function calls" (commit abbfe1e2616d), which is in GCC 13.4 and
    14.3. The release number is ``__GNUC_MINOR__`` (13.4.0 has minor 4, patch 0)."""
    if "__clang__" in macros or "__GNUC__" not in macros:
        return False
    try:
        major = int(macros["__GNUC__"])
        minor = int(macros.get("__GNUC_MINOR__", "0"))
    except ValueError:
        return False
    return (major == 13 and minor < 4) or (major == 14 and minor < 3)


#: The witness for the GCC masked-call miscompile (see :func:`gcc_maskcall_affected`): exit 0 on correct
#: code, 1 on wrong code. Ships with the package (and is the one the CI host-gcc job runs).
MASKED_CALL_WITNESS = Path(__file__).with_name("witness") / "gcc_masked_call.cpp"


_WITNESS_MEMO: dict = {}


_WITNESS_LOCK = threading.Lock()


def _witness_run(compiler: str, flags: list[str]) -> tuple[str, str]:
    """``(verdict, why)``: ``fixed`` (the witness ran correctly), ``miscompile``
    (it ran and gave wrong results) or ``unrun`` (it could not be built or run)."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="hawk-witness-") as tmp:
        exe = os.path.join(tmp, "witness")
        try:
            built = subprocess.run([compiler, *flags, "-std=c++17", str(MASKED_CALL_WITNESS), "-o", exe],
                                   capture_output=True, text=True, timeout=120)
            if built.returncode != 0:
                return "unrun", "the witness does not compile with the host flags"
            ran = subprocess.run([exe], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            return "unrun", f"{type(exc).__name__}: cannot build or run the witness"
    return {0: ("fixed", ""), 1: ("miscompile", "")}.get(ran.returncode, ("unrun", f"the witness exited {ran.returncode}"))


def masked_call_flag(flags: list[str], extra=()) -> list[str]:
    """``["-DAETHER_HVM_MASKED=1"]`` when the host compiler is a GCC whose version aether flags
    (:func:`gcc_maskcall_affected`) but which runs the witness correctly (a distribution compiler carrying the
    backport of the fix), else ``[]``: aether keeps its workaround. The witness is built with the host build's own
    ``flags`` and run once per compiler identity, CPU and flags: the verdict is kept in the process and in the
    compile cache directory. A macro the caller sets in ``extra`` wins without a probe; a compiler aether does
    not flag, clang, a compiler that cannot be probed, or a witness that cannot run costs nothing and changes nothing."""
    if any(str(f).startswith("-DAETHER_HVM_MASKED") for f in extra):
        return []
    try:
        compiler = host_compiler()
        if not gcc_maskcall_affected(compiler_macros(compiler)):
            return []
        key = hashlib.sha256("\n".join([_shared.compiler_identity(compiler), host_target_identity(), *flags]).encode()).hexdigest()[:16]
    except (RaptorError, OSError, subprocess.SubprocessError):
        return []
    with _WITNESS_LOCK:
        verdict = _WITNESS_MEMO.get(key)
    if verdict is None:
        stored = default_cache_dir() / "toolchain" / f"masked_call_{key}.json"
        try:
            verdict = json.loads(stored.read_text())["verdict"]
        except (OSError, ValueError, KeyError):
            verdict, why = _witness_run(compiler, flags)
            if verdict == "unrun":
                logging.getLogger("hawk").warning("hawk: the masked-call witness did not run (%s): aether's workaround stays", why)
            else:   # an unrun witness is not stored: the cause may be passing
                try:
                    stored.parent.mkdir(parents=True, exist_ok=True)
                    tmp = stored.with_suffix(f".{os.getpid()}.tmp")
                    tmp.write_text(json.dumps({"verdict": verdict}))
                    os.replace(tmp, stored)
                except OSError:
                    pass
        with _WITNESS_LOCK:
            verdict = _WITNESS_MEMO.setdefault(key, verdict)
    return ["-DAETHER_HVM_MASKED=1"] if verdict == "fixed" else []


def _is_x86_64() -> bool:
    return platform.machine().lower() in ("x86_64", "amd64")


def _is_arm64() -> bool:
    return platform.machine().lower() in ("aarch64", "arm64")


def _arm64_gcc() -> bool:
    """Whether the host is arm64 and the host compiler is GCC proper: the one place aether routes ``pow``
    (alone) to a vector variant on arm64. A compiler that cannot be resolved or probed counts as not GCC."""
    if not _is_arm64():
        return False
    try:
        return compiler_is_gcc(host_compiler())
    except (RaptorError, OSError, subprocess.SubprocessError):
        return False


def default_host_profile() -> str:
    """The profile used when neither the compile options nor
    ``$HAWK_HOST_PROFILE`` name one: ``native-vector-math`` when the host is
    x86-64 AND the resolved host compiler is GCC, or the host is arm64, else
    ``native``."""
    if _is_arm64() or (_is_x86_64() and compiler_is_gcc(host_compiler())):
        return "native-vector-math"
    return "native"


#: The FMA-contraction flag per profile (see :data:`HOST_PROFILES`): off for
#: the exact profiles, fast for the fast one.
_HOST_CONTRACTION = {"native": "-ffp-contract=off",
                     "portable": "-ffp-contract=off",
                     "native-vector-math": "-ffp-contract=fast",
                     "native-vector-math-nofma": "-ffp-contract=off"}


def host_profile(requested: str | None = None) -> str:
    """The host profile in effect: ``requested`` when given (the compile
    options' ``host_profile``), else ``$HAWK_HOST_PROFILE``, else
    :func:`default_host_profile`. An unknown name RAISES, and so does an
    explicit ``native-vector-math`` under a non-GCC host compiler on x86-64
    (its vector function ABI is GCC-only)."""
    explicit = requested or os.environ.get("HAWK_HOST_PROFILE")
    name = explicit or default_host_profile()
    if name not in HOST_PROFILES:
        raise RaptorError(
            f"unknown host profile {name!r} (from "
            f"{'the compile options' if requested else '$HAWK_HOST_PROFILE'}); "
            f"the profiles are {HOST_PROFILES}")
    if explicit and name in VECTOR_MATH_PROFILES and _is_x86_64():
        cxx = host_compiler()
        if not compiler_is_gcc(cxx):
            raise RaptorError(
                f"host profile {name!r} needs a GCC host compiler "
                f"(its vector function ABI is GCC-only), but {cxx!r} is not "
                f"GCC; use the 'native' profile, or point $HAWK_CXX at a GCC "
                f"g++")
    return name


def host_codegen_flags(profile: str | None = None, opt: str | None = None, extra=()) -> list[str]:
    """The code-generation half of the host recipe for ``profile``
    (resolved through :func:`host_profile`): optimisation level (resolved
    through :func:`opt_level`), target CPU, vector width and the
    FMA-contraction setting. On a vector-math profile it also carries
    :func:`masked_call_flag`'s answer for the host compiler (``extra`` is the
    caller's own flags, which can pin the macro). See :data:`HOST_PROFILES`."""
    name = host_profile(profile)
    level = _shared.opt_level(opt)
    machine = platform.machine().lower()
    x86 = machine in ("x86_64", "amd64")
    if name != "portable":
        target = (["-march=native", "-mprefer-vector-width=512"] if x86
                  else ["-mcpu=native"] if machine in ("aarch64", "arm64", "ppc64le")
                  else ["-march=native"])
    else:
        target = ["-march=x86-64-v2"] if x86 else []
    if name in VECTOR_MATH_PROFILES:
        if x86:
            target = [*target, "-fno-trapping-math", "-DAETHER_HOST_VECTOR_MATH"]
        elif name == "native-vector-math-nofma" or not _is_arm64():
            raise ToolchainAbsent(
                f"host profile {name!r} needs an x86-64 host "
                "(aether's vector math follows the x86-64 vector function ABI)", toolchain="x86-64")
        elif _arm64_gcc():
            # jump threading copies the pow call onto conditional paths, which GCC on aarch64 cannot vectorise
            target = [*target, "-fno-trapping-math", "-fno-thread-jumps", "-DAETHER_HOST_VECTOR_MATH"]
    # no kernel reads errno: without it a sqrt is one instruction, not a call that keeps a loop scalar
    flags = [f"-{level}", _HOST_CONTRACTION[name], "-fno-math-errno", *target]
    if name in VECTOR_MATH_PROFILES and x86:
        flags += masked_call_flag(flags, extra)
    return flags


#: Per-process memo for :func:`host_target_identity` (the CPU does not change
#: under a running process).
_TARGET_MEMO: list = []


def host_target_identity() -> str:
    """A digest naming the CPU a ``native`` (or ``native-vector-math``) build
    targets: architecture, CPU model and feature flags (``/proc/cpuinfo``'s
    first processor on Linux, ``platform.processor()`` elsewhere).
    ``-march=native`` is the same string on every machine, so it cannot tell
    two CPUs apart; this digest keeps a shared cache directory from serving
    a binary built for another CPU's instruction set."""
    if _TARGET_MEMO:
        return _TARGET_MEMO[0]
    parts = [platform.machine()]
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            seen = set()
            for line in fh:
                key = line.split(":", 1)[0].strip().lower()
                if key in ("vendor_id", "model name", "cpu family", "model",
                           "flags", "features", "cpu implementer", "cpu part",
                           "isa", "cpu") and key not in seen:
                    seen.add(key)
                    parts.append(line.strip())
                elif not line.strip() and seen:
                    break
    except OSError:
        parts.append(platform.processor())
    ident = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]
    _TARGET_MEMO.append(ident)
    return ident


def host_flags(*, extra=(), profile: str | None = None, opt: str | None = None) -> list[str]:
    """The host ``.so`` recipe: the profile's code-generation flags
    (:func:`host_codegen_flags`, carrying the resolved ``opt`` level), then
    PIC, shared, C++23, and the standalone-artifact build mode (one
    self-contained TU, ``dlopen``-ed by eagle, never linked against a
    CUDA-language aether TU)."""
    return [*host_codegen_flags(profile, opt, extra), f"-std={HOST_STD}", "-shared", "-fPIC",
            "-DAETHER_CPP_MODE", *_shared.include_flags(), *extra]



def reset_witness_memo() -> None:
    """Forget the memoised masked-call witness verdicts (:data:`_WITNESS_MEMO`), as a new process would."""
    _WITNESS_MEMO.clear()


# -- the machine's cache sizes: a fact of the machine the build runs on, stated through ``machine_facts`` ------------------------------------

_SYSFS_CACHES = "/sys/devices/system/cpu/cpu0/cache"


def _cache_bytes(text: str) -> int:
    """A sysfs ``size`` field (``32K``, ``1024K``, ``2M``) in bytes."""
    text = text.strip().upper()
    scale = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}.get(text[-1:], 1)
    return int(text.rstrip("KMG")) * scale


def _sysctl_caches():
    """``(l1d, l2, source)`` off macOS ``sysctl``: the performance cores' (``perflevel0``) where the machine has them, else the plain ``hw``
    keys; ``None`` if neither pair reads."""
    for prefix in ("hw.perflevel0.", "hw."):
        try:
            sizes = [int(subprocess.run(["sysctl", "-n", prefix + key], capture_output=True, text=True, timeout=5, check=True).stdout)
                     for key in ("l1dcachesize", "l2cachesize")]
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
        if all(n > 0 for n in sizes):
            return sizes[0], sizes[1], f"sysctl {prefix}l1dcachesize/l2cachesize"
    return None


@functools.cache
def host_caches(root: str = _SYSFS_CACHES):
    """CPU 0's ``(l1d, l2, source)`` in bytes off ``root`` (Linux sysfs; on macOS the default root reads ``sysctl`` instead), read once per
    process; ``None`` when a size cannot be read."""
    if platform.system() == "Darwin" and root == _SYSFS_CACHES:
        return _sysctl_caches()
    found: dict = {}
    try:
        for entry in sorted(os.listdir(root)):
            if not entry.startswith("index"):
                continue
            here = os.path.join(root, entry)

            def field(name, here=here):
                with open(os.path.join(here, name)) as fh:
                    return fh.read().strip()

            level, kind = int(field("level")), field("type")
            data = kind == "Data" if level == 1 else kind != "Instruction"
            if level in (1, 2) and data:
                found.setdefault(level, _cache_bytes(field("size")))
    except (OSError, ValueError):
        return None
    if not found.get(1) or not found.get(2):
        return None
    return found[1], found[2], root
