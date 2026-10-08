# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Include roots, compilers and the flag strings both drivers use.

One resolution of the two header roots, for every consumer: the emitted TUs,
the cache's flag string (a moved root is a key change), the sidecar's
recorded ``eagle_include``, and ``hawk/_core``'s compile line. The rule lives
in :mod:`hawk._roots`, a leaf module with no hawk imports, so
``hawk/CMakeLists.txt`` can load it by path without importing the hawk
package (whose self-check would import the extension CMake is building).
This module raises HAWK's own :class:`~hawk.ir.HawkError` instead.

Nothing here is skipped when a tool is missing: an unresolvable root or
compiler raises and names the variable that would fix it.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from .. import _roots
from ..ir import HawkError
from .cache import default_cache_dir

#: The workspace root the ``*-abi`` worktrees and ``aether`` hang off
#: (:data:`hawk._roots.WORKSPACE`, re-exported under its historical name).
WORKSPACE = _roots.WORKSPACE

#: ``g++`` compiles the host TU at C++23; nvcc caps the device TU at C++20
#: (matches the previous code generator's own convention).
HOST_STD = "c++23"
DEVICE_STD = "c++20"

#: The arch :func:`resolve_arch` falls back to only when neither a device
#: nor the device toolkit's own supported-arch list can be read: a virtual
#: (PTX-only) arch, never a CUBIN for a guessed device. With a toolkit
#: present the fallback is that toolkit's MINIMUM compute arch instead
#: (:func:`resolve_device_arch`). A plain literal, so setting
#: ``$HAWK_CUDA_ARCH`` after hawk is imported still takes effect.
#: ``hawk.artifact.arch`` is the public name for :func:`resolve_arch`.
FALLBACK_ARCH = "compute_61"

#: Per-process memo of the running device's own arch
#: (:func:`_probed_device_arch`): ``[]`` unprobed, ``[None]`` probed and no
#: device/binding reachable, ``[value]`` probed and resolved. A CUDA
#: context init/device query, not something to repeat on every build.
_DEVICE_ARCH_MEMO: list = []


def reset_device_arch_memo() -> None:
    """Forget the memoised device-arch probe (:func:`resolve_arch`) -- the
    door back to a fresh probe, for a row that changes the visible device
    mid-process."""
    _DEVICE_ARCH_MEMO.clear()


def _probed_device_arch() -> str | None:
    """The running GPU's own ``sm_XX``
    (:func:`hawk.compile.nvrtc.current_device_arch`: the current context's
    device, else ordinal 0), memoised per process, or ``None`` when no
    device is reachable: no cuda-bindings and no driver library, or no CUDA
    device/driver (:class:`~hawk.ir.HawkError`)."""
    if _DEVICE_ARCH_MEMO:
        return _DEVICE_ARCH_MEMO[0]
    from . import nvrtc as _nvrtc

    try:
        value = _nvrtc.current_device_arch()
    except ImportError:
        value = _driver_device_arch()
    except HawkError:
        value = None
    _DEVICE_ARCH_MEMO.append(value)
    return value


#: The CUDA driver library's file name per platform: what
#: :func:`_driver_device_arch` loads.
_DRIVER_LIBRARIES = {"win32": ("nvcuda.dll",)}
_DRIVER_LIBRARIES_DEFAULT = ("libcuda.so.1", "libcuda.so")

#: ``CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`` / ``_MINOR`` (cuda.h).
_CC_MAJOR, _CC_MINOR = 75, 76


def _driver_device_arch(device_index: int | None = None) -> str | None:
    """The running GPU's ``sm_XX`` read from the CUDA driver library itself
    (through :mod:`ctypes`), or ``None`` when no driver or device is there:
    the probe for an environment without cuda-bindings. The driver is present
    wherever a kernel can run at all, so this needs no Python package.

    ``device_index=None`` reads the CURRENT context's device
    (``cuCtxGetDevice``), falling back to ordinal 0 when no context is
    current; an explicit index reads that ordinal."""
    import ctypes

    lib = None
    for name in _DRIVER_LIBRARIES.get(sys.platform, _DRIVER_LIBRARIES_DEFAULT):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if lib is None or lib.cuInit(0) != 0:
        return None
    dev, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    if device_index is not None or lib.cuCtxGetDevice(ctypes.byref(dev)) != 0:
        if lib.cuDeviceGet(ctypes.byref(dev), device_index or 0) != 0:
            return None
    if (lib.cuDeviceGetAttribute(ctypes.byref(major), _CC_MAJOR, dev) != 0
            or lib.cuDeviceGetAttribute(ctypes.byref(minor), _CC_MINOR, dev) != 0):
        return None
    return f"sm_{major.value}{minor.value}"


#: ``nvcc --list-gpu-arch``'s entries, e.g. ``compute_75``.
_COMPUTE_RE = re.compile(r"compute_(\d+)")
#: An arch string's numeric part: ``sm_86``, ``compute_90``, ``sm_90a``.
_ARCH_RE = re.compile(r"^(sm|compute)_(\d+)([a-z]?)$")

#: Per-process memo for :func:`toolkit_archs`, keyed on the compiler kind
#: plus the nvcc binary's ``(path, size, mtime_ns)`` or NVRTC's version.
_TOOLKIT_ARCHS_MEMO: dict = {}


def _nvcc_supported_archs() -> tuple[str, tuple[int, ...]] | None:
    """``(label, archs)`` from ``nvcc --list-gpu-arch``, or ``None``."""
    try:
        nvcc = device_compiler()
        st = os.stat(nvcc)
    except (HawkError, OSError):
        return None
    key = ("nvcc", nvcc, st.st_size, st.st_mtime_ns)
    if key in _TOOLKIT_ARCHS_MEMO:
        return _TOOLKIT_ARCHS_MEMO[key]
    try:
        out = subprocess.run([nvcc, "--list-gpu-arch"], capture_output=True, text=True,
                             timeout=IDENTITY_TIMEOUT_S).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    archs = tuple(sorted({int(m) for m in _COMPUTE_RE.findall(out)}))
    version = nvcc_version(nvcc)
    label = (f"nvcc {version[0]}.{version[1]} ({nvcc})" if version
             else f"nvcc ({nvcc})")
    value = (label, archs) if archs else None
    _TOOLKIT_ARCHS_MEMO[key] = value
    return value


def _nvrtc_supported_archs() -> tuple[str, tuple[int, ...]] | None:
    """``(label, archs)`` from ``nvrtcGetSupportedArchs``, or ``None``."""
    from . import nvrtc as _nvrtc

    try:
        version = _nvrtc.nvrtc_version()
    except Exception:
        return None
    key = ("nvrtc", version)
    if key in _TOOLKIT_ARCHS_MEMO:
        return _TOOLKIT_ARCHS_MEMO[key]
    try:
        archs = tuple(sorted(_nvrtc.supported_archs()))
    except Exception:
        archs = ()
    value = (f"NVRTC {version[0]}.{version[1]}", archs) if archs else None
    _TOOLKIT_ARCHS_MEMO[key] = value
    return value


def toolkit_archs(kind: str | None = None) -> tuple[str, tuple[int, ...]] | None:
    """``(toolkit label, sorted compute archs)`` the device compiler can
    target, or ``None`` when it cannot be asked. ``kind`` is ``"nvcc"``
    (``nvcc --list-gpu-arch``) or ``"nvrtc"`` (``nvrtcGetSupportedArchs``);
    ``None`` follows :func:`device_compiler_kind`. Never raises."""
    if kind is None:
        try:
            kind = device_compiler_kind()
        except HawkError:
            return None
    return _nvcc_supported_archs() if kind == "nvcc" else _nvrtc_supported_archs()


def _device_label() -> str:
    """A human name for the probed device, for :func:`resolve_device_arch`'s
    refusal."""
    from . import nvrtc as _nvrtc

    try:
        return _nvrtc.current_device_name()
    except Exception:
        return "the current CUDA device"


#: The lowest arch a no-GPU build targets: the double ``atomicAdd`` the
#: emitted kernels use needs sm_60.
PORTABLE_MIN_ARCH = 60


def resolve_device_arch(device_arch: str = "", *, kind: str | None = None) -> str:
    """The arch policy behind :func:`resolve_arch`, against the device
    compiler ``kind``'s supported compute archs (:func:`toolkit_archs`).

    The requested arch is ``device_arch=`` > ``$HAWK_CUDA_ARCH`` > the
    running GPU (:func:`_probed_device_arch`). Then:

    * no GPU reachable (and nothing pinned): ``compute_<min>``, PTX at the
      toolkit's minimum compute arch, but never below
      :data:`PORTABLE_MIN_ARCH` (the double ``atomicAdd`` the kernels use
      needs it) -- never a CUBIN for a guessed arch;
    * an arch above the toolkit's maximum: ``compute_<max>``, PTX the
      driver JITs forward for the newer device;
    * an arch below the toolkit's minimum: :class:`~hawk.ir.HawkError`
      naming the device (or the pin), the arch and the toolkit;
    * otherwise the requested arch, unchanged.

    A ``compute_<N>`` answer is a virtual arch: every device compile door
    turns it into a PTX target (:func:`hawk.compile.nvrtc.forced_target`).
    With the toolkit's list unreadable, a requested arch passes through
    unchanged and the no-GPU answer is :data:`FALLBACK_ARCH`."""
    if device_arch:
        requested, who = device_arch, f"device_arch={device_arch!r}"
    elif os.environ.get("HAWK_CUDA_ARCH"):
        requested = os.environ["HAWK_CUDA_ARCH"]
        who = f"$HAWK_CUDA_ARCH={requested!r}"
    else:
        requested = _probed_device_arch()
        who = None
    toolkit = toolkit_archs(kind)
    if requested is None:
        if toolkit is None:
            return FALLBACK_ARCH
        portable = [a for a in toolkit[1] if a >= PORTABLE_MIN_ARCH]
        return f"compute_{portable[0] if portable else toolkit[1][-1]}"
    m = _ARCH_RE.match(requested)
    if toolkit is None or m is None:
        return requested
    label, archs = toolkit
    number = int(m.group(2))
    if number > archs[-1]:
        return f"compute_{archs[-1]}"
    if number < archs[0]:
        subject = who or f"the CUDA device {_device_label()!r}"
        raise HawkError(
            f"{subject} needs arch {requested}, below the minimum compute arch "
            f"{archs[0]} supported by the device toolkit {label}; use an older "
            "CUDA toolkit for this device, or pin a supported arch with "
            "device_arch=/$HAWK_CUDA_ARCH.")
    return requested


def resolve_arch(device_arch: str = "") -> str:
    """The CUDA arch a device artifact compiles for -- the ONE resolver
    every build door routes through (:func:`hawk.artifact.arch` is this
    same function, re-exported under the public name callers already use):
    ``device_arch=`` (an explicit pin) > ``$HAWK_CUDA_ARCH`` > the running
    GPU (:func:`_probed_device_arch`, probed once per process), each checked
    against the device toolkit's supported archs, and a PTX-only
    ``compute_<min>`` when no GPU is reachable (:func:`resolve_device_arch`).

    Call this only where ``"cuda"`` is actually a build target: every call
    may reach the device probe, so a host-only build must never call it at
    all (:mod:`hawk.artifact.bundle`/:mod:`hawk.artifact.unit_cache` gate it
    on ``targets`` before calling)."""
    return resolve_device_arch(device_arch)

#: The bounded per-process memo behind :func:`compiler_identity`, keyed on
#: the compiler binary's ``(path, size, mtime_ns)``. Needed because identity
#: is asked for on every compile — including cache hits — and costs a
#: ``--version`` subprocess; per-process only, like the content memo in
#: :mod:`hawk.compile.cache`.
_IDENTITY_MEMO: dict = {}
_IDENTITY_MEMO_LIMIT = 32

#: Seconds :func:`compiler_identity` waits for ``<compiler> --version``.
#:
#: Must tolerate a COLD filesystem, not a slow compiler: on ZFS, the first
#: exec of a compiler binary after it falls out of page cache is pure I/O
#: and can take far longer than any warm run — a too-short timeout fails a
#: suite's first identity query with "cannot identify compiler", which reads
#: like a broken toolchain and is not one. Costs nothing ordinarily: the
#: answer is MEMOISED per process (:data:`_IDENTITY_MEMO`).
IDENTITY_TIMEOUT_S = 180


def aether_include() -> str:
    """The aether header root (:func:`hawk._roots.aether_include`)."""
    return _resolve(_roots.aether_include)


def eagle_include() -> str:
    """The eagle header root — the directory ``plugin/gref_abi.h`` hangs off
    (:func:`hawk._roots.eagle_include`)."""
    return _resolve(_roots.eagle_include)


def _resolve(fn):
    """Run one :mod:`hawk._roots` resolver, re-raising its stdlib-only
    :class:`~hawk._roots.RootError` as HAWK's own refusal type."""
    try:
        return fn()
    except _roots.RootError as exc:
        raise HawkError(str(exc)) from None


#: The one process-lifetime directory a pip-only compile serves the sealed
#: payload into when neither real root resolves — set by
#: :func:`_pip_only_include_root`, read by :func:`pip_only_root`. Lives for
#: the whole process (``atexit``-cleaned), unlike
#: :mod:`hawk.compile.drivers`'s per-compile fallback, since
#: :func:`include_flags` has no natural "done with it" moment.
_PIP_ONLY_ROOT: str | None = None


def pip_only_root() -> str | None:
    """The pip-only served root this process is using as a real-root
    fallback, or ``None`` if nothing has needed one yet. Only reports; never
    triggers the fallback itself (see :func:`_pip_only_include_root`)."""
    return _PIP_ONLY_ROOT


def _pip_only_include_root() -> str:
    """Materialise :func:`~hawk.compile.payload.current_payload` into one
    private directory for the rest of this process's life, and hand back its
    path — used as both the aether and the eagle root, since the payload's
    own header names (``aether/...`` and ``plugin/...``) already sit at the
    top of one served tree.

    Entered once (``atexit``-cleaned): a compile that needs this twice in one
    process reuses the first directory rather than paying `serve()`'s
    copy-to-tmpfs cost again."""
    global _PIP_ONLY_ROOT
    if _PIP_ONLY_ROOT is None:
        import atexit

        from . import payload as _payload_module
        served = _payload_module.current_payload().serve()
        path = served.__enter__()
        atexit.register(served.__exit__, None, None, None)
        _PIP_ONLY_ROOT = str(path)
    return _PIP_ONLY_ROOT


def _resolved_or_sealed(resolver) -> str:
    """``resolver()`` (:func:`aether_include` or :func:`eagle_include`), or
    the pip-only served root when it raises. Scoped to :func:`include_flags`
    alone: the resolvers themselves stay unchanged and keep raising for
    every other caller — this is the one place a real compiler subprocess
    needs an ``-I`` root and has nowhere else to get one."""
    try:
        return resolver()
    except HawkError:
        return _pip_only_include_root()


def include_flags(*, aether_root: str | None = None,
                  eagle_root: str | None = None) -> list[str]:
    """``-I`` flags in (b)'s mandated order: aether first, eagle second.

    ``aether_root``/``eagle_root`` bypass the resolvers entirely when given:
    a caller that already knows its root (a sealed payload's served
    directory) must never call :func:`aether_include`/:func:`eagle_include`
    to confirm it, since a wheel-only install carries no C++ header tree for
    those resolvers to find. Omitted, this falls back to the sealed
    payload's own served directory for whichever root does not resolve."""
    resolved_aether = (aether_root if aether_root is not None
                      else _resolved_or_sealed(aether_include))
    resolved_eagle = (eagle_root if eagle_root is not None
                      else _resolved_or_sealed(eagle_include))
    return ["-I", resolved_aether, "-I", resolved_eagle]


def host_compiler() -> str:
    """The host compiler, in discovery order: ``$HAWK_CXX`` when set, else
    ``g++`` on ``PATH``, else the ``ziglang`` wheel's ``zig c++`` — a
    pip-only host tier for a box with no system C++ toolchain at all.

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
            raise HawkError(f"$HAWK_CXX={cxx!r} not found on PATH.")
        return found
    found = shutil.which("g++")
    if found:
        return found
    return _zig_cxx_wrapper()


def _zig_cxx_wrapper() -> str:
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
        raise HawkError(
            "no g++ on PATH and the ziglang wheel is not installed; set "
            "$HAWK_CXX, install g++, or `pip install ziglang` ("
            "host-compiler discovery order). A skipped compile certifies "
            "nothing.") from None
    zig_bin = Path(ziglang.__file__).resolve().parent / "zig"
    if not zig_bin.is_file():
        raise HawkError(
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


def device_compiler_kind() -> str:
    """``"nvcc"`` or ``"nvrtc"`` — which compiles the device backend.

    ``nvcc`` when found on ``PATH``/``$HAWK_NVCC`` (the developer/CI tree),
    else ``nvrtc``, the shipped Python tier's in-memory device compile
    (:mod:`hawk.compile.nvrtc`). ``$HAWK_DEVICE_COMPILER`` forces either
    value outright, skipping the PATH probe: ``"nvrtc"`` to rehearse the
    shipped path on a box that does have nvcc, or ``"nvcc"`` to refuse the
    fallback and get the ordinary "no nvcc found" error.
    """
    forced = os.environ.get("HAWK_DEVICE_COMPILER")
    if forced:
        if forced not in ("nvcc", "nvrtc"):
            raise HawkError(
                f"$HAWK_DEVICE_COMPILER={forced!r} must be 'nvcc' or 'nvrtc'.")
        return forced
    # Only $HAWK_NVCC and PATH count: a hard-coded toolkit path would be
    # this developer box's own.
    for candidate in (os.environ.get("HAWK_NVCC"), "nvcc"):
        if candidate and (Path(candidate).is_file() or shutil.which(candidate)):
            return "nvcc"
    return "nvrtc"


def device_compiler() -> str:
    """``nvcc`` — the developer/CI device compiler, or ``$HAWK_NVCC``.

    Call only after confirming :func:`device_compiler_kind` is ``"nvcc"``: it
    always looks for a real ``nvcc`` binary and raises when none exists; it
    is not itself the NVRTC/nvcc dispatch.

    Resolved, never the bare name: the unresolved ``"nvcc"`` used to reach
    :func:`compiler_identity` as the cache key's compiler term, so every
    box's PATH-found nvcc shared one literal string instead of its own
    install path, and ``os.stat("nvcc")`` resolves against the current
    working directory rather than ``PATH`` — it raised on every call and the
    per-process identity memo was never populated.
    """
    for candidate in (os.environ.get("HAWK_NVCC"), "nvcc"):   # same rule as _kind()
        if not candidate:
            continue
        if Path(candidate).is_file():
            return str(Path(candidate).resolve())
        found = shutil.which(candidate)
        if found:
            return found
    raise HawkError("no nvcc found; set $HAWK_NVCC.")


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
#: ``native-vector-math`` (fast) — the default on x86-64 only: ``native``
#: plus three trades. (1) ``-DAETHER_HOST_VECTOR_MATH`` and
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
HOST_PROFILES = ("native", "portable", "native-vector-math")
#: The profiles whose builds are bit-identical to the scalar build.
EXACT_HOST_PROFILES = ("native", "portable")
#: The profile used when neither the compile options nor ``$HAWK_HOST_PROFILE``
#: name one: ``native-vector-math`` on an x86-64 host, ``native`` on any other
#: architecture (where ``native-vector-math`` is refused).
DEFAULT_HOST_PROFILE = ("native-vector-math"
                        if platform.machine().lower() in ("x86_64", "amd64")
                        else "native")

#: The FMA-contraction flag per profile (see :data:`HOST_PROFILES`): off for
#: the exact profiles, fast for the fast one.
_HOST_CONTRACTION = {"native": "-ffp-contract=off",
                     "portable": "-ffp-contract=off",
                     "native-vector-math": "-ffp-contract=fast"}


def host_profile(requested: str | None = None) -> str:
    """The host profile in effect: ``requested`` when given (the compile
    options' ``host_profile``), else ``$HAWK_HOST_PROFILE``, else
    :data:`DEFAULT_HOST_PROFILE`. An unknown name RAISES."""
    name = requested or os.environ.get("HAWK_HOST_PROFILE") or DEFAULT_HOST_PROFILE
    if name not in HOST_PROFILES:
        raise HawkError(
            f"unknown host profile {name!r} (from "
            f"{'the compile options' if requested else '$HAWK_HOST_PROFILE'}); "
            f"the profiles are {HOST_PROFILES}")
    return name


#: The optimisation levels a compile accepts, host or device alike.
OPT_LEVELS = ("O0", "O1", "O2", "O3")
#: The level used when neither the compile options nor ``$HAWK_OPT_LEVEL``
#: name one.
DEFAULT_OPT_LEVEL = "O3"


def opt_level(requested: str | None = None) -> str:
    """The optimisation level in effect: ``requested`` when given (the
    compile options' ``opt_level``), else ``$HAWK_OPT_LEVEL``, else
    :data:`DEFAULT_OPT_LEVEL`. An unknown name RAISES — mirrors
    :func:`host_profile`'s own resolution order exactly.

    Feeds the host build's ``-O<n>`` (:func:`host_codegen_flags`, replacing
    what used to be a hard-coded ``-O3``) and the AOT device build's
    ``-O<n>``/``-Xptxas -O<n>`` (:func:`device_flags`, nvcc only — there was
    no explicit ``-O`` there before, so the default is now spelled out
    rather than left to nvcc's own). NVRTC always optimises and takes no
    ``-O`` option at all: this level does not reach it (see
    :func:`hawk.compile.nvrtc.options`)."""
    name = requested or os.environ.get("HAWK_OPT_LEVEL") or DEFAULT_OPT_LEVEL
    if name not in OPT_LEVELS:
        raise HawkError(
            f"unknown opt level {name!r} (from "
            f"{'the compile options' if requested else '$HAWK_OPT_LEVEL'}); "
            f"the levels are {OPT_LEVELS}")
    return name


def host_codegen_flags(profile: str | None = None, opt: str | None = None) -> list[str]:
    """The code-generation half of the host recipe for ``profile``
    (resolved through :func:`host_profile`): optimisation level (resolved
    through :func:`opt_level`), target CPU, vector width and the
    FMA-contraction setting. See :data:`HOST_PROFILES`."""
    name = host_profile(profile)
    level = opt_level(opt)
    machine = platform.machine().lower()
    x86 = machine in ("x86_64", "amd64")
    if name in ("native", "native-vector-math"):
        target = (["-march=native", "-mprefer-vector-width=512"] if x86
                  else ["-mcpu=native"] if machine in ("aarch64", "arm64", "ppc64le")
                  else ["-march=native"])
    else:
        target = ["-march=x86-64-v2"] if x86 else []
    if name == "native-vector-math":
        if not x86:
            raise HawkError(
                "host profile 'native-vector-math' needs an x86-64 host "
                "(aether's vector math follows the x86-64 vector function ABI)")
        target = [*target, "-fno-trapping-math", "-DAETHER_HOST_VECTOR_MATH"]
    return [f"-{level}", _HOST_CONTRACTION[name], *target]


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
    return [*host_codegen_flags(profile, opt), f"-std={HOST_STD}", "-shared", "-fPIC",
            "-DAETHER_CPP_MODE", *include_flags(), *extra]


def device_flags(*, arch: str = "", target: str = "ptx", extra=(),
                 opt: str | None = None) -> list[str]:
    """The device recipe: AOT ``nvcc -ptx`` (forward-compatible, JIT'd PTX)
    or ``-cubin`` (SASS for the concrete arch, no driver-side ISA ceiling) at
    the artifact's arch. ``target`` defaults to ``"ptx"``; only
    :mod:`hawk.compile.drivers` resolves a target first
    (:func:`hawk.compile.nvrtc.select_target`) and passes it here.

    ``opt`` (resolved through :func:`opt_level`) is spelled out explicitly
    for BOTH halves of the nvcc compile: ``-O<n>`` for nvcc's own host-side
    code generation, and ``-Xptxas -O<n>`` passed through to ptxas for the
    device side — there used to be no explicit ``-O`` here at all (nvcc's
    own default), so :data:`DEFAULT_OPT_LEVEL` now makes that default
    explicit instead of leaving it implicit. NVRTC (:mod:`hawk.compile.nvrtc`)
    never reaches this function; it always optimises and takes no ``-O``
    option, so ``opt_level`` has no effect there.

    ``arch`` is resolved through :func:`resolve_arch`: an explicit value is
    checked against the toolkit's supported archs, and an empty one falls to
    ``$HAWK_CUDA_ARCH``/the running GPU/the toolkit's minimum -- this
    function is reached only for the ``cuda`` backend, so that chain
    (including its device probe) is always in scope here. A virtual
    ``compute_<N>`` arch has no CUBIN and refuses ``target="cubin"``."""
    if target not in ("ptx", "cubin"):
        raise HawkError(
            f"device_flags(): target must be 'ptx' or 'cubin', got {target!r}")
    level = opt_level(opt)
    resolved = resolve_arch(arch)
    if target == "cubin" and resolved.startswith("compute_"):
        raise HawkError(
            f"device_flags(): arch {resolved} is virtual (PTX only); a CUBIN "
            "needs a real sm_<N> arch")
    return [f"-{target}", f"-std={DEVICE_STD}", f"-arch={resolved}",
            f"-{level}", "-Xptxas", f"-{level}", *include_flags(), *extra]


#: ``nvcc --version``'s own release line, e.g. ``"Cuda compilation tools,
#: release 12.6, V12.6.77"`` (see :func:`compiler_identity` for why the
#: version lives on this line and not the first).
_NVCC_RELEASE_RE = re.compile(r"release (\d+)\.(\d+)")

#: The memo behind :func:`nvcc_version`, keyed and bounded like
#: :data:`_IDENTITY_MEMO` — a separate dict because it holds a parsed
#: version tuple rather than an opaque digest string.
_NVCC_VERSION_MEMO: dict = {}


def nvcc_version(compiler: str) -> tuple[int, int] | None:
    """``(major, minor)`` parsed from ``<compiler> --version``'s "release"
    line, or ``None`` when the text carries no such line or the subprocess
    fails — an nvcc whose format this does not recognise is an unknown
    version to the caller (:func:`hawk.compile.nvrtc.select_target`'s own
    conservative answer), not a guess, and the real compile attempt right
    after this is what reports an actual failure loudly."""
    memo_key = None
    try:
        st = os.stat(compiler)
        memo_key = (compiler, st.st_size, st.st_mtime_ns)
    except OSError:
        pass
    if memo_key is not None and memo_key in _NVCC_VERSION_MEMO:
        return _NVCC_VERSION_MEMO[memo_key]
    try:
        out = subprocess.run([compiler, "--version"], capture_output=True, text=True,
                             timeout=IDENTITY_TIMEOUT_S).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = _NVCC_RELEASE_RE.search(out)
    version = (int(m.group(1)), int(m.group(2))) if m else None
    if memo_key is not None:
        if len(_NVCC_VERSION_MEMO) >= _IDENTITY_MEMO_LIMIT:
            _NVCC_VERSION_MEMO.clear()
        _NVCC_VERSION_MEMO[memo_key] = version
    return version


def compiler_identity(compiler: str) -> str:
    """``<path> || <first --version line> || <digest of the whole --version
    text>`` — the compiler term of the lookup key. A binary/version change
    is a key change.

    The digest, not just the first line, is what makes that true: nvcc's
    first ``--version`` line is version-invariant (the actual release lives
    on a later line), so keying on the first line alone gave two different
    nvcc releases the same identity, and the cache served one version's PTX
    to the other's run. Hashing the full text catches that line for nvcc and
    whatever line any other compiler puts its version on, without special-
    casing the word "release".

    Answered from :data:`_IDENTITY_MEMO` when this process has already asked
    about the same binary at the same size and mtime."""
    memo_key = None
    try:
        st = os.stat(compiler)
        memo_key = (compiler, st.st_size, st.st_mtime_ns)
    except OSError:
        pass                       # a name resolved through PATH, or gone: no memo
    if memo_key is not None:
        hit = _IDENTITY_MEMO.get(memo_key)
        if hit is not None:
            return hit
    try:
        out = subprocess.run([compiler, "--version"], capture_output=True, text=True,
                             timeout=IDENTITY_TIMEOUT_S).stdout
    except (OSError, subprocess.SubprocessError) as exc:   # pragma: no cover
        raise HawkError(f"cannot identify compiler {compiler!r}: {exc}") from exc
    first = out.splitlines()[0].strip() if out.strip() else ""
    version_digest = hashlib.sha256(out.encode()).hexdigest()[:16]
    identity = f"{compiler}||{first}||{version_digest}"
    if memo_key is not None:
        if len(_IDENTITY_MEMO) >= _IDENTITY_MEMO_LIMIT:  # pragma: no cover - bound
            _IDENTITY_MEMO.clear()
        _IDENTITY_MEMO[memo_key] = identity
    return identity


def reset_identity_memo() -> None:
    """Forget every memoised compiler identity (:data:`_IDENTITY_MEMO`) AND
    every memoised nvcc version (:data:`_NVCC_VERSION_MEMO`) — a compiler
    binary swapped mid-process should invalidate both alike."""
    _IDENTITY_MEMO.clear()
    _NVCC_VERSION_MEMO.clear()


def _ccbin_from_flags(flags: str) -> str | None:
    """The ``-ccbin``/``--compiler-bindir`` value in an nvcc flag string."""
    try:
        tokens = shlex.split(flags)
    except ValueError:
        return None
    for i, tok in enumerate(tokens):
        if tok in ("-ccbin", "--compiler-bindir") and i + 1 < len(tokens):
            return tokens[i + 1]
        for prefix in ("-ccbin=", "--compiler-bindir="):
            if tok.startswith(prefix):
                return tok[len(prefix):]
    return None


def _strip_ccbin(flags: str) -> list[str]:
    """``flags`` tokenised without any ``-ccbin``/``--compiler-bindir``."""
    try:
        tokens = shlex.split(flags)
    except ValueError:
        return []
    out, skip = [], False
    for tok in tokens:
        if skip:
            skip = False
            continue
        if tok in ("-ccbin", "--compiler-bindir"):
            skip = True
            continue
        if tok.startswith(("-ccbin=", "--compiler-bindir=")):
            continue
        out.append(tok)
    return out


def device_host_compiler(env: dict | None = None) -> str | None:
    """The host compiler ``nvcc`` will use under ``env`` (default
    :func:`subprocess_env`): ``-ccbin`` from ``NVCC_PREPEND_FLAGS`` or
    ``NVCC_APPEND_FLAGS``, else ``$NVCC_CCBIN``, else the ``g++`` on the
    subprocess ``PATH`` (nvcc's own default). A ``-ccbin`` directory is
    resolved to the ``g++`` inside it. ``None`` when none can be found."""
    env = subprocess_env() if env is None else env
    chosen = (_ccbin_from_flags(env.get("NVCC_PREPEND_FLAGS", ""))
              or _ccbin_from_flags(env.get("NVCC_APPEND_FLAGS", ""))
              or env.get("NVCC_CCBIN") or "g++")
    if Path(chosen).is_dir():
        chosen = str(Path(chosen) / "g++")
    if Path(chosen).is_file():
        return str(Path(chosen).resolve())
    return shutil.which(chosen, path=env.get("PATH"))


def device_host_compiler_identity() -> str:
    """The device cache key's host-compiler term: the
    :func:`compiler_identity` of :func:`device_host_compiler`, or
    ``"ccbin:unknown"`` when none is found (the nvcc compile then fails on
    its own)."""
    found = device_host_compiler()
    if not found:
        return "ccbin:unknown"
    try:
        return f"ccbin:{compiler_identity(found)}"
    except HawkError:
        return f"ccbin:{found}"


def subprocess_env() -> dict:
    """The environment a driver subprocess runs under: the device toolkit's
    ``bin`` first on ``PATH`` and ``CUDA_PATH`` at its root.

    The nvcc host compiler is chosen in this order: ``$HAWK_NVCC_CCBIN``
    (set as ``-ccbin`` in ``NVCC_PREPEND_FLAGS``, replacing any other
    ``-ccbin`` there and keeping the rest), else a user-set
    ``NVCC_PREPEND_FLAGS``/``NVCC_CCBIN`` left untouched, else nothing --
    nvcc's own default, the ``g++`` on ``PATH``. The choice is part of the
    device cache key (:func:`device_host_compiler_identity`)."""
    env = dict(os.environ)
    ccbin = os.environ.get("HAWK_NVCC_CCBIN")
    if ccbin:
        rest = _strip_ccbin(env.get("NVCC_PREPEND_FLAGS", ""))
        env["NVCC_PREPEND_FLAGS"] = shlex.join(["-ccbin", ccbin, *rest])
    try:
        cuda = Path(device_compiler()).resolve().parent
    except HawkError:                                       # pragma: no cover
        return env
    env["PATH"] = f"{cuda}:{env.get('PATH', '')}"
    env["CUDA_PATH"] = str(cuda.parent)
    env.setdefault("TMPDIR", os.environ.get("TMPDIR", "/tmp"))
    return env
