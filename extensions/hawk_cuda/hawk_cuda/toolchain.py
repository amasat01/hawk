# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The device toolchain: which compiler builds the device target, the arch it builds for, and its flags."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from hawk.compile.toolchain import (
    IDENTITY_TIMEOUT_S,
    _probe_stdout,
    compiler_identity,
    include_flags,
    nvcc_version,
    opt_level,
    subprocess_env,
)
from raptor.protocols.refusal import ToolchainAbsent
from raptor.errors import RaptorError

#: ``CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`` / ``_MINOR`` (cuda.h).
_CC_MAJOR, _CC_MINOR = 75, 76

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
    (:func:`hawk_cuda.nvrtc.current_device_arch`: the current context's
    device, else ordinal 0), memoised per process, or ``None`` when no
    device is reachable: no cuda-bindings and no driver library, or no CUDA
    device/driver (:class:`~raptor.errors.RaptorError`)."""
    if _DEVICE_ARCH_MEMO:
        return _DEVICE_ARCH_MEMO[0]
    from . import nvrtc as _nvrtc

    try:
        value = _nvrtc.current_device_arch()
    except ImportError:
        value = _driver_device_arch()
    except RaptorError:
        value = None
    _DEVICE_ARCH_MEMO.append(value)
    return value


#: The CUDA driver library's file name per platform: what
#: :func:`_driver_device_arch` loads.
_DRIVER_LIBRARIES = {"win32": ("nvcuda.dll",)}


_DRIVER_LIBRARIES_DEFAULT = ("libcuda.so.1", "libcuda.so")


def _driver_device_arch() -> str | None:
    """The running GPU's ``sm_XX`` read from the CUDA driver library itself
    (through :mod:`ctypes`), or ``None`` when no driver or device is there:
    the probe for an environment without cuda-bindings. The driver is present
    wherever a kernel can run at all, so this needs no Python package.

    Reads the CURRENT context's device (``cuCtxGetDevice``), falling back to
    ordinal 0 when no context is current."""
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
    if lib.cuCtxGetDevice(ctypes.byref(dev)) != 0:
        if lib.cuDeviceGet(ctypes.byref(dev), 0) != 0:
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
    except (RaptorError, OSError):
        return None
    key = ("nvcc", nvcc, st.st_size, st.st_mtime_ns)
    if key in _TOOLKIT_ARCHS_MEMO:
        return _TOOLKIT_ARCHS_MEMO[key]
    try:
        out = _probe_stdout(nvcc, ("--list-gpu-arch",), timeout=IDENTITY_TIMEOUT_S)[1]
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
        except RaptorError:
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
    * an arch below the toolkit's minimum: :class:`~raptor.errors.RaptorError`
      naming the device (or the pin), the arch and the toolkit;
    * otherwise the requested arch, unchanged.

    A ``compute_<N>`` answer is a virtual arch: every device compile door
    turns it into a PTX target (:func:`hawk_cuda.nvrtc.forced_target`).
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
        raise RaptorError(
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

    Call this only where a target compiles with the device toolchain: every call
    may reach the device probe, so a host-only build must never call it at
    all (:mod:`hawk.artifact.bundle`/:mod:`hawk.artifact.unit_cache` gate it
    on ``targets`` before calling)."""
    return resolve_device_arch(device_arch)


def device_compiler_kind() -> str:
    """``"nvcc"`` or ``"nvrtc"`` — which compiles the device target.

    ``nvcc`` when found on ``PATH``/``$HAWK_NVCC`` (the developer/CI tree),
    else ``nvrtc``, the shipped Python tier's in-memory device compile
    (:mod:`hawk_cuda.nvrtc`). ``$HAWK_DEVICE_COMPILER`` forces either
    value outright, skipping the PATH probe: ``"nvrtc"`` to rehearse the
    shipped path on a box that does have nvcc, or ``"nvcc"`` to refuse the
    fallback and get the ordinary "no nvcc found" error.
    """
    forced = os.environ.get("HAWK_DEVICE_COMPILER")
    if forced:
        if forced not in ("nvcc", "nvrtc"):
            raise RaptorError(
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
    raise ToolchainAbsent("no nvcc found; set $HAWK_NVCC.", toolchain="nvcc")


def device_flags(*, arch: str = "", target: str = "ptx", extra=(),
                 opt: str | None = None) -> list[str]:
    """The device recipe: AOT ``nvcc -ptx`` (forward-compatible, JIT'd PTX)
    or ``-cubin`` (SASS for the concrete arch, no driver-side ISA ceiling) at
    the artifact's arch. ``target`` defaults to ``"ptx"``; only
    :mod:`hawk.compile.drivers` resolves a target first
    (:func:`hawk_cuda.nvrtc.select_target`) and passes it here.

    ``opt`` (resolved through :func:`opt_level`) is spelled out explicitly
    for BOTH halves of the nvcc compile: ``-O<n>`` for nvcc's own host-side
    code generation, and ``-Xptxas -O<n>`` passed through to ptxas for the
    device side — there used to be no explicit ``-O`` here at all (nvcc's
    own default), so :data:`DEFAULT_OPT_LEVEL` now makes that default
    explicit instead of leaving it implicit. NVRTC (:mod:`hawk_cuda.nvrtc`)
    never reaches this function; it always optimises and takes no ``-O``
    option, so ``opt_level`` has no effect there.

    ``arch`` is resolved through :func:`resolve_arch`: an explicit value is
    checked against the toolkit's supported archs, and an empty one falls to
    ``$HAWK_CUDA_ARCH``/the running GPU/the toolkit's minimum -- this
    function is reached only for a device-compiled target, so that chain
    (including its device probe) is always in scope here. A virtual
    ``compute_<N>`` arch has no CUBIN and refuses ``target="cubin"``."""
    if target not in ("ptx", "cubin"):
        raise RaptorError(
            f"device_flags(): target must be 'ptx' or 'cubin', got {target!r}")
    level = opt_level(opt)
    resolved = resolve_arch(arch)
    if target == "cubin" and resolved.startswith("compute_"):
        raise RaptorError(
            f"device_flags(): arch {resolved} is virtual (PTX only); a CUBIN "
            "needs a real sm_<N> arch")
    # ptxas runs only for a CUBIN; `-v` has it report each kernel's registers,
    # spills and stack on stderr (:func:`hawk.compile.drivers.parse_ptxas_info`).
    verbose = ["-Xptxas", "-v"] if target == "cubin" else []
    return [f"-{target}", f"-std={DEVICE_STD}", f"-arch={resolved}",
            f"-{level}", "-Xptxas", f"-{level}", *verbose, *include_flags(), *extra]


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
    env = subprocess_env(("cuda",)) if env is None else env
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
    except RaptorError:
        return f"ccbin:{found}"


def process_env(env: dict) -> dict:
    """``env`` with the device toolkit's ``bin`` first on ``PATH`` and ``CUDA_PATH`` at its root.

    The nvcc host compiler is chosen in this order: ``$HAWK_NVCC_CCBIN``
    (set as ``-ccbin`` in ``NVCC_PREPEND_FLAGS``, replacing any other
    ``-ccbin`` there and keeping the rest), else a user-set
    ``NVCC_PREPEND_FLAGS``/``NVCC_CCBIN`` left untouched, else nothing --
    nvcc's own default, the ``g++`` on ``PATH``. The choice is part of the
    device cache key (:func:`device_host_compiler_identity`)."""
    ccbin = os.environ.get("HAWK_NVCC_CCBIN")
    if ccbin:
        rest = _strip_ccbin(env.get("NVCC_PREPEND_FLAGS", ""))
        env["NVCC_PREPEND_FLAGS"] = shlex.join(["-ccbin", ccbin, *rest])
    try:
        cuda = Path(device_compiler()).resolve().parent
    except RaptorError:                                       # pragma: no cover
        return env
    env["PATH"] = f"{cuda}:{env.get('PATH', '')}"
    env["CUDA_PATH"] = str(cuda.parent)
    env.setdefault("TMPDIR", os.environ.get("TMPDIR", "/tmp"))
    return env
