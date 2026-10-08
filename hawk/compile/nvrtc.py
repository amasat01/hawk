# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The NVRTC device compile.

The shipped Python tier's device compile: an in-memory NVRTC compile of
the device TU, serving the whole sealed :mod:`aether_dsc` payload as
NVRTC's header array — no header ever touches a filesystem. ``nvcc``
stays the developer/CI compiler; this module is reached only when
:func:`hawk.compile.toolchain.device_compiler_kind` resolves to
``"nvrtc"``.

The one translation this module owns: a payload header is stored under
an ``rtc/``-prefixed name but must be served to NVRTC under the bare
name a real ``#include <cstdint>`` asks for, since NVRTC's header lookup
is an exact string match. ``aether/...`` and ``plugin/...`` entries need
no translation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import platform
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

from ..ir import HawkError
from . import payload as _payload


class NvrtcUnavailable(HawkError):
    """NVRTC could not be loaded for a device compile. Raised in place
    of the binding import's own ``ImportError``, so a caller catches one
    hawk-owned type and the message can name the fix."""


def _import_nvrtc():
    """`cuda.bindings.nvrtc`, or :class:`NvrtcUnavailable`.

    A single import chokepoint so every NVRTC entry point in this module
    answers a missing binding the same way."""
    try:
        from cuda.bindings import nvrtc as _nvrtc
    except ImportError as exc:
        raise NvrtcUnavailable(
            "NVRTC could not be loaded for the device compile; install "
            "cuda-bindings with NVRTC and CCCL for your CUDA major version: "
            "pip install 'raptor-hawk[cuda12]' (cuda-bindings 12, "
            "nvidia-cuda-nvrtc-cu12, nvidia-cuda-cccl-cu12) or "
            "'raptor-hawk[cuda13]' (cuda-bindings 13, nvidia-cuda-nvrtc 13, "
            "nvidia-cuda-cccl 13)."
        ) from exc
    return _nvrtc


def served_name(payload_name: str) -> str:
    """The name NVRTC is asked to resolve for a payload entry — see the
    module docstring's translation note."""
    return (payload_name.split("/", 1)[1]
            if payload_name.startswith("rtc/") else payload_name)


#: The import names of the CCCL wheels: ``nvidia-cuda-cccl-cu12`` installs
#: ``nvidia/cuda_cccl/include``; the CUDA 13 wheels (``nvidia-cuda-cccl``
#: 13.x, no suffix: the ``-cu13`` names on PyPI are empty placeholders) install under
#: ``nvidia/cu13/include/cccl`` (or ``nvidia/cuda_cccl/include/cccl``).
_CCCL_WHEEL_MODULES = ("nvidia.cuda_cccl", "nvidia.cu13")

_CCCL_MISSING = (
    "no CCCL include directory found for the NVRTC device compile: install "
    "the CCCL wheel matching the loaded NVRTC (nvidia-cuda-cccl-cu12 for "
    "CUDA 12, nvidia-cuda-cccl 13.x for CUDA 13; both come with "
    "pip install 'raptor-hawk[cuda12]' / '[cuda13]'), or set $CUDA_PATH to a "
    "CUDA toolkit")


def _has_cccl(candidate: Path) -> bool:
    return (candidate / "cuda" / "std").is_dir()


def _include_candidates(base: Path) -> list[Path]:
    """One include directory's CCCL candidates: the flat CUDA 12 layout,
    then CUDA 13's ``cccl/`` subdirectory."""
    return [base, base / "cccl"]


def cccl_include_dir() -> str:
    """Where `<cuda/std/...>` lives. Tried in order:

    1. the CUDA root the NVRTC library actually LOADED in this process
       comes from (:func:`loaded_nvrtc_root`), so the headers match the
       compiler that will read them;
    2. a CCCL wheel (:data:`_CCCL_WHEEL_MODULES`, both layouts);
    3. toolkit roots: `$CUDA_PATH`, `$CUDA_HOME`, `$CONDA_PREFIX`, the
       toolkit holding the `nvcc` on `PATH`, then `/usr/local/cuda`.

    Under every root, `include/`, `include/cccl/` (the CUDA 13 layout) and
    the same two under `targets/*-linux/` are tried.

    `importlib.util.find_spec` on a dotted name imports the parent first
    and raises `ModuleNotFoundError`, rather than returning `None`, when
    that parent does not exist — a pip-only box with no `nvidia-*` wheel at
    all. Caught here as the same "not installed" answer as a wheel existing
    but lacking CCCL."""
    loaded = loaded_nvrtc_root()
    if loaded is not None:
        for candidate in _root_include_dirs(loaded):
            if _has_cccl(candidate):
                return str(candidate)
    for module in _CCCL_WHEEL_MODULES:
        try:
            spec = importlib.util.find_spec(module)
        except (ModuleNotFoundError, ValueError):
            spec = None
        if spec is None or not spec.submodule_search_locations:
            continue
        for loc in spec.submodule_search_locations:
            for candidate in _include_candidates(Path(loc) / "include"):
                if _has_cccl(candidate):
                    return str(candidate)
    for candidate in _toolkit_include_dirs():
        if _has_cccl(candidate):
            return str(candidate)
    raise HawkError(_CCCL_MISSING)


def _loaded_library_path(stem: str) -> str | None:
    """The file path of the shared library `stem` (``libnvrtc.so``) as
    mapped into this process (``/proc/self/maps``), or ``None``."""
    try:
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                path = line.rstrip("\n").split(None, 5)[-1]
                name = os.path.basename(path)
                if name == stem or name.startswith(stem + "."):
                    return path
    except OSError:
        return None
    return None


def loaded_nvrtc_root() -> Path | None:
    """The CUDA root of the NVRTC library this process has loaded: the
    parent of its ``lib``/``lib64`` directory (a toolkit root, a
    ``targets/<arch>-linux`` directory or a wheel's package root). ``None``
    when NVRTC is not importable or the mapping cannot be read."""
    try:
        nvrtc_version()                 # forces the library to load
    except Exception:
        return None
    path = _loaded_library_path("libnvrtc.so")
    if path is None:
        return None
    return Path(path).parent.parent


def _root_include_dirs(root: Path) -> list[Path]:
    """One CUDA root's candidate CCCL include directories, most specific
    first: `include/` and `include/cccl/`, then the same under
    `targets/<arch>-linux/` (this machine's arch first, then any other
    `targets/*-linux` the toolkit carries)."""
    targets = root / "targets"
    own = targets / f"{platform.machine()}-linux" / "include"
    others = sorted(targets.glob("*-linux/include")) if targets.is_dir() else []
    dirs: list[Path] = []
    for d in [root / "include", own, *others]:
        for c in _include_candidates(d):
            if c not in dirs:
                dirs.append(c)
    return dirs


def _toolkit_include_dirs() -> list[Path]:
    """Candidate toolkit include directories for :func:`cccl_include_dir`'s
    third step, per root in order (:func:`_root_include_dirs`)."""
    roots = [os.environ.get("CUDA_PATH"), os.environ.get("CUDA_HOME"),
             os.environ.get("CONDA_PREFIX")]
    nvcc = shutil.which("nvcc")
    if nvcc:
        roots.append(str(Path(nvcc).resolve().parent.parent))
    roots.append("/usr/local/cuda")
    dirs: list[Path] = []
    for root in dict.fromkeys(r for r in roots if r):
        for d in _root_include_dirs(Path(root)):
            if d not in dirs:
                dirs.append(d)
    return dirs


def cccl_digest(cccl_include: str | None = None) -> str:
    """A cheap identity for the served CCCL headers: sha256 over the
    `cuda/std/*` file names, sizes and mtimes. Content is not read: these
    files change only when the wheel/toolkit version changes, and
    name+size+mtime already tells one install apart from another."""
    root = Path(cccl_include or cccl_include_dir()) / "cuda" / "std"
    h = hashlib.sha256()
    for f in sorted(root.rglob("*")):
        if f.is_file():
            st = f.stat()
            h.update(str(f.relative_to(root)).encode())
            h.update(b"\0")
            h.update(f"{st.st_size}\0{st.st_mtime_ns}\0".encode())
    return h.hexdigest()


def nvrtc_version() -> tuple[int, int]:
    """`(major, minor)` of the loaded NVRTC — a compiler subprocess never
    runs for this (the `compiler_identity` timeout/RLIMIT_AS machinery is
    for a SUBPROCESS; NVRTC is an in-process library call)."""
    _nvrtc = _import_nvrtc()
    _, major, minor = _nvrtc.nvrtcVersion()
    return major, minor


def driver_cuda_version() -> tuple[int, int]:
    """`(major, minor)` of the CUDA version the loaded GPU driver
    supports — the ceiling on the PTX ISA the driver's JIT will accept,
    used by the `target="ptx"` guard in :func:`device`. Distinct from
    :func:`nvrtc_version`: NVRTC's own version can outrun an older
    driver on the same box."""
    try:
        from cuda.bindings import driver as _driver
    except ImportError as exc:
        raise HawkError(
            "cuda-bindings is not installed; the driver version is unknown.") from exc
    (err,) = _driver.cuInit(0)
    if err != _driver.CUresult.CUDA_SUCCESS:
        raise HawkError(f"cuInit failed ({err}); no CUDA driver reachable.")
    err, encoded = _driver.cuDriverGetVersion()
    if err != _driver.CUresult.CUDA_SUCCESS:
        raise HawkError(f"cuDriverGetVersion failed ({err}).")
    return encoded // 1000, (encoded % 1000) // 10


def _probed(fn):
    """``fn()``, or ``None`` when it raises a :class:`HawkError` — the
    "unknown" half of :func:`select_target`'s inputs: a version query
    needing a reachable driver must not turn a headless, GPU-less
    build-time compile into a hard failure."""
    try:
        return fn()
    except HawkError:
        return None


def select_target(compiler_version: tuple[int, int] | None,
                   driver_version: tuple[int, int] | None) -> str:
    """``"cubin"`` or ``"ptx"`` — the device artifact that loads on the
    driver present, by comparing the compiler's version (NVRTC's or AOT
    nvcc's) against the installed driver's.

    A compiler newer than the driver answers ``"cubin"``: a PTX image it
    emits would carry an ISA the driver's JIT refuses outright, while a
    CUBIN is already-lowered SASS with no such ceiling. A driver at
    least as new as the compiler answers ``"ptx"``, taking forward
    compatibility for free. Either version unknown answers ``"cubin"``,
    the conservative side: it either loads or fails to compile loudly,
    never the silent-at-compile / failed-at-load split a mismatched PTX
    produces."""
    if compiler_version is None or driver_version is None:
        return "cubin"
    return "cubin" if compiler_version > driver_version else "ptx"


def _current_device(device_index: int | None = None):
    """``(driver module, CUdevice)``: the CURRENT context's device
    (``cuCtxGetDevice``) when ``device_index`` is ``None`` and a context is
    current, else ordinal ``device_index`` (``0`` by default)."""
    from cuda.bindings import driver as _driver
    (err,) = _driver.cuInit(0)
    if err != _driver.CUresult.CUDA_SUCCESS:
        raise HawkError(f"cuInit failed ({err}); no CUDA device reachable.")
    if device_index is None:
        err, dev = _driver.cuCtxGetDevice()
        if err == _driver.CUresult.CUDA_SUCCESS:
            return _driver, dev
    ordinal = device_index or 0
    err, dev = _driver.cuDeviceGet(ordinal)
    if err != _driver.CUresult.CUDA_SUCCESS:
        raise HawkError(f"cuDeviceGet({ordinal}) failed ({err}).")
    return _driver, dev


def current_device_arch(device_index: int | None = None) -> str:
    """`"sm_XX"` of the device this process is running on: the current
    CUDA context's device, or ordinal 0 when no context is current (an
    explicit `device_index` reads that ordinal instead)."""
    _driver, dev = _current_device(device_index)
    err, major = _driver.cuDeviceGetAttribute(
        _driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)
    if err != _driver.CUresult.CUDA_SUCCESS:
        raise HawkError(f"cuDeviceGetAttribute(major) failed ({err}).")
    err, minor = _driver.cuDeviceGetAttribute(
        _driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)
    if err != _driver.CUresult.CUDA_SUCCESS:
        raise HawkError(f"cuDeviceGetAttribute(minor) failed ({err}).")
    return f"sm_{major}{minor}"


def current_device_name(device_index: int | None = None) -> str:
    """The marketing name of the device :func:`current_device_arch` reads."""
    _driver, dev = _current_device(device_index)
    err, raw = _driver.cuDeviceGetName(256, dev)
    if err != _driver.CUresult.CUDA_SUCCESS:
        raise HawkError(f"cuDeviceGetName failed ({err}).")
    return bytes(raw).split(b"\0", 1)[0].decode(errors="replace")


def supported_archs() -> tuple[int, ...]:
    """The compute archs the loaded NVRTC can target
    (``nvrtcGetSupportedArchs``), e.g. ``(75, 80, ..., 120)``."""
    _nvrtc = _import_nvrtc()
    err, archs = _nvrtc.nvrtcGetSupportedArchs()
    if err != _nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise HawkError(f"nvrtcGetSupportedArchs failed ({err}).")
    return tuple(int(a) for a in archs)


def forced_target(arch: str) -> str | None:
    """``"ptx"`` for a virtual ``compute_<N>`` arch (no CUBIN exists for
    one: the arch policy's no-GPU and above-the-toolkit answers,
    :func:`hawk.compile.toolchain.resolve_device_arch`), else ``None`` --
    the target is then :func:`select_target`'s."""
    return "ptx" if arch.startswith("compute_") else None


def identity(cccl_include: str | None = None) -> str:
    """The compiler TERM of the lookup key under NVRTC:
    `f"nvrtc {major}.{minor}"` plus the CCCL include dir's digest — a
    compiler upgrade OR a different CCCL wheel is a key change, exactly like
    an nvcc binary/version change is today."""
    major, minor = nvrtc_version()
    return f"nvrtc {major}.{minor}|cccl:{cccl_digest(cccl_include)}"


def options(arch: str, *, cccl_include: str | None = None,
            virtual_arch: bool = True) -> list[str]:
    """The NVRTC options this profile compiles under: C++20, `-arch=
    compute_<N>` (NVRTC's PTX target is virtual, so `sm_61` becomes
    `compute_61`), `-default-device` so a bare `AETHER_DEVICE()` function
    needs no `extern "C" __global__` wrapper, and the CCCL include root
    for the few headers NVRTC resolves from a real path.

    `virtual_arch=False` (the public CUBIN path) keeps `sm_<N>` literal
    instead: `nvrtcGetCUBIN` after a `compute_<N>` compile returns a
    zero-byte CUBIN with no error, and only a real `-arch=sm_<N>` asks
    NVRTC to lower to SASS.

    No ``-O`` option is in this list, by design: NVRTC always optimises
    (there is no equivalent of nvcc/ptxas's ``-O0``..``-O3`` ladder to
    select), so :func:`hawk.compile.toolchain.opt_level`/``opt_level=`` has
    no effect on an NVRTC compile and this function does not pretend
    otherwise by accepting or faking one."""
    device_arch = (arch.replace("sm_", "compute_")
                   if virtual_arch and arch.startswith("sm_") else arch)
    return ["-std=c++20", f"-arch={device_arch}", "-default-device",
            f"-I{cccl_include or cccl_include_dir()}"]


class NvrtcResult:
    """One NVRTC compile's outcome: the PTX bytes, or `None` on failure with
    `log` naming why."""

    __slots__ = ("ok", "ptx", "log")

    def __init__(self, ok: bool, ptx: bytes | None, log: str):
        self.ok = ok
        self.ptx = ptx
        self.log = log


def compile_ptx(source: str, name: str, arch: str, *,
                 headers: Mapping[str, bytes] | None = None) -> NvrtcResult:
    """Compile `source` (a HAWK-emitted device TU) through NVRTC, serving
    every header in `headers` (default: the sealed payload's own) from
    memory — no header is read from disk. Returns the PTX bytes on
    success; the diagnostic log always comes back, empty on a clean
    compile."""
    return _compile_image(source, name, arch, target="ptx", headers=headers)


class DeviceImage(NamedTuple):
    """One public :func:`device` compile's result: the compiled bytes, which
    `target` produced them, and the `sm_XX` they were compiled for."""

    image: bytes
    target: str
    arch: str


def _compile_image(source: str, name: str, arch: str, *, target: str,
                    headers: Mapping[str, bytes] | None = None) -> NvrtcResult:
    """The NVRTC mechanics behind :func:`compile_ptx`, :func:`compile_cubin`
    and :func:`device`: serve the headers from memory, compile, and fetch the
    PTX or the CUBIN per ``target``. A CUBIN is a binary ELF image and is not
    ``rstrip``'d the way the NUL-terminated PTX text is."""
    _nvrtc = _import_nvrtc()

    served = dict(headers if headers is not None
                  else _payload.current_payload().device_headers)
    names = sorted(served)
    header_bytes = [served[n] for n in names]
    include_names = [served_name(n).encode() for n in names]

    _, prog = _nvrtc.nvrtcCreateProgram(
        source.encode(), f"{name}.cu".encode(), len(names), header_bytes, include_names)
    try:
        opts = options(arch, virtual_arch=(target == "ptx"))
        (compile_result,) = _nvrtc.nvrtcCompileProgram(prog, len(opts),
                                                        [o.encode() for o in opts])
        _, log_size = _nvrtc.nvrtcGetProgramLogSize(prog)
        log_buf = bytearray(log_size)
        _nvrtc.nvrtcGetProgramLog(prog, log_buf)
        log = bytes(log_buf).decode(errors="replace").rstrip("\x00")
        if compile_result != _nvrtc.nvrtcResult.NVRTC_SUCCESS:
            return NvrtcResult(False, None, log)
        if target == "ptx":
            _, size = _nvrtc.nvrtcGetPTXSize(prog)
            buf = bytearray(size)
            _nvrtc.nvrtcGetPTX(prog, buf)
            image = bytes(buf).rstrip(b"\x00")
        else:
            _, size = _nvrtc.nvrtcGetCUBINSize(prog)
            buf = bytearray(size)
            _nvrtc.nvrtcGetCUBIN(prog, buf)
            image = bytes(buf)
        return NvrtcResult(True, image, log)
    finally:
        _nvrtc.nvrtcDestroyProgram(prog)


def compile_cubin(source: str, name: str, arch: str, *,
                   headers: Mapping[str, bytes] | None = None) -> NvrtcResult:
    """:func:`compile_ptx`'s sibling: compile `source` through NVRTC
    straight to a CUBIN (SASS) for the concrete `arch`, for a caller
    (:mod:`hawk.compile.drivers`) that wants the same in-memory compile
    :func:`device`/:func:`cubin` offer, without their `DeviceImage`/
    arch-resolution wrapper."""
    return _compile_image(source, name, arch, target="cubin", headers=headers)


def _guard_ptx_driver_support() -> None:
    """`target="ptx"` refuses outright when NVRTC would emit a PTX ISA
    newer than the loaded driver accepts — `cuModuleLoadData` on such a
    PTX fails at load time with an opaque error, so this names the
    refusal before the compile runs. `target="cubin"` needs no such
    guard: NVRTC lowers straight to SASS for the artifact's own arch."""
    major, minor = nvrtc_version()
    dmajor, dminor = driver_cuda_version()
    if (major, minor) > (dmajor, dminor):
        raise HawkError(
            f"NVRTC {major}.{minor} is newer than this driver's CUDA "
            f"{dmajor}.{dminor} — a PTX it emits would carry a virtual "
            "architecture the driver's JIT would refuse to load. Use "
            "target=\"cubin\" instead, or upgrade the GPU driver.")


def device(source: str, *, target: str | None = None, arch: str | None = None,
           name: str = "hawk_device_image",
           headers: Mapping[str, bytes] | None = None) -> DeviceImage:
    """Compile ``source`` (ordinary CUDA C++) for the device through
    NVRTC — the public entry point a pip-only caller (no `nvcc`, no
    aether/eagle checkout) reaches for instead of shelling out.

    ``target=None`` (the default) picks the artifact that loads on the
    driver present (:func:`select_target`); ``target="cubin"``/``"ptx"``
    force one outright, and ``"ptx"`` still refuses when the driver
    cannot JIT it (:func:`_guard_ptx_driver_support`).

    ``arch`` (``None``: ``$HAWK_CUDA_ARCH``, else the current device) goes
    through the arch policy against NVRTC's supported archs
    (:func:`hawk.compile.toolchain.resolve_device_arch`): no GPU reachable
    gives PTX at NVRTC's minimum compute arch, a device newer than NVRTC
    supports gives PTX at its maximum, an older one raises. A virtual
    ``compute_<N>`` arch always compiles to PTX (:func:`forced_target`).
    Headers are served from the sealed :mod:`hawk.compile.payload` (never
    disk).

    This does not go through :mod:`hawk.compile.cache` — it has no lookup
    key of its own; a caller wanting one builds it from the returned
    :class:`DeviceImage`.
    """
    if target is not None and target not in ("cubin", "ptx"):
        raise HawkError(
            f"hawk.compile.device(): target must be 'cubin' or 'ptx', got {target!r}")
    from . import toolchain as _tc

    resolved_arch = _tc.resolve_device_arch(arch or "", kind="nvrtc")
    forced = forced_target(resolved_arch)
    if forced is not None and target == "cubin":
        raise HawkError(
            f"hawk.compile.device(): arch {resolved_arch} is virtual (PTX only: "
            "no GPU reachable, or a device newer than this NVRTC supports); a "
            "CUBIN needs a real sm_<N> arch")
    resolved_target = target or forced or select_target(
        _probed(nvrtc_version), _probed(driver_cuda_version))
    if resolved_target == "ptx" and _probed(driver_cuda_version) is not None:
        _guard_ptx_driver_support()
    result = _compile_image(source, name, resolved_arch, target=resolved_target,
                            headers=headers)
    if not result.ok:
        raise HawkError(
            f"device compile of {name!r} failed under NVRTC (target={resolved_target}, "
            f"arch={resolved_arch}).\n{result.log[:4000]}")
    return DeviceImage(image=result.ptx, target=resolved_target, arch=resolved_arch)


def cubin(source: str, arch: str | None = None, **kwargs) -> DeviceImage:
    """`device(source, target="cubin", arch=arch)` — the common case named
    for itself."""
    return device(source, target="cubin", arch=arch, **kwargs)


def cubin_available() -> bool:
    """Whether :func:`cubin`/`device` could succeed on THIS process right
    now: NVRTC importable and a CUDA device present. Never raises — a
    caller's own "can I even try" probe, not a compile attempt."""
    try:
        _import_nvrtc()
    except NvrtcUnavailable:
        return False
    try:
        from cuda.bindings import driver as _driver
        (err,) = _driver.cuInit(0)
        if err != _driver.CUresult.CUDA_SUCCESS:
            return False
        err, count = _driver.cuDeviceGetCount()
        return err == _driver.CUresult.CUDA_SUCCESS and count > 0
    except Exception:
        return False
