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

import glob
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
from collections.abc import Mapping, Sequence
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
    return (candidate / "cuda/std").is_dir()


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
    when NVRTC is not importable or the mapping cannot be read. Answered
    from the persisted probe (:func:`_nvrtc_probe`) on a warm process."""
    try:
        nvrtc_version()                 # forces the library to load
        probe = _nvrtc_probe()
    except Exception:
        return None
    return Path(probe["root"]) if probe.get("root") else None


#: Per-process memo of :func:`_nvrtc_probe`.
_NVRTC_PROBE_MEMO: list = []


def _library_fingerprint() -> str:
    """A key for "which NVRTC would this process load", WITHOUT loading it (a load
    imports ``cuda.bindings`` and runs the pathfinder's ``ldconfig``): the
    path, size and mtime of every ``libnvrtc.so*`` the usual places hold (the
    NVRTC wheel, the conda/CUDA roots, ``LD_LIBRARY_PATH``), the bindings' own
    extension module, and the environment variables that steer the search. A new
    or replaced library, or a different search environment, is a new key."""
    dirs: list[str] = []
    try:
        spec = importlib.util.find_spec("nvidia.cuda_nvrtc")
    except (ImportError, ValueError):
        spec = None
    for loc in (spec.submodule_search_locations or ()) if spec else ():
        dirs += [os.path.join(loc, "lib"), os.path.join(loc, "bin")]
    for var in ("CUDA_PATH", "CUDA_HOME", "CONDA_PREFIX"):
        root = os.environ.get(var)
        if root:
            dirs += [os.path.join(root, d) for d in ("lib", "lib64")]
            dirs += glob.glob(os.path.join(root, "targets", "*", "lib"))
    dirs += [d for d in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if d]
    dirs += ["/usr/local/cuda/lib64", "/usr/lib64", "/usr/lib/x86_64-linux-gnu"]
    files: list[str] = []
    for d in dict.fromkeys(dirs):
        files += glob.glob(os.path.join(d, "libnvrtc.so*")) + glob.glob(os.path.join(d, "nvrtc64*.dll"))
    try:
        binding = importlib.util.find_spec("cuda.bindings.nvrtc")
        if binding and binding.origin:
            files.append(binding.origin)
    except (ImportError, ValueError):
        pass
    h = hashlib.sha256()
    for f in sorted(set(files)):
        try:
            st = os.stat(f)
            h.update(f"{f}\0{st.st_size}\0{st.st_mtime_ns}\0".encode())
        except OSError:
            pass
    for var in ("CUDA_PATH", "CUDA_HOME", "CONDA_PREFIX", "LD_LIBRARY_PATH"):
        h.update(f"{var}={os.environ.get(var, '')}\0".encode())
    return h.hexdigest()[:24]


def _nvrtc_probe() -> dict:
    """``{version, archs, root}`` of the NVRTC the import machinery loads. Asked
    of the library once per library fingerprint and kept in the cache directory:
    a fresh process answers from the file, so a cache HIT never imports
    ``cuda.bindings``. A miss in the file loads NVRTC (raising
    :class:`NvrtcUnavailable` when it cannot be)."""
    if _NVRTC_PROBE_MEMO:
        return _NVRTC_PROBE_MEMO[0]
    from .cache import default_cache_dir
    path = default_cache_dir() / "toolchain" / f"nvrtc-{_library_fingerprint()}.json"
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        record = None
    if record is None:
        _nvrtc = _import_nvrtc()
        _, major, minor = _nvrtc.nvrtcVersion()
        err, archs = _nvrtc.nvrtcGetSupportedArchs()
        ok = err == _nvrtc.nvrtcResult.NVRTC_SUCCESS
        lib = _loaded_library_path("libnvrtc.so")
        record = {"version": [major, minor], "archs": [int(a) for a in archs] if ok else None,
                  "root": str(Path(lib).parent.parent) if lib else None}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(json.dumps(record))
            os.replace(tmp, path)
        except OSError:
            pass
    _NVRTC_PROBE_MEMO.append(record)
    return record


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


#: Per-process memo of :func:`cccl_digest`, keyed on the include root.
_CCCL_DIGEST_MEMO: dict = {}


def cccl_digest(cccl_include: str | None = None) -> str:
    """A cheap identity for the served CCCL headers: sha256 over the
    `cuda/std/*` file names, sizes and mtimes. Content is not read: these
    files change only when the wheel/toolkit version changes, and
    name+size+mtime already tells one install apart from another. One
    ``scandir`` walk per process and root (it was a ``pathlib`` walk per
    compile, ~0.1 s for the ~1700 files)."""
    base = cccl_include or cccl_include_dir()
    known = _CCCL_DIGEST_MEMO.get(base)
    if known is not None:
        return known
    root = os.path.join(base, "cuda/std")
    rows: list[str] = []
    stack = [root]
    while stack:
        top = stack.pop()
        with os.scandir(top) as it:
            for e in it:
                if e.is_dir(follow_symlinks=True):
                    stack.append(e.path)
                elif e.is_file(follow_symlinks=True):
                    st = e.stat()
                    rows.append(f"{os.path.relpath(e.path, root)}\0{st.st_size}\0{st.st_mtime_ns}\0")
    h = hashlib.sha256()
    for r in sorted(rows):
        h.update(r.encode())
    return _CCCL_DIGEST_MEMO.setdefault(base, h.hexdigest())


def nvrtc_version() -> tuple[int, int]:
    """`(major, minor)` of the loaded NVRTC — a compiler subprocess never
    runs for this (the `compiler_identity` timeout/RLIMIT_AS machinery is
    for a SUBPROCESS; NVRTC is an in-process library call)."""
    major, minor = _nvrtc_probe()["version"]
    return major, minor


_DRIVER_VERSION_MEMO: list = []


def driver_cuda_version() -> tuple[int, int]:
    """`(major, minor)` of the CUDA version the loaded GPU driver
    supports — the ceiling on the PTX ISA the driver's JIT will accept,
    used by the `target="ptx"` guard in :func:`device`. Distinct from
    :func:`nvrtc_version`: NVRTC's own version can outrun an older
    driver on the same box.

    Asked of the driver library through :mod:`ctypes` first (no
    ``cuda.bindings`` import, ~40 ms), memoised for the process; the
    bindings answer only when the library cannot be loaded."""
    if _DRIVER_VERSION_MEMO:
        return _DRIVER_VERSION_MEMO[0]
    import ctypes
    for name in (("nvcuda.dll",) if sys.platform == "win32" else ("libcuda.so.1", "libcuda.so")):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        encoded = ctypes.c_int()
        if lib.cuInit(0) == 0 and lib.cuDriverGetVersion(ctypes.byref(encoded)) == 0:
            _DRIVER_VERSION_MEMO.append((encoded.value // 1000, (encoded.value % 1000) // 10))
            return _DRIVER_VERSION_MEMO[0]
        break
    try:
        _driver = _import_driver()
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


def _import_driver():
    """`cuda.bindings.driver`. An ImportError propagates (the caller may probe
    the driver by ctypes instead); a binding that imports but cannot load the
    driver library -- no NVIDIA driver on this machine -- is a HawkError, the
    same answer as no reachable device."""
    try:
        from cuda.bindings import driver as _driver
    except ImportError:
        raise
    except Exception as exc:
        raise HawkError(
            f"the CUDA driver library could not be loaded ({exc}); no CUDA device reachable.") from exc
    return _driver


def _current_device(device_index: int | None = None):
    """``(driver module, CUdevice)``: the CURRENT context's device
    (``cuCtxGetDevice``) when ``device_index`` is ``None`` and a context is
    current, else ordinal ``device_index`` (``0`` by default)."""
    _driver = _import_driver()
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
    explicit `device_index` reads that ordinal instead). Read through the
    driver library by :mod:`ctypes` first while the bindings' driver module is not yet
    loaded (no ``cuda.bindings`` import)."""
    if device_index is None and "cuda.bindings.driver" not in sys.modules:
        from .toolchain import _driver_device_arch
        probed = _driver_device_arch()
        if probed:
            return probed
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
    archs = _nvrtc_probe()["archs"]
    if archs is None:
        raise HawkError("nvrtcGetSupportedArchs failed.")
    return tuple(archs)


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
            virtual_arch: bool = True, extra: Sequence[str] = ()) -> list[str]:
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
    otherwise by accepting or faking one.

    `extra` is appended as given: the build's own ``-D<name>[=<value>]``
    defines and extra flags, so an NVRTC build sees the same macros an nvcc
    build of the same request does."""
    device_arch = (arch.replace("sm_", "compute_")
                   if virtual_arch and arch.startswith("sm_") else arch)
    # NVRTC has no __builtin_inf (the sample-sum identity spells +inf with it): a
    # function-like macro supplies it, so the emitted TEXT is the same under both compilers
    return ["-std=c++20", f"-arch={device_arch}", "-default-device",
            "-D__builtin_inf()=__longlong_as_double(0x7ff0000000000000LL)",
            f"-I{cccl_include or cccl_include_dir()}", *extra]


class NvrtcResult:
    """One NVRTC compile's outcome: the PTX bytes, or `None` on failure with
    `log` naming why."""

    __slots__ = ("ok", "ptx", "log")

    def __init__(self, ok: bool, ptx: bytes | None, log: str):
        self.ok = ok
        self.ptx = ptx
        self.log = log


def compile_ptx(source: str, name: str, arch: str, *,
                 headers: Mapping[str, bytes] | None = None,
                extra: Sequence[str] = ()) -> NvrtcResult:
    """Compile `source` (a HAWK-emitted device TU) through NVRTC, serving
    every header in `headers` (default: the sealed payload's own) from
    memory — no header is read from disk. Returns the PTX bytes on
    success; the diagnostic log always comes back, empty on a clean
    compile."""
    return _compile_image(source, name, arch, target="ptx", headers=headers,
                          extra=extra)


class DeviceImage(NamedTuple):
    """One public :func:`device` compile's result: the compiled bytes, which
    `target` produced them, and the `sm_XX` they were compiled for."""

    image: bytes
    target: str
    arch: str


def _compile_image(source: str, name: str, arch: str, *, target: str,
                    headers: Mapping[str, bytes] | None = None,
                    extra: Sequence[str] = ()) -> NvrtcResult:
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
        opts = options(arch, virtual_arch=(target == "ptx"), extra=extra)
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


_EIATTR_FRAME_SIZE, _EIATTR_REGCOUNT = 0x11, 0x2F


def cubin_resources(image: bytes) -> dict:
    """``{entry: {regs, stack_B, spill_store_B, spill_load_B}}`` read off a CUBIN itself: the
    EIATTR_REGCOUNT and EIATTR_FRAME_SIZE records of its global ``.nv.info`` section, each keyed by
    the entry's symbol index. The figures match nvcc's ``-Xptxas -v`` and need no tool and no
    device. The spill figures are ``None``: the image does not say how much of the frame is
    spills. (NVRTC's own ``-Xptxas -v`` log is not used: the driver's compute cache answers a
    repeat compile without printing it.)"""
    import struct
    if image[:4] != b"\x7fELF":
        return {}
    (shoff,) = struct.unpack_from("<Q", image, 0x28)
    shentsize, shnum, _ = struct.unpack_from("<HHH", image, 0x3A)
    sec = {}
    for i in range(shnum):
        name, kind, _, _, off, size, link, _ = struct.unpack_from("<IIQQQQII", image, shoff + i * shentsize)
        sec[i] = (name, kind, off, size, link)
    strtab_off = lambda link: sec[link][2]                       # noqa: E731
    shstr = struct.unpack_from("<H", image, 0x3E)[0]
    named = {image[sec[shstr][2] + n:image.index(b"\0", sec[shstr][2] + n)].decode(): (off, size)
             for n, _, off, size, _ in sec.values()}
    symtab = next(v for v in sec.values() if v[1] == 2)           # SHT_SYMTAB
    names = strtab_off(symtab[4])
    regs: dict = {}
    frames: dict = {}
    info_off, info_size = named.get(".nv.info", (0, 0))
    at = info_off
    while at + 4 <= info_off + info_size:
        fmt, attr, n = struct.unpack_from("<BBH", image, at)
        if fmt == 4 and n == 8 and attr in (_EIATTR_REGCOUNT, _EIATTR_FRAME_SIZE):
            index, value = struct.unpack_from("<II", image, at + 4)
            (regs if attr == _EIATTR_REGCOUNT else frames)[index] = value
        at += 4 + (n if fmt == 4 else 0)
    out = {}
    for index, count in regs.items():
        (st_name,) = struct.unpack_from("<I", image, symtab[2] + index * 24)
        entry = image[names + st_name:image.index(b"\0", names + st_name)].decode()
        out[entry] = {"regs": count, "stack_B": frames.get(index, 0),
                      "spill_store_B": None, "spill_load_B": None}
    return out


def compile_cubin(source: str, name: str, arch: str, *,
                   headers: Mapping[str, bytes] | None = None,
                  extra: Sequence[str] = ()) -> NvrtcResult:
    """:func:`compile_ptx`'s sibling: compile `source` through NVRTC
    straight to a CUBIN (SASS) for the concrete `arch`, for a caller
    (:mod:`hawk.compile.drivers`) that wants the same in-memory compile
    :func:`device`/:func:`cubin` offer, without their `DeviceImage`/
    arch-resolution wrapper."""
    return _compile_image(source, name, arch, target="cubin", headers=headers,
                          extra=extra)


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
