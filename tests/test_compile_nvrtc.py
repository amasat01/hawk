# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The PUBLIC device compile (``hawk.compile.device``/``cubin``) and the
pip-only path it exists for: a machine with a GPU driver, NVRTC from pip,
a host ``g++``, and NO ``nvcc``, NO aether/eagle checkout, NO headers under
any prefix.

``device``/``cubin`` are new; the in-memory PTX driver they share
(``hawk.compile.nvrtc.compile_ptx``, unchanged) is covered by
``test_random_op.py`` and friends through the ordinary emitted-kernel path.
This file is about the PUBLIC entry point itself: CUBIN vs. PTX, the
``arch=None`` default, the PTX/driver version guard, ``NvrtcUnavailable``,
and the pip-only probe (roots unresolvable, device + host compile still
succeed) that motivated all of it.
"""

from __future__ import annotations

import ctypes
import pathlib
import sys

import numpy as np
import pytest

from hawk.compile import (
    CompileOptions,
    DeviceImage,
    compile_source,
    cubin,
    cubin_available,
    device,
)
from hawk.compile import nvrtc as hnvrtc
from hawk.compile.nvrtc import NvrtcUnavailable
from _toolchain import real_arch
from hawk.ir import HawkError

#: A minimal `__global__` TU: no aether/eagle `#include` at all, so these
#: rows exercise `device`/`cubin`'s own plumbing (header serving, arch
#: resolution, the PTX guard) without depending on the emitted-kernel shape
#: `test_random_op.py` and friends already cover.
KERNEL_SRC = 'extern "C" __global__ void hawk_probe_kernel(int* out) { *out = 42; }\n'
KERNEL_NAME = "hawk_probe_kernel"


def _skip_without_nvrtc_and_gpu():
    if not cubin_available():
        pytest.skip("no NVRTC binding or no CUDA device on this box")


@pytest.mark.gpu
def test_cubin_compiles_loads_and_launches_on_the_device():
    """`cubin()` produces SASS `cuModuleLoadData` accepts outright -- no
    further JIT -- and the kernel it launches actually runs."""
    _skip_without_nvrtc_and_gpu()
    from cuda.bindings import driver as drv

    image = cubin(KERNEL_SRC)
    assert isinstance(image, DeviceImage)
    assert image.target == "cubin"
    assert image.arch.startswith("sm_")
    assert len(image.image) > 0

    (err,) = drv.cuInit(0)
    assert err == drv.CUresult.CUDA_SUCCESS
    err, dev = drv.cuDeviceGet(0)
    assert err == drv.CUresult.CUDA_SUCCESS
    err, ctx = drv.cuDevicePrimaryCtxRetain(dev)
    assert err == drv.CUresult.CUDA_SUCCESS
    try:
        (err,) = drv.cuCtxSetCurrent(ctx)
        assert err == drv.CUresult.CUDA_SUCCESS
        err, module = drv.cuModuleLoadData(image.image)
        assert err == drv.CUresult.CUDA_SUCCESS, (
            f"cuModuleLoadData refused a CUBIN hawk.compile.cubin() itself "
            f"produced ({err})")
        try:
            err, func = drv.cuModuleGetFunction(module, KERNEL_NAME.encode())
            assert err == drv.CUresult.CUDA_SUCCESS
            err, dptr = drv.cuMemAlloc(4)
            assert err == drv.CUresult.CUDA_SUCCESS
            try:
                kernel_args = ((int(dptr),), (ctypes.c_void_p,))
                (err,) = drv.cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, 0,
                                            kernel_args, 0)
                assert err == drv.CUresult.CUDA_SUCCESS
                (err,) = drv.cuCtxSynchronize()
                assert err == drv.CUresult.CUDA_SUCCESS
                host = np.zeros(1, dtype=np.int32)
                (err,) = drv.cuMemcpyDtoH(host.ctypes.data, dptr, 4)
                assert err == drv.CUresult.CUDA_SUCCESS
                assert int(host[0]) == 42, (
                    "the launched kernel did not write the value it was "
                    f"compiled to write: got {int(host[0])}")
            finally:
                drv.cuMemFree(dptr)
        finally:
            drv.cuModuleUnload(module)
    finally:
        drv.cuDevicePrimaryCtxRelease(dev)


@pytest.mark.gpu
def test_device_arch_none_resolves_to_the_current_device():
    _skip_without_nvrtc_and_gpu()
    image = cubin(KERNEL_SRC, arch=None)
    assert image.arch == hnvrtc.current_device_arch()


@pytest.mark.gpu
def test_device_ptx_target_either_compiles_or_hits_the_documented_guard():
    """The genuine, unmocked outcome on THIS box: `target="ptx"` succeeds
    when NVRTC is no newer than the driver, else it raises the SAME guard
    `test_the_ptx_guard_fires_when_nvrtc_outruns_the_driver` pins with
    monkeypatched versions -- this row is the real-versions leg of that
    same law, whichever way it falls on the box running it."""
    _skip_without_nvrtc_and_gpu()
    if hnvrtc.nvrtc_version() > hnvrtc.driver_cuda_version():
        with pytest.raises(HawkError, match="newer than this driver"):
            device(KERNEL_SRC, target="ptx")
        return
    image = device(KERNEL_SRC, target="ptx")
    assert image.target == "ptx"
    assert b"hawk_probe_kernel" in image.image


def test_the_ptx_guard_fires_when_nvrtc_outruns_the_driver(monkeypatch):
    """Deterministic, independent of THIS box's real NVRTC/driver versions
    (unlike the row above): a newer NVRTC than the driver must refuse
    `target="ptx"` outright, before any compile runs. `arch` is passed
    explicitly so this needs no real device at all."""
    monkeypatch.setattr(hnvrtc, "nvrtc_version", lambda: (99, 9))
    monkeypatch.setattr(hnvrtc, "driver_cuda_version", lambda: (1, 0))
    with pytest.raises(HawkError, match="newer than this driver"):
        device(KERNEL_SRC, target="ptx", arch=real_arch("nvrtc"))


def test_the_ptx_guard_does_not_fire_when_the_driver_is_current(monkeypatch):
    """The other direction of the same law: a driver at least as new as
    NVRTC must not be refused."""
    _skip_without_nvrtc_and_gpu()
    monkeypatch.setattr(hnvrtc, "nvrtc_version", lambda: (1, 0))
    monkeypatch.setattr(hnvrtc, "driver_cuda_version", lambda: (99, 9))
    image = device(KERNEL_SRC, target="ptx", arch=real_arch("nvrtc"))
    assert image.target == "ptx"


def test_device_rejects_an_unknown_target():
    with pytest.raises(HawkError, match="target"):
        device(KERNEL_SRC, target="fatbin", arch=real_arch("nvrtc"))


# --------------------------------------------------------------------------- #
# -- #31: the DEFAULT device artifact loads on the driver present. Pure
# -- selection logic first (faked versions, no compile/device at all), then
# -- device()'s own default resolution wired to it.
# --------------------------------------------------------------------------- #
def test_select_target_prefers_cubin_when_the_driver_is_old():
    assert hnvrtc.select_target((12, 9), (12, 6)) == "cubin"


def test_select_target_prefers_ptx_when_the_driver_is_new_enough():
    assert hnvrtc.select_target((12, 6), (12, 9)) == "ptx"
    assert hnvrtc.select_target((12, 6), (12, 6)) == "ptx", (
        "a driver exactly as new as the compiler is 'new enough' -- it can "
        "already JIT the newest ISA the compiler emits")


def test_select_target_answers_cubin_for_either_version_unknown():
    assert hnvrtc.select_target(None, (12, 6)) == "cubin"
    assert hnvrtc.select_target((12, 9), None) == "cubin"
    assert hnvrtc.select_target(None, None) == "cubin"


def test_device_default_target_follows_select_target(monkeypatch):
    """`device()` with no `target=` resolves it from the SAME law, not a
    static default -- the gap this closes was a HARDCODED 'cubin' that never
    looked at the driver at all, which happened to be safe but never checked
    the OTHER direction (a driver new enough deserves PTX's forward
    compatibility, not a blanket CUBIN)."""
    _skip_without_nvrtc_and_gpu()
    monkeypatch.setattr(hnvrtc, "nvrtc_version", lambda: (99, 9))
    monkeypatch.setattr(hnvrtc, "driver_cuda_version", lambda: (1, 0))
    image = device(KERNEL_SRC, arch=real_arch("nvrtc"))
    assert image.target == "cubin"

    monkeypatch.setattr(hnvrtc, "nvrtc_version", lambda: (1, 0))
    monkeypatch.setattr(hnvrtc, "driver_cuda_version", lambda: (99, 9))
    image = device(KERNEL_SRC, arch=real_arch("nvrtc"))
    assert image.target == "ptx"


class _BlockCudaBindingsNvrtc:
    """A `sys.meta_path` finder that makes `cuda.bindings.nvrtc` behave as
    though it were never installed -- ``ModuleNotFoundError``, never a
    `sys.modules[name] = None` shortcut (that shortcut makes `import`
    silently hand back `None` instead of raising, which is not what an
    actually-missing package does)."""

    def find_spec(self, name, path=None, target=None):
        if name == "cuda.bindings.nvrtc":
            raise ModuleNotFoundError(f"simulated: no module named {name!r}")
        return None


def test_nvrtc_unavailable_when_the_binding_cannot_be_imported(monkeypatch):
    for name in list(sys.modules):
        if name == "cuda.bindings.nvrtc" or name.startswith("cuda.bindings.nvrtc."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    # An already-imported submodule stays reachable as an ATTRIBUTE of its
    # parent package even after it is dropped from `sys.modules` -- `from
    # cuda.bindings import nvrtc` resolves that attribute directly and never
    # asks the meta path finder at all unless this is cleared too.
    parent = sys.modules.get("cuda.bindings")
    if parent is not None and hasattr(parent, "nvrtc"):
        monkeypatch.delattr(parent, "nvrtc", raising=False)

    finder = _BlockCudaBindingsNvrtc()
    sys.meta_path.insert(0, finder)
    try:
        with pytest.raises(NvrtcUnavailable, match="nvidia-cuda-nvrtc-cu12"):
            hnvrtc._import_nvrtc()
        assert cubin_available() is False, (
            "cubin_available() must never raise, and must answer False when "
            "NVRTC cannot be imported at all")
        with pytest.raises(NvrtcUnavailable):
            device(KERNEL_SRC, arch=real_arch("nvrtc"))
    finally:
        sys.meta_path.remove(finder)


def test_nvrtc_unavailable_is_a_hawk_error():
    assert issubclass(NvrtcUnavailable, HawkError)


def test_pip_only_roots_unresolvable_device_and_host_compile_still_succeed(
        monkeypatch, tmp_path):
    """The pip-only probe, pinned as a row: with NEITHER header root
    resolvable and no `nvcc` on the toolchain's chosen path, a HOST compile
    (against the sealed payload) and a DEVICE compile (through NVRTC) both
    still succeed -- both bypass the two real include roots entirely
    already (the host one through `aether_dsc.Payload.serve()`, the device
    one through NVRTC's in-memory header array), so an unresolvable root is
    not, by itself, a reason either one should fail."""
    import hawk._roots as _roots
    import hawk.compile.toolchain as tc

    for var in ("HAWK_AETHER_INCLUDE", "HAWK_EAGLE_INCLUDE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(_roots, "_prefixes", lambda: [])
    monkeypatch.setattr(_roots, "WORKSPACE", pathlib.Path("/nonexistent/hawk-probe"))
    monkeypatch.setenv("HAWK_DEVICE_COMPILER", "nvrtc")

    with pytest.raises(HawkError):
        tc.aether_include()
    with pytest.raises(HawkError):
        tc.eagle_include()

    # `include_flags()` is the one place a REAL nvcc/g++ compile still needs
    # an `-I` root even when neither resolves for real -- not exercised by
    # either compile below (the host one bypasses roots via the sealed
    # payload, the device one via NVRTC's in-memory headers), so this is its
    # only coverage.
    assert tc.pip_only_root() is None, (
        "nothing has asked for the pip-only fallback yet")
    flags = tc.include_flags()
    assert flags[0] == "-I" and flags[2] == "-I"
    served = tc.pip_only_root()
    assert served is not None and flags[1] == served and flags[3] == served, (
        "include_flags() must fall back to ONE served directory, used as "
        "BOTH the aether and the eagle root")
    assert (pathlib.Path(served) / "aether").is_dir()
    assert (pathlib.Path(served) / "plugin").is_dir()
    # Entered once: a second call must not re-serve the payload.
    assert tc.include_flags()[1] == served

    host_src = 'extern "C" int hawk_pip_only_probe(void) { return 42; }\n'
    host_result = compile_source(
        host_src, "pip_only_probe",
        CompileOptions(backend="host", cache_dir=str(tmp_path / "host_cache")))
    assert host_result.artifact.is_file()

    if not cubin_available():
        pytest.skip("no NVRTC binding or no CUDA device on this box for the "
                    "device half of this probe")
    device_result = compile_source(
        KERNEL_SRC, "pip_only_probe_device",
        CompileOptions(backend="cuda", cache_dir=str(tmp_path / "device_cache")))
    assert device_result.artifact.is_file()


def _hide_cccl_wheel(monkeypatch):
    """Hide every CCCL source but the toolkit roots under test: the CCCL
    wheels and the loaded NVRTC's own root."""
    real = hnvrtc.importlib.util.find_spec
    monkeypatch.setattr(hnvrtc.importlib.util, "find_spec",
                        lambda name, *a, **k: None if name in hnvrtc._CCCL_WHEEL_MODULES
                        else real(name, *a, **k))
    monkeypatch.setattr(hnvrtc, "loaded_nvrtc_root", lambda: None)
    monkeypatch.delenv("CUDA_HOME", raising=False)


def _toolkit(root: pathlib.Path, include: str) -> pathlib.Path:
    d = root / include
    (d / "cuda" / "std").mkdir(parents=True)
    return d


def test_cccl_found_in_a_conda_toolkit_targets_layout(monkeypatch, tmp_path):
    """A conda CUDA toolkit has no flat include/cuda/std; its CCCL lives
    under targets/<arch>-linux/include."""
    _hide_cccl_wheel(monkeypatch)
    want = _toolkit(tmp_path, f"targets/{hnvrtc.platform.machine()}-linux/include")
    monkeypatch.setenv("CUDA_PATH", str(tmp_path))
    assert hnvrtc.cccl_include_dir() == str(want)


def test_cccl_found_through_conda_prefix_when_cuda_path_is_unset(monkeypatch, tmp_path):
    _hide_cccl_wheel(monkeypatch)
    want = _toolkit(tmp_path, f"targets/{hnvrtc.platform.machine()}-linux/include")
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path))
    assert hnvrtc.cccl_include_dir() == str(want)


def test_cccl_prefers_a_flat_toolkit_include(monkeypatch, tmp_path):
    _hide_cccl_wheel(monkeypatch)
    want = _toolkit(tmp_path, "include")
    _toolkit(tmp_path, f"targets/{hnvrtc.platform.machine()}-linux/include")
    monkeypatch.setenv("CUDA_PATH", str(tmp_path))
    assert hnvrtc.cccl_include_dir() == str(want)
