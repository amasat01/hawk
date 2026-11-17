# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The AOT ``nvcc`` device compile (:mod:`hawk.compile.drivers`) picks the
SAME target (#31, ``hawk.compile.nvrtc.select_target``) the pip-only NVRTC
path does — CUBIN for the detected arch when the compiler outruns the
installed driver, PTX when the driver is new enough — so the developer/CI
tree is not left with the vacuous ``-ptx``-always recipe that produced
``CUDA_ERROR_UNSUPPORTED_PTX_VERSION`` whenever the box's ``nvcc`` was newer
than its driver (measured live on this box: nvcc 12.9, driver 12.6).

:mod:`test_compile_nvrtc` covers :func:`hawk.compile.nvrtc.select_target`
itself and :func:`hawk.compile.nvrtc.device`'s own default; this file is the
AOT nvcc leg — parsing nvcc's OWN version, the flag swap, and the actual
compiled bytes.
"""

from __future__ import annotations

import pytest
from _toolchain import real_arch

from hawk.compile import CompileOptions, compile_source
from hawk.compile import nvrtc as hnvrtc
from ospreys.cuda import toolchain as dtc
from raptor.errors import RaptorError as HawkError

KERNEL_SRC = 'extern "C" __global__ void hawk_probe_kernel(int* out) { *out = 42; }\n'


def _skip_without_nvcc():
    try:
        dtc.device_compiler()
    except HawkError:
        pytest.skip("no nvcc on this box")
    if dtc.device_compiler_kind() != "nvcc":
        pytest.skip("$HAWK_DEVICE_COMPILER forces nvrtc on this box")


def test_nvcc_version_parses_the_release_line():
    _skip_without_nvcc()
    major, minor = dtc.nvcc_version(dtc.device_compiler())
    assert (major, minor) >= (11, 0), "sanity: a real nvcc reports a real version"


def test_nvcc_version_is_none_for_a_binary_with_no_release_line():
    assert dtc.nvcc_version("/bin/ls") is None
    assert dtc.nvcc_version("/nonexistent/hawk-probe-binary") is None


def test_compile_source_emits_cubin_when_nvcc_outruns_the_driver(monkeypatch, tmp_path):
    """The end-to-end AOT leg: force the "nvcc newer than driver" condition
    deterministically and check the ACTUAL compiled bytes are an ELF CUBIN,
    not PTX text -- a CUBIN loads on the driver outright
    (``cuModuleLoad``/``cuModuleLoadData`` auto-detect the format; the file
    extension is cosmetic, confirmed against the real driver API on this box
    during development), which is the whole fix."""
    _skip_without_nvcc()
    monkeypatch.setattr(hnvrtc, "driver_cuda_version", lambda: (1, 0))
    result = compile_source(KERNEL_SRC, "hawk_probe_kernel",
                            CompileOptions(target="cuda", arch=real_arch(),
                                           cache_dir=str(tmp_path / "cache")))
    data = result.artifact.read_bytes()
    assert data[:4] == b"\x7fELF", (
        f"expected an ELF CUBIN when the compiler outruns the driver; got "
        f"{data[:16]!r}")


def test_compile_source_emits_ptx_when_the_driver_is_current(monkeypatch, tmp_path):
    _skip_without_nvcc()
    monkeypatch.setattr(hnvrtc, "driver_cuda_version", lambda: (999, 9))
    result = compile_source(KERNEL_SRC, "hawk_probe_kernel",
                            CompileOptions(target="cuda", arch=real_arch(),
                                           cache_dir=str(tmp_path / "cache")))
    data = result.artifact.read_bytes()
    assert data[:4] != b"\x7fELF" and b"hawk_probe_kernel" in data


def test_compile_source_keys_ptx_and_cubin_into_different_slots(monkeypatch, tmp_path):
    """The cache key must include the target (#31's own closing line): the
    SAME source, same compiler, same arch, differing only in the
    driver-forced target, must not collide on one cache slot."""
    _skip_without_nvcc()
    opts = CompileOptions(target="cuda", arch=real_arch(),
                          cache_dir=str(tmp_path / "cache"))
    monkeypatch.setattr(hnvrtc, "driver_cuda_version", lambda: (1, 0))
    cubin_result = compile_source(KERNEL_SRC, "hawk_probe_kernel", opts)
    monkeypatch.setattr(hnvrtc, "driver_cuda_version", lambda: (999, 9))
    ptx_result = compile_source(KERNEL_SRC, "hawk_probe_kernel", opts)
    assert cubin_result.key != ptx_result.key
    assert cubin_result.artifact != ptx_result.artifact


def test_compile_source_keys_differ_by_opt_level_for_the_AOT_nvcc_path(
        monkeypatch, tmp_path):
    """An O0 AOT nvcc build and an O3 one of the same source must never
    share a cache slot: ``opt_level`` rides inside the compile template's own
    ``-O<n>``/``-Xptxas -O<n>``, so this is the real compiler, not a flag
    string, proving the two never collide."""
    _skip_without_nvcc()
    monkeypatch.setattr(hnvrtc, "driver_cuda_version", lambda: (999, 9))  # PTX, same arm both times
    cache_dir = str(tmp_path / "cache")
    o3 = compile_source(KERNEL_SRC, "hawk_probe_kernel", CompileOptions(
        target="cuda", arch=real_arch(), cache_dir=cache_dir, opt_level="O3"))
    o0 = compile_source(KERNEL_SRC, "hawk_probe_kernel", CompileOptions(
        target="cuda", arch=real_arch(), cache_dir=cache_dir, opt_level="O0"))
    assert o3.key != o0.key
    assert o3.artifact != o0.artifact
    assert o3.artifact.is_file() and o0.artifact.is_file()


