# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Portability of the compile layer beyond the box it was developed on.

* CCCL lookup: the CUDA 13 layouts (``include/cccl``,
  ``targets/*/include/cccl``, the cu13 wheel), and the loaded NVRTC's own
  root preferred over ``$CUDA_PATH``.
* The arch policy (:func:`hawk.compile.toolchain.resolve_device_arch`): no
  GPU -> PTX at the toolkit minimum; above the toolkit -> PTX at its
  maximum; below it -> a refusal naming device, arch and toolkit.
* The device probe reads the CURRENT context's device.
* Cache writes are atomic (temporary in the slot + ``os.replace``).
* The nvcc host compiler is left to the caller unless ``$HAWK_NVCC_CCBIN``
  says otherwise, and is part of the device key; the host key carries libc.
"""

from __future__ import annotations

import importlib.machinery
import os
import platform
import types

import pytest

from hawk.compile import CompileOptions, compile_source
from hawk.compile import cache as hcache
from hawk.compile import drivers as hdrivers
from hawk.compile import nvrtc as hnvrtc
from hawk.compile import toolchain as tc
from hawk.ir import HawkError

KERNEL_SRC = 'extern "C" __global__ void hawk_probe_kernel(int* out) { *out = 42; }\n'
HOST_SRC = 'extern "C" int hawk_probe_host() { return 42; }\n'


def _cccl(root, include: str):
    d = root / include
    (d / "cuda" / "std").mkdir(parents=True)
    return d


def _isolate_cccl(monkeypatch, loaded=None):
    """Only the roots a row names are visible: no CCCL wheel, the loaded
    NVRTC root stubbed, the environment's toolkit roots cleared, and the
    PATH nvcc/``/usr/local/cuda`` fallbacks dropped."""
    real = hnvrtc.importlib.util.find_spec
    monkeypatch.setattr(hnvrtc.importlib.util, "find_spec",
                        lambda name, *a, **k: None if name in hnvrtc._CCCL_WHEEL_MODULES
                        else real(name, *a, **k))
    monkeypatch.setattr(hnvrtc, "loaded_nvrtc_root", lambda: loaded)
    for var in ("CUDA_PATH", "CUDA_HOME", "CONDA_PREFIX"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(hnvrtc.shutil, "which", lambda name: None)
    real_root_dirs = hnvrtc._root_include_dirs
    monkeypatch.setattr(hnvrtc, "_root_include_dirs",
                        lambda root: [] if str(root) == "/usr/local/cuda"
                        else real_root_dirs(root))


# --------------------------------------------------------------------------- #
# CCCL lookup.
# --------------------------------------------------------------------------- #
def test_cccl_found_in_a_cuda13_targets_layout(monkeypatch, tmp_path):
    _isolate_cccl(monkeypatch)
    want = _cccl(tmp_path, f"targets/{platform.machine()}-linux/include/cccl")
    monkeypatch.setenv("CUDA_PATH", str(tmp_path))
    assert hnvrtc.cccl_include_dir() == str(want)


def test_cccl_found_in_a_cuda13_flat_include_cccl(monkeypatch, tmp_path):
    _isolate_cccl(monkeypatch)
    want = _cccl(tmp_path, "include/cccl")
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    assert hnvrtc.cccl_include_dir() == str(want)


def test_cccl_prefers_the_loaded_nvrtc_root_over_cuda_path(monkeypatch, tmp_path):
    """The headers must match the compiler that reads them: the root of the
    NVRTC library actually loaded wins over ``$CUDA_PATH``."""
    other = tmp_path / "cuda-path"
    _cccl(other, "include")
    loaded = tmp_path / "loaded" / "targets" / f"{platform.machine()}-linux"
    want = _cccl(loaded, "include/cccl")
    _isolate_cccl(monkeypatch, loaded=loaded)
    monkeypatch.setenv("CUDA_PATH", str(other))
    assert hnvrtc.cccl_include_dir() == str(want)


def test_cccl_found_in_the_cu13_wheel_layout(monkeypatch, tmp_path):
    _isolate_cccl(monkeypatch)
    pkg = tmp_path / "nvidia" / "cu13"
    want = _cccl(pkg, "include/cccl")
    real = hnvrtc.importlib.util.find_spec

    def _find(name, *a, **k):
        if name == "nvidia.cu13":
            spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
            spec.submodule_search_locations = [str(pkg)]
            return spec
        if name in hnvrtc._CCCL_WHEEL_MODULES:
            return None
        return real(name, *a, **k)

    monkeypatch.setattr(hnvrtc.importlib.util, "find_spec", _find)
    assert hnvrtc.cccl_include_dir() == str(want)


def test_cccl_missing_names_both_wheel_generations(monkeypatch):
    _isolate_cccl(monkeypatch)
    monkeypatch.setattr(hnvrtc, "_toolkit_include_dirs", lambda: [])
    with pytest.raises(HawkError) as err:
        hnvrtc.cccl_include_dir()
    assert "nvidia-cuda-cccl-cu12" in str(err.value)
    assert "nvidia-cuda-cccl 13" in str(err.value)
    assert "-cu13" not in str(err.value)  # the -cu13 names on PyPI are empty placeholders


def test_loaded_nvrtc_root_is_the_mapped_library_root():
    """The real, unmocked root: the directory above the ``libnvrtc.so``
    mapped into this process, and the CCCL lookup answers beneath it when
    that root carries CCCL."""
    root = hnvrtc.loaded_nvrtc_root()
    if root is None:
        pytest.skip("NVRTC is not loadable in this process")
    lib = hnvrtc._loaded_library_path("libnvrtc.so")
    assert lib is not None and os.path.dirname(os.path.dirname(lib)) == str(root)
    if any(hnvrtc._has_cccl(d) for d in hnvrtc._root_include_dirs(root)):
        assert hnvrtc.cccl_include_dir().startswith(str(root))


# --------------------------------------------------------------------------- #
# The arch policy, with a stubbed probe and toolkit.
# --------------------------------------------------------------------------- #
@pytest.fixture
def stub_device(monkeypatch):
    """Set the probed device's arch (``None``: no GPU) and the toolkit's
    supported archs, with nothing real asked."""
    tc.reset_device_arch_memo()
    monkeypatch.delenv("HAWK_CUDA_ARCH", raising=False)

    def _set(probed, archs=(50, 90)):
        tc.reset_device_arch_memo()

        def _probe():
            if probed is None:
                raise HawkError("no CUDA device reachable (stub)")
            return probed

        monkeypatch.setattr(hnvrtc, "current_device_arch", _probe)
        monkeypatch.setattr(hnvrtc, "current_device_name", lambda: "Stub GPU")
        monkeypatch.setattr(tc, "toolkit_archs",
                            lambda kind=None: None if archs is None
                            else ("stub toolkit 1.0", tuple(archs)))

    yield _set
    tc.reset_device_arch_memo()


@pytest.mark.parametrize("probed, archs, want", [
    (None, (50, 61, 90), "compute_61"),       # no GPU: PTX at the minimum >= sm_60
    (None, (75, 80, 90), "compute_75"),       # no GPU, CUDA 13: the toolkit minimum
    ("sm_120", (50, 61, 90), "compute_90"),   # above the toolkit: PTX at max
    ("sm_61", (50, 61, 90), "sm_61"),         # supported: the device itself
    ("sm_90", (50, 61, 90), "sm_90"),         # the maximum itself: unchanged
    (None, None, tc.FALLBACK_ARCH),           # nothing readable: virtual
])
def test_arch_policy_table(stub_device, probed, archs, want):
    stub_device(probed, archs)
    assert tc.resolve_arch("") == want
    assert want.startswith("compute_") == (hnvrtc.forced_target(want) == "ptx")


def test_arch_policy_refuses_a_device_below_the_toolkit(stub_device):
    stub_device("sm_35", (50, 90))
    with pytest.raises(HawkError) as err:
        tc.resolve_arch("")
    message = str(err.value)
    assert "Stub GPU" in message and "sm_35" in message
    assert "stub toolkit 1.0" in message


def test_arch_policy_checks_explicit_pins_the_same_way(stub_device, monkeypatch):
    stub_device("sm_61", (50, 90))
    assert tc.resolve_arch("sm_100") == "compute_90"
    with pytest.raises(HawkError, match="device_arch='sm_35'"):
        tc.resolve_arch("sm_35")
    monkeypatch.setenv("HAWK_CUDA_ARCH", "sm_30")
    with pytest.raises(HawkError, match=r"\$HAWK_CUDA_ARCH='sm_30'"):
        tc.resolve_arch("")
    monkeypatch.setenv("HAWK_CUDA_ARCH", "sm_75")
    assert tc.resolve_arch("") == "sm_75"


def test_no_gpu_device_compile_is_ptx_at_the_nvrtc_minimum(monkeypatch):
    """The public NVRTC door with no GPU reachable: PTX at NVRTC's own
    minimum compute arch, never a CUBIN for a guessed device."""
    try:
        archs = hnvrtc.supported_archs()
        hnvrtc.cccl_include_dir()
    except HawkError:
        pytest.skip("NVRTC or CCCL not available in this process")
    monkeypatch.delenv("HAWK_CUDA_ARCH", raising=False)
    tc.reset_device_arch_memo()

    def _no_device():
        raise HawkError("no CUDA device reachable (stub)")

    monkeypatch.setattr(hnvrtc, "current_device_arch", _no_device)
    monkeypatch.setattr(hnvrtc, "driver_cuda_version", _no_device)
    try:
        image = hnvrtc.device(KERNEL_SRC)
    finally:
        tc.reset_device_arch_memo()
    low = min(a for a in archs if a >= tc.PORTABLE_MIN_ARCH)
    assert image.target == "ptx" and image.arch == f"compute_{low}"
    assert f".target sm_{low}".encode() in image.image
    with pytest.raises(HawkError, match="virtual"):
        tc.reset_device_arch_memo()
        hnvrtc.cubin(KERNEL_SRC)
    tc.reset_device_arch_memo()


# --------------------------------------------------------------------------- #
# The device probe reads the current context's device.
# --------------------------------------------------------------------------- #
def _fake_driver(ctx_device):
    ok, no_ctx = 0, 201
    archs = {0: (6, 1), 3: (8, 6)}
    attr = types.SimpleNamespace(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR="major",
                                 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR="minor")
    return types.SimpleNamespace(
        CUresult=types.SimpleNamespace(CUDA_SUCCESS=ok),
        CUdevice_attribute=attr,
        cuInit=lambda flags: (ok,),
        cuCtxGetDevice=lambda: (ok, ctx_device) if ctx_device is not None else (no_ctx, None),
        cuDeviceGet=lambda ordinal: (ok, ordinal),
        cuDeviceGetAttribute=lambda which, dev: (
            ok, archs[dev][0] if which == "major" else archs[dev][1]),
    )


@pytest.mark.parametrize("ctx_device, want", [(3, "sm_86"), (None, "sm_61")])
def test_probe_reads_the_current_context_device(monkeypatch, ctx_device, want):
    """A process whose current context is on device 3 compiles for device
    3, not ordinal 0; with no context current it falls back to ordinal 0."""
    cuda_bindings = pytest.importorskip("cuda.bindings")
    monkeypatch.setattr(cuda_bindings, "driver", _fake_driver(ctx_device), raising=False)
    monkeypatch.setitem(__import__("sys").modules, "cuda.bindings.driver",
                        cuda_bindings.driver)
    assert hnvrtc.current_device_arch() == want


# --------------------------------------------------------------------------- #
# Atomic cache writes.
# --------------------------------------------------------------------------- #
def test_write_atomic_never_exposes_a_partial_file(monkeypatch, tmp_path):
    """At the moment of the move the reader's path still holds the OLD
    content and the temporary already holds the WHOLE new one; afterwards
    the path holds the new content and no temporary is left."""
    target = tmp_path / "validity.json"
    target.write_text("old")
    new = "n" * 100_000
    real_replace = os.replace
    seen = []

    def _spy(src, dst):
        src, dst = os.fspath(src), os.fspath(dst)
        seen.append(src)
        assert os.path.dirname(src) == os.path.dirname(dst)
        assert os.path.basename(src).startswith(f".tmp.{os.getpid()}.")
        assert open(dst).read() == "old"
        assert open(src).read() == new
        real_replace(src, dst)

    monkeypatch.setattr(hcache.os, "replace", _spy)
    hcache.write_atomic(target, new)
    assert seen and target.read_text() == new
    assert not list(tmp_path.glob(".tmp.*"))


def test_write_atomic_failure_keeps_the_old_file(monkeypatch, tmp_path):
    target = tmp_path / "artifact.ptx"
    target.write_bytes(b"old")

    def _boom(src, dst):
        raise OSError("simulated failure at the move")

    monkeypatch.setattr(hcache.os, "replace", _boom)
    with pytest.raises(OSError):
        hcache.write_atomic(target, b"new")
    assert target.read_bytes() == b"old"
    assert not list(tmp_path.glob(".tmp.*"))


def test_compile_writes_through_a_temporary_and_keys_libc(monkeypatch, tmp_path):
    """A real host compile: the compiler writes a private temporary in the
    slot, which is then moved onto the artifact path; the host key carries
    the C library."""
    moves = []
    real_publish = hdrivers.publish_atomic

    def _spy(tmp, path):
        moves.append((tmp, path))
        assert tmp.is_file() and tmp.parent == path.parent
        assert tmp.name.startswith(f".tmp.{os.getpid()}.")
        real_publish(tmp, path)

    monkeypatch.setattr(hdrivers, "publish_atomic", _spy)
    result = compile_source(HOST_SRC, "hawk_probe_host",
                            CompileOptions(backend="host",
                                           cache_dir=str(tmp_path / "cache")))
    assert not result.hit
    assert any(path == result.artifact for _, path in moves)
    assert result.artifact.is_file()
    assert not list(result.artifact.parent.glob(".tmp.*"))
    record = hcache.Cache(str(tmp_path / "cache")).record(result.key)
    libc = "-".join(p for p in platform.libc_ver() if p) or "unknown"
    assert f"libc:{libc}" in record["key_terms"]


# --------------------------------------------------------------------------- #
# The nvcc host compiler.
# --------------------------------------------------------------------------- #
def test_ccbin_default_leaves_nvcc_prepend_flags_alone(monkeypatch):
    monkeypatch.delenv("HAWK_NVCC_CCBIN", raising=False)
    monkeypatch.setenv("NVCC_PREPEND_FLAGS", "-ccbin /opt/some/g++ -Xfoo")
    assert tc.subprocess_env()["NVCC_PREPEND_FLAGS"] == "-ccbin /opt/some/g++ -Xfoo"
    monkeypatch.delenv("NVCC_PREPEND_FLAGS")
    assert "NVCC_PREPEND_FLAGS" not in tc.subprocess_env(), (
        "with nothing set, nvcc's own default (the g++ on PATH) must stand")


def test_hawk_nvcc_ccbin_replaces_only_the_ccbin(monkeypatch):
    monkeypatch.setenv("HAWK_NVCC_CCBIN", "/opt/other/g++")
    monkeypatch.setenv("NVCC_PREPEND_FLAGS", "-ccbin /opt/some/g++ -Xfoo")
    assert tc.subprocess_env()["NVCC_PREPEND_FLAGS"] == "-ccbin /opt/other/g++ -Xfoo"


def test_device_host_compiler_follows_the_environment(tmp_path):
    fake = tmp_path / "g++"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    env = {"PATH": "/nonexistent", "NVCC_PREPEND_FLAGS": f"-ccbin {fake}"}
    assert tc.device_host_compiler(env) == str(fake.resolve())
    env = {"PATH": "/nonexistent", "NVCC_CCBIN": str(tmp_path)}
    assert tc.device_host_compiler(env) == str(fake.resolve())
    env = {"PATH": str(tmp_path)}
    assert tc.device_host_compiler(env) == str(fake)


def test_device_key_carries_the_host_compiler(monkeypatch, tmp_path):
    try:
        tc.device_compiler()
    except HawkError:
        pytest.skip("no nvcc on this box")
    if tc.device_compiler_kind() != "nvcc":
        pytest.skip("$HAWK_DEVICE_COMPILER forces nvrtc on this box")
    result = compile_source(KERNEL_SRC, "hawk_probe_kernel",
                            CompileOptions(backend="cuda",
                                           cache_dir=str(tmp_path / "cache")))
    record = hcache.Cache(str(tmp_path / "cache")).record(result.key)
    assert tc.device_host_compiler_identity() in record["key_terms"]
    assert any(t.startswith("ccbin:") for t in record["key_terms"])
