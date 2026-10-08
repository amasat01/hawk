# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.artifact.arch()``: the ONE device-arch resolver every build door
routes through -- ``device_arch=`` > ``$HAWK_CUDA_ARCH`` > the running GPU
(:func:`hawk.compile.nvrtc.current_device_arch`, memoised per process) >
PTX at the device toolkit's minimum compute arch -- and that it is reached ONLY when ``"cuda"`` is actually a
build target: a host-only build on a GPU-less sandbox must never probe.

Host-only: every row here runs with no GPU required. The two integration
rows (``test_build_cuda_target_resolves_through_the_same_chain``,
``test_build_cuda_target_prefers_the_env_over_the_probe``) go through a real
``build()`` call and a real (AOT nvcc) device compile, so the resolved arch
is checked where it actually lands -- the compiled ``-arch=`` flag -- not
only at the resolver's own door.
"""

from __future__ import annotations

import pytest

import hawk
from hawk.artifact import arch as hawk_arch
from hawk.artifact import build
from hawk.compile import nvrtc as hnvrtc
from hawk.compile import toolchain as tc
from hawk.ir import HawkError


def _pins() -> tuple[str, str, str]:
    """Three distinct real archs the device toolkit supports (its newest
    three), so the precedence rows name no box-specific arch."""
    toolkit = tc.toolkit_archs()
    if toolkit is None or len(toolkit[1]) < 3:
        return ("sm_70", "sm_75", "sm_80")
    a, b, c = toolkit[1][-3:]
    return (f"sm_{a}", f"sm_{b}", f"sm_{c}")


def _no_gpu_answer() -> str:
    """What the resolver answers with no GPU reachable: PTX (a virtual
    ``compute_<N>`` arch) at the toolkit's minimum compute arch, never below
    ``PORTABLE_MIN_ARCH``."""
    toolkit = tc.toolkit_archs()
    if not toolkit:
        return tc.FALLBACK_ARCH
    portable = [a for a in toolkit[1] if a >= tc.PORTABLE_MIN_ARCH]
    return f"compute_{portable[0] if portable else toolkit[1][-1]}"


@pytest.fixture(autouse=True)
def _fresh_probe_memo():
    """Every row starts and ends with the per-process probe memo clear, so
    one row's monkeypatched probe never leaks its answer into the next."""
    tc.reset_device_arch_memo()
    yield
    tc.reset_device_arch_memo()


@hawk.kernel
def _arch_probe_kernel(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    """The smallest possible unit: one Scalar in, one Scalar out."""
    y = x


# --------------------------------------------------------------------------- #
# The resolver itself: precedence and probe gating.
# --------------------------------------------------------------------------- #
def test_explicit_pin_beats_everything(monkeypatch):
    monkeypatch.delenv("HAWK_CUDA_ARCH", raising=False)
    monkeypatch.setattr(hnvrtc, "current_device_arch", lambda: (_ for _ in ()).throw(
        AssertionError("the probe must not run when device_arch= is given")))
    pin = _pins()[0]
    assert hawk_arch(pin) == pin


def test_env_beats_the_probe(monkeypatch):
    pin = _pins()[1]
    monkeypatch.setenv("HAWK_CUDA_ARCH", pin)
    monkeypatch.setattr(hnvrtc, "current_device_arch", lambda: (_ for _ in ()).throw(
        AssertionError("the probe must not run when $HAWK_CUDA_ARCH is set")))
    assert hawk_arch("") == pin


def test_probes_the_running_device_when_unpinned(monkeypatch):
    monkeypatch.delenv("HAWK_CUDA_ARCH", raising=False)
    calls = []

    def _fake_probe():
        calls.append(1)
        return "sm_80"

    monkeypatch.setattr(tc, "toolkit_archs", lambda kind=None: ("stub", (50, 90)))
    monkeypatch.setattr(hnvrtc, "current_device_arch", _fake_probe)
    assert hawk_arch("") == "sm_80"
    assert hawk_arch("") == "sm_80"
    assert len(calls) == 1, "the device probe must be memoised per process"


def test_falls_back_when_the_probe_raises_hawkerror(monkeypatch):
    monkeypatch.delenv("HAWK_CUDA_ARCH", raising=False)

    def _no_device():
        raise HawkError("no CUDA device reachable (simulated)")

    monkeypatch.setattr(hnvrtc, "current_device_arch", _no_device)
    assert hawk_arch("") == _no_gpu_answer()


def _no_binding():
    raise ImportError("cuda-bindings not installed (simulated)")


def test_falls_back_when_cuda_bindings_and_the_driver_are_missing(monkeypatch):
    monkeypatch.delenv("HAWK_CUDA_ARCH", raising=False)
    monkeypatch.setattr(hnvrtc, "current_device_arch", _no_binding)
    monkeypatch.setattr(tc, "_driver_device_arch", lambda: None)
    assert hawk_arch("") == _no_gpu_answer()


def test_asks_the_driver_when_cuda_bindings_are_missing(monkeypatch):
    """A GPU in an environment with no cuda-bindings compiles for its own
    arch, read through the driver library alone."""
    monkeypatch.delenv("HAWK_CUDA_ARCH", raising=False)
    monkeypatch.setattr(hnvrtc, "current_device_arch", _no_binding)
    monkeypatch.setattr(tc, "toolkit_archs", lambda kind=None: ("stub", (50, 90)))
    monkeypatch.setattr(tc, "_driver_device_arch", lambda: "sm_75")
    assert hawk_arch("") == "sm_75"


def test_driver_probe_reads_the_running_device():
    """The real driver probe, unmocked: it reads the arch the cuda-bindings
    probe reads, or ``None`` with no device visible."""
    try:
        expected = hnvrtc.current_device_arch()
    except (ImportError, HawkError):
        expected = None
    assert tc._driver_device_arch() == expected


def test_driver_probe_answers_without_cuda_bindings_or_the_fallback(monkeypatch):
    """With cuda-bindings unimportable and the fallback moved out of the way,
    a visible GPU is still read through the driver library alone."""
    import sys

    try:
        expected = hnvrtc.current_device_arch()
    except (ImportError, HawkError):
        pytest.skip("no CUDA device visible")
    monkeypatch.delenv("HAWK_CUDA_ARCH", raising=False)
    for name in ("cuda.bindings", "cuda.bindings.driver"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setattr(tc, "FALLBACK_ARCH", "compute_00")
    monkeypatch.setattr(tc, "toolkit_archs", lambda kind=None: ("stub", (1, 999)))
    assert hawk_arch("") == expected


def test_falls_back_for_real_with_no_gpu_visible(monkeypatch):
    """The un-monkeypatched path: on a sandbox with no CUDA device visible
    (``CUDA_VISIBLE_DEVICES=``), the real probe finds nothing and the
    resolver answers PTX at the toolkit's minimum -- no mocking at all. A
    process whose driver was already initialised still sees its device (the
    variable is read once, at ``cuInit``); then the answer is that device's
    own arch."""
    monkeypatch.delenv("HAWK_CUDA_ARCH", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    real = tc._probed_device_arch()
    tc.reset_device_arch_memo()
    assert hawk_arch("") == (real if real is not None else _no_gpu_answer())


# --------------------------------------------------------------------------- #
# Through a real build(): the device target's own compiled flags.
# --------------------------------------------------------------------------- #
def test_host_only_build_never_probes(monkeypatch, tmp_path):
    """The gate lives at the artifact call sites, not inside the resolver:
    a host-only build must not reach the device probe at all."""
    monkeypatch.delenv("HAWK_CUDA_ARCH", raising=False)
    monkeypatch.setattr(hnvrtc, "current_device_arch", lambda: (_ for _ in ()).throw(
        AssertionError("a host-only build probed the device")))

    build(_arch_probe_kernel, tmp_path / "unit", targets=("host",))


def test_build_cuda_target_resolves_through_the_same_chain(monkeypatch, tmp_path):
    """``build(..., targets=("cuda",))`` with no ``device_arch=`` and no
    ``$HAWK_CUDA_ARCH`` resolves through the probe -- spied at
    ``toolchain.device_flags``, the real function underneath
    :func:`hawk.compile.drivers`'s AOT nvcc compile -- and the compiled
    ``-arch=`` flag carries the probed value."""
    monkeypatch.delenv("HAWK_CUDA_ARCH", raising=False)
    probe = _pins()[2]
    monkeypatch.setattr(hnvrtc, "current_device_arch", lambda: probe)
    real_device_flags = tc.device_flags
    seen = []

    def _spy(*, arch="", **kw):
        seen.append(arch)
        return real_device_flags(arch=arch, **kw)

    monkeypatch.setattr(tc, "device_flags", _spy)

    build(_arch_probe_kernel, tmp_path / "unit", targets=("cuda",))

    assert seen and all(a == probe for a in seen), seen


def test_build_cuda_target_prefers_the_env_over_the_probe(monkeypatch, tmp_path):
    pin = _pins()[1]
    monkeypatch.setenv("HAWK_CUDA_ARCH", pin)
    monkeypatch.setattr(hnvrtc, "current_device_arch", lambda: (_ for _ in ()).throw(
        AssertionError("the probe must not run when $HAWK_CUDA_ARCH is set")))
    real_device_flags = tc.device_flags
    seen = []

    def _spy(*, arch="", **kw):
        seen.append(arch)
        return real_device_flags(arch=arch, **kw)

    monkeypatch.setattr(tc, "device_flags", _spy)

    build(_arch_probe_kernel, tmp_path / "unit", targets=("cuda",))

    assert seen and all(a == pin for a in seen), seen


def test_build_cuda_target_device_arch_beats_everything(monkeypatch, tmp_path):
    env_pin, pin, _ = _pins()
    monkeypatch.setenv("HAWK_CUDA_ARCH", env_pin)
    monkeypatch.setattr(hnvrtc, "current_device_arch", lambda: (_ for _ in ()).throw(
        AssertionError("device_arch= must beat the env and the probe both")))
    real_device_flags = tc.device_flags
    seen = []

    def _spy(*, arch="", **kw):
        seen.append(arch)
        return real_device_flags(arch=arch, **kw)

    monkeypatch.setattr(tc, "device_flags", _spy)

    build(_arch_probe_kernel, tmp_path / "unit", targets=("cuda",), device_arch=pin)

    assert seen and all(a == pin for a in seen), seen
