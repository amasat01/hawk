# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The autodiff example's host arm runs everywhere (numpy + the float64
finite difference alone); the device arm (hawk's vjp/jvp checked against
``torch.autograd.grad``/``torch.func.jvp``, plus the zero-copy crossings)
runs wherever a GPU, cupy and torch are all present and skips cleanly
otherwise.

The example needs eagle, so this row skips cleanly when eagle is absent.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.repo_local

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "autodiff_vs_torch.py"


def _load_example():
    """Import the example script by path; it is not part of the package."""
    spec = importlib.util.spec_from_file_location("autodiff_vs_torch", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example():
    """The example module, or a clean skip when its toolchain is absent."""
    pytest.importorskip("numpy")
    pytest.importorskip("eagle")
    pytest.importorskip("hawk")
    if shutil.which("g++") is None and importlib.util.find_spec("ziglang") is None:
        pytest.skip("no host C++ compiler available to build the kernel")
    assert EXAMPLE.is_file(), f"the example is missing: {EXAMPLE}"
    return _load_example()


def _gpu_available() -> bool:
    cupy = pytest.importorskip("cupy")
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _torch_and_gpu_available() -> bool:
    """Torch installed, a CUDA device visible to cupy, AND torch itself
    built with CUDA support -- the third check is the one that was missing:
    a pip CPU-only torch wheel installs cleanly and imports cleanly beside a
    real GPU cupy can see, so the first two checks alone said "yes" and
    every ``.cuda()`` call downstream raised "Torch not compiled with CUDA
    enabled" instead of a clean skip."""
    if importlib.util.find_spec("torch") is None:
        return False
    if not _gpu_available():
        return False
    import torch
    return torch.cuda.is_available()


def test_the_host_arm_runs_without_cupy_or_torch(example, monkeypatch):
    """The host arm never imports cupy, and torch is only an OPTIONAL
    cross-check: block both, the run must still succeed on the float64
    finite difference alone."""
    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setitem(sys.modules, "torch", None)

    results = example.run_demo(batch=64, dt=0.02, seed=0, target="host", verbose=False)

    assert results["target"] == "host"
    assert results["exec_targets"] == ["host"]
    assert results["torch_available"] is False
    assert results["vjp_fd_rel_err"] <= example.REL_TOL
    assert results["jvp_fd_rel_err"] <= example.REL_TOL


def test_the_host_arm_torch_cpu_cross_check_when_available(example):
    """When torch IS importable, the host arm's optional CPU cross-check
    must actually run and agree."""
    pytest.importorskip("torch")

    results = example.run_demo(batch=64, dt=0.02, seed=0, target="host", verbose=False)

    assert results["torch_available"] is True
    assert results["vjp_torch_rel_err"] <= example.REL_TOL
    assert results["jvp_torch_rel_err"] <= example.REL_TOL


@pytest.mark.gpu
def test_the_device_arm_matches_torch_and_zero_copy_holds(example):
    """Reverse (vjp) and forward (jvp) hawk derivatives agree with
    ``torch.autograd.grad``/``torch.func.jvp`` and the float64 FD, and every
    torch<->eagle crossing this arm exercises is a measured alias, not a
    copy."""
    if not _torch_and_gpu_available():
        pytest.skip("no CUDA-capable torch installation visible to this process")

    results = example.run_demo(batch=4096, dt=0.02, seed=0, target="device", reps=1,
                               verbose=False)

    assert results["target"] == "device"
    assert results["exec_targets"] == ["device"]
    assert results["primal_torch_rel_err"] <= example.REL_TOL
    assert results["vjp_torch_rel_err"] <= example.REL_TOL
    assert results["jvp_torch_rel_err"] <= example.REL_TOL
    assert results["vjp_fd_rel_err"] <= example.REL_TOL
    assert results["jvp_fd_rel_err"] <= example.REL_TOL
    for key in ("in_alias_x", "in_alias_v", "in_alias_t", "out_alias_x_next",
               "out_alias_bar_x", "out_alias_dot_next", "out_alias_x_next_after",
               "out_alias_bar_x_after", "out_alias_dot_next_after"):
        assert results[key] is True, f"{key}: expected a zero-copy alias, measured a copy"
    assert results["behavioural_ok"] is True


def test_a_wrong_cotangent_fails_the_same_agreement_check(example):
    """NON-VACUITY: the finite-difference agreement check this example uses
    must be able to say "no". Re-running it with the vjp/jvp cotangent and
    tangent SWAPPED -- ``bar_x`` was produced from ``v32``, so checking it
    against ``J.T @ t32`` instead is checking it against the WRONG seed --
    must report a mismatch, not a clean pass. Runs on the host arm alone, so
    it needs no GPU."""
    import tempfile

    import numpy as np
    with tempfile.TemporaryDirectory() as tmp:
        bundle = example.build_artifact(tmp, ("host",))
        state = example.seed_states(64, 0)
        results = example.run_host(bundle, state, 0.02)

    real_vjp_err, real_jvp_err = example._fd_directional_errors(
        results["state"].astype(np.float64), results["dt"], results["v32"],
        results["t32"], results["bar_x"], results["dot_next"])
    assert real_vjp_err <= example.REL_TOL
    assert real_jvp_err <= example.REL_TOL

    # swap v32/t32: bar_x is checked against J.T @ t32 (the WRONG cotangent),
    # and dot_next against J @ v32 (the WRONG tangent).
    wrong_vjp_err, wrong_jvp_err = example._fd_directional_errors(
        results["state"].astype(np.float64), results["dt"], results["t32"],
        results["v32"], results["bar_x"], results["dot_next"])
    assert wrong_vjp_err > example.REL_TOL, (
        "perturbing the cotangent did not make the agreement check fail -- "
        "the check is vacuous")
    assert wrong_jvp_err > example.REL_TOL, (
        "perturbing the tangent did not make the agreement check fail -- "
        "the check is vacuous")


def test_the_traced_derivative_kernels_declare_the_expected_planes(example):
    """The primal declares one vec_in and one uniform in, one mutable out;
    the vjp/jvp kernels' OWN planes are read from the built bundle's
    sidecars (never assumed) and must carry the roles the naming rule
    (``hawk.diff.ADJOINT_PREFIX``/``TANGENT_PREFIX``) implies."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        bundle = example.build_artifact(tmp, ("host",))
        primal_spec = example._sidecar(bundle, example.KERNEL_NAME)["arg_spec"]
        vjp_spec = example._sidecar(bundle, example.VJP_NAME)["arg_spec"]
        jvp_spec = example._sidecar(bundle, example.JVP_NAME)["arg_spec"]

    primal_roles = {name: role for role, name in primal_spec}
    vjp_roles = {name: role for role, name in vjp_spec}
    jvp_roles = {name: role for role, name in jvp_spec}

    assert primal_roles == {"x": "vec_in", "dt": "uniform", "x_next": "mutable"}

    names = example._plane_names()
    assert vjp_roles["x"] == "vec_in"
    assert vjp_roles["dt"] == "uniform"
    assert vjp_roles[names.bar_out] == "vec_in"
    assert vjp_roles[names.bar_wrt] == "mutable"

    assert jvp_roles["x"] == "vec_in"
    assert jvp_roles["dt"] == "uniform"
    assert jvp_roles[names.dot_wrt] == "vec_in"
    assert jvp_roles[names.dot_out] == "mutable"

    # each derivative's sidecar names its OWN primal and direction.
    vjp_sidecar = example._sidecar(bundle, example.VJP_NAME)
    jvp_sidecar = example._sidecar(bundle, example.JVP_NAME)
    assert vjp_sidecar["derivative"] == {"kind": "vjp", "wrt": list(example.WRT),
                                         "primal": example.KERNEL_NAME, "primal_unit": None}
    assert jvp_sidecar["derivative"] == {"kind": "jvp", "wrt": list(example.WRT),
                                         "primal": example.KERNEL_NAME, "primal_unit": None}
