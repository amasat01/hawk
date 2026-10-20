# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The hybrid orbit example runs end to end on the host arm, at tiny sizes.

The example needs eagle, torch and a C++ compiler, so this row skips cleanly when any is absent.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.repo_local

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "two_body_hybrid.py"


def _load_example():
    """Import the example script by path; it is not part of the package."""
    spec = importlib.util.spec_from_file_location("two_body_hybrid", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example():
    """The example module, or a clean skip when its toolchain is absent."""
    pytest.importorskip("numpy")
    pytest.importorskip("torch")
    pytest.importorskip("eagle")
    pytest.importorskip("hawk")
    if shutil.which("g++") is None and importlib.util.find_spec("ziglang") is None:
        pytest.skip("no host C++ compiler available to build the kernel")
    assert EXAMPLE.is_file(), f"the example is missing: {EXAMPLE}"
    return _load_example()


def test_the_host_arm_matches_torch_and_beats_uncorrected_two_body(example):
    """Both printed acceptances, at the smallest size that still means anything.

    ``run_demo`` asserts (a) and (b) itself; this row re-states them against
    the returned numbers so a failure names which one gave way.
    """
    results = example.run_demo(steps=8, batch=16, dt=0.01, seed=0,
                               target="host", verbose=False)

    assert results["exec_targets"] == ["host"]
    assert results["match"] <= example.MATCH_TOL, (
        f"(a) compiled vs torch eager: {results['match']:.3e}")
    assert results["hybrid_error"] < example.ERROR_RATIO * results["two_body_error"], (
        f"(b) corrected {results['hybrid_error']:.3e} vs two-body "
        f"{results['two_body_error']:.3e}")
    assert results["compiled_wall_per_step"] > 0.0


def test_the_traced_kernel_declares_one_state_in_and_one_state_out(example):
    """The kernel is ONE RK4 step: it reads a state plane and writes one; the weights are shared Tables."""
    kernel = example.step_kernel()
    roles = {name: role for role, name in kernel.arg_spec}

    assert kernel.name == example.KERNEL_NAME
    assert set(roles) == {"x", "w1", "b1", "w2", "b2", "dt", "x_next"}
    assert roles["x"] == "vec_in" and roles["x_next"] == "mutable"
    assert roles["w1"] == roles["b1"] == roles["w2"] == roles["b2"] == "lookup"
    assert roles["dt"] == "uniform"
