# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A kernel built in the ``banded`` scalar mode runs end to end: hawk emits
and compiles it, eagle runs it over planes of 8-byte cells packed by
``eagle.banded_pack``, and the unpacked answer is held against the float64
build of the same kernel on the device.

The kernel is a mul/add/div chain with two transcendentals and three
literals: six certified ops on one input path, each within 0.5 ULP53, so
the chain's bound is their sum, 3 ULP53. The inputs are float32-exact, so
the pack itself is exact and every difference is the kernel's."""

from __future__ import annotations

import numpy as np
import pytest

import hawk
import hawk.math as m

N = 4096
#: Six per-op bounds of 0.5 ULP53 along the longest input path.
CHAIN_ULP53 = 3.0


@hawk.kernel
def kq(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    t = x * x * 0.5 + y
    out = m.exp(t) / (y + 1.5) - m.sin(x) * 0.25


def _inputs():
    rng = np.random.default_rng(13)
    x = rng.uniform(-2.0, 2.0, N).astype(np.float32).astype(np.float64)
    y = rng.uniform(0.0, 3.0, N).astype(np.float32).astype(np.float64)
    return x, y


def _run(mode: str, target: str, tmp_path) -> tuple:
    """``kq`` built in ``mode`` and run on ``target``: ``(float64 results, cell words or None)``."""
    x, y = _inputs()
    return _run_kernel(kq, {"x": x, "y": y}, mode, target, tmp_path)


def _run_kernel(kernel, inputs: dict, mode: str, target: str, tmp_path) -> tuple:
    """``kernel`` (one ``out`` plane after its inputs) built in ``mode`` and run on ``target`` over ``inputs``."""
    import _deploy as L
    import eagle
    import eagle.exec as eexec
    from conftest import sidecar_of
    from eagle import arrays
    from eagle import plan as eplan
    from eagle._alloc import synchronize

    from hawk.artifact import build_bundle

    bundle = build_bundle([kernel], tmp_path / f"{mode}_{target}", mode=mode, targets=(target,), host_profile="native")
    sidecar = sidecar_of(bundle, kernel.name)
    plugin, structure = L.device_plugin(bundle.directory, kernel.name, sidecar), eexec.DeviceKernel
    planes = {k: arrays.asarray(v) for k, v in {**inputs, "out": np.zeros(N)}.items()}
    if mode == "banded":
        planes = {k: eagle.banded_pack(v) for k, v in planes.items()}
    if target == "cuda":
        planes = {k: arrays.asarray(v, device="cuda:0") for k, v in planes.items()}
    eplan.plan(plugin, structure=structure).bind(**planes).launch()
    if target == "cuda":
        synchronize()
    out = planes["out"].get() if target == "cuda" else planes["out"]
    if mode == "banded":
        return np.asarray(eagle.banded_unpack(out)).copy(), np.asarray(out).view(np.uint64).copy()
    return np.asarray(out).copy(), None


def _ulp53(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    return np.abs(got - want) / np.spacing(np.abs(want))


@pytest.mark.gpu
def test_a_banded_kernel_on_the_gpu_is_within_the_bound_of_its_float64_build(tmp_path):
    pytest.importorskip("eagle")
    got, _ = _run("banded", "cuda", tmp_path)
    want, _ = _run("float64", "cuda", tmp_path)
    assert _ulp53(got, want).max() <= CHAIN_ULP53
