# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Every ``Loop`` of a sample-local host kernel runs interchanged, and moves no bit.

The unit of a sample-local kernel with a ``Loop`` runs each top-level loop as ``for trip { for lane in tile { body } }``; the
same unit built with ``-DHAWK_HOST_INTERCHANGE=0`` runs the per-sample loop it replaced. Under the exact ``native`` profile both
builds of every kernel below leave every output plane byte-equal, for a count below one tile, one that is not a multiple of the
tile and several tiles: a 256-trip arithmetic chain and a ``sin`` chain in float64 and float32, a nested loop, a loop whose trip
count is per sample (a ``break`` on a plane value under a ``terminated`` guard), a break at the head and in the middle of the
body, and two loops in a row with a rank-1 carry.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk import Mutable, Param, Scalar, Terminated, Vector
from hawk import runtime as rt
from hawk.artifact import build_bundle

PLAIN = ("HAWK_HOST_INTERCHANGE=0",)
INTERCHANGED = ("HAWK_HOST_INTERCHANGE=1",)
#: The bit-identical host profile every twin build pins.
EXACT = "native"
TRIPS = 256


@hawk.kernel
def arith(x: Scalar, g: Scalar, y: Mutable[Scalar]):
    a = x
    for _k in range(TRIPS):
        a = a * g + 0.25
    y = a


@hawk.kernel
def sin_chain(x: Scalar, g: Scalar, y: Mutable[Scalar]):
    a = x
    for _k in range(16):
        a = m.sin(a * g) + 0.5
    y = a + x


@hawk.kernel
def nested(x: Scalar, g: Scalar, y: Mutable[Scalar]):
    a = x
    b = g * 0.5
    for _k in range(40):
        c = a
        for _j in range(24):
            c = c * b + 0.125
        a = a * 0.75 + c * 0.01
    y = a + b


@hawk.kernel
def trajectory(
    omega: Scalar,
    nstop: Scalar,
    dt: Param,
    terminated: Terminated,
    x: Mutable[Scalar],
    v: Mutable[Scalar],
    k: Mutable[Scalar],
):
    xs = x
    vs = v
    ks = k
    w2 = omega * omega
    for _s in range(300):
        if ks >= nstop:
            break
        a = -w2 * xs
        xs = xs + dt * vs
        vs = vs + dt * a
        ks = ks + 1.0
    x = xs
    v = vs
    k = ks


@hawk.kernel
def head(x: Scalar, g: Scalar, lim: Scalar, y: Mutable[Scalar]):
    s = x
    for _k in range(40):
        if s > lim:
            break
        s = m.sin(g * s) + s * 1.2
    y = s


@hawk.kernel
def middle(x: Scalar, g: Scalar, lim: Scalar, y: Mutable[Scalar]):
    s = x
    t = x
    for _k in range(40):
        s = s * 1.1 + g * t
        if s > lim:
            break
        t = m.tanh(t) + 0.1
    y = s + t


@hawk.kernel
def two_loops(w0: Vector[3], g: Scalar, out: Mutable[Vector[3]], y: Mutable[Scalar]):
    w = w0
    for _k in range(30):
        w = w * g + 0.1
    s = w[0] + w[1]
    acc = s
    for _k in range(20):
        acc = acc * 0.5 + w[2]
    out = w
    y = acc + s


KERNELS = {
    k.name: k for k in (arith, sin_chain, nested, trajectory, head, middle, two_loops)
}
#: (kernel, scalar mode) of every twin: float32 for the chains.
TWINS = [(name, "float64") for name in KERNELS] + [
    ("arith", "float32"),
    ("sin_chain", "float32"),
]


def _inputs(name: str, n: int, dtype) -> dict:
    rng = np.random.default_rng(11 + n)
    u = lambda lo, hi: rng.uniform(lo, hi, n).astype(dtype)  # noqa: E731
    if name == "arith":
        return {"x": u(-1.0, 1.0), "g": u(0.5, 0.99)}
    if name in ("sin_chain", "nested"):
        return {"x": u(-1.0, 1.0), "g": u(-1.5, 1.5)}
    if name == "trajectory":
        nstop = rng.integers(0, 320, n).astype(dtype)
        terminated = np.zeros(n, dtype=bool)
        terminated[::7] = True
        return {
            "omega": u(0.5, 2.0),
            "nstop": nstop,
            "dt": 0.01,
            "terminated": terminated,
        }
    if name in ("head", "middle"):
        x = u(0.02, 0.6)
        x[:2] = 5.0
        return {"x": x, "g": u(-0.9, 0.9), "lim": np.full(n, 3.0, dtype)}
    return {"w0": rng.uniform(-1.0, 1.0, (3, n)).astype(dtype), "g": u(0.2, 0.9)}


def _outputs(name: str, n: int, dtype) -> dict:
    if name == "trajectory":
        return {
            "x": np.linspace(1.0, 2.0, n).astype(dtype),
            "v": np.zeros(n, dtype),
            "k": np.zeros(n, dtype),
        }
    if name == "two_loops":
        return {"out": np.zeros((3, n), dtype), "y": np.zeros(n, dtype)}
    return {"y": np.zeros(n, dtype)}


@pytest.fixture(scope="module")
def twins(tmp_path_factory):
    """``{(mode, "interchanged"|"plain"): Bundle}``."""
    cache = str(tmp_path_factory.mktemp("loop_ic_cache"))
    out = {}
    for mode in ("float64", "float32"):
        kernels = list(KERNELS.values()) if mode == "float64" else [arith, sin_chain]
        for shape, defines in (("interchanged", INTERCHANGED), ("plain", PLAIN)):
            out[(mode, shape)] = build_bundle(
                kernels,
                tmp_path_factory.mktemp(f"{mode}_{shape}"),
                targets=("host",),
                cache_dir=cache,
                defines=defines,
                options={"host.profile": EXACT},
                mode=mode,
            )
    return out


def _run(bundle, name, n, dtype) -> dict:
    inputs = _inputs(name, n, dtype)
    planes = _outputs(name, n, dtype)
    art = rt.load(bundle.directory, name, sidecar_of(bundle, name))
    rt.run(art, **inputs, **planes)
    return planes


@pytest.mark.parametrize("n", [5, 200, 1000])
@pytest.mark.parametrize("name,mode", TWINS, ids=[f"{a}-{b}" for a, b in TWINS])
def test_interchanged_and_per_sample_loops_are_byte_identical(twins, name, mode, n):
    dtype = np.float64 if mode == "float64" else np.float32
    got = _run(twins[(mode, "interchanged")], name, n, dtype)
    want = _run(twins[(mode, "plain")], name, n, dtype)
    for plane in want:
        assert got[plane].tobytes() == want[plane].tobytes(), (name, plane)
    if name == "trajectory":  # non-vacuity: per-sample trip counts differ
        assert len(set(want["k"].tolist())) > 3
