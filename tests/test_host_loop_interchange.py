# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Every ``Loop`` of a sample-local host kernel runs interchanged, and moves no bit.

The host TU of a sample-local kernel with a ``Loop`` runs each top-level loop as
``for trip { for lane in tile { body } }`` (:func:`hawk.emit.host.loop_interchange`):
the carried values live in lane rows, a lane stops at its own ``break``, and the
tile spans eight vector registers of the kernel's ``Real``. The SAME TU built with
``-DHAWK_HOST_INTERCHANGE=0`` runs the per-sample loop it replaced. What is pinned:

* **Bit identity.** Under the exact ``native`` profile, both builds of every kernel
  below leave every output plane byte-equal, for N below one tile, N not a multiple
  of the tile and several tiles: a 256-trip arithmetic chain and a ``sin`` chain in
  float64 and float32, a nested loop (the outer one interchanged, the inner one each
  lane's own ``for``), a loop whose trip count is per sample (a ``break`` on a plane
  value, under a ``terminated`` guard), a break at the head and in the middle of the
  body, and two loops in a row with a rank-1 carry.
* **The vectoriser.** With AVX-512 the lane loop of the trips is reported
  "vectorized using 64 byte vectors" for the float64 and float32 arithmetic chains.
* **The flag.** ``vectorises`` and ``Source.vectorised`` answer what the compiler
  did with the lane loop, and the cost model's per-branch flag reads the same
  predicate.
* **Who keeps the per-sample loop.** A kernel that is not ``sample_local``, and a
  loop pair the lane form cannot hold (a derived loop reading its primal's count).
"""

from __future__ import annotations

import re
import subprocess

import numpy as np
import pytest
from conftest import avx512, sidecar_of

import hawk
import hawk.math as m
from hawk import Mutable, Param, Scalar, Terminated, Vector
from hawk import runtime as rt
from hawk.artifact import build_bundle
from hawk.compile import toolchain as tc
from hawk.diff import vjp
from hawk.emit import render_source, scalar_mode
from hawk.emit.host import INTERCHANGE_MACRO, loop_interchange
from hawk.ir.cost import kernel_cost
from hawk.ir.vectorise import interchanged_loops, vectorises
from hawk.targets import TARGETS

PLAIN = ("HAWK_HOST_INTERCHANGE=0",)
INTERCHANGED = ("HAWK_HOST_INTERCHANGE=1",)
#: The bit-identical host profile every twin build pins.
EXACT = "native"
TRIPS = 256


# --------------------------------------------------------------------------- #
# The kernels (copied here, never imported from a benchmark)
# --------------------------------------------------------------------------- #
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
def trajectory(omega: Scalar, nstop: Scalar, dt: Param, terminated: Terminated,
               x: Mutable[Scalar], v: Mutable[Scalar], k: Mutable[Scalar]):
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


@hawk.kernel
def gated_sum(x: Scalar, g: Scalar, lim: Scalar, y: Mutable[Scalar]):
    acc = 0.0
    for k in range(12):
        if x * k > lim:
            break
        acc = acc + m.sin(g * x * k)
    y = acc


KERNELS = {k.name: k for k in (arith, sin_chain, nested, trajectory, head, middle,
                               two_loops)}
#: (kernel, scalar mode) of every twin: float32 for the chains.
TWINS = [(name, "float64") for name in KERNELS] + [("arith", "float32"),
                                                   ("sin_chain", "float32")]


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
        return {"omega": u(0.5, 2.0), "nstop": nstop, "dt": 0.01,
                "terminated": terminated}
    if name in ("head", "middle"):
        x = u(0.02, 0.6)
        x[:2] = 5.0
        return {"x": x, "g": u(-0.9, 0.9), "lim": np.full(n, 3.0, dtype)}
    return {"w0": rng.uniform(-1.0, 1.0, (3, n)).astype(dtype), "g": u(0.2, 0.9)}


def _outputs(name: str, n: int, dtype) -> dict:
    if name == "trajectory":
        return {"x": np.linspace(1.0, 2.0, n).astype(dtype), "v": np.zeros(n, dtype),
                "k": np.zeros(n, dtype)}
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
                kernels, tmp_path_factory.mktemp(f"{mode}_{shape}"), targets=("host",),
                cache_dir=cache, defines=defines, host_profile=EXACT, mode=mode)
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
    if name == "trajectory":         # non-vacuity: per-sample trip counts differ
        assert len(set(want["k"].tolist())) > 3


def test_every_twin_tu_carries_the_interchange():
    for name, mode in TWINS:
        k = KERNELS[name]
        text = render_source(name, k.sinks, k.walk, TARGETS["host"],
                             mode=scalar_mode(mode)).text
        assert f"#if {INTERCHANGE_MACRO}" in text, name
        assert "hawk_abi::kHostLoopTile" in text, name
        assert len(interchanged_loops(k.walk)) == text.count("for (Int hawk_trip = 0;")


def test_a_nested_loop_interchanges_its_outermost_only():
    text = render_source("nested", nested.sinks, nested.walk, TARGETS["host"]).text
    trips = re.findall(r"for \(Int hawk_trip = 0; hawk_any != 0 && hawk_trip < (\d+);",
                       text)
    assert trips == ["40"]
    assert re.search(r"for \(Int hawk_k\d+__j = 0; hawk_k\d+__j < 24; \+\+hawk_k\d+__j\)",
                     text)


def test_who_keeps_the_per_sample_loop():
    # a derived loop pair: the reverse loop reads the primal's per-sample count
    derived = hawk.Kernel("gated_sum_vjp", vjp(gated_sum, wrt=("x", "g")))
    assert interchanged_loops(derived.walk) == ()
    assert loop_interchange(derived.sinks, derived.walk) is None
    # a loop-free kernel has nothing to interchange
    assert interchanged_loops(_loop_free.walk) == ()


@hawk.kernel
def _loop_free(x: Scalar, y: Mutable[Scalar]):
    y = x * 2.0


# --------------------------------------------------------------------------- #
# The vectoriser and the flag
# --------------------------------------------------------------------------- #
def _lane_loop_report(tmp_path, kernel, mode, profile):
    """(lines of the trips' lane loops, lines GCC vectorised with 64-byte vectors)."""
    src = tmp_path / f"{kernel.name}_{mode}_{profile}.cpp"
    so = src.with_suffix(".so")
    src.write_text(render_source(kernel.name, kernel.sinks, kernel.walk, TARGETS["host"],
                                 mode=scalar_mode(mode)).text)
    cc = subprocess.run(
        [tc.host_compiler(), *tc.host_flags(profile=profile,
                                            extra=["-fopt-info-vec-optimized"]),
         "-o", str(so), str(src)], capture_output=True, text=True)
    assert cc.returncode == 0, cc.stderr
    assert "warning" not in cc.stderr
    lines = src.read_text().splitlines()
    lane = set()
    for at, line in enumerate(lines):
        if "for (Int hawk_trip = 0;" in line:
            nxt = next(j for j in range(at + 1, len(lines))
                       if "for (aether::offset_t hawk_l = 0" in lines[j])
            lane.add(nxt + 1)
    wide = {int(g) for g in re.findall(
        rf"{re.escape(src.name)}:(\d+):\d+: optimized: loop vectorized using 64 byte vectors",
        cc.stderr)}
    return lane, wide


@pytest.mark.skipif(not avx512(), reason="the 64-byte vector verdict needs AVX-512")
@pytest.mark.parametrize("mode", ["float64", "float32"])
def test_the_lane_loop_of_an_arithmetic_chain_vectorises(tmp_path, mode):
    lane, wide = _lane_loop_report(tmp_path, arith, mode, EXACT)
    assert lane and lane <= wide, (lane, wide)


FLAG_CASES = [(arith, "float64", EXACT), (arith, "float32", EXACT),
              (nested, "float64", EXACT), (two_loops, "float64", EXACT),
              (sin_chain, "float64", EXACT), (sin_chain, "float64", "native-vector-math"),
              (trajectory, "float64", EXACT), (middle, "float64", "native-vector-math")]


@pytest.mark.skipif(not avx512(), reason="the 64-byte vector verdict needs AVX-512")
@pytest.mark.parametrize("kernel,mode,profile", FLAG_CASES,
                         ids=[f"{k.name}-{mo}-{p}" for k, mo, p in FLAG_CASES])
def test_the_flag_matches_the_lane_loop(request, tmp_path, kernel, mode, profile):
    macros = tc.compiler_macros(tc.host_compiler())
    if (kernel is trajectory and "__clang__" not in macros
            and int(macros.get("__GNUC__", "99")) < 13):
        request.applymarker(pytest.mark.xfail(
            strict=True, reason="GCC 12 does not vectorise this lane loop"))
    lane, wide = _lane_loop_report(tmp_path, kernel, mode, profile)
    assert vectorises(kernel.walk, profile, mode) is (lane <= wide), (lane, wide)


def test_the_cost_model_reads_the_same_predicate():
    for kernel in (arith, nested, sin_chain, trajectory):
        for profile in (EXACT, "native-vector-math"):
            assert kernel_cost(kernel.walk, profile).vectorised is vectorises(
                kernel.walk, profile)
    # an interchanged loop's trip count leaves the vectorised flag on
    assert kernel_cost(arith.walk, EXACT).vectorised is True
    # a nested loop above the unroll limit stays each lane's own `for`
    assert kernel_cost(nested.walk, EXACT).vectorised is False
