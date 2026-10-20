# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A float32 host loop has one width: its lane guards and tally are float32.

Three observations, each with a known answer:

* **The source.** The lane-guard arrays, the finish tally and the ABI structs
  that read them are spelled in the kernel's ``Real``: ``float`` for a float32
  unit, ``double`` for a float64 one.
* **The vectoriser.** With AVX-512 the compiler reports the float32 sample loop
  of a guarded, finishing kernel "vectorized using 64 byte vectors" and the
  entry carries packed float32 arithmetic on zmm; before, the mixed lane width
  (``float`` state, ``double`` guards) kept that loop scalar.
* **The flag.** ``vectorises`` answers what the compiler did with the default
  build, whose ``Loop`` runs interchanged (its trips outside, the samples'
  lanes inside), for float32 as for float64, and for a ``Loop`` above the
  unroll limit. The two rows above read the per-sample build
  (``-DHAWK_HOST_INTERCHANGE=0``), the loop the lane guards serve.
"""

from __future__ import annotations

import re
import subprocess

import pytest
from conftest import avx512

import hawk
from hawk import Mutable, Scalar, Terminated
from hawk.compile import toolchain as tc
from hawk.emit import render_source, scalar_mode
from hawk.emit.host import HOST
from hawk.ir.vectorise import vectorises


@hawk.kernel(steps=1)
def short_loop(x: Mutable[Scalar], t: Mutable[Scalar], terminated: Terminated):
    a = x
    for _i in range(16):
        a = a * 0.9999 + 0.01
    x = a
    t = t + 1.0
    terminated = t >= 3.0


@hawk.kernel(steps=1)
def long_loop(x: Mutable[Scalar], t: Mutable[Scalar], terminated: Terminated):
    a = x
    for _i in range(64):
        a = a * 0.9999 + 0.01
    x = a
    t = t + 1.0
    terminated = t >= 3.0


def _source(kernel, mode):
    return render_source("k", kernel.sinks, kernel.walk, HOST, mode=scalar_mode(mode))


def _compile(tmp_path, kernel, mode, interchanged=False):
    """Build the TU; ``sample`` is the line of the loop that runs the kernel's
    arithmetic across samples: the lane-guarded sample loop of the per-sample
    build (``-DHAWK_HOST_INTERCHANGE=0``), or the lane loop of the interchanged
    ``Loop``'s trips."""
    src = tmp_path / "k.cpp"
    so = tmp_path / "k.so"
    src.write_text(_source(kernel, mode).text)
    shape = [] if interchanged else ["-DHAWK_HOST_INTERCHANGE=0"]
    cc = subprocess.run(
        [tc.host_compiler(), *tc.host_flags(profile="native",
                                            extra=["-fopt-info-vec-optimized", *shape]),
         "-o", str(so), str(src)], capture_output=True, text=True)
    assert cc.returncode == 0, cc.stderr
    assert "warning" not in cc.stderr
    lines = src.read_text().splitlines()
    if interchanged:
        trips = next(i for i, ln in enumerate(lines) if "for (Int hawk_trip = 0;" in ln)
        sample = next(i + 1 for i, ln in enumerate(lines)
                      if i > trips and "for (aether::offset_t hawk_l = 0" in ln)
    else:
        sample = next(i + 1 for i, ln in enumerate(lines)
                      if "for (aether::offset_t hawk_j = hawk_blo" in ln)
    wide = {int(m.group(1)) for m in re.finditer(
        r"k\.cpp:(\d+):\d+: optimized: loop vectorized using 64 byte vectors", cc.stderr)}
    return so, sample, wide


def test_the_lane_guard_abi_follows_the_kernels_real():
    f32 = _source(short_loop, "float32").text
    f64 = _source(short_loop, "float64").text
    assert "HAWK_LANE_T" not in f32 + f64
    assert "alignas(64) float hawk_lanes0[" in f32
    assert "float hawk_finished = 0.0;" in f32
    assert "const float* lanes;" in f32 and "float* finished;" in f32
    assert "alignas(64) double hawk_lanes0[" in f64
    assert "const double* lanes;" in f64


@pytest.mark.skipif(not avx512(), reason="the 64-byte vector verdict needs AVX-512")
def test_the_float32_sample_loop_vectorises_with_packed_zmm_arithmetic(tmp_path):
    so, sample, wide = _compile(tmp_path, short_loop, "float32")
    assert sample in wide, (sample, wide)
    asm = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(so)],
                         capture_output=True, text=True, check=True).stdout
    assert re.search(r"\bv(?:mul|add|fmadd\w*)ps\b.*zmm", asm)


@pytest.mark.skipif(not avx512(), reason="the 64-byte vector verdict needs AVX-512")
@pytest.mark.parametrize("mode", ["float32", "float64"])
@pytest.mark.parametrize("kernel", [short_loop, long_loop])
def test_the_flag_matches_the_vectoriser(tmp_path, kernel, mode):
    _, sample, wide = _compile(tmp_path, kernel, mode, interchanged=True)
    assert vectorises(kernel.walk, "native", mode) is (sample in wide)
    assert _source(kernel, mode).vectorised is vectorises(kernel.walk, scalar_type=mode)
