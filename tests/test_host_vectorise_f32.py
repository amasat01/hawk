# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A float32 host loop vectorises with packed arithmetic.

With AVX-512 the compiler reports the float32 sample loop of a guarded, finishing kernel "vectorized using 64 byte
vectors" and the entry carries packed float32 arithmetic on zmm; before, the mixed lane width (``float`` state,
``double`` guards) kept that loop scalar. The row reads the per-sample build (``-DHAWK_HOST_INTERCHANGE=0``), the loop
the lane guards serve. It needs an AVX-512 host and the GCC vectoriser report, so it is proven on that machine.
The other rows of the old file (the source's lane-guard spelling, the flag against the vectoriser) were rows of the
deleted host writer and of ``vectorises``."""

from __future__ import annotations

import re
import subprocess

import ospreys.cpu.toolchain as ctc
import pytest
from conftest import avx512

import hawk
from hawk import Mutable, Scalar, Terminated
from hawk.artifact import emit_source


@hawk.kernel(steps=1)
def short_loop(x: Mutable[Scalar], t: Mutable[Scalar], terminated: Terminated):
    a = x
    for _i in range(16):
        a = a * 0.9999 + 0.01
    x = a
    t = t + 1.0
    terminated = t >= 3.0


def _source(kernel, mode):
    return emit_source(kernel, name="k", mode=mode, targets=("host",)).sources["host"]


def _compile(tmp_path, kernel, mode):
    """Build the TU per sample; ``sample`` is the line of the lane-guarded sample loop that runs the kernel's arithmetic."""
    src = tmp_path / "k.cpp"
    so = tmp_path / "k.so"
    src.write_text(_source(kernel, mode).text)
    cc = subprocess.run(
        [ctc.host_compiler(), *ctc.host_flags(profile="native",
                                              extra=["-fopt-info-vec-optimized", "-DHAWK_HOST_INTERCHANGE=0"]),
         "-o", str(so), str(src)], capture_output=True, text=True)
    assert cc.returncode == 0, cc.stderr
    assert "warning" not in cc.stderr
    lines = src.read_text().splitlines()
    sample = next(i + 1 for i, ln in enumerate(lines) if "for (aether::offset_t hawk_j = hawk_blo" in ln)
    wide = {int(m.group(1)) for m in re.finditer(
        r"k\.cpp:(\d+):\d+: optimized: loop vectorized using 64 byte vectors", cc.stderr)}
    return so, sample, wide


@pytest.mark.skipif(not avx512(), reason="the 64-byte vector verdict needs AVX-512")
def test_the_float32_sample_loop_vectorises_with_packed_zmm_arithmetic(tmp_path):
    so, sample, wide = _compile(tmp_path, short_loop, "float32")
    assert sample in wide, (sample, wide)
    asm = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(so)],
                         capture_output=True, text=True, check=True).stdout
    assert re.search(r"\bv(?:mul|add|fmadd\w*)ps\b.*zmm", asm)
