# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Second-order drag on the host: no stale stack read.

With aether-dsc 0.2.3 a nested ``select`` over value leaves (the guard of a
terminated-masked kernel, differentiated twice) was miscompiled by GCC's
points-to analysis: the kernel read a stack slot it had not written. The exact
result here is 0 (every tangent plane is zero), so a poisoned stack (1e200)
shows up verbatim. The kernel is the emitted ``jvp`` of ``jvp`` of a drag with a
``terminated`` guard, built with the production flags by the user's toolchain and
driven by ``stale_stack_harness.cpp`` with ``LD_BIND_NOW=1``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import hawk
from hawk import Mutable, Param, Terminated, Vector
from hawk.compile import toolchain as tc
from hawk.diff import jvp
from hawk.emit import BACKENDS, FLOAT64, render_source
from hawk.ir import canonical
from hawk.math import dot, sqrt
from hawk.trace import Kernel

HARNESS = Path(__file__).with_name("stale_stack_harness.cpp")


@hawk.kernel
def drag_guarded(v: Vector[3], cd: Param, terminated: Terminated, a: Mutable[Vector[3]]):
    a = -cd * sqrt(dot(v, v)) * v


@pytest.fixture(scope="module")
def kernel_so(tmp_path_factory):
    # a derived level is a kernel in its own right (its guard is applied again): two nested selects
    first = Kernel("drag_jvp", jvp(drag_guarded))
    sinks = jvp(first)
    text = render_source("drag_jj", sinks, canonical(sinks), BACKENDS["host"], mode=FLOAT64).text
    work = tmp_path_factory.mktemp("stale_stack")
    (work / "k.cpp").write_text(text)
    cxx = tc.host_compiler()
    subprocess.run([cxx, *tc.host_flags(), "-o", str(work / "k.so"), str(work / "k.cpp")],
                   check=True)
    subprocess.run([cxx, "-O1", "-std=c++17", "-o", str(work / "harness"), str(HARNESS), "-ldl"],
                   check=True)
    return work


@pytest.mark.parametrize("n", [6, 64])
@pytest.mark.parametrize("shared", [0, 1])
def test_second_order_guarded_drag_reads_no_stale_stack(kernel_so, n, shared):
    run = subprocess.run([str(kernel_so / "harness"), str(kernel_so / "k.so"), str(n), str(shared)],
                         capture_output=True, text=True,
                         env={**os.environ, "LD_BIND_NOW": "1"})
    assert run.returncode == 0, run.stdout + run.stderr
