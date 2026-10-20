# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``hawk.runtime.run`` on ONE shared HostKernel is one atomic call.

Two threads call ``run`` on the same kernel in lockstep, each with its own
planes and its own by-value uniform, and every result is checked against a
serial reference. Without the kernel's run lock one thread rebinds between the
other's bind and launch, so a launch writes the wrong planes (and, with unequal
sizes, the heap). Red on a GIL build too, not only on a free-threaded one. The
threads run in a CHILD process, so a crash is a red verdict and not the end of
the pytest session.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.ft

_CHILD = r"""
import faulthandler, json, sys, threading
import numpy as np
from hawk import runtime

faulthandler.enable()
faulthandler.dump_traceback_later(120, exit=True)
directory, sizes, calls = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3])
kernel = runtime.load(directory, "axpb")
sys.setswitchinterval(1e-6)             # a GIL build must switch inside run too
lockstep = threading.Barrier(len(sizes))
bad = [0] * len(sizes)

def work(i):
    n, a, b = sizes[i], float(i + 2), -1.0
    x = np.linspace(0.0, 1.0, n) + i
    want = a * x + b
    for _ in range(calls):
        y = np.zeros(n)
        lockstep.wait()
        runtime.run(kernel, x=x, a=a, b=b, y=y)
        bad[i] += not np.array_equal(y, want)

pool = [threading.Thread(target=work, args=(i,)) for i in range(len(sizes))]
for t in pool:
    t.start()
for t in pool:
    t.join()
print(json.dumps({"bad": bad}))
"""


@pytest.mark.parametrize("sizes", [(512, 512), (64, 1000)], ids=["equal_n", "unequal_n"])
def test_run_on_one_kernel_is_atomic(built, sizes):
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(built["axpb"].directory),
         json.dumps(sizes), "1500"],
        capture_output=True, text=True, timeout=240, check=False,
        cwd=str(Path(__file__).resolve().parent),
    )
    assert proc.returncode == 0, f"child died (rc={proc.returncode}):\n{proc.stderr[-2000:]}"
    bad = json.loads(proc.stdout.strip().splitlines()[-1])["bad"]
    assert bad == [0] * len(sizes), f"calls that read the other thread's planes: {bad}"
