# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Fills a row of ``hawk.ir.steps_class._ROWS``: K per-step reverse launches against one fused launch
(``step_vjp(hawk.steps(k, K), steps_reverse="fused")``) on the compiled host, per batch size N.

    python tests/_probe_steps_reverse.py [--K 4 8] [--N 1 64 1024 16384] [--reps 100]

Prints ``(width, K, n_max)``: the widest N where fused beat per-step by 10%. A device row needs the
same measurement on that card (launch it under the GPU pool); until then its class has no row."""

from __future__ import annotations

import argparse
import tempfile
import time

import numpy as np

import hawk
from hawk import Mutable, Param, Scalar, Terminated
from hawk import math as hm
from hawk.diff import step_vjp

STATE = ("x", "v", "y", "w")


@hawk.kernel(steps=1)
def osc4(dt: Param, limit: Scalar, terminated: Terminated, x: Mutable[Scalar], v: Mutable[Scalar],
         y: Mutable[Scalar], w: Mutable[Scalar]):
    xn = x + dt * v
    vn = v - dt * hm.sin(x) + 0.1 * dt * y
    yn = y + dt * (w - x)
    wn = w - dt * hm.sin(y)
    x, v, y, w = xn, vn, yn, wn
    terminated = xn > limit


def _go(art, **planes):
    names = {n for _, n in art.arg_spec}
    hawk.run(art, **{k: v for k, v in planes.items() if k in names})


def measure(K: int, ns, reps: int):
    fused = hawk.steps(osc4, K)
    per = hawk.Kernel("osc4_sv", step_vjp(osc4))
    one = hawk.Kernel("osc4_fv", step_vjp(fused, steps_reverse="fused"))
    out = tempfile.mkdtemp()
    hawk.build([osc4, fused, per, one], out, targets=("host",),
               derivative={"osc4_sv": ("osc4", "0" * 64), "osc4_fv": (fused.name, "0" * 64)})
    prim, rs, rf = (hawk.load(out, n) for n in ("osc4", "osc4_sv", "osc4_fv"))
    rows = []
    for n in ns:
        rng = np.random.default_rng(n)
        hist, cur = [], {s: rng.uniform(0.1, 1.0, n) for s in STATE}
        for _ in range(K):
            hist.append({s: a.copy() for s, a in cur.items()})
            _go(prim, dt=0.05, limit=np.full(n, 1e9), terminated=np.zeros(n, bool), **cur)
        seed = {f"bar_{s}_next": rng.uniform(-1, 1, n) for s in STATE}

        def per_step():
            nxt = {k: a.copy() for k, a in seed.items()}
            for k in reversed(range(K)):
                got = {f"bar_{s}": np.zeros(n) for s in STATE}
                _go(rs, dt=0.05, limit=np.full(n, 1e9), bar_dt=np.zeros(n), **hist[k], **nxt, **got)
                nxt = {f"bar_{s}_next": got[f"bar_{s}"] for s in STATE}

        def fused_once():
            got = {f"bar_{s}": np.zeros(n) for s in STATE}
            _go(rf, dt=0.05, limit=np.full(n, 1e9), terminated=np.zeros(n, bool), bar_dt=np.zeros(n),
                     **hist[0], **seed, **got)

        times = []
        for fn in (per_step, fused_once):
            fn()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            times.append((time.perf_counter() - t0) / reps)
        rows.append((n, times[0] / times[1]))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--N", type=int, nargs="+", default=[1, 64, 1024, 16384])
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()
    for K in a.K:
        rows = measure(K, a.N, a.reps)
        won = [n for n, ratio in rows if ratio >= 1.10]
        print(f"K={K}: " + ", ".join(f"N={n} {ratio:.2f}x" for n, ratio in rows)
              + f"  ->  row ({len(STATE)}, {K}, {max(won) if won else 0})")
