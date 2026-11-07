#!/usr/bin/env python
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Card A11 (recorded, not gated): a chain against the same kernels launched one by one.

Two chains from ``tests/test_chain.py``: ``pipe`` (20 members, each reads the one before) and ``fanin`` (10 members of
Vector[16], the last reads every earlier value). Each is timed as ONE chain and as its members in sequence, for each
scalar type and sample count; the median of ``--reps`` runs after a warm-up. The numbers are written by the run.

Usage::

    python fusion_sweep.py --out DIR [--target host|cuda] [--n 1 1000 400000] [--reps 7] [--label NAME]

writes ``DIR/fusion_sweep_<label>.json`` and ``.md``. It needs a repository checkout and, for ``cuda``, a device.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "tests"))
import _card_common as cc  # noqa: E402


def _cases():
    from test_chain import fan_in
    return {"pipe": (fan_in(20, pipe=True), 8), "fanin": (fan_in(10, 16), 16)}


def _time(fn, sync, reps: int) -> float:
    fn()
    sync()
    runs = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        sync()
        runs.append(time.perf_counter() - t)
    return statistics.median(runs) * 1e3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="host", choices=("host", "cuda"))
    ap.add_argument("--n", type=int, nargs="+", default=[1, 1000, 400000])
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--label", default="run")
    a = ap.parse_args()
    import eagle
    import hawk

    if a.target == "cuda":
        import cupy
        sync = cupy.cuda.Stream.null.synchronize
    else:
        def sync():
            pass
    rows = []
    for case, (members, w) in _cases().items():
        chained = hawk.chain(*members, name=case)
        for dtype in ("float32", "float64"):
            plans = {k.name: eagle.deploy(k, scalar_type=dtype, targets=(a.target,)) for k in (chained, *members)}
            for n in a.n:
                held = {}
                for k in (chained, *members):
                    for _, nm in k.arg_spec:
                        if nm not in held:
                            v = np.full(n, 0.5, dtype) if nm == "x" else np.zeros((w, n), dtype)
                            held[nm] = eagle.arrays.asarray(v, device="cuda:0") if a.target == "cuda" else v
                one = _time(lambda: plans[chained.name](**{nm: held[nm] for _, nm in chained.arg_spec}), sync, a.reps)
                seq = _time(lambda: [plans[k.name](**{nm: held[nm] for _, nm in k.arg_spec}) for k in members], sync, a.reps)
                rows.append({"case": case, "members": len(members), "dtype": dtype, "n": n,
                             "chain_ms": round(one, 4), "launches_ms": round(seq, 4), "ratio": round(one / seq, 3)})
    dev, _ = cc.device_facts()
    card = {"schema": cc.SCHEMA, "card": "A11 chain vs launches", "generated": cc.utc_now_iso(),
            "script_md5": cc.md5(__file__), "device": dev, "software": cc.software(("numpy", "cupy")),
            "target": a.target, "label": a.label, "reps": a.reps, "rows": rows}
    md = [f"# A11 chain vs launches ({a.label}, {a.target})", "", "| case | dtype | N | chain ms | launches ms | chain / launches |",
          "|---|---|---|---|---|---|"]
    md += [f"| {r['case']} | {r['dtype']} | {r['n']} | {r['chain_ms']} | {r['launches_ms']} | {r['ratio']} |" for r in rows]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cc.write_card(card, out / f"fusion_sweep_{a.label}.json", "\n".join(md) + "\n", out / f"fusion_sweep_{a.label}.md")
    print("\n".join(md))


if __name__ == "__main__":
    main()
