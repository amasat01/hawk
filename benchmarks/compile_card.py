#!/usr/bin/env python
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Card C3 (local only): compile turnaround of the device build.

Every number is written by the run: each row is a FRESH python process that
builds the kernels through ``hawk.artifact.build_bundle`` against a private
cache directory (``HAWK_CACHE_DIR``), so a ``cold`` row starts from an empty
cache and a ``warm`` row from the cache the cold row filled -- no memo of
any kind crosses a process. Each child also counts the subprocesses it
spawned (``subprocess.Popen``), by program name.

Rows: ``cold`` and ``warm`` builds of three kernels, per device compiler
(``HAWK_DEVICE_COMPILER`` nvcc / nvrtc), then one heavy translation unit (a
team-form LU) cold under both compilers.

Usage::

    python compile_card.py --out DIR [--reps 3]

writes ``DIR/compile_card.json`` and ``.md``. ``--child`` is internal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

_HERE = pathlib.Path(__file__).resolve().parent
_TESTS = _HERE.parent / "tests"
KERNELS = ("scale", "chain6", "scatter")
HEAVY = ("lu", 128)


def _kernels():
    sys.path.insert(0, str(_TESTS))
    import _kernels as K
    return {"scale": K.scale, "chain6": K.chain6, "scatter": K.scatter}


def _use_tree(tree: str | None) -> None:
    """Import hawk from the exported source tree ``tree`` instead of the editable
    install (whose finder would otherwise win): the BEFORE arm of a comparison."""
    global _TESTS
    if not tree:
        return
    sys.meta_path[:] = [f for f in sys.meta_path if "editable" not in type(f).__module__.lower()
                        and "editable" not in type(f).__name__.lower()]
    sys.path.insert(0, tree)
    _TESTS = pathlib.Path(tree) / "tests"
    import glob
    import site
    for so in glob.glob(os.path.join(site.getsitepackages()[0], "hawk", "_core*.so")):
        dst = pathlib.Path(tree) / "hawk" / pathlib.Path(so).name
        if not dst.exists():
            import shutil
            shutil.copy(so, dst)


def child(args) -> None:
    _use_tree(args.tree)
    import subprocess as sp
    spawned = []
    real = sp.Popen.__init__

    def counting(self, argv, *a, **k):
        spawned.append(pathlib.Path(argv[0] if isinstance(argv, (list, tuple)) else argv.split()[0]).name)
        real(self, argv, *a, **k)
    sp.Popen.__init__ = counting
    t0 = time.perf_counter()
    from hawk.artifact import build_bundle
    if args.heavy:
        sys.path.insert(0, str(_TESTS))
        from _linalg_kernels import lu_kernel
        units = {f"lu{HEAVY[1]}": (lu_kernel(HEAVY[1]), {"linalg_form": "team"})}
    else:
        units = {n: (k, {}) for n, k in _kernels().items()}
    imported = time.perf_counter() - t0
    out = {"import_s": round(imported, 3), "build_s": {}, "artifacts": {}}
    root = pathlib.Path(args.work)
    for name, (kern, kw) in units.items():
        t = time.perf_counter()
        b = build_bundle([kern], root / name, targets=(args.target,), device_arch="sm_61",
                         cache_dir=args.cache, host_profile="native", **kw)
        out["build_s"][name] = round(time.perf_counter() - t, 4)
        out["artifacts"][name] = hashlib.sha256(b"".join(
            p.read_bytes() for p in sorted((root / name).glob("*.cubin")) + sorted((root / name).glob("*.so")))).hexdigest()[:16]
    out["spawned"] = sorted(set(spawned)) if not spawned else {s: spawned.count(s) for s in sorted(set(spawned))}
    print("CARD_ROW " + json.dumps(out))


def run_child(compiler, cache, work, target="cuda", heavy=False, tree=None):
    env = dict(os.environ, HAWK_DEVICE_COMPILER=compiler, HAWK_CACHE_DIR=str(cache))
    argv = [sys.executable, str(pathlib.Path(__file__).resolve()), "--child", "--target", target,
            "--cache", str(cache), "--work", str(work)] + (["--heavy"] if heavy else []) + (["--tree", tree] if tree else [])
    done = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=550)
    for line in done.stdout.splitlines():
        if line.startswith("CARD_ROW "):
            return json.loads(line[9:])
    raise RuntimeError(f"child failed: {done.stderr[-800:]}")


def interleave(args) -> None:
    """Heavy TU, cold, BEFORE and AFTER trees alternated A B A B ... so a drift in
    machine load lands on both arms."""
    import statistics
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="compile_card_", dir=args.scratch))
    samples: dict = {}
    for i in range(args.interleave):
        for arm, tree in (("before", args.before_tree), ("after", args.tree)):
            for compiler in ("nvcc", "nvrtc"):
                r = run_child(compiler, scratch / f"c_{arm}_{compiler}_{i}", scratch / f"w_{arm}_{compiler}_{i}",
                              heavy=True, tree=tree)
                samples.setdefault((arm, compiler), []).append(next(iter(r["build_s"].values())))
    lines = ["| arm | compiler | cold lu128 team s (runs) | median |", "|---|---|---|---|"]
    for (arm, compiler), v in sorted(samples.items()):
        lines.append(f"| {arm} | {compiler} | {', '.join(f'{x:.2f}' for x in v)} | {statistics.median(v):.2f} |")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "compile_card_heavy_interleaved.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def parent(args) -> None:
    sys.path.insert(0, str(_HERE))
    import _card_common as cc
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="compile_card_", dir=args.scratch))
    rows = []
    for compiler in ("nvcc", "nvrtc"):
        cache = scratch / f"cache_{compiler}"
        cold = run_child(compiler, cache, scratch / f"w_cold_{compiler}", tree=args.tree)
        warms = [run_child(compiler, cache, scratch / f"w_warm_{compiler}_{i}", tree=args.tree)
                 for i in range(args.reps)]
        for tag, r in [("cold", cold)] + [("warm", w) for w in warms]:
            rows.append({"compiler": compiler, "row": tag, **r})
        hcache = scratch / f"hcache_{compiler}"
        h = run_child(compiler, hcache, scratch / f"w_heavy_{compiler}", heavy=True, tree=args.tree)
        rows.append({"compiler": compiler, "row": "heavy-cold", **h})
    same = {}
    for c in ("nvcc", "nvrtc"):
        mine = [r for r in rows if r["compiler"] == c and r["row"] in ("cold", "warm")]
        same[c] = all(r["artifacts"] == mine[0]["artifacts"] for r in mine)
    dev, _ = cc.device_facts()
    card = {"schema": cc.SCHEMA, "card": "C3 compile turnaround", "generated": cc.utc_now_iso(),
            "script_md5": cc.md5(__file__), "device": dev, "software": cc.software(("numpy", "cupy")),
            "kernels": list(KERNELS), "heavy": f"lu{HEAVY[1]} team", "label": args.label,
            "rows": rows, "cache_hit_artifacts_identical": same}
    md = [f"# C3 compile turnaround ({args.label})", "", "| compiler | row | build s per kernel | spawns | import s |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['compiler']} | {r['row']} | " + ", ".join(f"{k}={v}" for k, v in r["build_s"].items())
                  + f" | {r['spawned']} | {r['import_s']} |")
    cc.write_card(card, out / f"compile_card_{args.label}.json", "\n".join(md) + "\n", out / f"compile_card_{args.label}.md")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--heavy", action="store_true")
    ap.add_argument("--target", default="cuda")
    ap.add_argument("--cache")
    ap.add_argument("--work")
    ap.add_argument("--out")
    ap.add_argument("--label", default="run")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--scratch", default=os.path.join(tempfile.gettempdir(), "hawk_compile_card"))
    ap.add_argument("--tree", help="exported hawk source tree to import instead of the editable install")
    ap.add_argument("--before-tree", help="with --interleave: the BEFORE tree; the run alternates it with this one")
    ap.add_argument("--interleave", type=int, default=0, help="heavy-TU A/B: N alternations before/after")
    a = ap.parse_args()
    child(a) if a.child else interleave(a) if a.interleave else parent(a)
