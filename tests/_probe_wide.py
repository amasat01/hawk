# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The inner-loop probe of the wide-value pass: one command, one shape (or a sweep), seconds per case.

    python tests/_probe_wide.py --dtype float64 --widths 4,8 --patterns rtread,rotate --sizes 65536,262144

A case is ``(dtype, width, pattern)``. It is built in the REGISTER form (``wide_form='thread'``: the 0.7.2 form) and the
RESIDENT form (``'auto'``), and prints for each: regs, stack/spill/local bytes, a check of the values against NumPy, and ns/sample
at each size, the two forms launched alternately (``--reps``, median) so a load drift hits both. ``ratio`` = register ns / resident
ns: above 1 the resident form is faster. Patterns: rtread (runtime-index reads of a built value), fillread (a loop-written value read at a runtime index by another loop), loopwrite (a component written in a
loop), rotate (in-place rotate-add by a runtime count), rtwrite (two writes at runtime indices)."""

import argparse
import importlib.util
import os
import pathlib
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

PATTERNS = ("rtread", "fillread", "loopwrite", "rotate", "rtwrite")

TEMPLATE = '''
import hawk
import hawk.math as m
from hawk import Mutable, Scalar, Vector


def parts(a, n: int):
    return m.vec([a * (j + 1.5) for j in range(n)])


def zeros(n: int):
    return m.vec([0.0 for _ in range(n)])


def doubled(x, n: int):
    return m.vec([x[j] * 2.0 for j in range(n)])
'''

BODIES = {
    "rtread": '''
@hawk.kernel
def rtread{n}_{f}(a: Scalar, s: Scalar, y: Mutable[Scalar]):
    r = parts(a, {n})
    k = m.index(s, {n})
    acc = 0.0
    for j in range({n}):
        acc = acc + r[(j + k) % {n}] * (j + 1.0)
    y = acc
''',
    "fillread": '''
@hawk.kernel
def fillread{n}_{f}(a: Scalar, s: Scalar, y: Mutable[Vector[{n}]]):
    r = zeros({n})
    for j in range({n}):
        r[j] = a * (j + 1.5)
    k = m.index(s, {n})
    q = zeros({n})
    for j in range({n}):
        q[j] = r[(j + k) % {n}]
    y = q
''',
    "loopwrite": '''
@hawk.kernel
def loopwrite{n}_{f}(a: Scalar, y: Mutable[Vector[{n}]]):
    r = zeros({n})
    for j in range({n}):
        r[j] = a * (j + 1.5)
    y = r
''',
    "rotate": '''
@hawk.kernel
def rotate{n}_{f}(s: Scalar, x: Vector[{n}], y: Mutable[Vector[{n}]]):
    k = m.index(s, {n})
    q = doubled(x, {n})
    for j in range({n}):
        q[j] = q[j] + q[(j + k) % {n}]
    y = q
''',
    "rtwrite": '''
@hawk.kernel
def rtwrite{n}_{f}(a: Scalar, s: Scalar, y: Mutable[Vector[{n}]]):
    k = m.index(s, {n})
    r = zeros({n})
    r[k] = a
    r[(k + 1) % {n}] = a * 2.0
    y = r
''',
}


USES = {"rtread": ("a", "s", "y"), "fillread": ("a", "s", "y"), "loopwrite": ("a", "y"), "rotate": ("s", "x", "y"), "rtwrite": ("a", "s", "y")}


def _want(pattern, n, real, a, s, x):
    lanes = np.arange(a.shape[0])
    k = s.astype(int)
    if pattern == "rtread":
        r = a[None, :] * (np.arange(n, dtype=real)[:, None] + real(1.5))
        return sum(r[(j + k) % n, lanes] * real(j + 1.0) for j in range(n)).astype(real)
    if pattern == "fillread":
        r = a[None, :] * (np.arange(n, dtype=real)[:, None] + real(1.5))
        return np.stack([r[(j + k) % n, lanes] for j in range(n)]).astype(real)
    if pattern == "loopwrite":
        return (a[None, :] * (np.arange(n, dtype=real)[:, None] + real(1.5))).astype(real)
    if pattern == "rotate":
        q = (x * real(2.0)).astype(real)
        for j in range(n):
            q[j] = q[j] + q[(j + k) % n, lanes]
        return q
    r = np.zeros((n, a.shape[0]), real)
    r[k, lanes] = a
    r[(k + 1) % n, lanes] = a * real(2.0)
    return r


def _auto(kernel, dtype, cache, form):
    """``eagle.plan.auto`` with the build in ``form`` (the plan does not take a placement option: its build is wrapped)."""
    import functools

    import eagle._plan_deploy as pd
    import eagle.plan

    real = pd._hawk_api

    def api():
        build, *rest = real()
        return (functools.partial(build, wide_form=form), *rest)

    pd._hawk_api = api
    try:
        return eagle.plan.auto(kernel, targets=("cuda",), scalar_type=dtype, cache_dir=str(cache))
    finally:
        pd._hawk_api = real


def _load(path, tag, form):
    """The kernels of the generated module ``path``, traced under ``form``."""
    from hawk.emit import policy

    spec = importlib.util.spec_from_file_location(f"_probe_{tag}_{form}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    with policy.using(wide_form=form):
        spec.loader.exec_module(mod)
    return mod


#: The probe pattern each threshold of the class table is measured by (``loopwrite`` is measured only to show the register form never
#: takes local memory for it).
TABLE_PATTERN = {"rtread": "read", "fillread": "slot_read", "rotate": "rotate", "rtwrite": "write"}
TIE = 0.02


def dump_table(rows, arch) -> str:
    """The ``CLASSES`` entry the sweep ``rows`` imply for ``arch``. A pattern's threshold, by scalar bytes, is the widest width up to which
    the register form is preferred at every width: faster than resident (ratio below 1 - TIE), or a tie (within TIE) that spills
    nothing. Past the first width resident is preferred, the threshold stops. Values are what the sweep measured: widths it did not
    cover are not extrapolated, and ``wide_min`` (not measured here) is copied from the nearest known class, marked."""
    from hawk.ir import wide_class

    table = {}
    for dt in sorted({r[0] for r in rows}):
        for pat, name in TABLE_PATTERN.items():
            mine = sorted((r for r in rows if r[0] == dt and r[1] == pat), key=lambda r: r[2])
            if not mine:
                continue
            kept = 0 if mine[0][2] > 1 else 1
            for width, local, ok, ratio in ((r[2], r[3], r[4], r[5]) for r in mine):
                registers = ratio < 1 - TIE or (ratio <= 1 + TIE and local == 0)
                if not (registers and ok):
                    break
                kept = width
            table.setdefault(name, {})[4 if dt == "float32" else 8] = kept
    near = wide_class.CLASSES[wide_class.select(arch)][1].wide_min
    feats = wide_class.features_of(arch)
    return (f"CLASSES[{arch.replace('_', '')!r}] = ({feats!r}, Tables(\n    runtime_min={table!r},\n"
            f"    wide_min={near!r}))   # wide_min NOT measured: copied from the nearest class")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dtype", default="float64", help="float32, float64 or both (comma list)")
    ap.add_argument("--widths", default="2,3,4,6,8,12,16,24,32")
    ap.add_argument("--patterns", default=",".join(PATTERNS))
    ap.add_argument("--sizes", default="65536")
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--arch", default="sm_61")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--dump-table", action="store_true",
                    help="after the sweep, print the class table it implies (hawk.ir.wide_class.CLASSES entry for --arch)")
    ap.add_argument("--no-run", action="store_true", help="build and report resources only (no device)")
    args = ap.parse_args(argv)
    widths = [int(w) for w in args.widths.split(",")]
    patterns = args.patterns.split(",")
    sizes = [int(s) for s in args.sizes.split(",")]
    cache = pathlib.Path(os.environ.get("HAWK_PROBE_CACHE", pathlib.Path(tempfile.gettempdir()) / "hawk_probe_cache"))
    work = pathlib.Path(tempfile.mkdtemp(prefix="probe_", dir=os.environ.get("TMPDIR")))

    # the resident form is measured FORCED (every runtime-indexed vector placed), whatever the shipped table says
    from hawk.ir import wide_class

    real_select = wide_class.select
    zero = {pattern: {4: 0, 8: 0} for pattern in ("read", "slot_read", "rotate", "write")}
    wide_class.CLASSES["probe"] = (wide_class.features_of(args.arch), wide_class.Tables(zero, {4: 128, 8: 64}))
    wide_class.select = lambda *a, **k: "probe"
    mods = {}
    for form in ("thread", "auto"):        # a kernel's name keys eagle's plugin: the two forms must not share one
        src = TEMPLATE + "".join(BODIES[p].format(n=n, f=form) for p in patterns for n in widths)
        path = work / f"probe_kernels_{form}.py"
        path.write_text(src)
        mods[form] = _load(path, "k", form)

    from hawk.artifact import build_bundle

    import json

    def build(dtype, pattern, n, form):
        kernel = getattr(mods[form], f"{pattern}{n}_{form}")
        d = work / f"{dtype}_{pattern}{n}_{form}"
        b = build_bundle([kernel], d, mode=dtype, targets=("cuda",), device_arch=args.arch, cache_dir=cache / form,
                         wide_form=form)
        side = json.loads((pathlib.Path(b.directory) / f"{kernel.name}.json").read_text())
        row = side["ptxas"][kernel.name]
        return {"regs": row["regs"], "stack": row["stack_B"], "spill": row["spill_store_B"], "local": side["launch"]["local_bytes"],
                "placed": len(side["launch"].get("wide", {}))}

    rows = []        # (dtype, pattern, width, register-form local bytes, resident ok, ratio at the last size)
    cases = [(dt, p, n) for dt in args.dtype.split(",") for p in patterns for n in widths]
    jobs = [(c, f) for c in cases for f in ("thread", "auto")]
    with ThreadPoolExecutor(args.jobs) as pool:
        res = dict(zip(jobs, pool.map(lambda j: build(*j[0], j[1]), jobs)))

    head = f"{'dtype':8}{'pat':10}{'w':>3} | {'regs t/a':>9} {'local t/a':>10} {'spill t/a':>10} | values"
    print(head + "".join(f" | ns/smp thr,res,ratio @{s}" for s in sizes))
    if not args.no_run:
        from eagle import arrays
        from eagle._alloc import synchronize

        def up(a):
            return arrays.asarray(np.ascontiguousarray(a), device="cuda:0")

    for dt, p, n in cases:
        t, a_ = res[(dt, p, n), "thread"], res[(dt, p, n), "auto"]
        line = f"{dt:8}{p:10}{n:>3} | {t['regs']:>4}/{a_['regs']:<4} {t['local']:>5}/{a_['local']:<4} {t['spill']:>5}/{a_['spill']:<4} | placed {a_['placed']} |"
        if args.no_run:
            print(line)
            continue
        real = np.float32 if dt == "float32" else np.float64
        ok, cols = True, ""
        for size in sizes:
            rng = np.random.default_rng(3)
            a = rng.integers(0, 5, size).astype(real)
            s = rng.integers(0, n, size).astype(real)
            x = rng.integers(0, 8, (n, size)).astype(real)
            want = _want(p, n, real, a, s, x)
            shape = want.shape
            bounds = {}
            for form in ("thread", "auto"):
                kernel = getattr(mods[form], f"{p}{n}_{form}")
                plan = _auto(kernel, dt, cache / form, form)
                have = {"a": a, "s": s, "x": x, "y": np.zeros(shape, real)}
                kw = {k: up(have[k]) for k in USES[p]}
                bounds[form] = (plan.bind(**kw), kw)
                bounds[form][0].launch()
                synchronize()
                ok &= bool(np.array_equal(np.asarray(kw["y"].get()), want))
            times = {"thread": [], "auto": []}
            for _ in range(args.reps):
                for form in ("thread", "auto"):
                    synchronize()
                    t0 = time.perf_counter()
                    bounds[form][0].launch()
                    synchronize()
                    times[form].append(time.perf_counter() - t0)
            mt, ma = (statistics.median(times[f]) * 1e9 / size for f in ("thread", "auto"))
            cols += f" | {mt:7.2f} {ma:7.2f} {mt / ma:5.2f}"
        print(f"{line} {'ok' if ok else 'WRONG':5}{cols}", flush=True)
        rows.append((dt, p, n, t["local"], ok, mt / ma))
    if args.dump_table:
        wide_class.select = real_select
        print(dump_table(rows, args.arch))


if __name__ == "__main__":
    main()
