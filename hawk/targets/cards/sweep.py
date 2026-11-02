# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The wide sweep: measure the forms of a wide per-lane value and write the card rows that choose among them.

``python -m hawk.targets.cards.sweep [--dtype f32 f64] [--out FILE] [--quick] [--write]``

On the device it runs: a kernel that accumulates ``fold`` rows of a table into a vector of ``width`` elements, built in
each form (``thread``: the lane's own registers; ``team``: a warp per lane; ``staged``: a warp per lane, a block serving
several of them, outputs staged through shared memory), timed as the corpus gate times a launch; and the contraction shape
of the thread box for T in 2..16. It writes candidate rows for ``wide.json`` and ``thread_box.json`` to ``--out`` (never in
place unless ``--write``), with the timings, the registers, the local bytes and the spills that back each row. A form
within 10 % of the best at a point is a tie (``tie: true`` on the winner's row, for the cards' second pass to decide).

It needs a repository checkout (the corpus helpers live in ``tests/``) and a CUDA device; it is a tool, not a public name.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

from ... import HawkError

#: Forms measured, in the order a tie is reported.
FORMS = ("thread", "team", "staged")

#: A form within this fraction of the best is a tie.
TIE = 0.10

WIDTHS = (32, 64, 128, 256, 512)
FOLDS = (32, 128, 512)
QUICK_WIDTHS = (64, 128, 256)
QUICK_FOLDS = (128,)
BOXES = (2, 4, 8, 16)
LANES = 4096
BOX_LANES = 262144       # the box shape is small per lane: enough lanes that the launch is the work


def plan(dtypes: tuple, quick: bool) -> list:
    """The cells ``(form, width, fold, dtype)`` the sweep measures."""
    widths, folds = (QUICK_WIDTHS, QUICK_FOLDS) if quick else (WIDTHS, FOLDS)
    return [(form, w, f, d) for d in dtypes for w in widths for f in folds for form in FORMS]


def winners(cells: list) -> list:
    """The best form per ``(dtype, width, fold)`` of measured ``cells`` (dicts with form, width, fold, dtype, ns): the
    fastest, and with it the forms within :data:`TIE` of it (``tied``, the fastest included)."""
    by_point: dict = {}
    for c in cells:
        by_point.setdefault((c["dtype"], c["width"], c["fold"]), []).append(c)
    out = []
    for (dtype, width, fold), group in sorted(by_point.items()):
        best = min(group, key=lambda c: c["ns"])
        tied = [c["form"] for c in group if c["ns"] <= best["ns"] * (1 + TIE)]
        out.append({"dtype": dtype, "width": width, "fold": fold, "form": best["form"], "ns": best["ns"],
                    "tied": tied, "tie": len(tied) > 1})
    return out


def wide_row(cc: str, device: str, dtype: str, won: list, defaults: dict) -> dict:
    """The ``wide.json`` row for one scalar type from its :func:`winners`: SLICE_MIN is the narrowest width at which a warp
    form beats the thread form (at the longest fold measured), TEAM_MIN the shortest fold at which it does (at the
    widest width). A warp form beats the thread form when it is not in the thread form's tie. A threshold nothing
    reached is the longest swept value doubled, which keeps the form out; TC stays the default."""
    mine = [w for w in won if w["dtype"] == dtype]
    top_fold, top_width = max(w["fold"] for w in mine), max(w["width"] for w in mine)

    def warp_wins(w):
        return w["form"] != "thread" and "thread" not in w["tied"]

    widths = sorted(w["width"] for w in mine if w["fold"] == top_fold and warp_wins(w))
    folds = sorted(w["fold"] for w in mine if w["width"] == top_width and warp_wins(w))
    return {"cc": cc, "device": device, "dtype": dtype,
            "SLICE_MIN": widths[0] if widths else 2 * top_width,
            "TEAM_MIN": folds[0] if folds else 2 * top_fold,
            "TC": defaults["TC"],
            "tie": any(w["tie"] for w in mine)}


def box_row(cc: str, device: str, dtype: str, timed: list) -> dict:
    """The ``thread_box.json`` row from ``timed`` (``{"T", "ns"}`` per box width): the fastest T, ``tie`` when another is
    within :data:`TIE` of it."""
    best = min(timed, key=lambda t: t["ns"])
    return {"cc": cc, "device": device, "dtype": dtype, "T": best["T"],
            "tie": any(t is not best and t["ns"] <= best["ns"] * (1 + TIE) for t in timed)}


# ---- the device side ----------------------------------------------------------------------------------------------------

def _corpus_helpers():
    """The repository's corpus helpers (``tests/_probe_wide.py``, the gate's launch timing)."""
    tests = Path(__file__).resolve().parents[3] / "tests"
    if not (tests / "_probe_wide.py").is_file():
        raise HawkError("the wide sweep needs a checkout of the repository (tests/_probe_wide.py): run it from one")
    sys.path.insert(0, str(tests))
    import _probe_wide

    return _probe_wide


_SOURCE = """import hawk
import hawk.math as m
from hawk import Mutable, Table, Vector


def zeros(n):
    return m.vec([0.0 for _ in range(n)])


def row_vec(tbl, r, n):
    return m.vec([tbl.at(o=r, i=i) for i in range(n)])


def sweep_{width}_{fold}(s: Vector[{fold}], w: Table["o":{fold}, "i":{width}], y: Mutable[Vector[{width}]]):
    acc = zeros({width})
    for r in range({fold}):
        acc = acc + s[r] * row_vec(w, r, {width})
    y = acc


kernel = hawk.kernel(sweep_{width}_{fold})
"""


def _kernel(width: int, fold: int, where: Path):
    """``y[k] = sum over r < fold of s[r] * w[r, k]``: ``width`` outputs, a fold of ``fold`` terms. A kernel is traced from its
    source, so each is written to a module under ``where`` and imported."""
    import importlib.util

    where.mkdir(parents=True, exist_ok=True)
    path = where / f"sweep_{width}_{fold}.py"
    path.write_text(_SOURCE.format(width=width, fold=fold))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kernel


def _time(bound, synchronize, repeats: int) -> float:
    bound.launch()
    synchronize()
    times = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        bound.launch()
        synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times) * 1e9 / LANES


def measure(cells: list, cache: Path, repeats: int) -> list:
    """Each cell of ``cells`` built and timed on the device: ns per sample, registers, local bytes, spill bytes."""
    import numpy as np
    from eagle import arrays
    from eagle._alloc import synchronize

    from ...artifact import build_bundle
    from . import wide_override

    probe = _corpus_helpers()
    out = []
    warnings.filterwarnings("ignore", "hawk: kernel .* local frame", UserWarning)     # a form that spills is a result, not a defect
    for form, width, fold, dtype in cells:
        real = np.float32 if dtype == "float32" else np.float64
        kernel = _kernel(width, fold, cache / "kernels")
        # below the committed SLICE_MIN the sweep still slices; 'team' is a block per warp, 'staged' the card's tile budget
        force = {"SLICE_MIN": 32, "TEAM_MIN": 32, **({"TC": 1} if form == "team" else {})}
        with wide_override(force):
            bundle = build_bundle([kernel], cache / f"{form}_{width}_{fold}_{dtype}", targets=("cuda",), mode=dtype,
                                  cache_dir=cache / "units", wide_form="thread" if form == "thread" else "team")
            side = next(a.sidecar for a in bundle.artifacts if a.name == kernel.name)
            row = next(iter((side.get("ptxas") or {}).values()), {})
            plan_ = probe._auto(kernel, dtype, cache / "units", "thread" if form == "thread" else "team")
            rng = np.random.default_rng(1)
            planes = {"s": rng.integers(-3, 4, (fold, LANES)).astype(real), "w": rng.integers(-3, 4, (fold, width)).astype(real),
                      "y": np.zeros((width, LANES), real)}
            bound = plan_.bind(**{k: arrays.asarray(np.ascontiguousarray(v), device="cuda:0") for k, v in planes.items()})
            ns = _time(bound, synchronize, repeats)
        out.append({"form": form, "width": width, "fold": fold, "dtype": dtype, "ns": ns,
                    "regs": row.get("regs"), "local_B": side["launch"].get("local_bytes"), "spill_B": row.get("spill_store_B")})
    return out


def measure_boxes(dtype: str, cache: Path, repeats: int) -> list:
    """The contraction shape of the thread box (``tests/test_ir_lanes.kg_fwd``) timed for T in :data:`BOXES`."""
    import json as _json

    import numpy as np

    from ...artifact import build_bundle
    from . import THREAD_BOX_FILE
    import hawk.targets.cards as cards

    probe = _corpus_helpers()
    del probe
    from conftest import _launch
    from test_ir_lanes import BATCH, L_ALL, N_IN, S_ROW, kg_fwd

    import cupy

    real = np.float32 if dtype == "float32" else np.float64
    rows = max(1, BOX_LANES // BATCH)
    n = rows * BATCH
    rng = np.random.default_rng(rows)
    feat = rng.uniform(-1, 1, 8 * L_ALL).astype(real)
    feat[:L_ALL] = rng.integers(0, 5, L_ALL)
    th = rng.standard_normal(rows * N_IN * S_ROW).astype(real)
    original, timed = THREAD_BOX_FILE, []
    doc = _json.loads(original.read_text())
    try:
        for T in BOXES:
            doc["defaults"][dtype] = T
            path = cache / f"thread_box_{dtype}_{T}.json"
            path.write_text(_json.dumps(doc))
            cards.THREAD_BOX_FILE = path
            bundle = build_bundle([kg_fwd], cache / f"box_{dtype}_{T}", mode=dtype, targets=("cuda",))
            planes = {"lane": cupy.arange(n, dtype=np.int32), "feat": cupy.asarray(feat), "th": cupy.asarray(th),
                      "y": cupy.zeros(n, dtype=real)}
            bound = _launch(bundle, "kg_fwd", "cuda", planes)
            timed.append({"T": T, "ns": _time(bound, cupy.cuda.Device().synchronize, repeats) * LANES / n})   # _time divides by LANES
    finally:
        cards.THREAD_BOX_FILE = original
    return timed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hawk.targets.cards.sweep", description=__doc__.split("\n\n")[0])
    ap.add_argument("--dtype", nargs="+", default=["f32", "f64"], choices=["f32", "f64"])
    ap.add_argument("--out", type=Path, default=Path("wide_sweep.json"), help="where the candidate rows go")
    ap.add_argument("--quick", action="store_true", help="a small grid (about two minutes)")
    ap.add_argument("--write", action="store_true", help="also write the rows into wide.json and thread_box.json")
    ap.add_argument("--cache", type=Path, default=Path("wide_sweep_cache"))
    args = ap.parse_args(argv)
    from ...compile import nvrtc
    from . import THREAD_BOX_FILE, WIDE_FILE

    dtypes = tuple({"f32": "float32", "f64": "float64"}[d] for d in args.dtype)
    cc, device = nvrtc.current_device_arch(), nvrtc.current_device_name()
    args.cache.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cells = measure(plan(dtypes, args.quick), args.cache, 5 if args.quick else 9)
    won = winners(cells)
    defaults = json.loads(WIDE_FILE.read_text())["defaults"]
    wide_rows = [wide_row(cc, device, d, won, defaults) for d in dtypes]
    boxes = {d: measure_boxes(d, args.cache, 5 if args.quick else 9) for d in dtypes}
    box_rows = [box_row(cc, device, d, t) for d, t in boxes.items()]
    result = {"cc": cc, "device": device, "wall_s": round(time.perf_counter() - started, 1), "quick": args.quick,
              "wide": {"rows": wide_rows}, "thread_box": {"rows": box_rows}, "cells": cells, "winners": won, "boxes": boxes}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if args.write:
        for path, rows in ((WIDE_FILE, wide_rows), (THREAD_BOX_FILE, box_rows)):
            doc = json.loads(path.read_text())
            keep = [r for r in doc["rows"] if (r["cc"], r["device"]) != (cc, device) or r["dtype"] not in dtypes]
            doc["rows"] = keep + [{k: v for k, v in r.items() if k != "tie"} for r in rows]
            path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wide sweep on {device} ({cc}): {len(cells)} cells in {result['wall_s']} s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
