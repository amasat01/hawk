# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The KAN edge cell, at HAWK's own authoring surface — TEST SCAFFOLD.

ONE definition of the shape and were both ruled over, shared by the
memory row (``tests/test_ir_loop_digest_memory.py``) and the compile card
(``tests/test_loop_lowering_card.py``) so the two can never drift into
measuring different subjects and reporting them under one name.

WHY IT IS COPIED AND NOT IMPORTED. The real subject is a downstream
package's KAN edge kernel,
and ``hawk`` may not depend on it (the dependency runs the other way). So the shape is
reproduced here from HAWK's own vocabulary, with the two properties that made it
the subject preserved exactly:

* the ``for`` over EDGES is in the kernel body, so the ``ast`` pass lowers it to
  a real loop and the trip count never enters the IR;
* everything inside one edge — the Cox-de Boor basis recursion and the
  basis-term sum — is built by TRACE-TIME python in a helper, exactly as the
  downstream package's ``_edge_phi_loop_wide_hawk`` builds it, so the loop BODY is several hundred
  nodes deep. That depth is the whole point: a ``term1 + term2`` node whose two
  children are both large is what made an early body key double in size once
  per recursion level.

At ``(n_in=64, G=20, k=3)`` — the point measured here — the body is 727
nodes and the walk is 89.
"""

from __future__ import annotations

import hawk
import hawk.math as m

__all__ = ["knots", "basis_levels", "edge_phi", "kan_edge_cell",
           "shallow_kan_cell", "reference", "main"]


def knots(G: int, k: int) -> tuple:
    """``G + 2k + 1`` uniformly spaced knots extending ``[-1, 1]`` by ``k``
    intervals on each side — a downstream package's ``cox_de_boor_knots``. Uniform spacing makes
    every Cox-de Boor denominator the constant ``d * step``, so the recursion
    below needs no divide-by-zero guard."""
    step = 2.0 / G
    return tuple(-1.0 + (j - k) * step for j in range(G + 2 * k + 1))


def basis_levels(x, kn: tuple, k: int) -> list:
    """The order-``k`` Cox-de Boor basis at a traced scalar ``x``.

    The DEEP half of the body: each level's element combines TWO elements of the
    level below, so the dependency graph fans in at every one of the ``k``
    levels. Written against whatever ``x`` is — a traced value inside a lowered
    loop's scope, in this file's only caller."""
    level = [m.where(m.land(x >= kn[i], x < kn[i + 1]), 1.0, 0.0)
             for i in range(len(kn) - 1)]
    for d in range(1, k + 1):
        level = [
            ((x - kn[i]) / (kn[i + d] - kn[i])) * level[i]
            + ((kn[i + d + 1] - x) / (kn[i + d] - kn[i])) * level[i + 1]
            for i in range(len(level) - 1)
        ]
    return level


def edge_phi(x, theta, e, kn: tuple, k: int, n_basis: int):
    """One edge's spline value. A TRACE-TIME helper: a helper's body is ordinary
    Python the ``ast`` pass never sees, so its ``for`` runs once while the
    enclosing lowered ``for`` body is being traced and builds its chain INSIDE
    that scope — a downstream package's own ``_edge_phi_loop_wide_hawk`` arrangement."""
    phi = basis_levels(x, kn, k)
    spline = phi[0] * theta.at(edge=e, basis=0)
    for b in range(1, n_basis):
        spline = spline + phi[b] * theta.at(edge=e, basis=b)
    return spline


def kan_edge_cell(n_in: int, G: int, k: int):
    """The KAN edge cell at ``(n_in, G, k)``: ONE lowered ``for`` over edges
    whose body is a Cox-de Boor basis expansion folded through a nonlinearity."""
    kn, n_basis = knots(G, k), G + k

    @hawk.kernel
    def cell(x: hawk.Table["edge":n_in],
             theta: hawk.Table["edge":n_in, "basis":n_basis],
             g: hawk.Param,
             y: hawk.Mutable[hawk.Scalar]):
        total = 0.0
        for e in range(n_in):
            total = total + m.tanh(
                edge_phi(x.at(edge=e) * g, theta, e, kn, k, n_basis))
        y = total

    return cell


def reference(x, theta, g: float, n_in: int, G: int, k: int) -> float:
    """The same arithmetic in numpy, in the loop's own association order — the
    third arm beside the compiled kernel and the IR interpreter."""
    import numpy as np

    kn, n_basis = knots(G, k), G + k
    total = 0.0
    for e in range(n_in):
        xe = x[e] * g
        level = [1.0 if (xe >= kn[i] and xe < kn[i + 1]) else 0.0
                 for i in range(len(kn) - 1)]
        for d in range(1, k + 1):
            level = [
                ((xe - kn[i]) / (kn[i + d] - kn[i])) * level[i]
                + ((kn[i + d + 1] - xe) / (kn[i + d] - kn[i])) * level[i + 1]
                for i in range(len(level) - 1)
            ]
        spline = level[0] * theta[e * n_basis]
        for b in range(1, n_basis):
            spline = spline + level[b] * theta[e * n_basis + b]
        total = total + np.tanh(spline)
    return float(total)


def shallow_kan_cell(n_in: int, basis: int):
    """The card's ORIGINAL synthetic family — two nested lowered loops over a
    shallow body — kept beside the deep one so a row can cover both shapes."""
    @hawk.kernel
    def cell(x: hawk.Table["edge":n_in],
             theta: hawk.Table["edge":n_in, "basis":basis],
             g: hawk.Param,
             y: hawk.Mutable[hawk.Scalar]):
        total = 0.0
        for e in range(n_in):
            xe = x.at(edge=e)
            acc = 0.0
            for b in range(basis):
                acc = acc + theta.at(edge=e, basis=b) * m.cos(g * (b + 1) * xe)
            total = total + m.tanh(acc)
        y = total

    return cell


# --------------------------------------------------------------------------- #
# The capped child.
#
# WHY THIS ENTRY POINT EXISTS AT ALL. The KAN-scale point is the one that took
# a 31 GB box down before /, and rule is that a trace or
# canonicalisation of a body that size is exercised only behind an address-space
# cap. A `ulimit -v` is applied to a PROCESS, so the whole end-to-end — trace,
# canonicalise, emit, compile both targets, build the host artifact, run it,
# and evaluate the same IR with the scratch interpreter — has to happen in ONE
# child that the parent starts under that cap. `RLIMIT_AS` is inherited, so the
# compilers this child spawns run under the same ceiling without being told.
#
# It prints ONE json line, which `tests/test_loop_lowering_card.py` folds into
# CARD_LOOP_LOWERING.md. It measures; it decides nothing.
# --------------------------------------------------------------------------- #

#: Samples the compiled host kernel is run over for the timing column.
_SAMPLES = 256

#: Peak RSS, in KiB, out of `/usr/bin/time -v`'s own report. Read from the
#: COMPILER's own process rather than from `getrusage(RUSAGE_CHILDREN)`, which
#: reports the high-water mark over every child this process has ever reaped and
#: would therefore attribute the first compiler's peak to the second one too.
_RSS_LINE = "Maximum resident set size (kbytes):"


def _own_peak_kib() -> int:
    """THIS process's own peak RSS, in KiB, from `/proc/self/status`.

    NOT `getrusage(RUSAGE_SELF).ru_maxrss`, which is wrong here by two orders of
    magnitude and wrong in the flattering direction's opposite: this module runs
    as a CHILD forked from a pytest process holding ~330 MB, and the fork's
    high-water mark survives into the exec'd image's `ru_maxrss` — so the child
    reports its parent's footprint no matter how little it allocates (measured
    at 321 MiB by `getrusage` against 18 MiB of real usage, for the
    identical child). `VmHWM` is the mm's own high-water mark and the mm is
    replaced by `execve`, so it is this process's number and nobody else's.
    The COMPILER numbers above do not need this: `/usr/bin/time` forks them from
    itself, and itself is small."""
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    raise AssertionError("no VmHWM in /proc/self/status")


def _timed(argv: list) -> tuple:
    """Run ``argv`` under ``/usr/bin/time -v``; ``(rc, wall, peak KiB, stderr)``."""
    import subprocess
    import time

    from hawk.compile.toolchain import subprocess_env

    start = time.perf_counter()
    done = subprocess.run(["/usr/bin/time", "-v", *argv], capture_output=True,
                          text=True, env=subprocess_env())
    wall = time.perf_counter() - start
    peak = -1
    for line in done.stderr.splitlines():
        if _RSS_LINE in line:
            peak = int(line.split(":")[-1].strip())
    return done.returncode, wall, peak, done.stderr


def main(argv=None) -> int:
    """``python tests/_kan.py <n_in> <G> <k>`` — one KAN-scale end-to-end point."""
    import json
    import pathlib
    import sys
    import tempfile
    import time

    import _oracle as O
    import numpy as np
    from _eval import evaluate

    from hawk.artifact import build
    from hawk.artifact.layout import exports
    from hawk.compile import device_compiler, host_compiler, host_flags
    from hawk.compile.toolchain import resolve_arch, device_flags
    from hawk.emit import BACKENDS, FLOAT64, render_source
    from hawk.ir import Loop

    args = sys.argv[1:] if argv is None else list(argv)
    n_in, G, k = int(args[0]), int(args[1]), int(args[2])
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="hawk_kan_scale_"))

    start = time.perf_counter()
    cell = kan_edge_cell(n_in, G, k)          # traces AND canonicalises
    trace_wall = time.perf_counter() - start
    loops = [n for n in cell.walk.order if isinstance(n, Loop)]
    row = {
        "n_in": n_in, "G": G, "k": k, "n_basis": G + k,
        "trace_canonicalise_wall_s": round(trace_wall, 4),
        "trace_canonicalise_peak_rss_kib": _own_peak_kib(),
        "walk_nodes": len(cell.walk.order),
        "loop_body_nodes": sum(len(lp.body_nodes()) for lp in loops),
        "top_level_loops": len(loops),
        "device_arch": resolve_arch(),
    }

    sources = {}
    for backend in ("cuda", "host"):
        text = render_source(cell.name, cell.sinks, cell.walk, BACKENDS[backend],
                             mode=FLOAT64, exports=exports(backend)).text
        path = tmp / f"{cell.name}.{'cu' if backend == 'cuda' else 'cpp'}"
        path.write_text(text)
        sources[backend] = path
        row[f"{backend}_tu_bytes"] = path.stat().st_size

    ptx = tmp / f"{cell.name}.ptx"
    rc, wall, peak, err = _timed([device_compiler(), *device_flags(),
                                  str(sources["cuda"]), "-o", str(ptx)])
    if rc != 0:
        raise SystemExit(f"device compile failed (rc={rc}):\n{err[-3000:]}")
    row.update(device_wall_s=round(wall, 3), device_peak_rss_kib=peak,
               device_ptx_bytes=ptx.stat().st_size)

    rc, wall, peak, err = _timed([host_compiler(), *host_flags(),
                                  str(sources["host"]), "-o", str(tmp / "cell.so")])
    if rc != 0:
        raise SystemExit(f"host compile failed (rc={rc}):\n{err[-3000:]}")
    row.update(host_wall_s=round(wall, 3), host_peak_rss_kib=peak)

    rng = np.random.default_rng(20260906)
    x = rng.uniform(-0.9, 0.9, size=n_in)
    theta = rng.normal(size=n_in * (G + k))
    g = 0.9
    build(cell, tmp / "unit", targets=("host",), cache_dir=str(tmp / "cache"))
    loaded = O.load(tmp / "unit", cell.name)
    start = time.perf_counter()
    got = O.run_kernel(loaded, _SAMPLES, x=x, theta=theta, g=g)
    row["host_run_wall_s"] = round(time.perf_counter() - start, 4)
    start = time.perf_counter()
    interpreted = float(evaluate(cell.sinks, {"x": x, "theta": theta, "g": g})["y"])
    row["interpreter_wall_s"] = round(time.perf_counter() - start, 4)
    spread = float(np.max(np.abs(np.asarray(got) - interpreted)))
    row.update(
        samples=_SAMPLES,
        compiled_minus_interpreted_abs=spread,
        interpreted_minus_numpy_abs=abs(
            interpreted - reference(x, theta, g, n_in, G, k)),
        verdict="bit-identical" if spread == 0.0 else "banded",
        why=("bit-identical: the compiled body evaluates the same operations in "
             "the same association order the interpreter walks them"
             if spread == 0.0 else
             "banded: the emitted chain lets the compiler contract a "
             "multiply-add into an fma across the body's Cox-de Boor levels, "
             "which the interpreter's separate numpy operations cannot ("
             "the same low-bit consequence records for a fused ET)"),
        peak_rss_kib=_own_peak_kib(),
    )
    print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":                    # pragma: no cover - the child
    raise SystemExit(main())
