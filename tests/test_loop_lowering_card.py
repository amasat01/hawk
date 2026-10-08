# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""CARD_LOOP_LOWERING — what lowering a bounded ``for`` costs to COMPILE,
written by the run.

THE NUMBER THIS CARD EXISTS FOR. A 64-edge x 23-basis KAN cell, fully
unrolled by executing the loop at trace time, reached ptxas as ONE function
and took 25-27 GB of RSS on a 31 GB box — it did not assemble at all under a
22 GiB cap. Lowering that body to a real ``for`` instead means the SAME
point now assembles inside a 12 GiB address-space cap, the device compile is
measured with ``/usr/bin/time -v`` for its own peak RSS, and the 64x23 row is
asserted to have SUCCEEDED — a card whose largest point was quietly skipped
would be evidence of nothing.

WHAT IT DOES NOT DO. It decides nothing and gates no threshold: whether to
unroll is the C++/CUDA compiler's decision, and this card is the evidence
that HAWK hands it a translation unit it can make that decision about. There
is no cap on op counts anywhere in HAWK and none is derived here.

WHY THERE IS NO "UNROLLED TWIN" AT 64x23. The unroll-by-execution mechanism
was removed, and rebuilding it even as a control is exactly the thing that
took the machine down. The numerics arm instead compares the LOWERED
kernel's compiled answer against ``tests/_eval.py``, the scratch
interpreter, which evaluates the Python loop iteration by iteration and is
therefore the unrolled reading of the same body — and, at a size small
enough to write out by hand, against a straight-line twin authored term by
term. Each comparison is recorded as bit-identical or banded WITH ITS
REASON.

Earlier on, the whole capability was absent — a bounded ``for`` was refused
outright by ``hawk/trace/astpass.py`` after the deletion, so this module
could not build a single subject::

    hawk.ir.nodes.HawkError: …: a bounded `for` is NOT unrolled by executing
    it any more; loop lowering lands elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from _cards import card_path
from _eval import evaluate

import hawk
import hawk.math as m
from hawk.artifact.layout import exports
from hawk.compile import (
    aether_include,
    compiler_identity,
    device_compiler,
    eagle_include,
    host_compiler,
    host_flags,
)
from hawk.compile.drivers import ADDRESS_SPACE_CAP_BYTES
from hawk.compile.toolchain import resolve_arch, device_flags, subprocess_env
from hawk.emit import BACKENDS, FLOAT64, render_source
from hawk.ir import Loop

CARD = card_path("CARD_LOOP_LOWERING.md")

#: The (edges, basis) grid, ending at the point that did not assemble before.
GRID = ((8, 5), (32, 11), (64, 23))

#: The loop rows' own bounds.
A2_BOUNDS = (4, 16, 64)

#: The cap every compile below runs under, in KiB for ``ulimit -v`` — the same
#: 12 GiB ``hawk.compile.drivers`` installs with ``RLIMIT_AS``.
ULIMIT_KIB = 12_000_000

#: The KAN-scale end-to-end point: a ``kan_edge_kernel_loop`` shape, with
#: the DEEP Cox-de Boor body the ``GRID`` family above deliberately does
#: not have.
KAN_SCALE = (64, 20, 3)

#: The cap the KAN-scale child runs under, in KiB (6 GB). Tighter than
#: ``ULIMIT_KIB`` on purpose: it caps the TRACER as well as the compilers, and
#: the pre-keying of this exact body exhausted it — so the row is a real
#: gate on canonicalisation's memory, not only on ptxas's.
KAN_SCALE_ULIMIT_KIB = 6_000_000

_RSS = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")

#: How many samples the compiled host kernel is run over for the timing column.
N = 512


def kan_cell(n_in: int, basis: int):
    """The synthetic KAN edge cell: one lane per sample, ``n_in`` edges, each a
    ``basis``-term cosine expansion, folded through a nonlinearity.

    Two NESTED lowered loops — the shape exhibit has, at the sizes that
    broke it. Every trace-time constant (both bounds, both declared strides) is
    a literal in the emitted TU, which is what lets the compiler decide for
    itself whether any of it is worth unrolling."""
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


def _reference(x, theta, g, n_in, basis):
    """The same arithmetic in numpy, in the loop's own association order."""
    total = 0.0
    for e in range(n_in):
        acc = 0.0
        for b in range(basis):
            acc = acc + theta[e * basis + b] * np.cos(g * (b + 1) * x[e])
        total = total + np.tanh(acc)
    return total


def _source(kernel, backend: str, tmp: Path) -> Path:
    src = render_source(kernel.name, kernel.sinks, kernel.walk, BACKENDS[backend],
                        mode=FLOAT64, exports=exports(backend))
    path = tmp / f"{kernel.name}.{'cu' if backend == 'cuda' else 'cpp'}"
    path.write_text(src.text)
    return path


def _capped(argv: list) -> tuple:
    """Run ``argv`` under ``/usr/bin/time -v`` AND ``ulimit -v``; return
    ``(returncode, wall seconds, peak RSS in KiB, stderr)``.

    ``ulimit`` is set in the shell rather than through the driver's own
    ``preexec_fn`` on purpose: this row is measuring the compiler, so the cap it
    measures under must be the one a reader can reproduce from the card by
    typing the same line."""
    quoted = " ".join(f"'{a}'" for a in argv)
    line = f"ulimit -v {ULIMIT_KIB}; exec /usr/bin/time -v {quoted}"
    start = time.perf_counter()
    done = subprocess.run(["/bin/sh", "-c", line], capture_output=True, text=True,
                          env=subprocess_env())
    wall = time.perf_counter() - start
    found = _RSS.search(done.stderr)
    return done.returncode, wall, int(found.group(1)) if found else -1, done.stderr


def _grid_row(n_in: int, basis: int, tmp: Path) -> dict:
    kernel = kan_cell(n_in, basis)
    loops = [n for n in kernel.walk.order if isinstance(n, Loop)]
    device = _source(kernel, "cuda", tmp)
    host = _source(kernel, "host", tmp)
    d_rc, d_wall, d_rss, d_err = _capped(
        [device_compiler(), *device_flags(), str(device),
         "-o", str(tmp / f"{kernel.name}.ptx")])
    h_rc, h_wall, h_rss, h_err = _capped(
        [host_compiler(), *host_flags(), str(host),
         "-o", str(tmp / f"{kernel.name}.so")])
    assert d_rc == 0, f"device compile of {n_in}x{basis} failed:\n{d_err[-3000:]}"
    assert h_rc == 0, f"host compile of {n_in}x{basis} failed:\n{h_err[-3000:]}"
    ptx = tmp / f"{kernel.name}.ptx"
    return {
        "edges": n_in,
        "basis": basis,
        "walk_nodes": len(kernel.walk.order),
        "top_level_loops": len(loops),
        "device_wall_s": round(d_wall, 3),
        "device_peak_rss_kib": d_rss,
        "device_ptx_bytes": ptx.stat().st_size if ptx.is_file() else -1,
        "device_tu_bytes": device.stat().st_size,
        "host_wall_s": round(h_wall, 3),
        "host_peak_rss_kib": h_rss,
        "host_tu_bytes": host.stat().st_size,
    }


def _numerics_row(n_in: int, basis: int, tmp_path, cache_dir) -> dict:
    """The compiled host answer against the interpreter's, at one grid point."""
    import _oracle as O

    from hawk.artifact import build

    kernel = kan_cell(n_in, basis)
    rng = np.random.default_rng(20260906)
    x = rng.normal(size=n_in)
    theta = rng.normal(size=n_in * basis)
    g = 0.9
    build(kernel, tmp_path / f"kan{n_in}x{basis}", targets=("host",),
          cache_dir=cache_dir)
    loaded = O.load(tmp_path / f"kan{n_in}x{basis}", kernel.name)
    start = time.perf_counter()
    got = O.run_kernel(loaded, N, x=x, theta=theta, g=g)
    run_wall = time.perf_counter() - start
    start = time.perf_counter()
    interpreted = float(evaluate(kernel.sinks,
                                 {"x": x, "theta": theta, "g": g})["y"])
    interp_wall = time.perf_counter() - start
    reference = float(_reference(x, theta, g, n_in, basis))
    spread = float(np.max(np.abs(np.asarray(got) - interpreted)))
    return {
        "edges": n_in,
        "basis": basis,
        "samples": N,
        "host_run_wall_s": round(run_wall, 4),
        "interpreter_wall_s": round(interp_wall, 4),
        "compiled_minus_interpreted_abs": spread,
        "interpreted_minus_numpy_abs": abs(interpreted - reference),
        "verdict": ("bit-identical" if spread == 0.0 else "banded"),
        "why": ("bit-identical: the compiled body evaluates the same operations "
                "in the same association order as the interpreter walks them"
                if spread == 0.0 else
                "banded: the emitted chain lets the compiler contract a "
                "multiply-add into an fma across iterations, which the "
                "interpreter's separate numpy operations cannot (the "
                "same low-bit consequence a fused ET already records elsewhere)"),
    }


def _a2_row(bound: int) -> dict:
    """One additive-loop row: primal AND vjp against the interpreter.

    The loop bounds are the rows' own (4/16/64) and the
    body is that arm's ``acc = acc + g * norm(pos)``, so the row measures what
    the additive-loop card measures, at IR level, without standing up a benchmark."""
    from hawk.diff import vjp

    @hawk.kernel
    def additive(pos: hawk.Vector[3], g: hawk.Param, y: hawk.Mutable[hawk.Scalar]):
        acc = 0.0
        for _k in range(bound):
            acc = acc + g * m.norm(pos)
        y = acc

    pos = np.array([0.3, -0.7, 1.1])
    env = {"pos": pos, "g": 1.3}
    primal = float(evaluate(additive.sinks, env)["y"])
    reference = 0.0
    for _ in range(bound):
        reference = reference + env["g"] * float(np.linalg.norm(pos))
    derived = vjp(additive)
    got = evaluate(derived, {**env, "bar_y": 1.0})
    d_g = float(got["bar_g"])
    d_pos = np.asarray(got["bar_pos"], dtype=float)
    want_g = bound * float(np.linalg.norm(pos))
    want_pos = bound * env["g"] * pos / float(np.linalg.norm(pos))
    return {
        "bound": bound,
        "primal": primal,
        "primal_minus_reference_abs": abs(primal - reference),
        "primal_verdict": "bit-identical" if primal == reference else "banded",
        "vjp_g": d_g,
        "vjp_g_minus_reference_abs": abs(d_g - want_g),
        "vjp_pos_max_abs_error": float(np.max(np.abs(d_pos - want_pos))),
        "vjp_verdict": ("bit-identical" if d_g == want_g else "banded"),
        "why": ("the reverse of an accumulator loop is a forward loop over the "
                "same range with NOTHING stored, so both arms sum "
                "the same terms in the same order; any difference is the "
                "association a compiler's fma contraction chooses"),
    }


def _twin_row() -> dict:
    """A straight-line TWIN of the 2x2 cell, written out term by term, against
    the lowered loop's IR at the same size.

    Hand-authored on purpose: the unroll-by-execution mechanism is deleted and
    is not being rebuilt even as a control, so the only honest twin is one
    a human wrote. The seeds differ — the loop starts from a literal ``0.0`` the
    twin has no reason to write — which is exactly the kind of association
    difference the verdict below has to name rather than hide."""
    n_in, basis = 2, 2

    @hawk.kernel
    def twin(x: hawk.Table["edge":n_in],
             theta: hawk.Table["edge":n_in, "basis":basis],
             g: hawk.Param,
             y: hawk.Mutable[hawk.Scalar]):
        x0 = x.at(edge=0)
        x1 = x.at(edge=1)
        a0 = (0.0 + theta.at(edge=0, basis=0) * m.cos(g * 1 * x0)
              + theta.at(edge=0, basis=1) * m.cos(g * 2 * x0))
        a1 = (0.0 + theta.at(edge=1, basis=0) * m.cos(g * 1 * x1)
              + theta.at(edge=1, basis=1) * m.cos(g * 2 * x1))
        y = (0.0 + m.tanh(a0)) + m.tanh(a1)

    rng = np.random.default_rng(4242)
    x = rng.normal(size=n_in)
    theta = rng.normal(size=n_in * basis)
    env = {"x": x, "theta": theta, "g": 0.9}
    looped = float(evaluate(kan_cell(n_in, basis).sinks, env)["y"])
    straight = float(evaluate(twin.sinks, env)["y"])
    return {
        "edges": n_in,
        "basis": basis,
        "looped": looped,
        "straight_line_twin": straight,
        "abs_difference": abs(looped - straight),
        "verdict": "bit-identical" if looped == straight else "banded",
        "why": ("the twin is written with the loop's own seeds and association "
                "order, so the two evaluate the identical expression tree; a "
                "difference here would mean the lowering changed the arithmetic "
                "rather than where it is written"),
    }


def _kan_scale_row() -> dict:
    """The KAN-scale cell, END TO END, inside ONE child under a 6 GB cap.

    Trace, canonicalise, emit, compile BOTH targets, build the host artifact,
    run it and evaluate the same IR with ``tests/_eval.py`` — all in the child,
    because ``ulimit -v`` caps a PROCESS and the point of the row is that every
    one of those steps fits inside the cap TOGETHER. ``RLIMIT_AS`` is inherited
    across ``exec``, so the compilers the child spawns are capped without being
    told, which is the same mechanism ``hawk.compile.drivers`` uses in anger.

    A prior body-keying scheme exhausted 6 GB inside the tracer before
    reaching a compiler (the largest single body key at the smaller (8, 5, 2)
    point was 106 505 567 bytes), so there was no TU to compile and no answer
    to compare; a fixed-width body key fixed that.

    A child that dies FAILS the row. The alternative — reporting the point as
    absent — is exactly the shape of evidence this test exists to stop."""
    tests = str(Path(__file__).resolve().parent)
    n_in, G, k = KAN_SCALE
    line = (f"ulimit -v {KAN_SCALE_ULIMIT_KIB}; "
            f"exec {sys.executable} {tests}/_kan.py {n_in} {G} {k}")
    done = subprocess.run(["/bin/sh", "-c", line], capture_output=True, text=True,
                          env=subprocess_env())
    assert done.returncode == 0, (
        f"the KAN-scale child died (rc={done.returncode}) under "
        f"ulimit -v {KAN_SCALE_ULIMIT_KIB}\n{done.stdout[-2000:]}\n"
        f"{done.stderr[-4000:]}")
    payload = [ln for ln in done.stdout.splitlines() if ln.startswith("{")]
    assert payload, f"the child printed no json:\n{done.stdout}\n{done.stderr}"
    row = json.loads(payload[-1])
    row["ulimit_v_kib"] = KAN_SCALE_ULIMIT_KIB
    return row


def test_the_loop_lowering_card_is_written(tmp_path, cache_dir):
    grid = [_grid_row(n_in, basis, tmp_path) for n_in, basis in GRID]
    numerics = [_numerics_row(n_in, basis, tmp_path, cache_dir)
                for n_in, basis in GRID]
    payload = {
        "card": "HAWK loop lowering: what a lowered `for` costs to compile",
        "produced": time.strftime("%Y-%m-%d"),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "host_compiler": compiler_identity(host_compiler()),
        "device_compiler": compiler_identity(device_compiler()),
        "device_arch": resolve_arch(),
        "aether_include": Path(aether_include()).name,
        "eagle_include": Path(eagle_include()).name,
        "address_space_cap_bytes": ADDRESS_SPACE_CAP_BYTES,
        "ulimit_v_kib": ULIMIT_KIB,
        "note": ("every compile in `grid` ran under `ulimit -v 12000000` and "
                 "was measured with `/usr/bin/time -v`; the 64x23 point is the "
                 "one HW7d could not assemble at all when the same body was "
                 "unrolled by trace-time execution (25-27 GB of ptxas RSS). "
                 "`kan_scale_cell` is the point: a downstream package's own "
                 "`kan_edge_kernel_loop` shape (a DEEP Cox-de Boor body, not "
                 "the shallow `grid` family) at (n_in=64, G=20, k=3), taken "
                 "trace -> canonicalise -> emit -> compile both targets -> run, "
                 "all inside ONE child under `ulimit -v 6000000`, which caps "
                 "the TRACER as well as the compilers. Before the "
                 "fixed-width body key that child exhausted 6 GB during "
                 "canonicalisation and never reached a compiler. The card "
                 "DECIDES NOTHING: rules that whether to unroll is the "
                 "compiler's decision, and no threshold is derived here"),
        "grid": grid,
        "numerics": numerics,
        "a2_additive_loops": [_a2_row(b) for b in A2_BOUNDS],
        "straight_line_twin": _twin_row(),
        "kan_scale_cell": _kan_scale_row(),
    }
    body = json.dumps(payload, indent=2, sort_keys=True)
    fence = hashlib.md5(body.encode()).hexdigest()
    CARD.parent.mkdir(parents=True, exist_ok=True)
    CARD.write_text(
        "# CARD_LOOP_LOWERING — a bounded `for`, lowered\n\n"
        "Written BY THE RUN (`tests/test_loop_lowering_card.py`). Prose cites "
        "this file; no number here may be restated without citing it. It "
        "DECIDES NOTHING — rules that whether to unroll a lowered loop is "
        "the C++/CUDA compiler's decision, and this card is the evidence that "
        "HAWK hands it a translation unit it can make that decision about.\n\n"
        "Every compile in `grid` ran under `ulimit -v 12000000` (the same 12 "
        "GiB `hawk.compile.drivers` installs with `RLIMIT_AS`) and was "
        "measured with `/usr/bin/time -v`. `kan_scale_cell` — the point, "
        "a downstream package's own KAN edge shape at (n_in=64, G=20, k=3) — went trace to run "
        "inside ONE child under `ulimit -v 6000000`, a cap that binds the "
        "TRACER too.\n\n"
        f"md5: `{fence}`\n\n```json\n{body}\n```\n"
    )
    text = CARD.read_text()
    assert f"md5: `{fence}`" in text
    assert hashlib.md5(text.split("```json\n")[1].split("\n```")[0]
                       .encode()).hexdigest() == fence


def test_the_largest_point_assembles_under_the_cap():
    """The claim the card exists to carry, asserted rather than eyeballed: the
    64x23 cell — the exact point that took 25-27 GB and failed under a 22 GiB
    cap when the same body was unrolled — compiles for BOTH targets inside 12
    GiB, and its device peak RSS is recorded."""
    assert CARD.is_file(), "run test_the_loop_lowering_card_is_written first"
    payload = json.loads(CARD.read_text().split("```json\n")[1].split("\n```")[0])
    biggest = payload["grid"][-1]
    assert (biggest["edges"], biggest["basis"]) == GRID[-1]
    assert biggest["device_peak_rss_kib"] > 0, biggest
    assert biggest["device_peak_rss_kib"] * 1024 < payload["ulimit_v_kib"] * 1024
    assert biggest["device_ptx_bytes"] > 0 and biggest["host_wall_s"] > 0


def test_the_ir_does_not_grow_with_the_grid():
    """The structural claim beside the timing one: a LOWERED loop's IR does not
    track its trip count, so the 64x23 cell has the same node count as the 8x5
    one. Under the old unroll-by-execution the largest point's chain was ~37x
    the smallest's, which is what reached ptxas as one function."""
    counts = {(n, b): len(kan_cell(n, b).walk.order) for n, b in GRID}
    assert len(set(counts.values())) == 1, counts


@pytest.mark.parametrize("point", GRID, ids=[f"{n}x{b}" for n, b in GRID])
def test_the_compiled_answer_matches_the_interpreter(point):
    """The numerics arm, read off the card the run just wrote: the compiled
    kernel and the scratch interpreter must agree to the band allows, and
    the card must SAY which of bit-identical / banded it was and why."""
    assert CARD.is_file(), "run test_the_loop_lowering_card_is_written first"
    payload = json.loads(CARD.read_text().split("```json\n")[1].split("\n```")[0])
    row = next(r for r in payload["numerics"]
               if (r["edges"], r["basis"]) == point)
    assert row["verdict"] in ("bit-identical", "banded")
    assert row["why"]
    assert row["compiled_minus_interpreted_abs"] < 1e-12, row


def test_the_kan_scale_cell_goes_end_to_end_inside_six_gigabytes():
    """The card's claim, asserted rather than eyeballed, off the card the run wrote.

    The (n_in=64, G=20, k=3) Cox-de Boor edge cell — the KAN-scale
    ``kan_edge_kernel_loop`` shape — traces, canonicalises, emits, compiles
    for BOTH targets and RUNS, all inside one 6 GB address space, and the
    compiled answer agrees with the scratch interpreter. A prior body-keying
    scheme let the same child get stuck in canonicalisation: the tracer
    exhausted the cap building ONE key, before a fixed-width body key fixed it.

    Three separate things are pinned because three separate things could
    regress: the memory (the tracer's own peak, and each compiler's), the
    structure (the loop is ONE node over a several-hundred-node body, not an
    unroll), and the arithmetic (against the interpreter, which walks the loop
    iteration by iteration and is therefore the unrolled reading of the same
    body)."""
    assert CARD.is_file(), "run test_the_loop_lowering_card_is_written first"
    payload = json.loads(CARD.read_text().split("```json\n")[1].split("\n```")[0])
    row = payload["kan_scale_cell"]
    assert (row["n_in"], row["G"], row["k"]) == KAN_SCALE
    cap = row["ulimit_v_kib"]
    assert row["trace_canonicalise_peak_rss_kib"] < cap // 10, row
    assert 0 < row["device_peak_rss_kib"] < cap, row
    assert 0 < row["host_peak_rss_kib"] < cap, row
    assert row["device_ptx_bytes"] > 0 and row["host_wall_s"] > 0
    # the IR does not track the trip count: ONE loop node, a body far larger
    # than the whole flat walk, and a walk that does not grow with n_in.
    assert row["top_level_loops"] == 1, row
    assert row["loop_body_nodes"] > 5 * row["walk_nodes"], row
    assert row["verdict"] in ("bit-identical", "banded") and row["why"]
    assert row["compiled_minus_interpreted_abs"] < 1e-12, row
    assert row["interpreted_minus_numpy_abs"] < 1e-12, row
