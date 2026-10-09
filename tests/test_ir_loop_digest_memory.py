# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""/: a loop body's structural key is FIXED-WIDTH, so canonicalisation is
linear in body size — the memory row, the key-width row, and the two properties
the fix was not allowed to cost.

WHAT WAS WRONG AND HOW IT WAS MEASURED. Before the fix, ``hawk/ir/loop_nodes.py``'s
``_body_digest`` keyed each loop-body node with the ``repr`` of a tuple that
embedded its CHILDREN'S KEY STRINGS verbatim. One key was therefore the entire
sub-expression beneath it written out, and a body whose dependency chain has
depth ``d`` cost O(2**d) BYTES to key — all of it spent inside the tracer,
before any compiler ran. Every RED number below was produced by putting that one
line back into ``hawk/ir/loop_nodes.py::_body_keys`` and running THESE rows against
it, so both arms measured the same subject through the same path; the plant is
not kept in the tree, because a keying nothing calls is a keying nothing keeps
honest.

Measured on this box (`ulimit -v 2000000`, i.e. 2 GB, one child process each):

=============================== ================== =====================
 subject legacy keying fixed-width keying
=============================== ================== =====================
 4-trip loop, depth-8 chain 0.13 s / 54 MiB 0.023 s / 19 MiB
 4-trip loop, depth-10 chain 2.22 s / 603 MiB 0.017 s / 18 MiB
 4-trip loop, depth-12 chain MemoryError 0.017 s / 18 MiB
 4-trip loop, depth-14 chain MemoryError 0.018 s / 18 MiB
=============================== ================== =====================

and, at a larger scale (`ulimit -v 6000000`), the Cox-de Boor edge cell this
file builds:

=============================== ============================ ==============
 subject legacy keying fixed width
=============================== ============================ ==============
 n_in=8, G=5, k=2 2.47 s / 629 MiB, 0.004 s,
 (180 body nodes) largest key 106 505 567 B largest key 32 B
 n_in=64, G=20, k=3 MemoryError 0.011 s,
 (727 body nodes) (exhausts 6 GB) largest key 32 B
=============================== ============================ ==============

WHY EVERY MEASUREMENT RUNS IN A CHILD UNDER ``ulimit -v``. A tracer that can
take the machine down must be exercised only behind a cap: the legacy keying
exhausted a 31 GB box on the second row of that table, and a row that reproduces
it inside the pytest process would take the whole suite with it. So the subject
is traced in a SUBPROCESS whose address space is capped with ``ulimit -v``, and
the child reports its own peak RSS out of ``/proc/self/status``'s ``VmHWM`` —
NOT ``getrusage``, which carries the forking parent's high-water mark into the
exec'd child and so reported 321 MiB for a child that really used 18. A child
that dies is a FAILING row, never a skipped one — a skipped memory row
certifies nothing.

WHAT THE FIX WAS NOT ALLOWED TO COST. The key is still STRUCTURAL, POSITIONAL
and PROCESS-STABLE: the last two rows pin exactly that, because a
"fixed-width key" that hashed with Python's own salted ``hash()`` would be
smaller and completely wrong — two processes would disagree about the digest of
one kernel, and the whole contract is that they do not.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from _kan import kan_edge_cell, shallow_kan_cell

import hawk
import hawk.math as m
from hawk.ir import Loop
from hawk.ir.loop_nodes import BODY_KEY_HEX, body_key_widths

#: The repo root a child process has to put on ``sys.path`` to import hawk.
HAWK_ROOT = str(Path(__file__).resolve().parent.parent)

#: The address-space cap the chain rows run their child under, in KiB (2 GB).
#: Chosen so the LEGACY keying dies at depth 12 — the row would be vacuous under
#: a cap so generous that both arms fit.
CHAIN_CAP_KIB = 2_000_000

#: The cap the KAN-scale rows use (6 GB): the legacy keying exhausts it at
#: (n_in=64, G=20, k=3), which is the point was ruled over.
KAN_CAP_KIB = 6_000_000

#: The width, in hex characters, one body key is allowed to be. The row asserts
#: against 128 BYTES rather than against this constant so that widening the
#: constant cannot silently widen the gate as well.
MAX_KEY_BYTES = 128


# --------------------------------------------------------------------------- #
# The subjects live in ``tests/_kan.py`` — ONE definition, shared with the
# compile card (``tests/test_loop_lowering_card.py``), so the memory row and the
# compile row can never drift into measuring different bodies under one name.
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Running one measurement inside a capped child.
# --------------------------------------------------------------------------- #
def _child(body: str, cap_kib: int, tmp_path: Path) -> dict:
    """Run ``body`` in a fresh interpreter under ``ulimit -v cap_kib`` and parse
    the ONE json line it prints. A non-zero return code, or no json, FAILS.

    The cap is applied by the shell rather than by ``resource.setrlimit`` in the
    driver's cap trampoline so that a reader can reproduce the row by typing the same
    line — the same reason ``tests/test_loop_lowering_card.py`` sets it that
    way for the compilers."""
    script = tmp_path / "probe.py"
    script.write_text(
        "import json, sys, time\n"
        f"sys.path.insert(0, {HAWK_ROOT!r})\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent)!r})\n"
        # THE CHILD'S OWN PEAK, and `getrusage` will not give it. `ru_maxrss`
        # is read from `signal_struct`, which SURVIVES `execve` — so a child
        # forked from a 330 MB pytest process reports 330 MB no matter how
        # little it goes on to allocate, and the whole row would be measuring
        # its parent. `VmHWM` in `/proc/self/status` is the mm's own high-water
        # mark and IS reset by exec, so it is this process's number and nobody
        # else's (measured: 18 MiB here, 321 MiB through `getrusage`, for the
        # identical child).
        "def _rss():\n"
        "    for line in open('/proc/self/status'):\n"
        "        if line.startswith('VmHWM:'):\n"
        "            return int(line.split()[1]) // 1024\n"
        "    raise AssertionError('no VmHWM in /proc/self/status')\n"
        + textwrap.dedent(body)
    )
    line = f"ulimit -v {cap_kib}; exec {sys.executable} {script}"
    done = subprocess.run(["/bin/sh", "-c", line], capture_output=True, text=True)
    assert done.returncode == 0, (
        f"the capped child died (rc={done.returncode}) under ulimit -v {cap_kib}"
        f"; a memory row that cannot run is a FAILING row, never a skipped one."
        f"\nstdout:\n{done.stdout[-2000:]}\nstderr:\n{done.stderr[-4000:]}"
    )
    payload = [ln for ln in done.stdout.splitlines() if ln.startswith("{")]
    assert payload, f"the child printed no json:\n{done.stdout}\n{done.stderr}"
    return json.loads(payload[-1])


_CHAIN_PROBE = '''
    DEPTH = {depth}

    src = (
        "import hawk\\n"
        "@hawk.kernel\\n"
        "def chain(tab: hawk.Table['row':4], y: hawk.Mutable[hawk.Scalar]):\\n"
        "    acc = 0.0\\n"
        "    for i in range(4):\\n"
        "        t = tab.at(row=i) + 1.0\\n"
        + "        t = t * t\\n" * DEPTH
        + "        acc = acc + t\\n"
        "    y = acc\\n"
    )
    import pathlib, importlib.util
    tmp = pathlib.Path({chain_dir!r})
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / "chain.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("chain_probe_mod", path)
    mod = importlib.util.module_from_spec(spec)
    t0 = time.time()
    spec.loader.exec_module(mod)          # @hawk.kernel traces AND canonicalises
    digest = mod.chain.walk.digest
    print(json.dumps({{"wall_s": time.time() - t0, "rss_mib": _rss(),
                      "digest": digest, "nodes": len(mod.chain.walk.order)}}))
'''


# --------------------------------------------------------------------------- #
# row 1: the memory row.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("depth", [12, 14])
def test_a_deep_loop_body_canonicalises_inside_two_gigabytes(depth, tmp_path):
    """THE row. A 4-trip ``for`` whose body is a ``depth``-long multiply
    chain — 20-odd nodes — must trace AND canonicalise in a fraction of a second
    inside a 2 GB address space.

    Both points were a ``MemoryError`` under this exact cap before the fix (see
    the module table): the legacy key of ``t = t * t`` embedded BOTH children's
    key text, so the key doubled once per line of the chain and depth 12 alone
    wanted 9 GB. The thresholds are deliberately far from both arms — a fixed
    width key makes this body cost tens of milliseconds and tens of megabytes,
    and the legacy one could not finish at any wall-clock budget — so the row is
    about which SIDE of the gap the implementation is on, not about a margin."""
    got = _child(_CHAIN_PROBE.format(depth=depth, chain_dir=str(tmp_path / "chain")),
                CHAIN_CAP_KIB, tmp_path)
    assert got["wall_s"] < 2.0, got
    assert got["rss_mib"] < 200, got
    assert got["nodes"] > 0 and len(got["digest"]) == 64


# --------------------------------------------------------------------------- #
# row 2: the key-width row.
# --------------------------------------------------------------------------- #
def _all_body_widths(loop: Loop) -> tuple:
    """Every key width in ``loop``'s body, nested loops included — the widths
    the digest itself built, read back through
    :func:`hawk.ir.loop_nodes.body_key_widths` rather than recomputed here."""
    widths = list(body_key_widths(loop))
    for node in loop.body_nodes():
        if isinstance(node, Loop):
            widths.extend(_all_body_widths(node))
    return tuple(widths)


KEY_WIDTH_CASES = [
    ("kan_edge_cell_64x20x3", lambda: kan_edge_cell(64, 20, 3)),
    ("kan_edge_cell_8x5x2", lambda: kan_edge_cell(8, 5, 2)),
    ("card_shallow_cell_64x23", lambda: shallow_kan_cell(64, 23)),
]


@pytest.mark.parametrize("case", KEY_WIDTH_CASES, ids=[c[0] for c in KEY_WIDTH_CASES])
def test_no_body_key_is_wider_than_a_fixed_hash(case):
    """The structural claim, stated as a number a reader can check: NO single
    body key exceeds 128 bytes, at any body size.

    This is the observable the defect hid behind. A key that embeds its
    children's key text is invisible from every other angle — the digest is the
    same 64 hex characters, the walk returns the same nodes, the emitted body is
    byte-identical — so only the SIZE of the intermediate keys distinguishes an
    O(n) digest from an O(2**depth) one. Measured before the fix: 106 505 567
    bytes for the largest key of the 8x5x2 cell, and the 64x20x3 cell could not
    be keyed inside 6 GB at all.

    The three cases are the deep KAN cell at the size was ruled over, the
    same cell at the size that fits in memory both ways (so the row still has a
    subject if the largest one is ever removed), and the card's own shallow
    two-loop family."""
    _id, build = case
    kernel = build()
    loops = [n for n in kernel.walk.order if isinstance(n, Loop)]
    assert loops, "the subject must contain a lowered loop"
    widths = tuple(w for loop in loops for w in _all_body_widths(loop))
    assert widths, "the subject's loop bodies must have nodes to key"
    assert max(widths) <= MAX_KEY_BYTES, (
        f"the widest body key is {max(widths)} bytes over {len(widths)} keyed "
        "nodes; a body key must be a FIXED-WIDTH hash of its children's keys "
        "never their text")
    assert set(widths) == {BODY_KEY_HEX}, sorted(set(widths))


def test_the_body_key_cost_does_not_track_the_body_size():
    """The same claim as a RATIO rather than a bound: the deep cell at
    (64, 20, 3) has four times the body nodes of the one at (8, 5, 2), so its
    total keying cost must be about four times as large — LINEAR. Under the
    legacy keying the same pair differed by more than the memory of the machine
    (213 083 226 bytes against a total that never completed)."""
    small = kan_edge_cell(8, 5, 2)
    large = kan_edge_cell(64, 20, 3)

    def total(kernel):
        loops = [n for n in kernel.walk.order if isinstance(n, Loop)]
        widths = tuple(w for loop in loops for w in _all_body_widths(loop))
        return len(widths), sum(widths)

    n_small, b_small = total(small)
    n_large, b_large = total(large)
    assert n_large > 3 * n_small, (n_small, n_large)
    assert b_large / b_small == pytest.approx(n_large / n_small, rel=1e-12), (
        "total key bytes must be exactly proportional to the node count — that "
        "is what a fixed-width key means")


# --------------------------------------------------------------------------- #
# row 3: the properties the fix was not allowed to cost.
# --------------------------------------------------------------------------- #
_DIGEST_PROBE = '''
    import hawk.math as m
    import hawk.trace as T
    import hawk

    @hawk.kernel
    def cell(t: hawk.Table["row":8], g: hawk.Param, y: hawk.Mutable[hawk.Scalar]):
        acc = 0.0
        for r in range(8):
            acc = acc + m.tanh(g * t.at(row=r)) - m.sin(t.at(row=r) / g)
        y = acc

    print(json.dumps({"digest": cell.walk.digest, "rss_mib": _rss()}))
'''


def test_the_digest_is_byte_identical_in_a_SECOND_process(tmp_path):
    """The one way a "fixed-width key" can be quietly wrong: a key
    hashed with Python's own ``hash()`` would be short, structural, positional
    — and DIFFERENT in every process, because ``PYTHONHASHSEED`` randomises the
    hash of a string. So the digest of one kernel is computed HERE and again in
    a fresh interpreter, and the two must be the same bytes."""
    @hawk.kernel
    def cell(t: hawk.Table["row":8], g: hawk.Param, y: hawk.Mutable[hawk.Scalar]):
        acc = 0.0
        for r in range(8):
            acc = acc + m.tanh(g * t.at(row=r)) - m.sin(t.at(row=r) / g)
        y = acc

    got = _child(_DIGEST_PROBE, CHAIN_CAP_KIB, tmp_path)
    assert got["digest"] == cell.walk.digest


def test_swapping_two_children_inside_the_body_changes_the_digest():
    """POSITIONAL, still. A Merkle key over an ORDERED tuple of child keys must
    separate ``a - b`` from ``b - a`` inside a loop body; a key that hashed an
    unordered set of children would pass every other row in this file and merge
    two different kernels into one cache slot."""
    def cell(swap: bool):
        @hawk.kernel
        def k(t: hawk.Table["row":4], g: hawk.Param, y: hawk.Mutable[hawk.Scalar]):
            acc = 0.0
            for r in range(4):
                a = t.at(row=r)
                b = g * 2.0
                acc = acc + (b - a if swap else a - b)
            y = acc
        return k

    assert cell(False).walk.digest != cell(True).walk.digest
    assert cell(False).walk.digest == cell(False).walk.digest


def test_two_structurally_equal_bodies_still_dedup_to_one_loop():
    """The property the digest exists FOR, unchanged by the fix: two loops with
    the same header, the same boundary operands and structurally equal bodies
    are ONE node in the canonical order. If the fixed-width key had lost
    any structure, this would silently become two."""
    @hawk.kernel
    def twice(t: hawk.Table["row":4], g: hawk.Param, y: hawk.Mutable[hawk.Scalar]):
        acc = 0.0
        for r in range(4):
            acc = acc + g * t.at(row=r)
        first = acc
        acc = 0.0
        for r in range(4):
            acc = acc + g * t.at(row=r)
        y = first + acc

    loops = [n for n in twice.walk.order if isinstance(n, Loop)]
    assert len(loops) == 1, (
        "two structurally equal loop bodies must dedup to one node; got "
        f"{len(loops)}")
