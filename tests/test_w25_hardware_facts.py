# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""One home per fact: a machine number (warp size, lane block, bank count, cache size, tile, SIMD width, thread count) is data of a
variant, never an UPPER_CASE constant in code. One test per piece; no list of names, no allow-list, no expected count: a piece that holds
such a constant is red and the message lists each ``file:line``. The scan reads the source and judges names only; arithmetic inside a
function (``32 * 1024``) cannot be judged statically and is NOT CHECKABLE here. The extensions are backends: a leaf may name its hardware,
so they are scanned for the number constants only. The scanner is tools/_hardware_facts.py (raptor's tests/test_hardware_free.py holds the same rule)."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import _hardware_facts as facts  # noqa: E402
import _w25_scan as S  # noqa: E402

PIECES = {"hawk": Path(importlib.util.find_spec("hawk").origin).resolve().parent}
PIECES |= {pid: root for pid, (root, _) in S.pieces().items() if pid.startswith(("wing:", "osprey:"))}


def test_the_pieces_are_found_by_discovery():
    assert {"hawk", "wing:raptor.base", "wing:raptor.simt", "osprey:c", "osprey:cuda"} <= set(PIECES)


@pytest.mark.parametrize("piece", sorted(PIECES))
def test_no_machine_number_constant(piece):
    root = PIECES[piece]
    hits = facts.piece_constants(root, root.parent)
    assert not hits, f"{piece}: {len(hits)} machine number constant(s) outside a variant:\n" + "\n".join(hits)


def test_the_scan_sees_a_module_and_a_class_constant_and_not_function_arithmetic(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("WARP_SIZE = 32\nclass K:\n    L1_BYTES = 32768\nOTHER = 5\n\ndef g():\n    return 32 * 1024\n")
    got = [h.split(": ", 1)[1] for h in facts.machine_constants(f, "m.py")]
    assert got == ["WARP_SIZE = 32", "L1_BYTES = 32768"], got
