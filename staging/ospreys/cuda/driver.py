# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The device compile facts: the ptxas report parser."""

from __future__ import annotations

import re

_PTXAS_ENTRY = re.compile(r"Compiling entry function '([^']+)'")


_PTXAS_FRAME = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")


_PTXAS_REGS = re.compile(r"Used (\d+) registers")


def parse_ptxas_info(stderr: str) -> dict:
    """``{kernel: {regs, spill_store_B, spill_load_B, stack_B}}`` from the
    ``ptxas info`` lines of an nvcc ``-Xptxas -v`` stderr; ``{}`` when it carries
    none. A kernel compiled for several archs keeps the largest figures."""
    found: dict = {}
    entry = None
    for line in stderr.splitlines():
        if (m := _PTXAS_ENTRY.search(line)):
            entry = found.setdefault(m.group(1), {})
        elif entry is not None and (m := _PTXAS_FRAME.search(line)):
            _grow(entry, stack_B=m.group(1), spill_store_B=m.group(2),
                  spill_load_B=m.group(3))
        elif entry is not None and (m := _PTXAS_REGS.search(line)):
            _grow(entry, regs=m.group(1))
    return found


def _grow(entry: dict, **values: str) -> None:
    for key, value in values.items():
        entry[key] = max(entry.get(key, 0), int(value))

