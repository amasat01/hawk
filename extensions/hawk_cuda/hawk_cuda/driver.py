# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The device compile facts: the ptxas report parser and the AOT target choice."""

from __future__ import annotations

import re

from . import nvrtc as _nvrtc_driver
from . import toolchain as dtc

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


def _device_target(compiler: str, arch: str) -> str:
    """``"cubin"`` or ``"ptx"`` for an AOT nvcc device compile at the
    resolved ``arch``. A virtual ``compute_<N>`` arch (the arch policy's
    no-GPU and above-the-toolkit answers) is always PTX
    (:func:`hawk_cuda.nvrtc.forced_target`); otherwise the device
    artifact that loads on the driver present
    (:func:`hawk_cuda.nvrtc.select_target`), compared against nvcc's own
    parsed version rather than NVRTC's."""
    forced = _nvrtc_driver.forced_target(arch)
    if forced is not None:
        return forced
    return _nvrtc_driver.select_target(
        dtc.nvcc_version(compiler),
        _nvrtc_driver._probed(_nvrtc_driver.driver_cuda_version))
