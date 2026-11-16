# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""What a banded build checks of its own output: no FP64 anywhere in it, and no register spills on the device.

The FP64 check reads the device object (the SASS of a CUBIN, else the opcodes of PTX) and refuses by name; it is held RED
against a planted FP64, so a check that finds nothing is not a blind one."""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest
from conftest import sidecar_of

import hawk
import hawk.math as m
from hawk.artifact import build_bundle
from ospreys.cuda.backend import _refuse_fp64
from raptor.errors import RaptorError as HawkError


@hawk.kernel
def kq(x: hawk.Scalar, y: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    t = x * x * 0.5 + y
    out = m.exp(t) / (y + 1.5) - m.sin(x) * 0.25


@hawk.kernel
def stepping(x: hawk.Scalar, out: hawk.Mutable[hawk.Scalar]):
    acc = x
    for _k in range(16):
        acc = acc * 0.5 + m.sin(acc) * 0.25
    out = acc


def test_a_device_object_with_an_fp64_instruction_is_refused_by_name(tmp_path):
    ptx = tmp_path / "planted.ptx"
    ptx.write_text(".visible .entry k()\n{\n    add.f32 %f1, %f2, %f3;\n    add.f64 %fd1, %fd2, %fd3;\n}\n")
    with pytest.raises(HawkError, match=r"contains the FP64 PTX instruction 'add\.f64"):
        _refuse_fp64("k", "cuda", ptx, "ptx")
    clean = tmp_path / "clean.ptx"
    clean.write_text(".visible .entry k()\n{\n    add.f32 %f1, %f2, %f3;\n}\n")
    _refuse_fp64("k", "cuda", clean, "ptx")


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", (kq, stepping), ids=lambda k: k.name)
def test_a_banded_device_build_has_no_fp64_and_no_spills(kernel, tmp_path):
    """The build itself refuses an FP64 instruction; the sidecar's ptxas record says the frame does not spill."""
    bundle = build_bundle([kernel], tmp_path / kernel.name, mode="banded", targets=("cuda",), host_profile="native")
    ptxas = sidecar_of(bundle, kernel.name).get("ptxas")
    if ptxas:
        for entry, row in ptxas.items():
            assert row["spill_store_B"] == 0 and not row["spills"], (entry, row)
        return
    # the device compile returned PTX: ptxas gives the frame the driver would build
    tool, ptx = shutil.which("ptxas"), next((tmp_path / kernel.name).rglob("*.ptx"))
    if tool is None:
        pytest.skip("PTX only and no ptxas on PATH: no spill figures to read")
    arch = re.search(r"^\.target (sm_\d+)", ptx.read_text(), re.M).group(1)
    log = subprocess.run([tool, "-v", f"-arch={arch}", str(ptx), "-o", os.devnull],
                         capture_output=True, text=True, check=True).stderr
    stores = [int(n) for n in re.findall(r"(\d+) bytes spill stores", log)]
    assert stores and set(stores) == {0}, log
