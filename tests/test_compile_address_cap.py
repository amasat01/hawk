# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Every compiler subprocess runs under an address-space cap — a MACHINE
guard, checked by making a child actually hit it.

WHY THE GUARD EXISTS AND WHAT IT IS NOT. A real KAN cell with 64 edges and
23 basis functions reached ptxas as ONE function and took 25-27 GB of RSS on
a 31 GB box. The answer to that SIZE problem is loop lowering; this file is
about the other half: whatever HAWK emits, a pathological compile must fail
LOUDLY instead of taking the machine down. So ``hawk.compile.drivers``
installs ``RLIMIT_AS`` in the child (an exec trampoline, so it is thread-safe) before the compiler runs — inherited by everything
``nvcc`` spawns, ptxas included — and an over-large compile comes back as a
non-zero return code the driver already turns into a named refusal.

It is deliberately NOT a codegen policy. An earlier idea to cap the emission
budget directly was dropped for exactly that reason: a threshold in the
emitter decides what a kernel may cost, and it is the compiler, not the
emitter, that decides how to lower a loop.
This cap rejects no kernel, inspects no IR and has no opinion about op counts;
it is a ceiling on one subprocess, and ``$HAWK_COMPILE_ADDRESS_CAP`` (bytes; 0
disables) is how a machine with a different budget says so.

WRITTEN TO FAIL FIRST. Before ``hawk/compile/drivers.py`` applied any cap at
all, so :func:`test_a_child_under_the_cap_cannot_allocate_past_it` had nothing
to import (``ImportError: cannot import name 'limited'``) and
:func:`test_the_cap_reaches_the_real_driver` compiled happily under a 64 MB
setting. The plant for the GREEN direction is kept live below: every row that
asserts the cap BITES is paired with the same run under ``cap=0``, so a cap that
silently did nothing would fail the pair rather than pass the assertion.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from hawk.compile import drivers
from hawk.compile.drivers import CompileOptions, compile_source
from hawk.ir import HawkError

#: A cap small enough that one modest allocation crosses it, and an allocation
#: comfortably past it. Both are far below the real default, so the row costs
#: milliseconds and never approaches the machine's own memory.
SMALL_CAP = 256 * 1024 * 1024
OVERSHOOT = 512 * 1024 * 1024

_ALLOCATE = (
    "import sys\n"
    "try:\n"
    f"    b = bytearray({OVERSHOOT}); b[0] = 1\n"
    "except MemoryError:\n"
    "    sys.exit(9)\n"
    "sys.exit(0)\n"
)


def _child(cap: str | None) -> int:
    """Run a child that allocates :data:`OVERSHOOT` bytes under ``cap``."""
    held = os.environ.get("HAWK_COMPILE_ADDRESS_CAP")
    if cap is None:
        os.environ.pop("HAWK_COMPILE_ADDRESS_CAP", None)
    else:
        os.environ["HAWK_COMPILE_ADDRESS_CAP"] = cap
    try:
        done = subprocess.run(drivers.limited([sys.executable, "-c", _ALLOCATE]),
                              capture_output=True)
        return done.returncode
    finally:
        if held is None:
            os.environ.pop("HAWK_COMPILE_ADDRESS_CAP", None)
        else:
            os.environ["HAWK_COMPILE_ADDRESS_CAP"] = held


def test_a_child_under_the_cap_cannot_allocate_past_it():
    """The mechanism, with its own negative control beside it. The SAME child,
    the SAME allocation: it fails under the cap and succeeds without one, so a
    cap that had quietly stopped applying would fail this row rather
    than pass it by doing nothing."""
    assert _child(str(SMALL_CAP)) != 0, (
        "a child under a 256 MB address-space cap allocated 512 MB — the "
        "RLIMIT_AS the driver installs is not reaching the child")
    assert _child("0") == 0, (
        "with the cap disabled the same allocation must succeed; if it does "
        "not, this row is measuring the machine and not the cap")


def test_the_cap_is_read_from_the_environment_and_validated():
    held = os.environ.get("HAWK_COMPILE_ADDRESS_CAP")
    try:
        os.environ.pop("HAWK_COMPILE_ADDRESS_CAP", None)
        assert drivers.address_space_cap() == drivers.ADDRESS_SPACE_CAP_BYTES
        os.environ["HAWK_COMPILE_ADDRESS_CAP"] = "0"
        assert drivers.address_space_cap() == 0
        assert drivers.limited(["x"]) == ["x"]
        os.environ["HAWK_COMPILE_ADDRESS_CAP"] = str(SMALL_CAP)
        assert drivers.address_space_cap() == SMALL_CAP
        assert drivers.limited(["x"])[-1] == "x" and len(drivers.limited(["x"])) > 1
        os.environ["HAWK_COMPILE_ADDRESS_CAP"] = "not-a-number"
        with pytest.raises(HawkError, match="not an integer"):
            drivers.address_space_cap()
        os.environ["HAWK_COMPILE_ADDRESS_CAP"] = "-1"
        with pytest.raises(HawkError, match="negative"):
            drivers.address_space_cap()
    finally:
        if held is None:
            os.environ.pop("HAWK_COMPILE_ADDRESS_CAP", None)
        else:
            os.environ["HAWK_COMPILE_ADDRESS_CAP"] = held


def test_the_cap_reaches_the_real_driver(tmp_path):
    """The INJECTION half: not that a flag is set, but that a real
    :func:`hawk.compile.drivers.compile_source` inherits it. Under a cap no
    compiler can start in, the compile must FAIL and the refusal must name the
    cap — otherwise the guard is a setting nobody applies."""
    source = "extern \"C\" int hawk_cap_probe(void) { return 41 + 1; }\n"
    held = os.environ.get("HAWK_COMPILE_ADDRESS_CAP")
    os.environ["HAWK_COMPILE_ADDRESS_CAP"] = str(4 * 1024 * 1024)
    try:
        with pytest.raises(HawkError) as excinfo:
            compile_source(source, "cap_probe", CompileOptions(
                backend="host", cache_dir=str(tmp_path / "capped")))
        assert "address-space cap" in str(excinfo.value), str(excinfo.value)
    finally:
        if held is None:
            os.environ.pop("HAWK_COMPILE_ADDRESS_CAP", None)
        else:
            os.environ["HAWK_COMPILE_ADDRESS_CAP"] = held
    # the positive control: the SAME source, the same driver, no cap -> it
    # compiles. Without this the row above could be passing because the source
    # is broken rather than because the cap bit.
    os.environ["HAWK_COMPILE_ADDRESS_CAP"] = "0"
    try:
        result = compile_source(source, "cap_probe", CompileOptions(
            backend="host", cache_dir=str(tmp_path / "free")))
        assert result.artifact.is_file()
    finally:
        if held is None:
            os.environ.pop("HAWK_COMPILE_ADDRESS_CAP", None)
        else:
            os.environ["HAWK_COMPILE_ADDRESS_CAP"] = held
