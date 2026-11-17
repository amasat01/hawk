# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The exact host profiles stay exact: the lowering library's host compile template for a profile carries what it says.

``native``, ``portable`` and ``native-vector-math-nofma`` contract nothing (``-ffp-contract=off``); the shipped default,
``native-vector-math``, lets the compiler fuse (``-ffp-contract=fast``). Flipping an exact entry makes the bit-for-bit
profiles differ from NumPy without any other row moving.
"""

from __future__ import annotations

import platform

import pytest
from _toolchain import host_template

EXACT = ("native", "portable", "native-vector-math-nofma")
DEFAULT = "native-vector-math"
X86 = platform.machine().lower() in ("x86_64", "amd64")


@pytest.mark.parametrize("profile", EXACT)
def test_an_exact_profile_compiles_with_contraction_off(profile):
    if profile == "native-vector-math-nofma" and not X86:
        pytest.skip("native-vector-math-nofma is x86-64 only")
    flags = host_template(profile)
    assert "-ffp-contract=off" in flags and not [f for f in flags if f.startswith("-ffp-contract=") and f != "-ffp-contract=off"], flags


def test_the_default_profile_compiles_with_contraction_fast():
    flags = host_template(DEFAULT)
    assert [f for f in flags if f.startswith("-ffp-contract=")] == ["-ffp-contract=fast"], flags
