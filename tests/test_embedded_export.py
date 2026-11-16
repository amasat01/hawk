# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The standalone C target's export door and its traceability.

Rows:

* **files** — :func:`hawk.embedded.export` writes exactly ``<name>.c``,
  ``<name>.h`` and ``<name>.embedded.json``;
* **report** — the report's keys come in the spec's order, its traceability
  stamps (hawk version, walk digest, kernel source digest) agree with the
  header's TRACEABILITY block, and it records the toolchain deviations;
* **digest** — ``Kernel.source_digest`` is the sha256 of the dedented
  source of the traced function, carried through ``hawk.steps`` and
  ``resolve_dispatch``; a hand-built ``Kernel`` reports ``"unavailable"``;
* **determinism** — a re-export is byte-identical, with and without the
  toolchain checks (no timestamps, no temp paths, no durations);
* **checks** — ``check=True`` runs the C compiler and cppcheck and records a
  zero-finding cppcheck block.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import textwrap

import _kernels as K
import pytest

import hawk
from hawk import Index, Int32, Mutable, Param, Scalar, Vector
from hawk.math import norm, where
from hawk.trace.kernel import Kernel

#: The report's keys in the spec's order (``int_division_sites`` and
#: ``int_ub_sites`` sit where the verify pass puts them, after
#: ``fp_equality_sites``).
REPORT_KEYS = [
    "kernel", "generator", "hawk_version", "walk_digest", "source_digest",
    "scalar_mode", "profile", "interface", "budgets", "indices", "fp_domain",
    "fp_equality_sites", "int_division_sites", "int_ub_sites", "literal_specials",
    "libm",
    "helpers", "core", "core_version", "deviations", "unsupported_checked",
    "audit", "checks",
]


def _probe(state: Vector[3], rho: Param, out: Mutable[Vector[3]]):
    out = -rho * state * norm(state)

PROBE = hawk.kernel(_probe)


def _sha(fn) -> str:
    return hashlib.sha256(textwrap.dedent(inspect.getsource(fn)).encode()).hexdigest()


def _drag_source() -> str:
    """``K.drag``'s decorated definition, as ``inspect.getsource`` returns it
    (a :class:`Kernel` does not hold the traced function)."""
    text = inspect.getsource(K)
    m = re.search(r"^@kernel\ndef drag\(.*?(?=\n\n)", text, re.S | re.M)
    return m.group(0) + "\n"


# --------------------------------------------------------------------------- #
# files and report
# --------------------------------------------------------------------------- #


@hawk.kernel
def float_equal(x: Scalar, z: Scalar, y: Mutable[Scalar]):
    y = where(x == z, x, z)


@hawk.kernel
def index_add(a: Index, b: Index, y: Mutable[Index]):
    y = a + b


@hawk.kernel
def int32_add(a: Int32, b: Int32, y: Mutable[Int32]):
    y = a + b


# --------------------------------------------------------------------------- #
# the kernel source digest
# --------------------------------------------------------------------------- #
def test_source_digest_is_the_sha256_of_the_dedented_source():
    assert PROBE.source_digest == _sha(_probe)
    assert K.drag.source_digest == hashlib.sha256(_drag_source().encode()).hexdigest()


def test_the_digest_rides_through_derived_kernels():
    import test_finish_own_sample as F

    from hawk.trace.steps import steps

    four = steps(F.osc_finish, 4)
    assert F.osc_finish.source_digest is not None
    assert four.source_digest == F.osc_finish.source_digest


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def _bytes(directory):
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir())}


